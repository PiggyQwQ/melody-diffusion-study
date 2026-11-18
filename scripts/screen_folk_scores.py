"""Audit the ACFS release and extract clean runs of eight-bar 2/4 candidates."""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import file_digest, sha256
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


METADATA_ARCHIVE = "ACFS-v261103-metadata.zip"
METHOD_VERSION = "acfs-screen-v1"
BAR_SLOTS = 8
WINDOW_BARS = 8
PITCH_STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NOTE_SLOTS = {
    "whole": 16, "half": 8, "quarter": 4, "eighth": 2, "16th": 1,
    "32nd": Fraction(1, 2), "64th": Fraction(1, 4),
}
SONG_FIELDS = (
    "id", "volume", "title", "opening_meter", "meter_review_status", "xml_meters",
    "meter_evidence", "status", "source_bars", "usable_bars", "longest_usable_run",
    "candidate_windows", "issue_counts", "source_split_group", "clean_melody_hash",
    "pitch_reference", "musicxml", "midi", "preview",
)
ISSUE_FIELDS = ("id", "volume", "bar_index", "measure_number", "parsed_slots", "reasons")


@dataclass(frozen=True)
class Event:
    pitch: int | None
    slots: int
    tie_stop: bool
    tie_start: bool


@dataclass
class Measure:
    number: str
    slots: int | None
    events: list[Event | None]
    issues: set[str] = field(default_factory=set)


def parse_event(note, divisions):
    if note.find("chord") is not None:
        return None, "chord"
    if note.find("grace") is not None:
        return None, "grace_note"
    try:
        duration = int(note.findtext("duration", "0"))
    except ValueError:
        return None, "invalid_duration"
    if duration <= 0:
        return None, "nonpositive_duration"
    slots, remainder = divmod(4 * duration, divisions)
    if remainder:
        return None, "off_grid_duration"

    note_type = note.findtext("type")
    if note_type is not None and note.find("rest[@measure='yes']") is None:
        if note_type not in NOTE_SLOTS:
            return None, "unsupported_note_type"
        dots = len(note.findall("dot"))
        expected = NOTE_SLOTS[note_type] * Fraction(2 ** (dots + 1) - 1, 2 ** dots)
        modifier = note.find("time-modification")
        if modifier is not None:
            try:
                expected *= Fraction(
                    int(modifier.findtext("normal-notes", "0")),
                    int(modifier.findtext("actual-notes", "0")),
                )
            except (ValueError, ZeroDivisionError):
                return None, "invalid_tuplet_ratio"
        if slots != expected:
            return None, "notation_duration_mismatch"

    pitch = None
    if note.find("rest") is None:
        pitch_node = note.find("pitch")
        if pitch_node is None:
            return None, "missing_pitch"
        try:
            step = PITCH_STEPS[pitch_node.findtext("step")]
            octave = int(pitch_node.findtext("octave"))
            alteration = Fraction(pitch_node.findtext("alter", "0"))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None, "invalid_pitch"
        pitch = 12 * (octave + 1) + step + alteration
        if pitch.denominator != 1 or not 0 <= pitch <= 127:
            return None, "invalid_pitch"
        pitch = int(pitch)

    ties = {
        node.get("type") for node in note.findall("tie") + note.findall("notations/tied")
    }
    if not ties <= {"start", "stop", "continue"} or (ties and pitch is None):
        return None, "invalid_tie"
    return Event(
        pitch, slots, "stop" in ties or "continue" in ties,
        "start" in ties or "continue" in ties,
    ), None


def musicxml_meter(time):
    beats = time.findall("beats")
    beat_type = time.findall("beat-type")
    if len(beats) != 1 or len(beat_type) != 1:
        return "other"
    return f"{beats[0].text}/{beat_type[0].text}"


def read_measures(part):
    source_measures = part.findall("measure")
    declared = {musicxml_meter(time) for time in part.findall("./measure/attributes/time")}
    inferred = declared == {"4/4"}
    divisions = 0
    active_meter = ""
    measures = []
    for index, source in enumerate(source_measures, 1):
        timings = source.findall("attributes/time")
        if timings:
            active_meter = musicxml_meter(timings[-1])
        division_values = source.findall("attributes/divisions")
        if division_values:
            try:
                divisions = int(division_values[-1].text)
            except (TypeError, ValueError):
                divisions = 0
        notes = source.findall("note")
        issues = set()
        if not inferred and active_meter != "2/4":
            issues.add("no_2_4_meter_at_bar")
        if divisions <= 0 or len(division_values) > 1:
            issues.add("invalid_divisions")
        if len(timings) > 1:
            issues.add("meter_change_in_measure")
        voices = {note.findtext("voice", "1") for note in notes}
        staves = {note.findtext("staff", "1") for note in notes}
        if len(voices) > 1 or len(staves) > 1 or any(
            node.tag in {"backup", "forward"} for node in source
        ):
            issues.add("multiple_voices_or_cursor_shift")

        events = []
        if divisions > 0:
            for note in notes:
                event, issue = parse_event(note, divisions)
                events.append(event)
                if issue:
                    issues.add(issue)
        slots = None
        if divisions > 0 and None not in events:
            slots = sum(event.slots for event in events)
            if slots < BAR_SLOTS:
                issues.add("short_bar")
            elif slots > BAR_SLOTS:
                issues.add("long_bar")
        measures.append(Measure(source.get("number", str(index)), slots, events, issues))

    # In this release a tie is represented on both adjacent same-pitch notes.
    timeline = [
        (measure, event)
        for measure in measures
        for event in (measure.events or [None])
    ]
    for i, (measure, event) in enumerate(timeline):
        if event is None:
            continue
        previous = timeline[i - 1][1] if i else None
        following = timeline[i + 1][1] if i + 1 < len(timeline) else None
        if event.tie_stop and not (
            previous and previous.tie_start and previous.pitch == event.pitch
        ):
            measure.issues.add("unmatched_tie_stop")
        if event.tie_start and not (
            following and following.tie_stop and following.pitch == event.pitch
        ):
            measure.issues.add("unmatched_tie_start")
    return measures, sorted(declared), inferred


def clean_runs(measures):
    runs = []
    begin = 0
    for end in range(len(measures) + 1):
        if end < len(measures) and not measures[end].issues:
            continue
        if end > begin:
            runs.append((begin + 1, measures[begin:end]))
        begin = end + 1
    return runs


def candidate_windows(first_bar, measures):
    starts = []
    for index in range(len(measures) - WINDOW_BARS + 1):
        window = measures[index:index + WINDOW_BARS]
        if window[0].events[0].tie_stop or window[-1].events[-1].tie_start:
            continue
        if any(event.pitch is not None for measure in window for event in measure.events):
            starts.append(first_bar + index)
    return starts


def relative_melody_hash(runs):
    # Use all surviving runs to flag exact matches under global transposition.
    origin = next(
        event.pitch
        for run in runs for measure in run["bars"] for event in measure.events
        if event.pitch is not None
    )
    relative = [
        [
            [
                [None if event.pitch is None else event.pitch - origin,
                 event.slots, event.tie_stop, event.tie_start]
                for event in measure.events
            ]
            for measure in run["bars"]
        ]
        for run in runs
    ]
    return sha256(json.dumps(relative, separators=(",", ":")).encode()).hexdigest()


def screen(record, archive):
    fields = record["fields"]
    meter = fields["meter"]["value"]
    assets = record["assets"]
    row = dict(
        id=record["id"], volume=record["volume_id"], title=record["title"],
        opening_meter=meter, meter_review_status=fields["meter"]["review_status"],
        pitch_reference=fields["tonic"]["value"],
        source_split_group=record["suggested_split_group"],
        **assets,
    )
    if record["empty_score"]:
        return {**row, "status": "empty_score"}, [], None
    if meter != "2/4":
        reason = "other_opening_meter" if meter else "unknown_opening_meter"
        return {**row, "status": reason}, [], None

    try:
        root = ET.fromstring(archive.read(assets["musicxml"]))
    except ET.ParseError:
        return {**row, "status": "invalid_musicxml"}, [], None
    parts = root.findall("part")
    if root.tag != "score-partwise" or len(parts) != 1:
        return {**row, "status": "unsupported_score_structure"}, [], None

    measures, declared, inferred = read_measures(parts[0])
    runs = clean_runs(measures)
    issue_counts = Counter()
    issues = []
    for index, measure in enumerate(measures, 1):
        if measure.issues:
            issue_counts.update(measure.issues)
            issues.append({
                "id": row["id"], "volume": row["volume"], "bar_index": index,
                "measure_number": measure.number, "parsed_slots": measure.slots,
                "reasons": "|".join(sorted(measure.issues)),
            })

    usable_runs = []
    for first, bars in runs:
        starts = candidate_windows(first, bars)
        if starts:
            usable_runs.append({"first_bar": first, "bars": bars, "window_starts": starts})
    windows = sum(len(run["window_starts"]) for run in usable_runs)
    longest = max((len(bars) for _, bars in runs), default=0)
    status = "kept"
    if not windows:
        status = "no_8_bar_run" if longest < WINDOW_BARS else "no_uncut_sounding_window"
    row.update(
        xml_meters="|".join(declared),
        meter_evidence="opening_metadata_and_bar_duration" if inferred else "xml_and_metadata",
        source_bars=len(measures), usable_bars=sum(len(bars) for _, bars in runs),
        longest_usable_run=longest, candidate_windows=windows,
        issue_counts=json.dumps(dict(sorted(issue_counts.items())), separators=(",", ":")),
        status=status,
    )
    if not usable_runs:
        return row, issues, None

    row["clean_melody_hash"] = relative_melody_hash(usable_runs)
    cleaned = {
        "id": row["id"], "volume": row["volume"], "meter": "2/4",
        "source_split_group": row["source_split_group"],
        "clean_melody_hash": row["clean_melody_hash"],
        "runs": [
            {
                "first_bar": run["first_bar"], "window_starts": run["window_starts"],
                "bars": [
                    {
                        "source_number": bar.number,
                        "events": [
                            [event.pitch, event.slots, event.tie_stop, event.tie_start]
                            for event in bar.events
                        ],
                    }
                    for bar in run["bars"]
                ],
            }
            for run in usable_runs
        ],
    }
    return row, issues, cleaned


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(archive_dir, output_dir):
    songs = []
    issues = []
    cleaned = []
    archive_hashes = {}
    seen_ids = set()
    with ZipFile(archive_dir / METADATA_ARCHIVE) as metadata:
        release = json.loads(metadata.read("release-assets.json"))
        index = json.loads(metadata.read("metadata/index.json"))
        statuses = {volume: Counter() for volume in index["volumes"]}
        counts = Counter()
        for package in release["packages"]:
            filename = package["filename"]
            archive_path = archive_dir / filename
            with archive_path.open("rb") as stream:
                digest = file_digest(stream, "sha256").hexdigest()
            if digest != package["sha256"] or archive_path.stat().st_size != package["bytes"]:
                raise ValueError(f"Release checksum or size mismatch: {filename}")
            archive_hashes[filename] = digest
            with ZipFile(archive_path) as scores:
                files = {item.filename for item in scores.infolist() if not item.is_dir()}
                if len(files) != package["archive_members"]:
                    raise ValueError(f"Release member count mismatch: {filename}")
                referenced_files = set()
                for volume in package["volumes"]:
                    info = index["volumes"][volume]
                    manifest = info["files"]["curated"]
                    data = metadata.read(manifest["path"])
                    if sha256(data).hexdigest() != manifest["sha256"]:
                        raise ValueError(f"Metadata checksum mismatch: {volume}")
                    records = data.splitlines()
                    if len(records) != info["records"]:
                        raise ValueError(f"Metadata record count mismatch: {volume}")
                    for line in records:
                        item = json.loads(line)
                        if item["id"] in seen_ids or item["volume_id"] != volume:
                            raise ValueError(f"Invalid song ID: {item['id']}")
                        seen_ids.add(item["id"])
                        referenced_files.update(path for path in item["assets"].values() if path)
                        row, bar_issues, melody = screen(item, scores)
                        songs.append(row)
                        issues.extend(bar_issues)
                        statuses[volume][row["status"]] += 1
                        counts[volume] += row.get("candidate_windows", 0)
                        if melody:
                            cleaned.append(melody)
                    empty = statuses[volume]["empty_score"]
                    if len(records) - empty != info["nonempty_scores"]:
                        raise ValueError(f"Nonempty score count mismatch: {volume}")
                if referenced_files != files:
                    raise ValueError(f"Archive inventory mismatch: {filename}")
            print(f"{package['province_id']}: checked {package['records']} records", flush=True)

    if len(songs) != index["records"] or any(not stats for stats in statuses.values()):
        raise ValueError("Incomplete release coverage")

    total_statuses = Counter()
    breakdown = []
    for volume, stats in statuses.items():
        total_statuses.update(stats)
        breakdown.append({
            "volume": volume, "release_records": stats.total(),
            "nonempty_scores": stats.total() - stats["empty_score"],
            "opening_2_4": stats.total() - sum(
                stats[key] for key in ("empty_score", "other_opening_meter", "unknown_opening_meter")
            ),
            "kept": stats["kept"], "excluded": stats.total() - stats["kept"],
            "candidate_windows": counts[volume],
        })

    fingerprints = Counter(row["clean_melody_hash"] for row in cleaned)
    summary = {
        "method_version": METHOD_VERSION,
        "dataset_version": index["dataset_version"], "score_archives_sha256": archive_hashes,
        "release_records": len(songs), "nonempty_scores": index["nonempty_scores"],
        "volumes": breakdown, "statuses": dict(sorted(total_statuses.items())),
        "candidate_windows": counts.total(), "measures_flagged": len(issues),
        "measure_issue_counts": dict(Counter(
            reason for issue in issues for reason in issue["reasons"].split("|")
        ).most_common()),
        "same_clean_melody_groups": sum(count > 1 for count in fingerprints.values()),
        "songs_in_same_clean_melody_groups": sum(
            count for count in fingerprints.values() if count > 1
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "song_audit.csv", SONG_FIELDS, songs)
    write_csv(output_dir / "measure_issues.csv", ISSUE_FIELDS, issues)
    write_csv(output_dir / "volume_summary.csv", tuple(breakdown[0]), breakdown)
    with (output_dir / "cleaned_runs.jsonl").open("w", encoding="utf-8") as stream:
        for row in cleaned:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Kept {len(cleaned)} songs with {counts.total()} candidate windows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    run(args.archive_dir, args.output_dir)
