"""Build a split, 64-token ACFS dataset from the audited screen-v1 windows."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from dataclasses import dataclass
from hashlib import file_digest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from folk_window_groups import SPLITS, assign_splits, fingerprint, group_songs, normalized_title, relative_notes
from screen_folk_scores import BAR_SLOTS, METHOD_VERSION as SCREEN_VERSION, WINDOW_BARS, write_csv


FORMAT_VERSION = "acfs-split-encode-v1"
DEFAULT_SEED = 20261118
WINDOW_SLOTS = BAR_SLOTS * WINDOW_BARS
REST = 0
HOLD = 1
PITCH_OFFSET = 2


@dataclass(frozen=True)
class Window:
    song_id: str
    start_bar: int
    bars: tuple
    spans: tuple

    @property
    def window_id(self):
        return f"{self.song_id}@{self.start_bar}"

    @property
    def events(self):
        return (event for bar in self.bars for event in bar["events"])


def source_spans(events):
    """Interpret the source as (slot, MIDI or None, duration) spans, joining ties."""
    spans = []
    position = 0
    pending_tie = False
    for pitch, duration, tie_stop, tie_start in events:
        if type(duration) is not int or duration <= 0:
            raise ValueError(f"Invalid duration at slot {position}: {duration}")
        if pitch is not None and (type(pitch) is not int or not 0 <= pitch <= 127):
            raise ValueError(f"Invalid MIDI pitch at slot {position}: {pitch}")
        if (pitch is None and (tie_stop or tie_start)) or tie_stop != pending_tie:
            raise ValueError(f"Unpaired tie at slot {position}")
        if pending_tie:
            start, previous_pitch, previous_duration = spans[-1]
            if pitch != previous_pitch:
                raise ValueError(f"Pitch changes inside a tie at slot {position}")
            spans[-1] = (start, pitch, previous_duration + duration)
        else:
            spans.append((position, pitch, duration))
        position += duration
        pending_tie = tie_start
    if pending_tie:
        raise ValueError("Window ends inside a tie")
    return spans


def song_windows(song):
    seen = set()
    for run in song["runs"]:
        bars = run["bars"]
        for start in run["window_starts"]:
            offset = start - run["first_bar"]
            if start < 1 or offset < 0 or offset + WINDOW_BARS > len(bars) or start in seen:
                raise ValueError(f"{song['id']}: invalid window start {start}")
            seen.add(start)
            excerpt = tuple(bars[offset:offset + WINDOW_BARS])
            if any(sum(event[1] for event in bar["events"]) != BAR_SLOTS for bar in excerpt):
                raise ValueError(f"{song['id']}@{start}: invalid bar length")
            spans = source_spans(event for bar in excerpt for event in bar["events"])
            if not any(pitch is not None for _, pitch, _ in spans):
                raise ValueError(f"{song['id']}@{start}: silent window")
            yield Window(song["id"], start, excerpt, tuple(spans))


def encode_events(events, midi_min, midi_max):
    tokens = []
    for pitch, duration, tie_stop, _ in events:
        if pitch is None:
            first = REST
        elif not midi_min <= pitch <= midi_max:
            raise ValueError(f"MIDI pitch outside vocabulary: {pitch}")
        else:
            first = HOLD if tie_stop else pitch - midi_min + PITCH_OFFSET
        tokens.append(first)
        tokens.extend([HOLD] * (duration - 1))
    return tokens


def decode(tokens, midi_min, midi_max):
    spans = []
    largest_id = midi_max - midi_min + PITCH_OFFSET
    for position, token in enumerate(tokens):
        if type(token) is not int or not 0 <= token <= largest_id:
            raise ValueError(f"Invalid token at slot {position}: {token}")
        if token == HOLD:
            if not spans:
                raise ValueError("Initial HOLD has no previous state")
            start, pitch, duration = spans[-1]
            spans[-1] = (start, pitch, duration + 1)
        else:
            pitch = None if token == REST else midi_min + token - PITCH_OFFSET
            spans.append((position, pitch, 1))
    return spans


def synthetic_tie_fixture(midi_min, midi_max):
    # Explicit reference for a tied note that crosses a barline and is then
    # immediately rearticulated at the same pitch.
    lower = (midi_min + midi_max) // 2
    upper = min(lower + 4, midi_max)
    bars = [
        [(None, 2, False, False), (lower, 2, False, False), (upper, 4, False, True)],
        [(upper, 2, True, False), (upper, 2, False, False), (None, 4, False, False)],
    ] + [[(None, BAR_SLOTS, False, False)] for _ in range(WINDOW_BARS - 2)]
    events = [event for bar in bars for event in bar]
    tokens = encode_events(events, midi_min, midi_max)
    spans = decode(tokens, midi_min, midi_max)
    expected = [(0, None, 2), (2, lower, 2), (4, upper, 6), (10, upper, 2), (12, None, 4)]
    expected.extend((start, None, BAR_SLOTS) for start in range(2 * BAR_SLOTS, WINDOW_SLOTS, BAR_SLOTS))
    if spans != expected or source_spans(events) != expected or tokens[BAR_SLOTS] != HOLD:
        raise ValueError("Synthetic cross-bar tie did not round trip")
    return dict(source="synthetic fixture", source_bars=bars, tokens=tokens, decoded_spans=spans)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def read_screen(screen_dir):
    summary = json.loads((screen_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["method_version"] != SCREEN_VERSION:
        raise ValueError(f"Expected {SCREEN_VERSION}, got {summary['method_version']}")
    with (screen_dir / "song_audit.csv").open(encoding="utf-8-sig", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    songs = {row["id"]: row for row in all_rows if row["status"] == "kept"}
    if (len({row["id"] for row in all_rows}) != summary["release_records"]
            or len(songs) != summary["statuses"]["kept"]):
        raise ValueError("Song audit counts differ from the screen summary")

    windows = []
    seen = set()
    for song in read_jsonl(screen_dir / "cleaned_runs.jsonl"):
        song_id = song["id"]
        audited = songs[song_id]
        if song_id in seen or song["meter"] != "2/4" or audited["opening_meter"] != "2/4":
            raise ValueError(f"Unexpected ID or time signature: {song_id}")
        for field in ("volume", "source_split_group", "clean_melody_hash"):
            if song[field] != audited[field]:
                raise ValueError(f"{song_id}: {field} differs from song audit")
        candidates = list(song_windows(song))
        if len(candidates) != int(audited["candidate_windows"]):
            raise ValueError(f"{song_id}: candidate window count differs from song audit")
        windows.extend(candidates)
        seen.add(song_id)
    if seen != songs.keys() or len(windows) != summary["candidate_windows"]:
        raise ValueError("Cleaned runs do not cover the complete screened corpus")
    return dict(sorted(songs.items())), sorted(windows, key=lambda w: (w.song_id, w.start_bar)), summary


def window_metadata(window, song, group_id, split):
    return dict(window_id=window.window_id, song_id=window.song_id, volume=song["volume"],
                group_id=group_id, split=split, start_bar=window.start_bar)


def audit_vocab(windows, memberships, midi_min, midi_max):
    pitches = {split: Counter() for split in SPLITS}
    excluded = {}
    affected = {split: set() for split in SPLITS}
    for window in windows:
        split = memberships[window.song_id]
        notes = Counter(pitch for _, pitch, _ in window.spans if pitch is not None)
        pitches[split].update(notes)
        outside = {pitch: count for pitch, count in notes.items() if not midi_min <= pitch <= midi_max}
        if outside:
            excluded[window.window_id] = outside
            affected[split].add(window.song_id)

    stats = {}
    for split, counts in pitches.items():
        below = {pitch: counts[pitch] for pitch in sorted(counts) if pitch < midi_min}
        above = {pitch: counts[pitch] for pitch in sorted(counts) if pitch > midi_max}
        stats[split] = dict(observed_midi_min=min(counts), observed_midi_max=max(counts),
                            note_onsets=counts.total(), below=below, above=above,
                            out_of_range_onsets=sum(below.values()) + sum(above.values()),
                            affected_songs=len(affected[split]))
    return stats, excluded


def verify_saved_sequences(output, windows, songs, groups, memberships, excluded, midi_min, midi_max):
    per_split = {split: [] for split in SPLITS}
    for window in windows:
        if window.window_id not in excluded:
            per_split[memberships[window.song_id]].append(window)
    counts = {}
    seen = {}
    for split, expected in per_split.items():
        symbol_counts = Counter()
        for window, row in zip(expected, read_jsonl(output / f"{split}.jsonl"), strict=True):
            metadata = window_metadata(window, songs[window.song_id], groups[window.song_id], split)
            tokens = row.pop("tokens")
            spans = decode(tokens, midi_min, midi_max)
            if row != metadata or len(tokens) != WINDOW_SLOTS or spans != list(window.spans):
                raise ValueError(f"Persisted data failed round trip: {window.window_id}")
            key = relative_notes(spans)
            if seen.setdefault(key, split) != split:
                raise ValueError(f"Cross-split exact transposed match: {window.window_id}")
            symbol_counts.update(tokens)
        counts[split] = dict(sorted(symbol_counts.items()))
    return counts


def build_report(screen_dir, summary, seed, windows, songs, memberships, groups, exact,
                 edges, review, ranges, excluded, symbols, samples, midi_min, midi_max,
                 real_tie_windows, synthetic):
    split_stats = {}
    for split in SPLITS:
        assigned = [s for s in songs if memberships[s] == split]
        candidates = [w for w in windows if memberships[w.song_id] == split]
        encoded = [w for w in candidates if w.window_id not in excluded]
        split_stats[split] = dict(songs=len(assigned), encoded_songs=len({w.song_id for w in encoded}),
                                 groups=len({groups[s] for s in assigned}), candidate_windows=len(candidates),
                                 encoded_windows=len(encoded), excluded_windows=len(candidates) - len(encoded),
                                 tokens=len(encoded) * WINDOW_SLOTS, token_counts=symbols[split], **ranges[split])
    groups_size = Counter(groups.values())
    inputs = {}
    for name in ("cleaned_runs.jsonl", "song_audit.csv", "summary.json"):
        with (screen_dir / name).open("rb") as stream:
            inputs[name] = file_digest(stream, "sha256").hexdigest()
    return dict(
        format_version=FORMAT_VERSION, dataset_version=summary["dataset_version"],
        screen_version=summary["method_version"], screen_input_sha256=inputs,
        screen_verified_score_archives_sha256=summary["score_archives_sha256"], seed=seed,
        target_split_fractions=SPLITS, songs=len(songs), candidate_windows=len(windows),
        groups=len(groups_size), largest_group=max(groups_size.values()),
        group_size_histogram=dict(sorted(Counter(groups_size.values()).items())),
        grouping_evidence_edges=dict(Counter(edge["reason"] for edge in edges)),
        exact_multi_song_window_fingerprints=len(exact),
        cross_split_exact_fingerprints_before_oov=0, cross_split_exact_fingerprints_after_encoding=0,
        near_song_pairs=len(review), cross_split_near_song_pairs=sum(r["cross_split"] for r in review),
        pitch_coordinate="MusicXML MIDI; no transposition", midi_min=midi_min, midi_max=midi_max,
        vocabulary_size=midi_max - midi_min + PITCH_OFFSET + 1,
        out_of_range_policy="exclude entire affected window; preserve original pitches and song assignment",
        splits=split_stats, sampled_songs=len(samples), sampled_volumes=len({r["volume"] for r in samples}),
        round_trip_samples=len(samples), round_trip_exported_windows=sum(s["encoded_windows"] for s in split_stats.values()),
        sample_feature_counts=dict(Counter(f for r in samples for f in r["features"].split("|") if f)),
        real_cross_bar_tie_windows=real_tie_windows, synthetic_cross_bar_tie_fixture=bool(synthetic),
    )


def build(screen_dir, output, seed):
    from folk_window_samples import export_samples, select_samples

    songs, windows, summary = read_screen(screen_dir)
    print(f"Auditing {len(windows)} approved windows from {len(songs)} songs", flush=True)
    exact = defaultdict(list)
    for window in windows:
        exact[relative_notes(window.spans)].append(window)
    exact = {key: copies for key, copies in exact.items() if len({w.song_id for w in copies}) > 1}
    groups, edges, near, variants = group_songs(songs, windows, exact)
    memberships = assign_splits(songs, groups, seed)
    if any(memberships[e["song_a"]] != memberships[e["song_b"]] for e in edges):
        raise ValueError("Grouping constraint crossed a split boundary")
    repeats = []
    for key, copies in exact.items():
        if len({memberships[w.song_id] for w in copies}) != 1:
            raise ValueError("Exact transposed match crossed a split boundary")
        for window in copies:
            repeats.append({"fingerprint": fingerprint(key), "first_midi": next(
                pitch for _, pitch, _ in window.spans if pitch is not None),
                **window_metadata(window, songs[window.song_id], groups[window.song_id], memberships[window.song_id])})

    review = []
    for (_, a, b, match) in near.values():
        sa, sb = a.song_id, b.song_id
        ta, tb = songs[sa]["title"], songs[sb]["title"]
        review.append(dict(
            song_a=sa, title_a=ta, start_bar_a=a.start_bar, split_a=memberships[sa], group_a=groups[sa],
            song_b=sb, title_b=tb, start_bar_b=b.start_bar, split_b=memberships[sb], group_b=groups[sb],
            shift_b_minus_a=match.shift, dice=round(match.dice, 6), matched_onsets=match.matches,
            onsets_a=match.note_count_a, onsets_b=match.note_count_b, matched_pitches=match.matched_pitch_count,
            same_normalized_title=bool(ta and normalized_title(ta) == normalized_title(tb)),
            cross_split=memberships[sa] != memberships[sb],
            status=("grouped_as_variant" if (sa, sb) in variants else
                    "already_same_group" if groups[sa] == groups[sb] else "review"),
        ))
    review.sort(key=lambda r: (not r["cross_split"], -r["dice"], r["song_a"], r["song_b"]))
    observed_train = sorted({pitch for window in windows if memberships[window.song_id] == "train"
                             for _, pitch, _ in window.spans if pitch is not None})
    midi_min, midi_max = observed_train[0], observed_train[-1]
    ranges, excluded = audit_vocab(windows, memberships, midi_min, midi_max)
    candidates = Counter(memberships[w.song_id] for w in windows)
    print(f"Grouped split fixed: {dict(candidates)}; training MIDI {midi_min}..{midi_max}; "
          f"{len(excluded)} windows outside the vocabulary", flush=True)

    real_tie_windows = sum(any(bar["events"][0][2] for bar in w.bars[1:]) for w in windows)
    selections = []
    for window, features in select_samples(windows, songs, groups, memberships, excluded, seed):
        tokens = encode_events(window.events, midi_min, midi_max)
        spans = decode(tokens, midi_min, midi_max)
        if len(tokens) != WINDOW_SLOTS or spans != list(window.spans):
            raise ValueError(f"Sample failed round trip: {window.window_id}")
        song_id = window.song_id
        metadata = window_metadata(window, songs[song_id], groups[song_id], memberships[song_id])
        selections.append((window, tokens, spans, features, metadata))
    synthetic = (None if any("cross_bar_tie" in selection[3] for selection in selections)
                 else synthetic_tie_fixture(midi_min, midi_max))
    samples = export_samples(output / "samples", screen_dir.parent, selections, songs, source_spans, synthetic)
    print(f"Verified {len(samples)} replay samples; encoding all eligible windows", flush=True)
    rejected_counts = Counter()
    rejected_rows = []
    with ExitStack() as stack:
        streams = {s: stack.enter_context((output / f"{s}.jsonl").open("w", encoding="utf-8", newline="\n"))
                   for s in SPLITS}
        for window in windows:
            song_id = window.song_id
            split = memberships[song_id]
            metadata = window_metadata(window, songs[song_id], groups[song_id], split)
            if window.window_id in excluded:
                rejected_counts[song_id] += 1
                rejected_rows.append({**metadata, "midi_onset_counts": json.dumps(excluded[window.window_id])})
                continue
            metadata["tokens"] = encode_events(window.events, midi_min, midi_max)
            streams[split].write(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n")
    symbols = verify_saved_sequences(output, windows, songs, groups, memberships, excluded, midi_min, midi_max)

    song_manifest = [dict(song_id=song_id, volume=song["volume"], title=song["title"],
                          group_id=groups[song_id], split=memberships[song_id],
                          source_split_group=song["source_split_group"], clean_melody_hash=song["clean_melody_hash"],
                          pitch_reference=song["pitch_reference"], musicxml=song["musicxml"],
                          candidate_windows=int(song["candidate_windows"]), excluded_windows=rejected_counts[song_id],
                          encoded_windows=int(song["candidate_windows"]) - rejected_counts[song_id])
                     for song_id, song in songs.items()]
    meta_fields = ("window_id", "song_id", "volume", "group_id", "split", "start_bar")
    write_csv(output / "songs.csv", tuple(song_manifest[0]), song_manifest)
    by_volume = defaultdict(Counter)
    for row in song_manifest:
        counts = by_volume[(row["volume"], row["split"])]
        counts["songs"] += 1
        for field in ("candidate_windows", "encoded_windows", "excluded_windows"):
            counts[field] += row[field]
    volume_rows = [dict(volume=volume, split=split, **counts)
                   for (volume, split), counts in sorted(by_volume.items())]
    write_csv(output / "volume_split.csv", tuple(volume_rows[0]), volume_rows)
    write_csv(output / "group_edges.csv", ("song_a", "song_b", "reason", "evidence"), edges)
    write_csv(output / "exact_repeats.csv", ("fingerprint", "first_midi", *meta_fields), repeats)
    write_csv(output / "near_repeats_review.csv", REVIEW_FIELDS, review)
    write_csv(output / "out_of_range.csv", (*meta_fields, "midi_onset_counts"), rejected_rows)

    id_to_token = ["REST", "HOLD"] + [f"MIDI_{pitch}" for pitch in range(midi_min, midi_max + 1)]
    write_json(output / "vocab.json", dict(
        format_version=FORMAT_VERSION, window_slots=WINDOW_SLOTS, slots_per_quarter=4, meter="2/4",
        size=len(id_to_token), midi_min=midi_min, midi_max=midi_max, id_to_token=id_to_token,
        token_to_id={name: i for i, name in enumerate(id_to_token)},
        midi_to_id={p: p - midi_min + PITCH_OFFSET for p in range(midi_min, midi_max + 1)},
        midi_observed_in_train=observed_train,
    ))
    report = build_report(screen_dir, summary, seed, windows, songs, memberships, groups, exact,
                          edges, review, ranges, excluded, symbols, samples, midi_min, midi_max,
                          real_tie_windows, synthetic)
    write_json(output / "summary.json", report)
    return report


REVIEW_FIELDS = (
    "song_a", "title_a", "start_bar_a", "split_a", "group_a", "song_b", "title_b", "start_bar_b", "split_b",
    "group_b", "shift_b_minus_a", "dice", "matched_onsets", "onsets_a", "onsets_b", "matched_pitches",
    "same_normalized_title", "cross_split", "status",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen_dir", type=Path, help="ACFS screen-v1 directory")
    parser.add_argument("output_dir", type=Path, help="new version directory outside the code repository")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo) or output.exists():
        parser.error("output_dir must be a new directory outside the code repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as staging:
        version = Path(staging) / "version"
        version.mkdir()
        report = build(args.screen_dir, version, args.seed)
        version.rename(output)
    print(f"Saved {report['round_trip_exported_windows']} verified windows to {output}")


if __name__ == "__main__":
    main()
