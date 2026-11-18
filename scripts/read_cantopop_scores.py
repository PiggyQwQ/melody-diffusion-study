"""Read explicitly bounded 2/4 melody windows from the Cantopop **kern scores."""

from collections import Counter
import csv
from dataclasses import dataclass
from fractions import Fraction
from io import StringIO
from pathlib import PurePosixPath
import re
from zipfile import ZipFile

from encode_folk_windows import Window, source_spans
from screen_folk_scores import BAR_SLOTS, WINDOW_BARS


SCORE_TYPES = {"**kern", "**text", "**jyutping"}
SPINE_PATH_TOKENS = {"*^", "*v", "*x", "*+", "*-"}
METER = re.compile(r"\*M(\d+/\d+)")
KEY = re.compile(r"\*[A-Ga-g](?:#+|-+)?:")
BAR_NUMBER = re.compile(r"=+(\d+[a-z]?)")
RECIPROCAL = re.compile(r"(?P<divisor>\d+)(?:%(?P<multiplier>\d+))?(?P<dots>\.*)")
PITCH = re.compile(r"(?P<name>(?P<letter>[A-Ga-gr])(?P=letter)*)(?P<accidental>#+|-+|n)?(?P<dots>\.*)")
PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NOTE_MARKS = "{}()[]_&hHo;pPUsz'\"\x60~^Iuv\\/LJkKijlNVZ,xyXY?MmS$TtWwO"
WINDOW_STAGES = {
    "meter_windows": "songs_with_2_4_windows",
    "grid_windows": "songs_with_grid_windows",
    "uncut_sounding_windows": "songs_with_uncut_sounding_windows",
    "in_vocab_windows": "songs_with_in_vocab_windows",
    "stable_key_windows": "songs_with_stable_key_windows",
}


@dataclass
class Measure:
    source_number: str | None
    source_line: int
    bounded: bool
    meter: str | None
    key: str | None
    events: list | None

    def window_bar(self):
        return dict(source_number=self.source_number, source_line=self.source_line,
                    key=self.key, events=self.events)


def melody_records(text):
    """Read a single stable kern spine, allowing only the corpus's companion spines."""
    records = [(number, line.split("\t")) for number, line in enumerate(text.splitlines(), 1)
               if line.strip() and not line.startswith("!!")]
    if not records:
        return None
    header = records[0][1]
    if header.count("**kern") != 1 or set(header) - SCORE_TYPES:
        return None
    melody = header.index("**kern")
    if records[-1][1] != ["*-"] * len(header):
        return None
    body = []
    for number, fields in records[1:-1]:
        if (len(fields) != len(header)
                or any(field.startswith("**") or field in SPINE_PATH_TOKENS for field in fields)):
            return None
        body.append((number, fields[melody]))
    return body


def read_event(token):
    """Read an explicitly timed pitch/rest, or raise for a syntax or grid failure."""
    rhythms = list(RECIPROCAL.finditer(token))
    pitches = list(PITCH.finditer(token))
    if len(rhythms) != 1 or len(pitches) != 1:
        raise ValueError(token)
    rhythm, pitch = rhythms[0], pitches[0]
    left, right = sorted((rhythm, pitch), key=lambda match: match.start())
    marks = token[:left.start()] + token[left.end():right.start()] + token[right.end():]
    ties = [mark for mark in marks if mark in "[]_"]
    if marks.strip(NOTE_MARKS) or len(ties) > 1:
        raise ValueError(token)

    divisor = int(rhythm["divisor"])
    multiplier = int(rhythm["multiplier"] or 1)
    # A few source files place augmentation dots after the pitch rather than
    # after the reciprocal (e.g. 8f.L). Humlib also reads these as augmentation.
    if rhythm["dots"] and pitch["dots"]:
        raise ValueError(token)
    dots = len(rhythm["dots"] or pitch["dots"])
    whole = 2 * BAR_SLOTS
    base = (Fraction(whole * multiplier, divisor) if divisor else
            Fraction(whole * 2 ** len(rhythm["divisor"]) * multiplier))
    slots = base * Fraction(2 ** (dots + 1) - 1, 2 ** dots)
    if slots.denominator != 1 or slots <= 0:
        raise ValueError(token)

    name = pitch["name"]
    accidental = pitch["accidental"] or ""
    if name == "r" and not accidental and not ties:
        midi = None
    elif name[0] == "r":
        raise ValueError(token)
    else:
        octave_base = 60 + 12 * (len(name) - 1) if name.islower() else 48 - 12 * (len(name) - 1)
        midi = octave_base + PITCH_CLASS[name[0].upper()] + accidental.count("#") - accidental.count("-")
        if not 0 <= midi <= 127:
            raise ValueError(token)
    return [midi, int(slots), bool(ties and ties[0] in "]_"), bool(ties and ties[0] in "[_")]


def read_bars(records):
    """Keep physical measure order, including pickups and partial measures."""
    bars = []
    meter = key = None
    opening = opening_line = None
    data = []
    has_2_4 = False

    def finish(bounded_end):
        # Before the first barline or after the last, only an interval with
        # data represents a measure. Between barlines even empty bars matter.
        if not data and not (opening is not None and bounded_end):
            return
        measure_meters = {meter for _, _, meter, _ in data}
        measure_keys = {key for _, _, _, key in data}
        try:
            events = [read_event(token) for _, token, _, _ in data]
        except ValueError:
            events = None
        label = BAR_NUMBER.match(opening or "")
        bars.append(Measure(
            source_number=label[1] if label else None,
            source_line=opening_line if opening_line is not None else data[0][0],
            bounded=opening is not None and bounded_end,
            meter=next(iter(measure_meters)) if len(measure_meters) == 1 else None,
            key=next(iter(measure_keys)) if len(measure_keys) == 1 else None,
            events=events,
        ))

    for number, token in records:
        if token.startswith("="):
            finish(bounded_end=True)
            opening, opening_line = token, number
            data.clear()
        elif token.startswith("*M") and not token.startswith("*MM"):
            match = METER.fullmatch(token)
            meter = match[1] if match else None
            has_2_4 |= meter == "2/4"
        elif KEY.fullmatch(token):
            key = token
        elif re.fullmatch(r"\*[A-Ga-g][#-]*", token):
            # The release occasionally has a tonic-like token without ":".
            key = None
        elif not token.startswith(("*", "!")) and token != ".":
            data.append((number, token, meter, key))
        # A null record in the sole timed spine has no attack or extra time.

    finish(bounded_end=False)
    return bars, has_2_4


def extract_windows(song_id, bars, midi_min, midi_max):
    counts = Counter()
    windows = []
    bounded_2_4 = [bar.bounded and bar.meter == "2/4" for bar in bars]
    on_grid = [bar.events is not None and sum(event[1] for event in bar.events) == BAR_SLOTS
               for bar in bars]
    for index in range(len(bars) - WINDOW_BARS + 1):
        stop = index + WINDOW_BARS
        if not all(bounded_2_4[index:stop]):
            continue
        counts["meter_windows"] += 1
        if not all(on_grid[index:stop]):
            continue
        counts["grid_windows"] += 1
        measures = bars[index:stop]
        try:
            spans = source_spans(event for bar in measures for event in bar.events)
        except ValueError:
            continue
        pitches = [pitch for _, pitch, _ in spans if pitch is not None]
        if not pitches:
            continue
        counts["uncut_sounding_windows"] += 1
        if not all(midi_min <= pitch <= midi_max for pitch in pitches):
            continue
        counts["in_vocab_windows"] += 1
        if measures[0].key is not None and all(bar.key == measures[0].key for bar in measures):
            counts["stable_key_windows"] += 1
        windows.append(Window(song_id, index + 1, tuple(bar.window_bar() for bar in measures), tuple(spans)))
    return windows, counts


def read_cantopop(archive_path, midi_min, midi_max):
    """Return composer-credited song records, eligible Window objects and stage counts."""
    if not 0 <= midi_min <= midi_max <= 127:
        raise ValueError("MIDI range must lie within 0..127")
    songs = {}
    windows = []
    audit = Counter({name: 0 for name in ("single_kern_songs", "songs_with_2_4", "source_bars",
                                          *WINDOW_STAGES, *WINDOW_STAGES.values())})
    with ZipFile(archive_path) as archive:
        catalog = [name for name in archive.namelist()
                   if PurePosixPath(name).name == "Cantopop corpus list_final.csv"]
        if len(catalog) != 1:
            raise ValueError("Expected one author-supplied Cantopop song catalog")
        rows = list(csv.DictReader(StringIO(archive.read(catalog[0]).decode("utf-8-sig"), newline="")))
        members = [name for name in archive.namelist() if name.endswith(".krn")]
        scores = {PurePosixPath(name).stem: name for name in members}
        ids = {row["ID"] for row in rows}
        if ids != scores.keys() or len(ids) != len(rows) or len(scores) != len(members):
            raise ValueError("Song catalog and score archive must have a one-to-one ID correspondence")
        audit["metadata_songs"] = len(rows)
        audit["score_files"] = len(scores)
        for row in sorted(rows, key=lambda row: row["ID"]):
            member = scores[row["ID"]]
            song_id = f"cantopop:{row['ID']}"
            records = melody_records(archive.read(member).decode("utf-8-sig"))
            bars, has_2_4 = read_bars(records) if records is not None else ([], False)
            candidates, counts = extract_windows(song_id, bars, midi_min, midi_max)
            audit["single_kern_songs"] += records is not None
            audit["songs_with_2_4"] += has_2_4
            audit["source_bars"] += len(bars)
            audit.update(counts)
            audit.update({song_counter: bool(counts[stage]) for stage, song_counter in WINDOW_STAGES.items()})
            songs[song_id] = dict(
                id=song_id, volume="cantopop", title=row["Song Title"], artist=row["Singer"], year=row["Year"],
                composer=row["Composer"], candidate_windows=len(candidates), source_bars=len(bars),
                source_split_group=song_id, clean_melody_hash="", source_kern=member,
            )
            windows.extend(candidates)
    return songs, windows, dict(audit)
