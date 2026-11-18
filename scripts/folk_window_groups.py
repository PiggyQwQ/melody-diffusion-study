"""Transposition-invariant overlap audit and reproducible grouped splits."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import random
import re
import unicodedata

from screen_folk_scores import BAR_SLOTS, WINDOW_BARS


SPLITS = {"train": 0.8, "validation": 0.1, "test": 0.1}
MIN_NEAR_DICE = 0.80
MIN_VARIANT_DICE = 0.90
MIN_VARIANT_COVERAGE = 0.60


def relative_notes(spans):
    notes = [(start, pitch, slots) for start, pitch, slots in spans if pitch is not None]
    origin = notes[0][1]
    # The complement of the timed notes defines rest occupancy. Adjacent rests
    # engraved as separate symbols do not distinguish otherwise identical music.
    return tuple((start, pitch - origin, slots) for start, pitch, slots in notes)


def fingerprint(relative):
    return sha256(json.dumps(relative, separators=(",", ":")).encode()).hexdigest()


def normalized_title(title):
    title = unicodedata.normalize("NFKC", title).strip()
    title = re.sub(r"\s*\((?:之?[一二三四五六七八九十〇零\d]+)\)$", "", title)
    return "".join(c for c in title.casefold() if not c.isspace())


@dataclass(frozen=True)
class Match:
    shift: int
    matches: int
    note_count_a: int
    note_count_b: int
    matched_pitch_count: int

    @property
    def dice(self):
        return 2 * self.matches / (self.note_count_a + self.note_count_b)


def near_matches(windows):
    """Find near matches seeded by two-bar blocks allowing one omitted attack per side."""
    block_size = 2 * BAR_SLOTS
    index = defaultdict(list)
    note_sets = []
    for i, window in enumerate(windows):
        notes = tuple(span for span in window.spans if span[1] is not None)
        note_sets.append(set(notes))
        candidates = set()
        for block_start in range(0, WINDOW_BARS * BAR_SLOTS, block_size):
            block = tuple(note for note in notes if block_start <= note[0] < block_start + block_size)
            if len(block) < 4:
                continue
            seeds = [block]
            if len(block) > 4:
                seeds.extend(block[:omit] + block[omit + 1:] for omit in range(len(block)))
            for seed in seeds:
                anchor = seed[0][1]
                if all(pitch == anchor for _, pitch, _ in seed):
                    continue
                key = bytes(value for start, pitch, duration in seed
                            for value in (start, pitch - anchor + 127, duration))
                candidates.update((j, anchor - old_anchor) for j, old_anchor in index[key]
                                  if windows[j].song_id != window.song_id)
                index[key].append((i, anchor))
        for j, shift in sorted(candidates):
            old = note_sets[j]
            size = len(old) + len(notes)
            if min(len(old), len(notes)) < max(10, MIN_NEAR_DICE * size / 2):
                continue
            pitches = [pitch for start, pitch, dur in notes if (start, pitch - shift, dur) in old]
            diversity = len(set(pitches))
            if len(pitches) < 10 or 2 * len(pitches) < MIN_NEAR_DICE * size or diversity < 3:
                continue
            if 2 * len(pitches) == size:  # Already audited by the exact index.
                continue
            yield windows[j], window, Match(shift, len(pitches), len(old), len(notes), diversity)


def group_songs(songs, windows, exact):
    parent = {song_id: song_id for song_id in songs}
    edges = []

    def root(song_id):
        while parent[song_id] != song_id:
            parent[song_id] = parent[parent[song_id]]
            song_id = parent[song_id]
        return song_id

    def link(ids, reason, evidence):
        first, *others = sorted(set(ids))
        for other in others:
            low, high = sorted((root(first), root(other)))
            parent[high] = low
            edges.append(dict(song_a=first, song_b=other, reason=reason, evidence=evidence))

    for field in ("source_split_group", "clean_melody_hash"):
        buckets = defaultdict(list)
        for song_id, song in songs.items():
            if song[field]:
                buckets[song[field]].append(song_id)
        for value, ids in sorted(buckets.items()):
            if len(ids) > 1:
                link(ids, field, value)
    for key, copies in exact.items():
        ids = {window.song_id for window in copies}
        if len(ids) > 1:
            link(ids, "exact_transposed_window", fingerprint(key))

    titles = {song_id: normalized_title(song["title"]) for song_id, song in songs.items()}
    covered = defaultdict(set)
    for window in windows:
        covered[window.song_id].update(range(window.start_bar, window.start_bar + WINDOW_BARS))
    best = {}
    variant_evidence = {}
    for left, right, match in near_matches(windows):
        a, b = left.song_id, right.song_id  # The input is sorted by song and start.
        pair = (a, b)
        rank = (match.dice, match.matches, match.matched_pitch_count)
        if pair not in best or rank > best[pair][0]:
            best[pair] = rank, left, right, match
        if match.dice >= MIN_VARIANT_DICE and match.matches >= 12:
            alignment = (a, b, match.shift, right.start_bar - left.start_bar)
            if alignment not in variant_evidence:
                variant_evidence[alignment] = set(), set(), (rank, left, right, match)
            bars_a, bars_b, witness = variant_evidence[alignment]
            bars_a.update(range(left.start_bar, left.start_bar + WINDOW_BARS))
            bars_b.update(range(right.start_bar, right.start_bar + WINDOW_BARS))
            if rank > witness[0]:
                variant_evidence[alignment] = bars_a, bars_b, (rank, left, right, match)

    variants = set()
    for (a, b, shift, offset), (bars_a, bars_b, witness) in sorted(variant_evidence.items()):
        _, left, right, match = witness
        coverage_a = len(bars_a) / len(covered[a])
        coverage_b = len(bars_b) / len(covered[b])
        source_coverage_a = len(bars_a) / int(songs[a]["source_bars"])
        source_coverage_b = len(bars_b) / int(songs[b]["source_bars"])
        shared_title = titles[a] and titles[a] == titles[b]
        title_evidence = shared_title and (
            min(coverage_a, coverage_b) >= MIN_VARIANT_COVERAGE or (match.dice >= 0.95 and match.matches >= 20)
        )
        sustained_evidence = min(coverage_a, coverage_b) >= 0.80 and min(source_coverage_a, source_coverage_b) >= 0.60
        if title_evidence or sustained_evidence:
            variants.add((a, b))
            evidence = (
                f"{titles[a]} | " if title_evidence else ""
            ) + (f"bars={left.start_bar},{right.start_bar} | shift={shift:+d} | offset={offset:+d} | "
                 f"dice={match.dice:.6f} | matched={match.matches} | "
                 f"candidate_coverage={coverage_a:.3f},{coverage_b:.3f} | "
                 f"source_coverage={source_coverage_a:.3f},{source_coverage_b:.3f}")
            link((a, b), "title_and_melody_variant" if title_evidence else "sustained_melody_variant", evidence)
    return {song_id: root(song_id) for song_id in parent}, edges, best, variants


def assign_splits(songs, groups, seed):
    """Balance song counts and window counts globally and within each volume."""
    weights = defaultdict(Counter)
    for song_id, song in songs.items():
        windows = int(song["candidate_windows"])
        counts = weights[groups[song_id]]
        for volume in ("all", song["volume"]):
            counts[(volume, "songs")] += 1
            counts[(volume, "windows")] += windows

    totals = sum(weights.values(), Counter())
    targets = {split: {key: share * total for key, total in totals.items()}
               for split, share in SPLITS.items()}
    current = {split: Counter() for split in SPLITS}
    rng = random.Random(seed)
    order = sorted(weights)
    rng.shuffle(order)
    order.sort(key=lambda group: max(value / totals[key] for key, value in weights[group].items()),
               reverse=True)

    def cost(split, counts):
        # Delta of the weighted squared deviation from the target.
        return sum(value * (2 * (current[split][key] - targets[split][key]) + value)
                   / targets[split][key] * (2 if key[0] == "all" else 1)
                   for key, value in counts.items())

    assignment = {}
    for group in order:
        counts = weights[group]
        split = min(SPLITS, key=lambda s: cost(s, counts))
        current[split].update(counts)
        assignment[group] = split
    return {song_id: assignment[group] for song_id, group in groups.items()}
