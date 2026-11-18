"""Choose an ACFS training control with the Cantonese pop window and song budget."""

from collections import defaultdict
from dataclasses import dataclass
from math import dist
from random import Random
from statistics import fmean, pstdev

from encode_folk_windows import WINDOW_SLOTS, decode


FEATURES = ("note_attacks", "mean_attack_midi", "midi_span", "rest_slots")


@dataclass(frozen=True)
class Window:
    row: dict
    features: tuple
    midi_min: int
    midi_max: int


def _song_windows(rows, midi_min, midi_max):
    songs = defaultdict(list)
    song_groups = {}
    window_ids = set()
    for row in sorted(rows, key=lambda row: (row["song_id"], row["start_bar"], row["window_id"])):
        window_id = row["window_id"]
        song_id = row["song_id"]
        if row["split"] != "train" or len(row["tokens"]) != WINDOW_SLOTS:
            raise ValueError(f"Expected {WINDOW_SLOTS}-token train windows: {window_id}")
        if window_id in window_ids or song_groups.setdefault(song_id, row["group_id"]) != row["group_id"]:
            raise ValueError(f"Duplicate window or inconsistent song group: {window_id}")
        window_ids.add(window_id)
        spans = decode(row["tokens"], midi_min, midi_max)
        pitches = [pitch for _, pitch, _ in spans if pitch is not None]
        if not pitches:
            raise ValueError(f"Expected sounding train windows: {window_id}")
        lower, upper = min(pitches), max(pitches)
        rests = sum(duration for _, pitch, duration in spans if pitch is None)
        features = len(pitches), fmean(pitches), upper - lower, rests
        songs[song_id].append(Window(row, features, lower, upper))
    return songs


def _summarize(windows):
    columns = list(zip(*(window.features for window in windows)))
    return dict(songs=len({window.row["song_id"] for window in windows}),
                groups=len({window.row["group_id"] for window in windows}),
                windows=len(windows), tokens=len(windows) * WINDOW_SLOTS,
                observed_midi_min=min(window.midi_min for window in windows),
                observed_midi_max=max(window.midi_max for window in windows),
                mean=dict(zip(FEATURES, map(fmean, columns))),
                sd=dict(zip(FEATURES, map(pstdev, columns))))


def match_folk_train(folk_rows, pop_rows, midi_min, midi_max, seed):
    """Return original ACFS rows, song-level pair provenance, and balance statistics.

    Match pop songs in descending window-count order. For each, select the eligible
    folk song whose best k windows have the smallest mean squared distance to the
    pop song's feature centroid. Scale features by their pop train window standard
    deviation with a one-unit floor. Each ACFS group is used at most once.
    """
    folk = _song_windows(folk_rows, midi_min, midi_max)
    pop = _song_windows(pop_rows, midi_min, midi_max)
    if not pop:
        raise ValueError("Pop training windows are required for matching")
    pop_windows = [window for windows in pop.values() for window in windows]
    pop_summary = _summarize(pop_windows)
    scales = tuple(max(1.0, value) for value in pop_summary["sd"].values())

    rng = Random(seed)
    candidate_songs = [(windows, [tuple(value / scale for value, scale in zip(window.features, scales))
                                  for window in windows]) for windows in folk.values()]
    rng.shuffle(candidate_songs)
    pop_songs = list(pop.values())
    rng.shuffle(pop_songs)
    pop_songs.sort(key=len, reverse=True)

    # Descending counts preserve feasibility: a group eligible now would also be
    # eligible for any later count, regardless of which earlier group was chosen.
    used_groups = set()
    selected = []
    pairing = []
    for pop_song in pop_songs:
        count = len(pop_song)
        centroid = tuple(fmean(column) / scale for column, scale in
                         zip(zip(*(w.features for w in pop_song)), scales))
        best_score = float("inf")
        best_windows = None
        for folk_song, coordinates in candidate_songs:
            if len(folk_song) < count or folk_song[0].row["group_id"] in used_groups:
                continue
            nearest = sorted((dist(centroid, window) ** 2, index)
                             for index, window in enumerate(coordinates))[:count]
            score = fmean(distance for distance, _ in nearest)
            if score < best_score:
                best_score = score
                best_windows = [folk_song[index] for _, index in nearest]
        pop_first = pop_song[0].row
        if best_windows is None:
            raise ValueError("Cannot match distinct ACFS training groups: "
                             f"{pop_first['song_id']} needs {count} windows from one unused song")
        folk_first = best_windows[0].row
        used_groups.add(folk_first["group_id"])
        selected.extend(best_windows)
        pairing.append(dict(pop_song_id=pop_first["song_id"], pop_group_id=pop_first["group_id"],
                            folk_song_id=folk_first["song_id"], folk_group_id=folk_first["group_id"],
                            windows=count, mean_window_squared_distance=best_score,
                            pop_window_ids=[w.row["window_id"] for w in pop_song],
                            folk_window_ids=[w.row["window_id"] for w in best_windows]))

    selected.sort(key=lambda window: (window.row["song_id"], window.row["start_bar"], window.row["window_id"]))
    pairing.sort(key=lambda pair: pair["pop_song_id"])
    matched_summary = _summarize(selected)
    stats = dict(
        method="Descending pop song window counts; nearest windows to pop song centroid by mean squared "
               "standardized feature distance; one unused ACFS song/group per pop song; seeded song tie order",
        seed=seed,
        feature_scales=dict(zip(FEATURES, scales)),
        eligible_folk_train=_summarize([window for windows in folk.values() for window in windows]),
        pop_train=pop_summary,
        matched_folk_train=matched_summary,
        mean_differences_folk_minus_pop={feature: matched_summary["mean"][feature] - pop_summary["mean"][feature]
                                         for feature in FEATURES},
    )
    return [window.row for window in selected], pairing, stats
