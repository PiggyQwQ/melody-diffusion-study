"""Encode Cantopop scores and build a size-matched ACFS training subset."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import file_digest
from html import escape
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from encode_folk_windows import (
    DEFAULT_SEED, WINDOW_SLOTS, Window, decode, encode_events,
    read_jsonl, source_spans, verify_saved_sequences, window_metadata, write_json,
)
from folk_window_groups import SPLITS, assign_splits, group_songs, near_matches, relative_notes
from screen_folk_scores import write_csv


SOURCE_COMMIT = "018e88540b917c40dc3f7248ed98908208544b54"
SOURCE_SHA256 = "21e2cef97635e63f20a7a2207935e74ca4c6cc6c1ba9e93278bb2c06434ec85c"
FORMAT_VERSION = "cantopop-split-encode-v1"
PENTATONIC_STEPS = (0, 2, 4, 7, 9)


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def exact_cross_corpus(folk_windows, pop_windows):
    by_melody = defaultdict(list)
    for window in pop_windows:
        by_melody[relative_notes(window.spans)].append(window)
    return [(folk, pop) for folk in folk_windows
            for pop in by_melody.get(relative_notes(folk.spans), ())]


def informative_melody(relative):
    return len(relative) >= 10 and len({pitch for _, pitch, _ in relative}) >= 3


def describe_training(rows, midi_min, midi_max):
    """Clean-data priors with one descriptive best-fit template per song."""
    histograms = defaultdict(lambda: (Counter(), Counter()))
    pitch_minima = []
    pitch_maxima = []
    intervals = Counter()
    for row in rows:
        notes = [(p, d) for _, p, d in decode(row["tokens"], midi_min, midi_max) if p is not None]
        pitches = [p for p, _ in notes]
        pitch_minima.append(min(pitches))
        pitch_maxima.append(max(pitches))
        onset, duration = histograms[row["song_id"]]
        onset.update(p % 12 for p in pitches)
        for p, d in notes:
            duration[p % 12] += d
        intervals.update(b - a for a, b in zip(pitches, pitches[1:]))

    onset_total, duration_total = Counter(), Counter()
    covered_onsets = covered_duration = 0
    song_onset_coverage, song_duration_coverage = [], []
    templates = [tuple((root + delta) % 12 for delta in PENTATONIC_STEPS) for root in range(12)]
    for onset, duration in histograms.values():
        onset_total.update(onset)
        duration_total.update(duration)
        n, d = max((sum(onset[p] for p in pcs), sum(duration[p] for p in pcs)) for pcs in templates)
        covered_onsets += n
        covered_duration += d
        song_onset_coverage.append(n / onset.total())
        song_duration_coverage.append(d / duration.total())
    slots = sum(len(row["tokens"]) for row in rows)
    return dict(
        songs=len(histograms), groups=len({row.get("group_id", row["song_id"]) for row in rows}),
        windows=len(rows), slots=slots, note_onsets=onset_total.total(),
        mean_note_onsets=onset_total.total() / len(rows),
        rest_fraction=1 - duration_total.total() / slots,
        midi_min=min(pitch_minima), midi_max=max(pitch_maxima),
        pitch_class_onsets=[onset_total[i] for i in range(12)],
        pitch_class_duration_slots=[duration_total[i] for i in range(12)],
        interval_counts=dict(sorted(intervals.items())),
        best_pentatonic_fit=dict(pooled_onset_coverage=covered_onsets / onset_total.total(),
                                  pooled_duration_coverage=covered_duration / duration_total.total(),
                                  song_mean_onset_coverage=sum(song_onset_coverage) / len(histograms),
                                  song_mean_duration_coverage=sum(song_duration_coverage) / len(histograms)),
    )


def read_folk(folk_dir):
    vocab = json.loads((folk_dir / "vocab.json").read_text(encoding="utf-8"))
    low, high = vocab["midi_min"], vocab["midi_max"]
    names = ["REST", "HOLD"] + [f"MIDI_{pitch}" for pitch in range(low, high + 1)]
    if (vocab["id_to_token"] != names or vocab["window_slots"] != WINDOW_SLOTS or vocab["size"] != len(names)
            or vocab.get("meter") != "2/4" or vocab.get("slots_per_quarter") != 4):
        raise ValueError("Expected the shared 2/4 vocabulary with 64 slots at four slots per quarter")
    windows, rows_by_id = [], {}
    for split in SPLITS:
        for row in read_jsonl(folk_dir / f"{split}.jsonl"):
            spans = tuple(decode(row["tokens"], low, high))
            window = Window(row["song_id"], row["start_bar"], (), spans)
            if (window.window_id != row["window_id"] or row["split"] != split
                    or len(row["tokens"]) != WINDOW_SLOTS or window.window_id in rows_by_id):
                raise ValueError(f"Invalid ACFS window {row['window_id']}")
            rows_by_id[window.window_id] = row
            windows.append(window)
    return vocab, windows, rows_by_id


def audit_cross_corpus(folk_windows, folk_rows, pop_windows):
    def match_row(folk, pop, kind, dice, shift, attacks):
        return dict(folk_window_id=folk.window_id, folk_song_id=folk.song_id,
                    folk_group_id=folk_rows[folk.window_id]["group_id"], folk_split=folk_rows[folk.window_id]["split"],
                    pop_window_id=pop.window_id, pop_song_id=pop.song_id,
                    match=kind, dice=round(dice, 6), matched_attacks=attacks, midi_shift_pop_minus_folk=shift)

    def first_pitch(window):
        return next(p for _, p, _ in window.spans if p is not None)

    exact = exact_cross_corpus(folk_windows, pop_windows)
    rows = [match_row(a, b, "exact", 1, first_pitch(b) - first_pitch(a), len(relative_notes(b.spans)))
            for a, b in exact]
    near = {}
    pop_ids = {w.song_id for w in pop_windows}
    for a, b, match in near_matches([*folk_windows, *pop_windows]):
        if a.song_id in pop_ids or b.song_id not in pop_ids:
            continue
        pair = (a.song_id, b.song_id)
        if pair not in near or match.dice > near[pair]["dice"]:
            near[pair] = match_row(a, b, "near_review", match.dice, match.shift, match.matches)
    rows.extend(sorted(near.values(), key=lambda row: (-row["dice"], row["pop_window_id"], row["folk_window_id"])))
    related_songs = {pop.song_id for _, pop in exact if informative_melody(relative_notes(pop.spans))}
    return rows, {pop.window_id for _, pop in exact}, related_songs, {row["folk_group_id"] for row in rows}


def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def folk_review_groups(folk_dir):
    with (folk_dir / "near_repeats_review.csv").open(encoding="utf-8-sig", newline="") as stream:
        return {row[f"group_{side}"] for row in csv.DictReader(stream)
                if row["split_a"] != row["split_b"]
                for side in ("a", "b") if row[f"split_{side}"] == "train"}


def one_song_per_group(rows):
    by_group = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_group[row["group_id"]][row["song_id"]].append(row)
    selected_ids = {max(sorted(songs), key=lambda song_id: len(songs[song_id])) for songs in by_group.values()}
    return [row for row in rows if row["song_id"] in selected_ids]


def export_samples(output, windows, songs, groups, memberships, low, high, seed):
    import verovio
    from folk_window_samples import export_score, select_samples

    output.mkdir()
    toolkit = verovio.toolkit(False)
    if not toolkit.setResourcePath(str(Path(verovio.__file__).with_name("data"))):
        raise ValueError("Could not load Verovio resources; install it to an ASCII-only path on Windows")
    toolkit.setOptions({"breaks": "encoded", "adjustPageHeight": True, "pageWidth": 2000,
                        "scale": 45, "header": "none", "footer": "none"})
    rows, cards = [], []
    for i, (window, features) in enumerate(select_samples(windows, songs, groups, memberships, {}, seed), 1):
        song = songs[window.song_id]
        stem = f"{i:02d}-{window.song_id.split(':')[-1]}-bar{window.start_bar}"
        tokens = encode_events(window.events, low, high)
        spans = decode(tokens, low, high)
        if spans != list(window.spans):
            raise ValueError(f"Sample mismatch: {window.window_id}")
        files = export_score(output, stem, spans, toolkit, source_spans)
        rows.append(dict(window_id=window.window_id, title=song["title"], artist=song["artist"],
                         split=memberships[window.song_id], source_kern=song["source_kern"],
                         features="|".join(features), **files))
        caption = escape(f"{song['title']} · {song['artist']} · {window.window_id}")
        cards.append(f'<section><h2>{caption}</h2><p><a href="{files["midi"]}">MIDI</a> · '
                     f'<a href="{files["musicxml"]}">MusicXML</a></p>'
                     f'<img src="{files["svg"]}" alt="{caption}"></section>')
    write_csv(output / "index.csv", tuple(rows[0]), rows)
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Cantopop 编码回放</title>'
            '<style>body{max-width:960px;margin:32px auto;padding:0 16px;font:16px/1.5 system-ui}'
            'section{border-top:1px solid #ccc;margin:2rem 0}img{width:100%;height:auto}</style>'
            '<h1>Cantopop 编码回放</h1><p>拍号沿用作者转录谱，MIDI 为 90 BPM 钢琴回放。</p>'
            + "\n".join(cards) + "</html>\n")
    (output / "index.html").write_text(html, encoding="utf-8", newline="\n")
    return rows


def build(archive, folk_dir, output, seed):
    from cantopop_match_folk import match_folk_train
    from read_cantopop_scores import read_cantopop

    archive_hash = sha256(archive)
    if archive_hash != SOURCE_SHA256:
        raise ValueError(f"Expected pinned Cantopop ZIP SHA-256 {SOURCE_SHA256}, got {archive_hash}")
    vocab, folk_windows, folk_rows = read_folk(folk_dir)
    low, high = vocab["midi_min"], vocab["midi_max"]
    all_songs, screened, audit = read_cantopop(archive, low, high)
    screened.sort(key=lambda w: (w.song_id, w.start_bar))
    songs = {song_id: row for song_id, row in sorted(all_songs.items()) if row["candidate_windows"]}
    exact = defaultdict(list)
    for window in screened:
        exact[relative_notes(window.spans)].append(window)
    shared = {key: copies for key, copies in exact.items() if len({w.song_id for w in copies}) > 1}
    shared_informative = {key: copies for key, copies in shared.items() if informative_melody(key)}
    sparse_duplicates = {w.window_id for key, copies in shared.items() if key not in shared_informative for w in copies}
    groups, edges, near, variants = group_songs(songs, screened, shared_informative)
    cross_review, cross_duplicates, pop_clashes, folk_groups_for_review = audit_cross_corpus(folk_windows, folk_rows, screened)
    excluded_groups = {groups[song_id] for song_id in pop_clashes}
    removed = {}
    for window in screened:
        reasons = []
        if window.window_id in sparse_duplicates:
            reasons.append("sparse_exact_pop_overlap")
        if window.window_id in cross_duplicates:
            reasons.append("exact_acfs_overlap")
        if groups[window.song_id] in excluded_groups:
            reasons.append("informative_exact_acfs_group_overlap")
        if reasons:
            removed[window.window_id] = "|".join(reasons)
    windows = [window for window in screened if window.window_id not in removed]
    retained = Counter(w.song_id for w in windows)
    songs = {song_id: dict(song, candidate_windows=retained[song_id])
             for song_id, song in songs.items() if retained[song_id]}
    print(f"Screened {len(all_songs)} Cantopop scores; encoding {len(windows)} windows from {len(songs)} songs", flush=True)
    memberships = assign_splits(songs, {song_id: groups[song_id] for song_id in songs}, seed)
    dataset = {split: [] for split in SPLITS}
    window_index = []
    for window in windows:
        song = songs[window.song_id]
        split = memberships[window.song_id]
        metadata = window_metadata(window, song, groups[window.song_id], split)
        dataset[split].append({**metadata, "tokens": encode_events(window.events, low, high)})
        keys = {bar.get("key") for bar in window.bars}
        key = next(iter(keys)) if len(keys) == 1 else None
        window_index.append(dict(**metadata, source_kern=song["source_kern"],
                                 printed_start_bar=window.bars[0]["source_number"],
                                 end_bar=window.start_bar + len(window.bars) - 1,
                                 printed_end_bar=window.bars[-1]["source_number"],
                                 source_line=window.bars[0].get("source_line"), score_key=key))
    for split, rows in dataset.items():
        if not rows:
            raise ValueError(f"Empty Cantopop split: {split}")
        save_jsonl(output / f"{split}.jsonl", rows)
    verify_saved_sequences(output, windows, songs, groups, memberships, {}, low, high)
    print("Verified all 64-slot round trips and within-Cantopop split isolation", flush=True)

    blocked_by_folk_review = folk_review_groups(folk_dir)
    blocked = folk_groups_for_review | blocked_by_folk_review
    folk_train = [row for row in folk_rows.values() if row["split"] == "train" and row["group_id"] not in blocked]
    pop_train = one_song_per_group(dataset["train"])
    pop_matched_file = "train.jsonl"
    if len(pop_train) != len(dataset["train"]):
        pop_matched_file = "cantopop_matched_train.jsonl"
        save_jsonl(output / pop_matched_file, pop_train)
    matched, pairing, match_stats = match_folk_train(folk_train, pop_train, low, high, seed)
    save_jsonl(output / "acfs_matched_train.jsonl", matched)
    write_json(output / "training_pairs.json", pairing)
    write_json(output / "training_priors.json", dict(
        cantopop=describe_training(pop_train, low, high),
        acfs_matched=describe_training(matched, low, high),
        definition="Template maximizes pooled clean onsets per song, then duration; no verified mode label implied.",
    ))
    samples = export_samples(output / "samples", windows, songs, groups, memberships, low, high, seed)

    manifest = []
    for song_id, song in sorted(all_songs.items()):
        split = memberships.get(song_id, "")
        manifest.append(dict(song_id=song_id, title=song["title"], artist=song["artist"], year=song["year"],
                             source_kern=song["source_kern"], group_id=groups.get(song_id, ""), split=split,
                             source_bars=song["source_bars"], candidate_windows=song["candidate_windows"],
                             encoded_windows=retained[song_id],
                             status="kept" if split else "no_window_after_overlap_audit" if song_id in groups else "no_window"))
    within_review = [dict(song_a=a, song_b=b, window_a=left.window_id, window_b=right.window_id,
                          dice=round(match.dice, 6), matched_notes=match.matches,
                          midi_shift_b_minus_a=match.shift, merged=(a, b) in variants,
                          split_a=memberships.get(a, ""), split_b=memberships.get(b, ""))
                     for (a, b), (_, left, right, match) in sorted(near.items())]
    write_csv(output / "songs.csv", tuple(manifest[0]), manifest)
    write_csv(output / "windows.csv", tuple(window_index[0]), window_index)
    write_csv(output / "group_edges.csv", ("song_a", "song_b", "reason", "evidence"), edges)
    write_csv(output / "near_repeats_review.csv", ("song_a", "song_b", "window_a", "window_b", "dice",
              "matched_notes", "midi_shift_b_minus_a", "merged", "split_a", "split_b"), within_review)
    write_csv(output / "acfs_overlap_review.csv", ("folk_window_id", "folk_song_id", "folk_group_id", "folk_split",
              "pop_window_id", "pop_song_id", "match", "dice", "matched_attacks", "midi_shift_pop_minus_folk"), cross_review)
    write_csv(output / "discarded_windows.csv", ("window_id", "song_id", "start_bar", "reason"),
              [dict(window_id=w.window_id, song_id=w.song_id, start_bar=w.start_bar, reason=removed[w.window_id])
               for w in screened if w.window_id in removed])
    (output / "vocab.json").write_bytes((folk_dir / "vocab.json").read_bytes())
    write_json(output / "screen_audit.json", audit)
    write_json(output / "summary.json", dict(
        format_version=FORMAT_VERSION, source_commit=SOURCE_COMMIT, source_zip_sha256=archive_hash, seed=seed,
        folk_inputs_sha256={name: sha256(folk_dir / name) for name in
                            ("vocab.json", "near_repeats_review.csv", *(f"{s}.jsonl" for s in SPLITS))},
        meter_basis="2/4 in the publisher's Humdrum transcriptions", midi_min=low, midi_max=high,
        vocabulary_size=vocab["size"], window_slots=WINDOW_SLOTS,
        splits={s: dict(songs=len({r["song_id"] for r in rows}), groups=len({r["group_id"] for r in rows}),
                        windows=len(rows), slots=len(rows) * WINDOW_SLOTS) for s, rows in dataset.items()},
        screened_songs=len(all_songs), encoded_songs=len(songs), encoded_windows=len(windows),
        discarded_overlap_windows=len(removed), sparse_pop_overlap_windows=len(sparse_duplicates),
        stable_score_key_windows=sum(bool(row["score_key"]) for row in window_index),
        pop_cross_split_exact_matches=0, pop_near_pairs=len(within_review),
        pop_cross_split_near_pairs=sum(r["split_a"] != r["split_b"] and bool(r["split_a"] and r["split_b"])
                                       for r in within_review),
        acfs_cross_corpus_exact_pairs=sum(r["match"] == "exact" for r in cross_review),
        acfs_cross_corpus_near_song_pairs=sum(r["match"] == "near_review" for r in cross_review),
        excluded_cantopop_groups=len(excluded_groups),
        excluded_folk_train_groups_for_cross_corpus_review=sorted(folk_groups_for_review & {r["group_id"]
                                                         for r in folk_rows.values() if r["split"] == "train"}),
        excluded_folk_train_groups_for_existing_review=sorted(blocked_by_folk_review),
        round_trip_windows=len(windows), replay_examples=len(samples),
        pop_matched_training_file=pop_matched_file, matched_training=match_stats,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cantopop_zip", type=Path)
    parser.add_argument("folk_data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("output_dir must be a new directory outside the code repository")
    output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as staging:
        destination = Path(staging) / "version"
        destination.mkdir()
        build(args.cantopop_zip, args.folk_data_dir, destination, args.seed)
        destination.rename(output)
    print(f"Saved Cantopop and matched ACFS data to {output}")


if __name__ == "__main__":
    main()
