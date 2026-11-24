"""Export the existing melody experiments as scripts for the offline website."""

import argparse
from collections import defaultdict
import csv
import gzip
from hashlib import file_digest
import json
from pathlib import Path

TOKEN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
LEVELS = (10, 25, 50, 75, 90)
SPLITS = ("train", "validation", "test")
COMPARISON_MODELS = {"folk": "h3-acfs_matched-s0", "pop": "h3-cantopop-s0"}
CORPORA = (
    ("folk", "acfs-v261103", "中国民歌", 4247, 40795),
    ("pop", "cantopop-018e885", "粤语流行", 31, 1684),
)
VOLUMES = {
    "beijing": "北京卷",
    "guangdong": "广东卷",
    "hainan": "海南卷",
    "hebei1": "河北卷（一）",
    "hebei2": "河北卷（二）",
    "henan": "河南卷",
    "jiangsu1": "江苏卷（一）",
    "jiangsu2": "江苏卷（二）",
    "jilin": "吉林卷",
    "shaanxi1": "陕西卷（一）",
    "shaanxi2": "陕西卷（二）",
    "shanghai": "上海卷",
    "sichuan1": "四川卷",
    "tianjin": "天津卷",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            yield json.loads(line)


def sha256(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def pop_middle_windows(source):
    songs = defaultdict(list)
    for row in read_jsonl(source):
        if row["split"] != "test":
            raise ValueError(f"Not a held-out window: {row['window_id']}")
        songs[row["song_id"]].append(row)
    if len(songs) != 3:
        raise ValueError("Expected three held-out Cantopop songs")
    return [
        sorted(windows, key=lambda row: row["start_bar"])[len(windows) // 2]
        for _, windows in sorted(songs.items())
    ]


def pack(tokens):
    if len(tokens) != 64 or any(type(token) is not int or not 0 <= token <= 50 for token in tokens):
        raise ValueError("Expected 64 integer tokens in range 0–50")
    return "".join(TOKEN_CHARS[token] for token in tokens)


def write_script(path, prefix, data):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    path.write_text(f"{prefix} = {payload};\n", encoding="utf-8", newline="\n")


def read_song_table(dataset):
    with (dataset / "songs.csv").open(encoding="utf-8-sig", newline="") as source:
        return {row["song_id"]: row for row in csv.DictReader(source)}


def folk_detail(song, bar=None):
    detail = VOLUMES[song["volume"]]
    if bar is not None:
        detail += f" · 从原谱第 {bar} 小节开始"
    return detail


def build_library(dataset, song_table, corpus_id, expected_songs, expected_windows):
    fragments = defaultdict(list)
    for split in SPLITS:
        for row in read_jsonl(dataset / f"{split}.jsonl"):
            song = song_table[row["song_id"]]
            if song["split"] != split or row["split"] != split:
                raise ValueError(f"Split mismatch for {row['window_id']}")
            if row["tokens"][0] == 1:
                raise ValueError(f"Clean fragment starts with HOLD: {row['window_id']}")
            fragments[row["song_id"]].append(
                {"bar": row["start_bar"], "tokens": pack(row["tokens"])}
            )

    songs = []
    window_count = 0
    for song_id, song in song_table.items():
        windows = sorted(fragments.get(song_id, []), key=lambda window: window["bar"])
        if (
            len(windows) != int(song["encoded_windows"])
            or len({window["bar"] for window in windows}) != len(windows)
        ):
            raise ValueError(f"Window count or order mismatch for {song_id}")
        if not windows:
            continue
        detail = folk_detail(song) if corpus_id == "folk" else " · ".join(
            item for item in (song["artist"], song["year"]) if item
        )
        songs.append({
            "id": song_id,
            "title": song["title"],
            "detail": detail,
            "split": song["split"],
            "fragments": windows,
        })
        window_count += len(windows)
    if (len(songs), window_count) != (expected_songs, expected_windows):
        raise ValueError(f"Unexpected {corpus_id} totals: {len(songs)} songs, {window_count} windows")
    return {"songs": songs}


def collect_frozen_cases(frozen_inputs, catalogue):
    wanted = {
        f"{case['window_id']}|t{level}|n0"
        for case in catalogue
        for level in LEVELS
    }
    rows = {}
    for row in read_jsonl(frozen_inputs):
        if row["case_id"] in wanted:
            if row["noise_seed"] != 0 or row["case_id"] in rows:
                raise ValueError(f"Nonunique or unexpected noise input: {row['case_id']}")
            rows[row["case_id"]] = row
    if rows.keys() != wanted:
        raise ValueError("Frozen cases are missing from the selected five noise levels")
    return rows


def collect_predictions(reconstructions, run, model_dataset, cases):
    wanted = {
        f"{case['window_id']}|t{level}|n0"
        for case in cases
        for level in LEVELS
    }
    path = reconstructions / f"recon__{run}-s0.jsonl.gz"
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            case_id = row["case_id"]
            if case_id in wanted and row["method"] == "diffusion" and row["sample_index"] == 0:
                if (
                    row["train_seed"] != 0
                    or row["model_dataset"] != model_dataset
                    or case_id in rows
                ):
                    raise ValueError(f"Invalid prediction in {path.name}: {case_id}")
                rows[case_id] = pack(row["decoded_tokens"])
    if rows.keys() != wanted:
        raise ValueError(f"Missing predictions in {path.name}")
    return rows


def case_kind(slug):
    if slug.startswith("h2_high_pent_"):
        return "音集保留 · 原旋律变化"
    if slug.startswith("h2_rhythm_"):
        return "节奏变化"
    if slug.startswith("h2_high_"):
        return "恢复得好"
    if slug.startswith("h2_median_"):
        return "恢复得一般"
    if slug.startswith("h2_low_"):
        return "恢复困难"
    return "民歌与粤语流行对比"


def build_examples(cases, frozen, predictions, folk_songs):
    examples = []
    for case in cases:
        window_id = case["window_id"]
        song_id, bar = window_id.rsplit("@", 1)
        clean = pack(frozen[f"{window_id}|t10|n0"]["clean_tokens"])
        levels = {}
        for level in LEVELS:
            case_id = f"{window_id}|t{level}|n0"
            row = frozen[case_id]
            if row["window_id"] != window_id or row["t"] != level or pack(row["clean_tokens"]) != clean:
                raise ValueError(f"Frozen cases for {window_id} do not share the same clean input")
            levels[str(level)] = {
                "noisy": pack(row["noisy_tokens"]),
                **{name: values[case_id] for name, values in predictions.items()},
            }
        examples.append({
            "id": case["case_slug"],
            "title": case["title"],
            "kind": case_kind(case["case_slug"]),
            "windowId": window_id,
            "detail": folk_detail(folk_songs[song_id], bar),
            "clean": clean,
            "levels": levels,
        })
    return examples


def build_trajectories(trajectories_file, folk_songs):
    rows = read_json(trajectories_file)["windows"]
    if len(rows) != 3 or len({row["window_id"] for row in rows}) != len(rows):
        raise ValueError("Expected three distinct saved trajectories")
    trajectories = []
    for row in rows:
        forward = row["forward_states"]
        reverse50 = row["reverse_from_050"]["states"]
        reverse100 = row["reverse_from_100"]["states"]
        if (
            (len(forward), len(reverse50), len(reverse100)) != (101, 51, 101)
            or forward[0] != row["clean_tokens"]
            or forward[50] != reverse50[0]
            or forward[100] != reverse100[0]
        ):
            raise ValueError(f"Broken forward-to-reverse link for {row['window_id']}")
        song = folk_songs[row["song_id"]]
        bar = row["window_id"].rsplit("@", 1)[1]
        trajectories.append({
            "id": row["window_id"],
            "title": song["title"],
            "windowId": row["window_id"],
            "detail": folk_detail(song, bar),
            "forward": [pack(tokens) for tokens in forward],
            "reverse50": [pack(tokens) for tokens in reverse50],
            "reverse100": [pack(tokens) for tokens in reverse100],
        })
    return trajectories


def build_pop_comparisons(run_path, reconstructions, dataset, songs):
    receipt = run_path / "inputs.meta.json"
    if not receipt.is_file():
        raise ValueError("Run visualization/prepare_pop_comparison.py to prepare the Cantonese cases.")
    source = dataset / "test.jsonl"
    picked = pop_middle_windows(source)
    inputs = run_path / "xt-test-pop.jsonl"
    meta = read_json(receipt)
    if meta["source_sha256"] != sha256(source) or meta["input_sha256"] != sha256(inputs):
        raise ValueError("The Cantopop comparison input has changed")
    expected = {
        f"{row['window_id']}|t{step}|n0": (row, step) for row in picked for step in LEVELS
    }
    frozen = {}
    for case in read_jsonl(inputs):
        key = case["case_id"]
        row, step = expected[key]
        if (case["window_id"] != row["window_id"] or case["t"] != step or case["noise_seed"] != 0
                or case["clean_tokens"] != row["tokens"] or key in frozen):
            raise ValueError(f"Invalid Cantopop comparison input: {key}")
        frozen[key] = pack(case["noisy_tokens"])
    if frozen.keys() != expected.keys():
        raise ValueError("Missing Cantopop comparison inputs")
    predictions = {}
    for role, run in COMPARISON_MODELS.items():
        basename = f"recon__{run}"
        original = read_json(reconstructions / f"{basename}.meta.json")
        summary = read_json(run_path / f"{basename}.meta.json")
        output = run_path / f"{basename}.jsonl.gz"
        if (summary["input_sha256"] != meta["input_sha256"]
                or summary["checkpoint_sha256"] != original["checkpoint_sha256"]
                or summary["output_sha256_gzip"] != sha256(output)
                or summary["cases"] != len(expected) or summary["train_seed"] != 0):
            raise ValueError(f"Mismatched Cantopop comparison metadata: {run}")
        results = {}
        with gzip.open(output, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["method"] != "diffusion":
                    continue
                key = row["case_id"]
                if (key not in expected or key in results or row["sample_index"] != 0
                        or row["checkpoint_sha256"] != original["checkpoint_sha256"]
                        or row["model_dataset"] != original["model_dataset"]):
                    raise ValueError(f"Unexpected Cantopop prediction: {run} {key}")
                results[key] = pack(row["decoded_tokens"])
        if results.keys() != expected.keys():
            raise ValueError(f"Incomplete Cantopop predictions: {run}")
        predictions[role] = results

    examples = []
    for row in picked:
        window = row["window_id"]
        song = songs[row["song_id"]]
        bar = row["start_bar"]
        levels = {}
        for step in LEVELS:
            key = f"{window}|t{step}|n0"
            levels[str(step)] = {"noisy": frozen[key], **{role: rows[key] for role, rows in predictions.items()}}
        examples.append({
            "id": window, "title": song["title"], "kind": "粤语流行输入", "corpus": "pop",
            "windowId": window, "detail": f"{song['artist']} · 从原谱第 {bar} 小节开始",
            "clean": pack(row["tokens"]), "levels": levels,
        })
    return examples


def add_input_arguments(parser):
    parser.add_argument("--case-catalog", required=True, type=Path, help="Selected restoration and comparison cases")
    parser.add_argument("--frozen-inputs", required=True, type=Path, help="Frozen noisy input JSONL")
    parser.add_argument("--reconstructions", required=True, type=Path, help="Reconstruction exports and metadata directory")
    parser.add_argument("--trajectories", required=True, type=Path, help="Saved forward and reverse trajectories JSON")
    parser.add_argument("--pop-comparison", required=True, type=Path, help="Supplemental Cantopop comparison exports directory")


def build_data(source: Path, output: Path, inputs):
    """Write the case sets, saved trajectories, and complete libraries."""
    datasets = {
        corpus_id: source / directory / "split-encode-v1"
        for corpus_id, directory, _, _, _ in CORPORA
    }
    folk_songs = read_song_table(datasets["folk"])
    pop_songs = read_song_table(datasets["pop"])
    catalogue = read_json(inputs.case_catalog)
    effects = [case for case in catalogue if case["case_slug"].startswith("h2_")]
    comparisons = [case for case in catalogue if case["case_slug"].startswith("h3_")]
    if (
        (len(effects), len(comparisons), len(catalogue)) != (12, 3, 15)
        or len({case["window_id"] for case in catalogue}) != len(catalogue)
    ):
        raise ValueError("Expected 12 distinct restoration cases and 3 model comparisons")

    frozen = collect_frozen_cases(inputs.frozen_inputs, catalogue)
    effects = build_examples(effects, frozen, {
        "model": collect_predictions(inputs.reconstructions, "h2-acfs_full", "acfs_full", effects),
    }, folk_songs)
    comparisons = build_examples(comparisons, frozen, {
        "folk": collect_predictions(inputs.reconstructions, "h3-acfs_matched", "acfs_matched", comparisons),
        "pop": collect_predictions(inputs.reconstructions, "h3-cantopop", "cantopop", comparisons),
    }, folk_songs)
    for example in comparisons:
        example["corpus"] = "folk"
    comparisons.extend(build_pop_comparisons(inputs.pop_comparison, inputs.reconstructions, datasets["pop"], pop_songs))
    trajectories = build_trajectories(inputs.trajectories, folk_songs)

    output.mkdir(parents=True, exist_ok=True)
    libraries = []
    for corpus_id, _, title, song_count, window_count in CORPORA:
        dataset = datasets[corpus_id]
        table = folk_songs if corpus_id == "folk" else pop_songs
        library = build_library(dataset, table, corpus_id, song_count, window_count)
        filename = f"library-{corpus_id}.js"
        write_script(output / filename,
                     f"window.MELODY_LIBRARIES ??= {{}};\nwindow.MELODY_LIBRARIES.{corpus_id}",
                     library)
        libraries.append({
            "id": corpus_id,
            "title": title,
            "songs": song_count,
            "windows": window_count,
            "file": f"data/{filename}",
        })

    write_script(output / "experiments.js", "window.MELODY_DATA", {
        "effects": effects,
        "comparisons": comparisons,
        "trajectories": trajectories,
        "libraries": libraries,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output directory for the three local data scripts")
    add_input_arguments(parser)
    args = parser.parse_args()
    build_data(Path(__file__).resolve().parents[1] / "data", args.output, args)
