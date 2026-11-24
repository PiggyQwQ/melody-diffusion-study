"""Compare two models on the same held-out Cantopop excerpts."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from build_data import COMPARISON_MODELS, LEVELS, pop_middle_windows, read_json, sha256


def prepare(repo, reconstructions, training, output):
    source = repo / "data" / "cantopop-018e885" / "split-encode-v1" / "test.jsonl"
    selected = pop_middle_windows(source)
    matched = {}
    for run in COMPARISON_MODELS.values():
        expected = read_json(reconstructions / f"recon__{run}.meta.json")["checkpoint_sha256"]
        actual = sha256(training / "checkpoints" / f"{run}.pt")
        if actual != expected:
            raise ValueError(f"{run} differs from the checkpoint recorded for the formal comparison.")
        matched[run] = actual

    sys.path.insert(0, str(training))
    # The training CLI reads ACFS. Reuse its unchanged case builder with the Cantopop split.
    from make_xt import build_cases

    windows = [row["window_id"] for row in selected]
    cases = build_cases(selected, windows, LEVELS, [0])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    decoder_path = [str(repo / "scripts"), os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, decoder_path))}
    with tempfile.TemporaryDirectory(prefix="melody-pop-", dir=output.parent) as temporary:
        work = Path(temporary)
        inputs = work / "xt-test-pop.jsonl"
        with inputs.open("w", encoding="utf-8", newline="\n") as stream:
            for case in cases:
                stream.write(json.dumps(case, ensure_ascii=False) + "\n")
        subprocess.run([
            sys.executable, "-X", "utf8", str(training / "reconstruct.py"),
            "--xt", str(inputs), "--out", str(work), "--runs", ",".join(COMPARISON_MODELS.values()),
            "--samples", "1", "--device", "cpu", "--threads", "4", "--batch-size", "500",
            "--seed", "20261121",
        ], check=True, env=env)
        receipt = {
            "source": "data/cantopop-018e885/split-encode-v1/test.jsonl",
            "source_sha256": sha256(source),
            "selection": "One middle start-bar window per held-out song; later of two middle windows.",
            "windows": windows,
            "times": LEVELS,
            "noise_seeds": [0],
            "input_sha256": sha256(inputs),
            "checkpoint_sha256": matched,
        }
        (work / "inputs.meta.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8", newline="\n")
        output.mkdir()
        for path in work.iterdir():
            shutil.move(path, output / path.name)
    print(f"{len(cases)} shared input cases for both models: {output}")


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True, help="training/ directory from train/c")
    parser.add_argument("--reconstructions", type=Path, required=True, help="Formal reconstruction metadata directory")
    parser.add_argument("--output", type=Path, default=repo / "dist" / "pop-comparison")
    args = parser.parse_args()
    prepare(repo, args.reconstructions.resolve(), args.training.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
