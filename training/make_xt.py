"""Build a corrupted-input (xt) file from the ACFS validation or test split.

Every case is a window corrupted once at step t: each of the 64 slots is
redrawn uniformly from the 51 classes with probability t/100 and kept
otherwise (the closed-form forward process used in training). The random
numbers of a case depend only on (window_id, t, noise_seed).

Example (from the repository root):
    python training/make_xt.py --split validation --window-ids ids.txt \
        --times 10,25,50,75,90 --noise-seeds 0,1,2,3,4 --out xt-validation.jsonl
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch
except ImportError:
    sys.exit("PyTorch is not installed. Install the dependencies first: "
             "python -m pip install -r training/requirements.txt")

import diffusion as D  # noqa: E402


def case_id(window_id, t, noise_seed):
    return f"{window_id}|t{t}|n{noise_seed}"


def case_seed(window_id, t, noise_seed):
    """First 8 bytes (big-endian) of SHA-256 of the case_id string."""
    return int.from_bytes(sha256(case_id(window_id, t, noise_seed).encode("utf-8")).digest()[:8], "big")


def corrupt_case(tokens, t, seed):
    generator = torch.Generator().manual_seed(seed)
    x_0 = torch.tensor([tokens], dtype=torch.long)
    return D.corrupt(x_0, torch.tensor([t]), generator)[0].tolist()


def int_list(text, name, low, high):
    try:
        values = [int(part) for part in text.split(",") if part.strip() != ""]
    except ValueError:
        sys.exit(f"--{name} must be comma-separated integers, got {text!r}")
    if not values or len(set(values)) != len(values) or any(not low <= v <= high for v in values):
        sys.exit(f"--{name} needs distinct integers between {low} and {high}, got {text!r}")
    return values


def build_cases(rows, window_ids, times, noise_seeds):
    by_id = {row["window_id"]: row for row in rows}
    cases = []
    for window_id in window_ids:
        row = by_id[window_id]
        for t in times:
            for noise_seed in noise_seeds:
                cases.append(dict(
                    case_id=case_id(window_id, t, noise_seed), window_id=window_id, song_id=row["song_id"],
                    group_id=row["group_id"], split=row["split"], t=t, rho=t / D.T, noise_seed=noise_seed,
                    noisy_tokens=corrupt_case(row["tokens"], t, case_seed(window_id, t, noise_seed)),
                    clean_tokens=row["tokens"]))
    return cases


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=("validation", "test"), required=True)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--window-ids", type=Path, help="Text file with one window_id per line")
    which.add_argument("--all", action="store_true", help="Use every window of the split, in file order")
    p.add_argument("--times", default="10,25,50,75,90", help="Comma-separated diffusion steps (0-100)")
    p.add_argument("--noise-seeds", default="0,1,2,3,4", help="Comma-separated non-negative integers")
    p.add_argument("--out", type=Path, required=True, help="Output .jsonl file (must not exist)")
    p.add_argument("--data-root", type=Path, default=D.DATA_ROOT)
    args = p.parse_args(argv)

    times = int_list(args.times, "times", 0, D.T)
    noise_seeds = int_list(args.noise_seeds, "noise-seeds", 0, 2 ** 31 - 1)
    if args.out.exists():
        sys.exit(f"Output file already exists: {args.out}. Choose a new name; files are never overwritten.")
    meta_path = args.out.with_name(args.out.name.removesuffix(".jsonl") + ".meta.json")
    if meta_path.exists():
        sys.exit(f"Metadata file already exists: {meta_path}. Choose a new output name.")

    source = Path(args.data_root) / D.ACFS / f"{args.split}.jsonl"
    if not source.is_file():
        sys.exit(f"Cannot find {source}. Run the command from the repository root of melody-diffusion-study.")
    rows = D.read_rows(source)
    D.clean_tensor(rows)
    if args.all:
        window_ids = [row["window_id"] for row in rows]
        selection = "all windows of the split, in file order"
    else:
        if not args.window_ids.is_file():
            sys.exit(f"Window list not found: {args.window_ids}")
        window_ids = [line.strip() for line in args.window_ids.read_text(encoding="utf-8-sig").splitlines()
                      if line.strip()]
        known = {row["window_id"] for row in rows}
        unknown = [w for w in window_ids if w not in known]
        if unknown:
            sys.exit(f"{len(unknown)} window_id(s) are not in {args.split}.jsonl, e.g. {unknown[:3]}")
        if len(set(window_ids)) != len(window_ids):
            sys.exit("The window list contains duplicates")
        if not window_ids:
            sys.exit("The window list is empty")
        selection = "window list file, in its line order"

    cases = build_cases(rows, window_ids, times, noise_seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")

    meta = dict(
        tool="training/make_xt.py", split=args.split, selection=selection,
        window_ids_file=args.window_ids.name if args.window_ids else None,
        window_ids_file_sha256=D.sha256_file(args.window_ids) if args.window_ids else None,
        times=times, noise_seeds=noise_seeds, windows=len(window_ids), cases=len(cases),
        case_order="windows in selection order; within a window, times in the given order, then noise seeds",
        case_id_format="{window_id}|t{t}|n{noise_seed}",
        noise_rule="seed = first 8 bytes (big-endian) of SHA-256 of the UTF-8 case_id; torch.Generator (CPU) "
                   "seeded with it draws torch.rand(1,64) < t/100 for replacement, then torch.randint(51, (1,64)) "
                   "for the new symbols",
        source=f"data/{D.ACFS}/{args.split}.jsonl", source_sha256=D.sha256_file(source),
        output=args.out.name, output_sha256=D.sha256_file(args.out),
        python=platform.python_version(), torch=torch.__version__)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(cases)} cases ({len(window_ids)} windows x {len(times)} t x {len(noise_seeds)} seeds) "
          f"to {args.out}")
    print(f"SHA-256 {meta['output_sha256']}")


if __name__ == "__main__":
    main()
