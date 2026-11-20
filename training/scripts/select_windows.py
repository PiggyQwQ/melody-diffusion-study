"""Write a fixed random selection of ACFS window_ids, one per line, in file order.

The selection is sorted(random.Random(seed).sample(range(N), count)), read as
0-based line indices of data/acfs-v261103/split-encode-v1/<split>.jsonl.

Example:
    python training/scripts/select_windows.py --split validation --count 100 --seed 20261120 --out ids.txt
"""

import argparse
import json
from pathlib import Path
import random

DATA = Path(__file__).resolve().parents[2] / "data" / "acfs-v261103" / "split-encode-v1"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=("validation", "test"), required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output text file (must not exist)")
    args = p.parse_args()
    with (DATA / f"{args.split}.jsonl").open(encoding="utf-8") as stream:
        ids = [json.loads(line)["window_id"] for line in stream if line.strip()]
    if args.out.exists():
        raise SystemExit(f"Output file already exists: {args.out}")
    if not 0 < args.count <= len(ids):
        raise SystemExit(f"--count must be between 1 and {len(ids)}")
    chosen = [ids[i] for i in sorted(random.Random(args.seed).sample(range(len(ids)), args.count))]
    args.out.write_bytes("".join(w + chr(10) for w in chosen).encode("utf-8"))
    print(f"Wrote {len(chosen)} window_ids to {args.out}")


if __name__ == "__main__":
    main()
