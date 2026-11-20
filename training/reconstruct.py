"""Reconstruct every case of an xt file with the trained models.

For each case and each model the output has 7 rows: 3 reverse-diffusion
samples (method "diffusion"), 1 one-shot argmax read-out ("oneshot_argmax")
and 3 one-shot samples ("oneshot_sample"). The model sees only noisy_tokens
and t; clean_tokens and every other field are discarded on reading.

Example (from the repository root):
    python training/reconstruct.py --xt xt-validation.jsonl --out recon-validation
Writes recon-validation/recon__<run>.jsonl.gz and recon__<run>.meta.json.
"""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch
except ImportError:
    sys.exit("Error: PyTorch is not installed.\n"
             "Fix: activate the virtual environment and run  python -m pip install -r training/requirements.txt")

import diffusion as D  # noqa: E402

CHECKPOINTS = Path(__file__).resolve().parent / "checkpoints"
RUNS = ("h2-acfs_full-s0", "h2-acfs_full-s1", "h2-acfs_full-s2",
        "h3-acfs_matched-s0", "h3-acfs_matched-s1", "h3-acfs_matched-s2",
        "h3-cantopop-s0", "h3-cantopop-s1", "h3-cantopop-s2")
GROUPS = {"all": RUNS, "h2": RUNS[:3], "h3": RUNS[3:]}
XT_FIELDS = ("case_id", "window_id", "song_id", "group_id", "split", "t", "rho", "noise_seed", "noisy_tokens")
METHODS = ("diffusion", "oneshot_argmax", "oneshot_sample")
DEFAULT_SEED = 20261121


class UserError(Exception):
    """An input or setup problem with a message saying how to fix it."""

    def __init__(self, problem, fix):
        super().__init__(problem)
        self.problem, self.fix = problem, fix


# ------------------------------------------------------------------ inputs

def open_text(path):
    path = Path(path)
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_xt(path):
    """Read cases and keep only the documented xt fields (clean_tokens is dropped)."""
    path = Path(path)
    if not path.is_file():
        raise UserError(f"xt file not found: {path}", "Check the path given to --xt.")
    cases, seen = [], set()
    try:
        with open_text(path) as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise UserError(f"{path.name} line {number} is not valid JSON ({error.msg})",
                                    "Each line must be one JSON object; regenerate the file with make_xt.py.")
                missing = [field for field in XT_FIELDS if field not in row]
                if missing:
                    raise UserError(f"{path.name} line {number} lacks the field(s) {missing}",
                                    "Use an xt file written by training/make_xt.py (see INTERFACE.md).")
                case = {field: row[field] for field in XT_FIELDS}
                tokens, t = case["noisy_tokens"], case["t"]
                if (not isinstance(tokens, list) or len(tokens) != D.LENGTH
                        or any(type(x) is not int or not 0 <= x < D.K for x in tokens)):
                    raise UserError(f"{path.name} line {number}: noisy_tokens must be 64 integers in 0..50",
                                    "Regenerate the file with make_xt.py.")
                if type(t) is not int or not 0 <= t <= D.T:
                    raise UserError(f"{path.name} line {number}: t must be an integer in 0..100",
                                    "Regenerate the file with make_xt.py.")
                if case["case_id"] in seen:
                    raise UserError(f"{path.name} line {number}: duplicate case_id {case['case_id']}",
                                    "Every case_id must be unique in one file.")
                seen.add(case["case_id"])
                cases.append(case)
    except (OSError, EOFError, UnicodeDecodeError) as error:
        raise UserError(f"Cannot read {path}: {error}", "Check that the file is a complete .jsonl or .jsonl.gz file.")
    if not cases:
        raise UserError(f"{path} contains no cases", "Check the xt file.")
    return cases


def processing_batches(cases, batch_size):
    """Stable sort by t (descending), then input line; consecutive blocks of batch_size."""
    order = sorted(range(len(cases)), key=lambda i: (-cases[i]["t"], i))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def select_runs(text):
    if text in GROUPS:
        return list(GROUPS[text])
    runs = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [run for run in runs if run not in RUNS]
    if unknown or not runs:
        raise UserError(f"Unknown run name(s): {unknown or text!r}",
                        "Use all, h2, h3 or comma-separated names from: " + ", ".join(RUNS))
    return list(dict.fromkeys(runs))


def load_manifest():
    path = CHECKPOINTS / "manifest.json"
    if not path.is_file():
        raise UserError(f"Missing {path}", "Clone the repository again and check out the train/c branch.")
    return {entry["run"]: entry for entry in json.loads(path.read_text(encoding="utf-8"))["runs"]}


def load_run(run, manifest, device):
    path = CHECKPOINTS / f"{run}.pt"
    if not path.is_file():
        raise UserError(f"Checkpoint not found: {path}",
                        "Check out the train/c branch (git switch train/c) from the repository root.")
    digest = D.sha256_file(path)
    if digest != manifest[run]["sha256"]:
        raise UserError(f"{path.name} has SHA-256 {digest}, but manifest.json lists {manifest[run]['sha256']}",
                        "The file was changed or only partly downloaded. Run  git checkout -- training/checkpoints  "
                        "or clone the repository again.")
    model, state = D.load_checkpoint(path, device)
    config = state["config"]
    return model, dict(run=run, checkpoint=f"training/checkpoints/{path.name}", checkpoint_sha256=digest,
                       checkpoint_step=state["step"], model_dataset=config["dataset"], train_seed=config["seed"])


# -------------------------------------------------------------- sampling

class Progress:
    def __init__(self, label, total, every=5.0):
        self.label, self.total, self.every = label, max(total, 1), every
        self.done, self.start, self.last = 0, time.time(), 0.0

    def advance(self, units, note=""):
        self.done += units
        now = time.time()
        if now - self.last >= self.every or self.done >= self.total:
            self.last = now
            elapsed = now - self.start
            fraction = self.done / self.total
            left = elapsed * (1 - fraction) / fraction if fraction > 0 else float("nan")
            print(f"  {self.label}: {100 * fraction:5.1f}%  elapsed {elapsed:6.0f} s  "
                  f"about {left:6.0f} s left  {note}", flush=True)


def reconstruct_rows(model, info, cases, samples, batch_size, base_seed, device, progress=None):
    """Sample every case, then yield the output rows.

    Rows come in input line order; within a case diffusion, oneshot_argmax,
    oneshot_sample, each by sample_index. Results are kept as uint8 tensors
    until writing, so memory stays small for large xt files.
    """
    batches = processing_batches(cases, batch_size)
    plan = ([("diffusion", s) for s in range(samples)] + [("oneshot_argmax", 0)]
            + [("oneshot_sample", s) for s in range(samples)])
    outputs = []
    for method, sample_index in plan:
        seed = None if method == "oneshot_argmax" else D.stable_seed(base_seed, method, sample_index)
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        tokens = torch.empty((len(cases), D.LENGTH), dtype=torch.uint8)
        for batch in batches:
            x_t = torch.tensor([cases[i]["noisy_tokens"] for i in batch], device=device)
            t = torch.tensor([cases[i]["t"] for i in batch], device=device)
            if method == "diffusion":
                x_0, _ = D.reverse_chain(model, x_t, t, generator)
                units = int(t.sum().item())
            else:
                x_0 = D.one_shot(model, x_t, t, generator, mode=method.split("_")[1])
                units = len(batch)
            tokens[torch.tensor(batch)] = x_0.to("cpu", torch.uint8)
            if progress:
                progress.advance(units, f"{method} {sample_index + 1}")
        outputs.append((method, sample_index, seed, tokens))
    for i, case in enumerate(cases):
        for method, sample_index, seed, tokens in outputs:
            yield dict(case_id=case["case_id"], window_id=case["window_id"], t=case["t"], method=method,
                       model_dataset=info["model_dataset"], train_seed=info["train_seed"],
                       checkpoint_sha256=info["checkpoint_sha256"], sample_index=sample_index,
                       sample_seed=seed, **D.describe(tokens[i].tolist()))


def work_units(cases, samples):
    return samples * sum(c["t"] for c in cases) + (1 + samples) * len(cases)


def write_gzip_jsonl(path, rows):
    """Deterministic gzip (mtime 0, no file name), streamed.

    Returns the row count, SHA-256 and size of the uncompressed text and of the gzip file.
    """
    raw_hash, raw_size, count = sha256(), 0, 0
    tmp = Path(str(path) + ".partial")
    with tmp.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, compresslevel=9, mtime=0) as stream:
            for row in rows:
                line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                raw_hash.update(line)
                raw_size += len(line)
                count += 1
                stream.write(line)
    tmp.replace(path)
    return count, raw_hash.hexdigest(), raw_size, D.sha256_file(path), Path(path).stat().st_size


def environment(device, threads):
    return dict(os=platform.platform(), cpu=D.cpu_name(), python=platform.python_version(), torch=torch.__version__,
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                torch_threads=torch.get_num_threads() if device.type == "cpu" else None,
                requested_threads=threads)


def pick_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise UserError("--device cuda was requested, but this PyTorch build sees no CUDA GPU",
                        "Use --device cpu (the default auto also falls back to the CPU).")
    return torch.device(name if name != "auto" else "cuda" if torch.cuda.is_available() else "cpu")


def run(args):
    runs = select_runs(args.runs)
    if args.samples < 1 or args.batch_size < 1 or args.threads < 1:
        raise UserError("--samples, --batch-size and --threads must be positive", "Use positive integers.")
    out = Path(args.out)
    targets = [(out / f"recon__{r}.jsonl.gz", out / f"recon__{r}.meta.json") for r in runs]
    existing = [str(p) for pair in targets for p in pair if p.exists()]
    if existing:
        raise UserError(f"Output file(s) already exist: {existing[:3]}",
                        "Choose a new --out directory or move the old files away; nothing is overwritten.")
    device = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    manifest = load_manifest()
    cases = read_xt(args.xt)
    input_sha = D.sha256_file(args.xt)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(cases)} cases from {args.xt} (SHA-256 {input_sha[:12]}...), {len(runs)} model(s), "
          f"device {device}, threads {torch.get_num_threads()}", flush=True)
    order_digest = sha256("\n".join(cases[i]["case_id"] for b in processing_batches(cases, args.batch_size)
                                    for i in b).encode("utf-8")).hexdigest()
    overall = time.time()
    for k, run_name in enumerate(runs, 1):
        gz_path, meta_path = out / f"recon__{run_name}.jsonl.gz", out / f"recon__{run_name}.meta.json"
        model, info = load_run(run_name, manifest, device)
        print(f"[{k}/{len(runs)}] {run_name} (checkpoint step {info['checkpoint_step']})", flush=True)
        clock = time.time()
        progress = Progress(run_name, work_units(cases, args.samples))
        rows = reconstruct_rows(model, info, cases, args.samples, args.batch_size, args.seed, device, progress)
        count, raw_sha, raw_size, gz_sha, gz_size = write_gzip_jsonl(gz_path, rows)
        seconds = round(time.time() - clock, 1)
        meta = dict(
            tool="training/reconstruct.py", run=run_name, input=Path(args.xt).name, input_sha256=input_sha,
            cases=len(cases), rows=count, **{k2: info[k2] for k2 in
                                                ("checkpoint", "checkpoint_sha256", "checkpoint_step",
                                                 "model_dataset", "train_seed")},
            rows_per_case=dict(diffusion=args.samples, oneshot_argmax=1, oneshot_sample=args.samples),
            samples=args.samples, base_seed=args.seed,
            sample_seed_rule="stable_seed(base_seed, method, sample_index): first 4 bytes of SHA-256 of "
                             "'base_seed|method|sample_index', masked to 31 bits; one generator per "
                             "(method, sample_index), consumed batch by batch in the processing order; "
                             "oneshot_argmax is deterministic and has sample_seed null",
            batch_size=args.batch_size,
            processing_order="cases stably sorted by descending t, then by input line; consecutive blocks "
                             "of batch_size",
            processing_order_sha256=order_digest,
            output_order="input line order; within a case diffusion, oneshot_argmax, oneshot_sample, "
                         "each by sample_index",
            ignored_input_fields="clean_tokens and every field outside " + ", ".join(XT_FIELDS),
            output=gz_path.name, output_sha256_uncompressed=raw_sha, output_sha256_gzip=gz_sha,
            output_bytes_uncompressed=raw_size, output_bytes_gzip=gz_size,
            gzip="compresslevel 9, mtime 0, empty file name", environment=environment(device, args.threads),
            seconds=seconds)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"  wrote {gz_path} ({count} rows) in {seconds} s", flush=True)
    print(f"Finished {len(runs)} model(s) in {time.time() - overall:.0f} s", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xt", type=Path, required=True, help="Input xt file (.jsonl or .jsonl.gz)")
    p.add_argument("--out", type=Path, required=True, help="Output directory")
    p.add_argument("--runs", default="all", help="all, h2, h3 or comma-separated run names (default all)")
    p.add_argument("--samples", type=int, default=3, help="Samples per case for diffusion and oneshot_sample")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=4, help="CPU threads (default 4)")
    p.add_argument("--batch-size", type=int, default=500, help="Cases per batch (default 500)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    try:
        run(args)
    except UserError as error:
        print(f"Error: {error.problem}\nFix: {error.fix}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("Interrupted. Files of finished models are complete; delete the unfinished output directory "
              "or choose a new --out before running again.", file=sys.stderr)
        sys.exit(130)
    except MemoryError:
        print("Error: out of memory.\nFix: use a smaller --batch-size, e.g. 100.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
