"""Train the x_0-predicting denoiser on one experiment entry.

Example (from the repository root):
    python training/train.py --dataset acfs_full --steps 20000 --width 64 --layers 2 \
        --seed 0 --run-dir ../runs/h2-acfs_full-s0
"""

import argparse
import csv
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diffusion as D  # noqa: E402


def git_state():
    def run(*args):
        return subprocess.run(["git", *args], cwd=D.REPO_ROOT, capture_output=True, text=True).stdout.strip()

    commit = run("rev-parse", "HEAD")
    dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    return commit or None, dirty


def fixed_corruption(clean, times=D.VALIDATION_TIMES, seed=D.VALIDATION_SEED):
    """The reference implementation's fixed validation corruption.

    Draw order: for each t, for each block of VALIDATION_BATCH windows.
    Returns a list of (t, x_0, x_t) on the CPU.
    """
    generator = torch.Generator().manual_seed(seed)
    out = []
    for step in times:
        parts = []
        for x_0 in clean.split(D.VALIDATION_BATCH):
            t = torch.full((len(x_0),), step, dtype=torch.long)
            parts.append(D.corrupt(x_0, t, generator))
        out.append((step, clean, torch.cat(parts)))
    return out


@torch.no_grad()
def fixed_loss(model, fixed, device, batch=2048):
    """Mean cross-entropy per slot over all fixed (t, x_t) pairs, and per t."""
    was_training = model.training
    model.eval()
    per_t = {}
    for step, x_0, x_t in fixed:
        total = 0.0
        for a, b in zip(x_0.split(batch), x_t.split(batch)):
            a, b = a.to(device), b.to(device)
            t = torch.full((len(a),), step, dtype=torch.long, device=device)
            logits = model(b, t)
            total += F.cross_entropy(logits.reshape(-1, D.K), a.reshape(-1), reduction="sum").item()
        per_t[step] = total / x_0.numel()
    model.train(was_training)
    return sum(per_t.values()) / len(per_t), per_t


@torch.no_grad()
def memorizer_floor(clean, times, seed=D.VALIDATION_SEED):
    """Cross-entropy of the exact posterior of a model that knows the training set.

    For a corrupted window x_t the posterior over the N training windows is
    proportional to prod_j Q̄_t[w_j, x_j]; the per-slot prediction mixes the
    windows by that posterior. This is the lowest loss a perfect memoriser can
    reach on the same corruption, so it is the target of the overfitting check.
    """
    fixed = fixed_corruption(clean, times, seed)
    floor = {}
    for step, x_0, x_t in fixed:
        r = step / D.T
        keep, other = math.log(1 - r + r / D.K), math.log(r / D.K)
        match = (x_t[:, None, :] == clean[None, :, :])               # [M, N, L]
        log_like = torch.where(match, keep, other).double().sum(-1)  # [M, N]
        post = log_like.log_softmax(dim=-1).exp()                    # [M, N]
        hit = (clean[None, :, :] == x_0[:, None, :]).double()        # [M, N, L]
        prob = (post[:, :, None] * hit).sum(1).clamp_min(1e-300)
        floor[step] = (-prob.log()).mean().item()
    return floor


class WindowSampler:
    """Epoch-wise shuffled batches with a resumable state (drop_last)."""

    def __init__(self, n, batch_size, seed):
        self.n, self.batch_size = n, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order, self.position = None, n

    def next(self):
        if self.position + self.batch_size > self.n:
            self.order = torch.randperm(self.n, generator=self.generator)
            self.position = 0
        index = self.order[self.position:self.position + self.batch_size]
        self.position += self.batch_size
        return index

    def state_dict(self):
        return dict(generator=self.generator.get_state(), order=self.order, position=self.position)

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"])
        self.order, self.position = state["order"], state["position"]


def lr_factor(step, warmup, total):
    """Linear warm-up to 1, then cosine decay to 0 at `total` updates."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def environment(device):
    info = dict(python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                host=socket.gethostname(), device=str(device), torch_threads=torch.get_num_threads(),
                omp_num_threads=os.environ.get("OMP_NUM_THREADS"), cpu_count=os.cpu_count())
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def save_checkpoint(path, model, optimizer, scheduler, sampler, noise, step, val_loss, config):
    state = dict(config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), sampler=sampler.state_dict(),
                 noise_rng=noise.get_state(), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 step=step, val_loss=val_loss)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(args):
    device = torch.device(args.device if args.device != "auto" else
                          "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    run = Path(args.run_dir)
    if run.exists() and any(run.iterdir()) and not args.resume:
        raise SystemExit(f"Run directory is not empty: {run}")
    run.mkdir(parents=True, exist_ok=True)

    train_path, val_path = D.dataset_paths(args.dataset, args.data_root)
    vocab_path = train_path.parent / "vocab.json"
    vocab_sha = D.verify_vocab(vocab_path)
    if D.verify_vocab(val_path.parent / "vocab.json") != vocab_sha:
        raise ValueError("Train and validation vocabularies differ")
    train_rows = D.read_rows(train_path)
    val_rows = D.read_rows(val_path)
    clean = D.clean_tensor(train_rows[:args.train_windows or None])
    val = D.clean_tensor(val_rows[:args.validation_windows or None])
    if args.batch_size > len(clean):
        raise ValueError("Batch size exceeds the number of training windows")

    model_cfg = dict(width=args.width, layers=args.layers, heads=args.heads,
                     ff_mult=args.ff_mult, dropout=args.dropout)
    torch.manual_seed(args.seed)
    model = D.build_model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, args.warmup, args.schedule_steps))
    sampler = WindowSampler(len(clean), args.batch_size, args.seed)
    noise = torch.Generator(device=device).manual_seed(D.stable_seed(args.seed, "training-noise"))

    commit, dirty = git_state()
    config = dict(
        dataset=args.dataset, seed=args.seed, model=model_cfg, parameters=D.parameter_count(model),
        optimizer=dict(name="AdamW", lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)),
        schedule=dict(warmup=args.warmup, decay="cosine to 0", schedule_updates=args.schedule_steps),
        grad_clip=args.grad_clip, batch_size=args.batch_size, steps=args.steps,
        validate_every=args.validate_every, save_steps=args.save_steps,
        validation_times=list(D.VALIDATION_TIMES), validation_seed=D.VALIDATION_SEED,
        validation_batch=D.VALIDATION_BATCH,
        train_path=str(train_path.relative_to(D.REPO_ROOT)), validation_path=str(val_path.relative_to(D.REPO_ROOT)),
        train_sha256=D.sha256_file(train_path), validation_sha256=D.sha256_file(val_path),
        vocab_path=str(vocab_path.relative_to(D.REPO_ROOT)), vocab_sha256=vocab_sha,
        train_windows=len(clean), validation_windows=len(val),
        train_windows_total=len(train_rows), validation_windows_total=len(val_rows),
        T=D.T, K=D.K, length=D.LENGTH, git_commit=commit, git_dirty=dirty,
        environment=environment(device), command=" ".join(sys.argv).replace(str(Path.home()), "~"),
    )

    start_step, best, best_step, elapsed_before = 0, float("inf"), None, 0.0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.load_state_dict(state["sampler"])
        noise.set_state(state["noise_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step = state["step"]
        summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
        best, best_step = summary.get("best_val_loss", best), summary.get("best_step")
        elapsed_before = summary.get("elapsed_seconds", 0.0)
        config["resumed_from"] = str(args.resume)
    (run / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fixed_val = fixed_corruption(val)
    fixed_train = fixed_corruption(clean[:args.train_eval_windows]) if args.train_eval_windows else None
    if args.memorizer_floor:
        floor = memorizer_floor(clean, D.VALIDATION_TIMES)
        floor_all = memorizer_floor(clean, range(1, D.T + 1))
        (run / "memorizer_floor.json").write_text(json.dumps(dict(
            note="Cross-entropy of the exact posterior of a model that has memorised the training windows",
            windows=len(clean), per_t=floor, mean_validation_times=sum(floor.values()) / len(floor),
            mean_t_1_to_100=sum(floor_all.values()) / len(floor_all)), indent=2) + "\n")

    fields = ["step", "train_loss", "val_loss", "lr", "elapsed_seconds", "validation_seconds"] +         [f"val_t{t}" for t in D.VALIDATION_TIMES]
    if fixed_train:
        fields += ["train_eval_loss"] + [f"train_eval_t{t}" for t in D.VALIDATION_TIMES]
    log_path = run / "train_log.csv"
    new_log = not log_path.exists()
    log_file = log_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(log_file, fieldnames=fields)
    if new_log:
        writer.writeheader()

    print(f"{args.dataset} on {device} | {len(clean)} train / {len(val)} validation windows | "
          f"{config['parameters']:,} parameters | commit {commit} dirty={dirty}", flush=True)
    clock = time.time()
    running, count, val_seconds = 0.0, 0, 0.0
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        x_0 = clean[sampler.next()].to(device, non_blocking=True)
        t = torch.randint(1, D.T + 1, (len(x_0),), device=device, generator=noise)
        x_t = D.corrupt(x_0, t, noise)
        logits = model(x_t, t)
        loss = F.cross_entropy(logits.reshape(-1, D.K), x_0.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        running += loss.item()
        count += 1
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"Non-finite loss at step {step}")

        if step % args.validate_every == 0 or step == args.steps:
            val_clock = time.time()
            val_loss, per_t = fixed_loss(model, fixed_val, device)
            val_seconds += time.time() - val_clock
            row = dict(step=step, train_loss=f"{running / count:.6f}", val_loss=f"{val_loss:.6f}",
                       lr=f"{lr:.3e}", elapsed_seconds=f"{elapsed_before + time.time() - clock:.1f}",
                       validation_seconds=f"{val_seconds:.1f}",
                       **{f"val_t{s}": f"{v:.6f}" for s, v in per_t.items()})
            if fixed_train:
                train_eval, train_per_t = fixed_loss(model, fixed_train, device)
                row["train_eval_loss"] = f"{train_eval:.6f}"
                row.update({f"train_eval_t{s}": f"{v:.6f}" for s, v in train_per_t.items()})
            writer.writerow(row)
            log_file.flush()
            print(" | ".join(f"{k} {v}" for k, v in row.items()), flush=True)
            running, count = 0.0, 0
            if val_loss < best:
                best, best_step = val_loss, step
                save_checkpoint(run / "best.pt", model, optimizer, scheduler, sampler, noise, step, val_loss, config)
            last_val = val_loss
        if step in args.save_steps:
            save_checkpoint(run / f"step{step}.pt", model, optimizer, scheduler, sampler, noise, step,
                            None, config)
        if step == args.steps:
            save_checkpoint(run / "last.pt", model, optimizer, scheduler, sampler, noise, step, last_val, config)
    log_file.close()

    summary = dict(best_step=best_step, best_val_loss=best, last_step=args.steps, last_val_loss=last_val,
                   elapsed_seconds=round(elapsed_before + time.time() - clock, 1),
                   parameters=config["parameters"], selection_rule="lowest fixed-corruption validation CE")
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=D.DATASETS, required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-root", type=Path, default=D.DATA_ROOT)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--steps", type=int, required=True, help="Number of updates to run")
    p.add_argument("--schedule-steps", type=int,
                   help="Length of the cosine schedule (default: --steps); larger values stop before the decay ends")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--validate-every", type=int, default=500)
    p.add_argument("--save-steps", type=int, nargs="*", default=[])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-windows", type=int, default=0, help="Use only the first N training windows")
    p.add_argument("--validation-windows", type=int, default=0, help="Use only the first N validation windows")
    p.add_argument("--train-eval-windows", type=int, default=0,
                   help="Also log the fixed-corruption CE on the first N training windows")
    p.add_argument("--memorizer-floor", action="store_true",
                   help="Write the exact-posterior loss floor of a memoriser of the training windows")
    p.add_argument("--resume", type=Path)
    args = p.parse_args(argv)
    args.schedule_steps = args.schedule_steps or args.steps
    if args.schedule_steps < args.steps:
        p.error("--schedule-steps must be at least --steps")
    if min(args.steps, args.batch_size, args.validate_every, args.warmup) <= 0:
        p.error("steps, batch size, validation interval and warm-up must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
