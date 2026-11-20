"""Reconstruction, one-shot ablation, unconditional generation and trajectories.

Subcommands (run from the repository root):
    make-xt      build a provisional xt file from the ACFS validation split
    reconstruct  reverse-sample every case of an xt file (and the one-shot ablation)
    generate     sample from pure noise and find training-set nearest neighbours
    trajectory   forward and reverse trajectories of fixed validation windows
See training/INTERFACE.md for the file formats.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diffusion as D  # noqa: E402

XT_FIELDS = ("case_id", "window_id", "song_id", "group_id", "split", "t", "rho", "noise_seed", "noisy_tokens")
METHODS = ("diffusion", "oneshot_argmax", "oneshot_sample")
EXPORT_TIMES = (0, 10, 25, 50, 75, 90, 100)


def pick_device(name):
    return torch.device(name if name != "auto" else "cuda" if torch.cuda.is_available() else "cpu")


def fresh_output(path):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"Output exists, choose a new path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def checkpoint_info(path, state):
    config = state["config"]
    return dict(checkpoint=str(path), checkpoint_sha256=D.sha256_file(path), checkpoint_step=state["step"],
                model_dataset=config["dataset"], train_seed=config["seed"])


# ------------------------------------------------------------------- make-xt

def make_xt(args):
    """100 validation windows x 5 t x 2 noise repetitions, closed-form corruption."""
    out = fresh_output(args.output)
    _, val_path = D.dataset_paths("acfs_full", args.data_root)
    if val_path.name != "validation.jsonl":
        raise AssertionError
    rows = D.read_rows(val_path)
    D.clean_tensor(rows)
    chosen = sorted(random.Random(args.seed).sample(range(len(rows)), args.windows))
    cases = []
    for index in chosen:
        row = rows[index]
        for t in args.times:
            for rep in range(args.repeats):
                noise_seed = D.stable_seed("provisional-xt", row["window_id"], t, rep)
                generator = torch.Generator().manual_seed(noise_seed)
                x_0 = torch.tensor([row["tokens"]])
                x_t = D.corrupt(x_0, torch.tensor([t]), generator)[0].tolist()
                cases.append(dict(case_id=f"prov-{row['window_id']}-t{t:03d}-r{rep}", window_id=row["window_id"],
                                  song_id=row["song_id"], group_id=row["group_id"], split=row["split"], t=t,
                                  rho=t / D.T, noise_seed=noise_seed, noisy_tokens=x_t,
                                  clean_tokens=row["tokens"]))
    write_jsonl(out, cases)
    meta = dict(kind="provisional xt (pipeline check only)", source=str(val_path.relative_to(D.REPO_ROOT)),
                source_sha256=D.sha256_file(val_path), window_selection_seed=args.seed, windows=args.windows,
                times=args.times, repeats=args.repeats, cases=len(cases),
                noise_seed_rule="stable_seed('provisional-xt', window_id, t, repetition): first 4 bytes of "
                                "SHA-256 of the '|'-joined labels, masked to 31 bits; one CPU torch.Generator per case",
                corruption="diffusion.corrupt (closed form Q̄_t)")
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {out}")


# --------------------------------------------------------------- reconstruct

def read_xt(path):
    """Read cases and drop every field the model must not see."""
    cases = []
    seen = set()
    for number, row in enumerate(D.read_rows(path), 1):
        missing = [field for field in XT_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{path}:{number} lacks {missing}")
        case = {field: row[field] for field in XT_FIELDS}  # clean_tokens is never copied
        D.check_tokens(case["noisy_tokens"], case["case_id"])
        if type(case["t"]) is not int or not 0 <= case["t"] <= D.T:
            raise ValueError(f"{path}:{number} has t outside 0..{D.T}")
        if case["case_id"] in seen:
            raise ValueError(f"Duplicate case_id {case['case_id']}")
        seen.add(case["case_id"])
        cases.append(case)
    return cases


def processing_batches(cases, batch_size):
    """Stable sort by t (descending), then file order; consecutive blocks of batch_size."""
    order = sorted(range(len(cases)), key=lambda i: (-cases[i]["t"], i))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def reconstruct(args):
    device = pick_device(args.device)
    out = fresh_output(args.output)
    model, state = D.load_checkpoint(args.checkpoint, device)
    info = checkpoint_info(args.checkpoint, state)
    cases = read_xt(args.xt)
    batches = processing_batches(cases, args.batch_size)
    methods = args.methods
    results = {i: [] for i in range(len(cases))}
    clock = time.time()
    for method in methods:
        runs = 1 if method == "oneshot_argmax" else args.samples
        for sample_index in range(runs):
            seed = None if method == "oneshot_argmax" else D.stable_seed(args.seed, method, sample_index)
            generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
            for batch in batches:
                x_t = torch.tensor([cases[i]["noisy_tokens"] for i in batch], device=device)
                t = torch.tensor([cases[i]["t"] for i in batch], device=device)
                if method == "diffusion":
                    x_0, _ = D.reverse_chain(model, x_t, t, generator)
                else:
                    x_0 = D.one_shot(model, x_t, t, generator, mode=method.split("_")[1])
                for i, tokens in zip(batch, x_0.tolist()):
                    case = cases[i]
                    results[i].append(dict(
                        case_id=case["case_id"], window_id=case["window_id"], t=case["t"], method=method,
                        model_dataset=info["model_dataset"], train_seed=info["train_seed"],
                        checkpoint_sha256=info["checkpoint_sha256"], sample_index=sample_index,
                        sample_seed=seed, **D.describe(tokens)))
    write_jsonl(out, (row for i in range(len(cases)) for row in results[i]))
    meta = dict(
        input=str(args.xt), input_sha256=D.sha256_file(args.xt), cases=len(cases),
        **info, methods=methods, samples_per_case={m: 1 if m == "oneshot_argmax" else args.samples for m in methods},
        base_seed=args.seed,
        sample_seed_rule="stable_seed(base_seed, method, sample_index); one device generator per "
                         "(method, sample_index), consumed batch by batch in the processing order",
        batch_size=args.batch_size,
        processing_order="cases stably sorted by descending t, then by input line; consecutive blocks of batch_size",
        batches=[[cases[i]["case_id"] for i in batch] for batch in batches],
        output_order="input line order; within a case: method order, then sample_index",
        sampler="inverse-CDF categorical draw from diffusion.reverse_probs at every step t..1; rows with t=0 are copied",
        device=str(device), torch=torch.__version__, seconds=round(time.time() - clock, 1),
        ignored_input_fields="clean_tokens and any field outside the documented xt schema")
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {sum(map(len, results.values()))} rows for {len(cases)} cases to {out} "
          f"in {meta['seconds']} s")


# ------------------------------------------------------------------ generate

def shifted(tokens, shift):
    """Transpose pitch tokens by `shift` semitones; out-of-range pitches become -1."""
    pitch = tokens >= 2
    moved = tokens + shift
    moved = torch.where((moved < 2) | (moved >= D.K), torch.full_like(moved, -1), moved)
    return torch.where(pitch, moved, tokens)


@torch.no_grad()
def nearest_neighbours(generated, train, device, chunk=4096):
    """Exact Hamming nearest neighbour, and the minimum over transpositions -12..+12."""
    gen = generated.to(device)
    shifts = list(range(-12, 13))
    variants = [shifted(gen, s) for s in shifts]
    n = len(gen)
    best = {name: [torch.full((n,), 10 ** 6, device=device), torch.zeros(n, dtype=torch.long, device=device),
                   torch.zeros(n, dtype=torch.long, device=device)] for name in ("plain", "transposed")}
    for start in range(0, len(train), chunk):
        block = train[start:start + chunk].to(device)
        dist = torch.stack([(g[:, None, :] != block[None, :, :]).sum(-1) for g in variants])  # [S, G, chunk]
        per_window, shift_index = dist.min(dim=0)            # first (lowest) shift on ties
        candidates = dict(plain=(dist[shifts.index(0)], torch.full_like(per_window, 0)),
                          transposed=(per_window, torch.tensor(shifts, device=device)[shift_index]))
        for name, (values, window_shift) in candidates.items():
            value, index = values.min(dim=1)                  # first (earliest) window on ties
            d, i, s = best[name]
            better = value < d                                # earlier chunks win ties
            d[better] = value[better]
            i[better] = index[better] + start
            s[better] = window_shift[better, index[better]]
    return {name: tuple(x.cpu().tolist() for x in value) for name, value in best.items()}


def generate(args):
    device = pick_device(args.device)
    out = fresh_output(args.output)
    model, state = D.load_checkpoint(args.checkpoint, device)
    info = checkpoint_info(args.checkpoint, state)
    seed = D.stable_seed(args.seed, "generate", info["model_dataset"], info["train_seed"])
    generator = torch.Generator(device=device).manual_seed(seed)
    clock = time.time()
    x_T = torch.randint(D.K, (args.count, D.LENGTH), device=device, generator=generator)
    t = torch.full((args.count,), D.T, dtype=torch.long, device=device)
    x_0, _ = D.reverse_chain(model, x_T, t, generator)
    rows = [dict(case_id=f"gen-{info['model_dataset']}-s{info['train_seed']}-{n:03d}", window_id=None, t=D.T,
                 method="diffusion", model_dataset=info["model_dataset"], train_seed=info["train_seed"],
                 checkpoint_sha256=info["checkpoint_sha256"], sample_index=n, sample_seed=seed,
                 initial_tokens=x_T[n].tolist(), **D.describe(tokens)) for n, tokens in enumerate(x_0.tolist())]

    train_path = Path(state["config"]["train_path"])
    train_path = train_path if train_path.is_absolute() else D.REPO_ROOT / train_path
    train_rows = D.read_rows(train_path)
    train = D.clean_tensor(train_rows)
    decoded = torch.tensor([row["decoded_tokens"] for row in rows])
    nn_result = nearest_neighbours(decoded, train, device)
    for n, row in enumerate(rows):
        d, i, _ = (v[n] for v in nn_result["plain"])
        td, ti, ts = (v[n] for v in nn_result["transposed"])
        row.update(nn_hamming=d, nn_window_id=train_rows[i]["window_id"],
                   nn_transposed_hamming=td, nn_transposed_window_id=train_rows[ti]["window_id"],
                   nn_transposed_shift=ts)
    write_jsonl(out, rows)
    meta = dict(**info, count=args.count, base_seed=args.seed, sample_seed=seed,
                sample_seed_rule="stable_seed(base_seed, 'generate', model_dataset, train_seed); one device "
                                 "generator draws x_T (uniform over 51 classes) and then every reverse step",
                batch_size=args.count, nearest_neighbour_reference=str(train_path.relative_to(D.REPO_ROOT)),
                nearest_neighbour_reference_sha256=D.sha256_file(train_path), reference_windows=len(train_rows),
                distance="Hamming distance over the 64 decoded tokens; transposed: pitch tokens of the generated "
                         "window shifted by -12..+12 semitones (pitches leaving 38-86 never match), minimum over "
                         "shifts; ties keep the first training window and the lowest shift",
                device=str(device), torch=torch.__version__, seconds=round(time.time() - clock, 1))
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} generated windows to {out} in {meta['seconds']} s")


# ---------------------------------------------------------------- trajectory

def choose_trajectory_windows(rows, count, seed):
    """Shuffle the validation windows with `seed`; take the first `count` from distinct groups."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    chosen, groups = [], set()
    for index in order:
        if rows[index]["group_id"] not in groups:
            chosen.append(index)
            groups.add(rows[index]["group_id"])
        if len(chosen) == count:
            return chosen
    raise ValueError("Not enough distinct groups")


def trajectory(args):
    device = pick_device(args.device)
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, state = D.load_checkpoint(args.checkpoint, device)
    info = checkpoint_info(args.checkpoint, state)
    _, val_path = D.dataset_paths("acfs_full", args.data_root)
    rows = D.read_rows(val_path)
    D.clean_tensor(rows)
    chosen = choose_trajectory_windows(rows, args.windows, args.seed)
    records = []
    for index in chosen:
        row = rows[index]
        tag = row["window_id"].replace(":", "-").replace("@", "-bar")
        forward_seed = D.stable_seed(args.seed, "forward", row["window_id"])
        g = torch.Generator().manual_seed(forward_seed)
        x = torch.tensor([row["tokens"]])
        forward = [x[0].tolist()]
        for step in range(1, D.T + 1):
            x = D.forward_step(x, step, g)
            forward.append(x[0].tolist())
        reverse = {}
        for start in (50, 100):
            seed = D.stable_seed(args.seed, "reverse", row["window_id"], start)
            rg = torch.Generator(device=device).manual_seed(seed)
            x_t = torch.tensor([forward[start]], device=device)
            _, states = D.reverse_chain(model, x_t, torch.tensor([start], device=device), rg, keep_states=True)
            reverse[start] = dict(sample_seed=seed, states={s: states[s][0].tolist() for s in sorted(states)})
        exports = []
        for name, states in (("forward", dict(enumerate(forward))),
                             ("reverse_from_050", reverse[50]["states"]),
                             ("reverse_from_100", reverse[100]["states"])):
            for s in EXPORT_TIMES:
                if s in states:
                    prefix = out_dir / tag / f"{name}_t{s:03d}"
                    repaired = D.export_score(prefix, states[s])
                    exports.append(dict(chain=name, t=s, midi=f"{tag}/{prefix.name}.mid",
                                        musicxml=f"{tag}/{prefix.name}.musicxml", leading_hold_repaired=repaired))
        records.append(dict(
            window_id=row["window_id"], song_id=row["song_id"], group_id=row["group_id"],
            clean_tokens=row["tokens"], forward_seed=forward_seed,
            forward_states=forward,  # index s holds x_s, s = 0..100, sampled step by step with Q_s
            reverse_from_050=dict(sample_seed=reverse[50]["sample_seed"],
                                  states=[reverse[50]["states"][s] for s in range(50, -1, -1)]),
            reverse_from_100=dict(sample_seed=reverse[100]["sample_seed"],
                                  states=[reverse[100]["states"][s] for s in range(100, -1, -1)]),
            final_from_050=D.describe(reverse[50]["states"][0]),
            final_from_100=D.describe(reverse[100]["states"][0]),
            exports=exports))
    payload = dict(
        **info, selection_seed=args.seed,
        selection_rule=f"shuffle ACFS validation windows with random.Random({args.seed}); keep the first "
                       f"{args.windows} windows from distinct group_id",
        forward_rule="x_s ~ Q_s(x_{s-1}, .) for s = 1..100 with a CPU generator seeded by "
                     "stable_seed(selection_seed, 'forward', window_id)",
        reverse_rule="from the forward state at t = 50 and t = 100, reverse_chain to 0 with a device generator "
                     "seeded by stable_seed(selection_seed, 'reverse', window_id, start); reverse states are "
                     "listed from the start step down to 0",
        export_rule="MIDI and MusicXML at t in (0, 10, 25, 50, 75, 90, 100) where the chain has that state; a "
                    "leading HOLD is written as REST only for the score",
        windows=records)
    (out_dir / "trajectories.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote trajectories for {[rows[i]['window_id'] for i in chosen]} to {out_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=D.DATA_ROOT)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("make-xt")
    m.add_argument("--output", required=True)
    m.add_argument("--windows", type=int, default=100)
    m.add_argument("--times", type=int, nargs="+", default=list(D.VALIDATION_TIMES))
    m.add_argument("--repeats", type=int, default=2)
    m.add_argument("--seed", type=int, default=20261120)

    r = sub.add_parser("reconstruct")
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--xt", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    r.add_argument("--samples", type=int, default=3)
    r.add_argument("--batch-size", type=int, default=500)
    r.add_argument("--seed", type=int, default=20261121)

    g = sub.add_parser("generate")
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--count", type=int, default=100)
    g.add_argument("--seed", type=int, default=20261122)

    j = sub.add_parser("trajectory")
    j.add_argument("--checkpoint", type=Path, required=True)
    j.add_argument("--output-dir", type=Path, required=True)
    j.add_argument("--windows", type=int, default=3)
    j.add_argument("--seed", type=int, default=20261123)

    args = p.parse_args(argv)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    commands = {"make-xt": make_xt, "reconstruct": reconstruct, "generate": generate, "trajectory": trajectory}
    commands[args.command](args)


if __name__ == "__main__":
    main()
