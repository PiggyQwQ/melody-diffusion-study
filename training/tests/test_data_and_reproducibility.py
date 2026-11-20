"""Data loading, the xt builder, reconstruction and same-seed reproducibility."""

import gzip
import hashlib
import json
import shutil

import pytest
import torch

import diffusion as D
import make_xt as MX
import reconstruct as RC
import sample as S
import train as TR


@pytest.mark.parametrize("name", D.DATASETS)
def test_training_and_validation_files_load(name):
    train_path, val_path = D.dataset_paths(name)
    for path in (train_path, val_path):
        assert path.name != "test.jsonl"
        D.verify_vocab(path.parent / "vocab.json")
        x = D.clean_tensor(D.read_rows(path))
        assert x.dim() == 2 and x.shape[1] == 64 and x.shape[0] > 0
        assert x.dtype == torch.long
        assert 0 <= x.min().item() and x.max().item() <= 50
        assert (x[:, 0] != D.HOLD).all()


def test_documented_window_counts():
    counts = {name: len(D.read_rows(D.dataset_paths(name)[0])) for name in D.DATASETS}
    assert counts == dict(acfs_full=32_656, acfs_matched=1_323, cantopop=1_323, cantopop_standalone=1_323)
    assert len(D.read_rows(D.dataset_paths("acfs_full")[1])) == 4_074


def test_clean_tensor_rejects_leading_hold():
    with pytest.raises(ValueError):
        D.clean_tensor([dict(window_id="x", tokens=[1] + [0] * 63)])


# ------------------------------------------------------------------ make_xt

def validation_ids(n):
    return [row["window_id"] for row in D.read_rows(D.dataset_paths("acfs_full")[1])[:n]]


def make(tmp_path, name, ids, times="10,90", seeds="0,1"):
    listing = tmp_path / f"{name}.txt"
    listing.write_text("\n".join(ids) + "\n", encoding="utf-8")
    out = tmp_path / f"{name}.jsonl"
    MX.main(["--split", "validation", "--window-ids", str(listing), "--times", times, "--noise-seeds", seeds,
             "--out", str(out)])
    return out, [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def test_make_xt_noise_depends_only_on_window_t_and_seed(tmp_path):
    ids = validation_ids(4)
    _, full = make(tmp_path, "full", ids)
    _, reversed_ = make(tmp_path, "reversed", ids[::-1])
    _, single = make(tmp_path, "single", [ids[2]], times="90", seeds="1")
    by_id = {case["case_id"]: case["noisy_tokens"] for case in full}
    assert {c["case_id"]: c["noisy_tokens"] for c in reversed_} == by_id
    assert single[0]["case_id"] == f"{ids[2]}|t90|n1"
    assert single[0]["noisy_tokens"] == by_id[single[0]["case_id"]]
    # different seeds give different corruptions
    assert by_id[f"{ids[0]}|t90|n0"] != by_id[f"{ids[0]}|t90|n1"]


def test_make_xt_fields_and_meta(tmp_path):
    out, cases = make(tmp_path, "x", validation_ids(3))
    assert len(cases) == 3 * 2 * 2
    assert list(cases[0]) == ["case_id", "window_id", "song_id", "group_id", "split", "t", "rho", "noise_seed",
                              "noisy_tokens", "clean_tokens"]
    assert cases[0]["case_id"] == f"{cases[0]['window_id']}|t10|n0" and cases[0]["rho"] == 0.1
    meta = json.loads((tmp_path / "x.meta.json").read_text(encoding="utf-8"))
    assert meta["output_sha256"] == D.sha256_file(out) and meta["cases"] == 12 and meta["windows"] == 3
    with pytest.raises(SystemExit):  # never overwrite
        make(tmp_path, "x", validation_ids(3))


def test_make_xt_matches_the_closed_form_marginal():
    tokens = [7] * 64
    hits = sum(x == 7 for seed in range(400) for x in MX.corrupt_case(tokens, 60, seed))
    expected = 400 * 64 * (0.4 + 0.6 / 51)
    assert abs(hits - expected) < 5 * (expected * (1 - expected / 25600)) ** 0.5


# -------------------------------------------------------------- training

def tiny_run(tmp_path, name):
    args = TR.parse_args(["--dataset", "acfs_matched", "--run-dir", str(tmp_path / name), "--device", "cpu",
                          "--steps", "4", "--batch-size", "8", "--train-windows", "32",
                          "--validation-windows", "16", "--validate-every", "2", "--warmup", "2",
                          "--width", "16", "--layers", "1", "--seed", "3"])
    TR.train(args)
    return tmp_path / name


def test_training_is_reproducible(tmp_path):
    a, b = tiny_run(tmp_path, "a"), tiny_run(tmp_path, "b")
    rows = [line.split(",")[:4] for line in (a / "train_log.csv").read_text().splitlines()]
    assert rows == [line.split(",")[:4] for line in (b / "train_log.csv").read_text().splitlines()]
    sa, sb = torch.load(a / "last.pt", weights_only=False), torch.load(b / "last.pt", weights_only=False)
    for key in sa["model"]:
        assert torch.equal(sa["model"][key], sb["model"][key])
    for key in ("optimizer", "scheduler", "sampler", "noise_rng", "torch_rng"):
        assert key in sa
    config = json.loads((a / "config.json").read_text())
    for key in ("git_commit", "train_sha256", "validation_sha256", "vocab_sha256", "parameters", "environment"):
        assert key in config
    assert set(json.loads((a / "summary.json").read_text())) >= {"best_step", "best_val_loss", "elapsed_seconds"}


# ---------------------------------------------------------- reconstruction

@pytest.fixture
def tiny_checkpoints(tmp_path, monkeypatch):
    run = tiny_run(tmp_path, "m")
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    shutil.copy(run / "last.pt", folder / "h2-acfs_full-s0.pt")
    manifest = dict(runs=[dict(run="h2-acfs_full-s0", sha256=D.sha256_file(folder / "h2-acfs_full-s0.pt"))])
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(RC, "CHECKPOINTS", folder)
    return folder


def recon(xt, out):
    RC.main(["--xt", str(xt), "--out", str(out), "--runs", "h2-acfs_full-s0", "--device", "cpu",
             "--batch-size", "5", "--threads", "2"])
    return out / "recon__h2-acfs_full-s0.jsonl.gz"


def test_reconstruction_is_reproducible_and_ignores_clean_tokens(tmp_path, tiny_checkpoints):
    xt, cases = make(tmp_path, "xt-provisional", validation_ids(3))
    scrambled = tmp_path / "xt-scrambled.jsonl.gz"
    with gzip.open(scrambled, "wt", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(dict(case, clean_tokens=[2] * 64)) + "\n")
    outputs = [recon(xt, tmp_path / "r1"), recon(xt, tmp_path / "r2"), recon(scrambled, tmp_path / "r3")]
    data = [path.read_bytes() for path in outputs]
    assert data[0] == data[1] == data[2]  # byte-identical gzip, clean_tokens has no effect

    with gzip.open(outputs[0], "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == len(cases) * 7
    assert [r["method"] for r in rows[:7]] == ["diffusion"] * 3 + ["oneshot_argmax"] + ["oneshot_sample"] * 3
    assert [r["sample_index"] for r in rows[:7]] == [0, 1, 2, 0, 0, 1, 2]
    assert list(rows[0]) == ["case_id", "window_id", "t", "method", "model_dataset", "train_seed",
                             "checkpoint_sha256", "sample_index", "sample_seed", "raw_generated_tokens",
                             "decoded_tokens", "leading_hold_repaired", "has_sounding_note"]
    for r in rows:
        assert r["decoded_tokens"][0] != D.HOLD
        assert r["leading_hold_repaired"] == (r["raw_generated_tokens"][0] == D.HOLD)
    meta = json.loads((tmp_path / "r1" / "recon__h2-acfs_full-s0.meta.json").read_text(encoding="utf-8"))
    assert meta["input_sha256"] == D.sha256_file(xt) and meta["batch_size"] == 5
    assert meta["output_sha256_gzip"] == D.sha256_file(outputs[0])
    with gzip.open(outputs[0], "rb") as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == meta["output_sha256_uncompressed"]


def test_reconstruction_refuses_to_overwrite(tmp_path, tiny_checkpoints):
    xt, _ = make(tmp_path, "xt", validation_ids(1), times="10", seeds="0")
    recon(xt, tmp_path / "out")
    with pytest.raises(SystemExit):
        recon(xt, tmp_path / "out")


def test_reconstruction_rejects_bad_input(tmp_path, tiny_checkpoints):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(dict(case_id="a", t=10)) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        recon(bad, tmp_path / "out")


def test_run_selection():
    assert RC.select_runs("all") == list(RC.RUNS) and len(RC.RUNS) == 9
    assert RC.select_runs("h2") == ["h2-acfs_full-s0", "h2-acfs_full-s1", "h2-acfs_full-s2"]
    assert len(RC.select_runs("h3")) == 6
    assert RC.select_runs("h3-cantopop-s2,h2-acfs_full-s0") == ["h3-cantopop-s2", "h2-acfs_full-s0"]
    with pytest.raises(RC.UserError):
        RC.select_runs("h4")


def test_repository_checkpoints_match_manifest():
    manifest = json.loads((RC.CHECKPOINTS / "manifest.json").read_text(encoding="utf-8"))["runs"]
    assert [entry["run"] for entry in manifest] == list(RC.RUNS)
    for entry in manifest:
        path = RC.CHECKPOINTS / f"{entry['run']}.pt"
        assert D.sha256_file(path) == entry["sha256"]
        _, state = D.load_checkpoint(path, "cpu")
        assert state["step"] == entry["step"]
        assert state["config"]["dataset"] == entry["dataset"] and state["config"]["seed"] == entry["seed"]


# -------------------------------------------------------------- generation

def test_generation_is_reproducible(tmp_path):
    run = tiny_run(tmp_path, "g")
    texts = []
    for name in ("g1", "g2"):
        out = tmp_path / f"{name}.jsonl"
        S.main(["--device", "cpu", "generate", "--checkpoint", str(run / "last.pt"), "--output", str(out),
                "--count", "4"])
        texts.append(out.read_text(encoding="utf-8"))
    assert texts[0] == texts[1]
    first = json.loads(texts[0].splitlines()[0])
    assert 0 <= first["nn_transposed_hamming"] <= first["nn_hamming"] <= 64


def test_nearest_neighbour_finds_transposed_copy():
    train = torch.tensor([[0] * 64, [10, 1, 12, 1] * 16, [20, 1, 0, 1] * 16])
    query = torch.tensor([[13, 1, 15, 1] * 16, [0] * 64])  # window 1 moved up 3 semitones; window 0
    result = S.nearest_neighbours(query, train, torch.device("cpu"))
    distances, windows, _ = result["transposed"]
    assert distances == [0, 0] and windows == [1, 0]
    assert result["transposed"][2][0] == -3
    assert result["plain"][0][1] == 0


def test_reverse_chain_keeps_t0_rows_and_reaches_zero():
    torch.manual_seed(0)
    model = D.build_model(dict(width=16, layers=1, heads=4, ff_mult=4, dropout=0.0)).eval()
    x = torch.randint(D.K, (3, 64))
    t = torch.tensor([0, 5, 2])
    out, states = D.reverse_chain(model, x, t, torch.Generator().manual_seed(0), keep_states=True)
    assert torch.equal(out[0], x[0])
    assert sorted(states) == list(range(0, 6))
    assert torch.equal(states[5], x)
