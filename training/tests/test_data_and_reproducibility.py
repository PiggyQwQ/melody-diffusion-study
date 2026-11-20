"""Data loading checks and same-seed reproducibility of training and sampling."""

import json

import pytest
import torch

import diffusion as D
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
    row = dict(window_id="x", tokens=[1] + [0] * 63)
    with pytest.raises(ValueError):
        D.clean_tensor([row])


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


def test_sampling_is_reproducible_and_ignores_clean_tokens(tmp_path):
    run = tiny_run(tmp_path, "m")
    xt = tmp_path / "xt-provisional.jsonl"
    S.main(["--device", "cpu", "make-xt", "--output", str(xt), "--windows", "3", "--times", "10", "90",
            "--repeats", "2"])
    cases = [json.loads(line) for line in xt.read_text(encoding="utf-8").splitlines()]
    assert len(cases) == 12 and all(len(c["noisy_tokens"]) == 64 for c in cases)
    # same inputs with the answers scrambled must give identical outputs
    scrambled = tmp_path / "xt-scrambled.jsonl"
    with scrambled.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(dict(case, clean_tokens=[2] * 64)) + "\n")

    outputs = []
    for name, source in (("r1", xt), ("r2", xt), ("r3", scrambled)):
        out = tmp_path / f"{name}.jsonl"
        S.main(["--device", "cpu", "reconstruct", "--checkpoint", str(run / "last.pt"), "--xt", str(source),
                "--output", str(out), "--batch-size", "5"])
        outputs.append([json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()])
    assert outputs[0] == outputs[1] == outputs[2]
    rows = outputs[0]
    assert len(rows) == 12 * (3 + 1 + 3)
    assert {r["method"] for r in rows} == set(S.METHODS)
    for r in rows:
        assert r["decoded_tokens"][0] != D.HOLD
        assert r["leading_hold_repaired"] == (r["raw_generated_tokens"][0] == D.HOLD)
    meta = json.loads((tmp_path / "r1.meta.json").read_text(encoding="utf-8"))
    assert meta["batch_size"] == 5 and meta["input_sha256"] == D.sha256_file(xt)

    gens = []
    for name in ("g1", "g2"):
        out = tmp_path / f"{name}.jsonl"
        S.main(["--device", "cpu", "generate", "--checkpoint", str(run / "last.pt"), "--output", str(out),
                "--count", "4"])
        gens.append(out.read_text(encoding="utf-8"))
    assert gens[0] == gens[1]
    first = json.loads(gens[0].splitlines()[0])
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
