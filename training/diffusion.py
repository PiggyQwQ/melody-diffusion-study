"""Uniform-replacement discrete diffusion for 64-slot melody windows.

Refactored from the group's reference implementation (reference_diffusion.py).
The transition matrices, corruption, network and reverse posterior keep the
reference definitions; this module adds explicit matrices for testing, a
single-step forward kernel for trajectories and batched samplers.
"""

from hashlib import file_digest, sha256
import json
from pathlib import Path
import sys

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
SCRIPTS_DIR = REPO_ROOT / "scripts"
ACFS = "acfs-v261103/split-encode-v1"
POP = "cantopop-018e885/split-encode-v1"

LENGTH, K, T = 64, 51, 100
REST, HOLD = 0, 1
MIDI_MIN, MIDI_MAX = 38, 86
VALIDATION_TIMES = (10, 25, 50, 75, 90)
VALIDATION_SEED = 9182
VALIDATION_BATCH = 64  # fixes the order of the validation corruption draws


# ---------------------------------------------------------------- data access

def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def check_tokens(tokens, label):
    if (len(tokens) != LENGTH
            or any(type(token) is not int or not 0 <= token < K for token in tokens)):
        raise ValueError(f"Invalid token sequence: {label}")


def clean_tensor(rows):
    for row in rows:
        check_tokens(row["tokens"], row["window_id"])
        if row["tokens"][0] == HOLD:
            raise ValueError(f"Clean window starts with HOLD: {row['window_id']}")
    return torch.tensor([row["tokens"] for row in rows], dtype=torch.long)


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def verify_vocab(path):
    vocab = json.loads(Path(path).read_text(encoding="utf-8"))
    names = ["REST", "HOLD"] + [f"MIDI_{pitch}" for pitch in range(MIDI_MIN, MIDI_MAX + 1)]
    if (vocab["id_to_token"] != names or vocab["size"] != K
            or vocab["window_slots"] != LENGTH or vocab["slots_per_quarter"] != 4):
        raise ValueError(f"Incompatible melody vocabulary: {path}")
    return sha256_file(path)


DATASETS = ("acfs_full", "acfs_matched", "cantopop", "cantopop_standalone")


def dataset_paths(name, data_root=DATA_ROOT):
    """Training file and validation file for an experiment entry.

    H2 and both H3 models validate on the ACFS validation split; only the
    stand-alone Cantopop check uses the Cantopop validation split.
    """
    folk = Path(data_root) / ACFS
    pop = Path(data_root) / POP
    trains = {
        "acfs_full": folk / "train.jsonl",
        "acfs_matched": pop / "acfs_matched_train.jsonl",
        "cantopop": pop / "train.jsonl",
        "cantopop_standalone": pop / "train.jsonl",
    }
    validation = pop if name == "cantopop_standalone" else folk
    return trains[name], validation / "validation.jsonl"


def show_path(path):
    """Path for records: the home directory is written as ~."""
    return str(path).replace(str(Path.home()), "~")


def cpu_name():
    """Processor model name, best effort."""
    import platform
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if Path("/proc/cpuinfo").is_file():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def stable_seed(*parts):
    """Deterministic 31-bit seed from arbitrary labels."""
    digest = sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


# ------------------------------------------------------------ the noise process

def beta(step):
    return 1.0 / (T - step + 1)


def rho(step):
    return step / T


def step_matrix(step, dtype=torch.float64):
    """Q_t[i, j] = p(x_t = j | x_{t-1} = i)."""
    b = beta(step)
    return (1 - b) * torch.eye(K, dtype=dtype) + b * torch.full((K, K), 1 / K, dtype=dtype)


def cumulative_matrix(step, dtype=torch.float64):
    """Closed form of Q_1 ... Q_t: (1 - t/T) I + (t/T) U."""
    r = rho(step)
    return (1 - r) * torch.eye(K, dtype=dtype) + r * torch.full((K, K), 1 / K, dtype=dtype)


def corrupt(x_0, t, generator=None):
    """Sample x_t ~ Q̄_t(x_0, .) in one draw; t has shape [B]."""
    replace = torch.rand(x_0.shape, device=x_0.device, generator=generator) < (t / T)[:, None]
    replacement = torch.randint(K, x_0.shape, device=x_0.device, generator=generator)
    return torch.where(replace, replacement, x_0)


def forward_step(x_prev, step, generator=None):
    """Sample x_t ~ Q_t(x_{t-1}, .): resample uniformly with probability beta_t."""
    replace = torch.rand(x_prev.shape, device=x_prev.device, generator=generator) < beta(step)
    replacement = torch.randint(K, x_prev.shape, device=x_prev.device, generator=generator)
    return torch.where(replace, replacement, x_prev)


def reverse_probs(logits, x_t, step):
    """p(x_{t-1} = i | x_t) = sum_k f(k) Q_t[i, j] Q̄_{t-1}[k, i] / Q̄_t[k, j].

    `step` is a Python int shared by the batch. The expression is the
    reference implementation's; it sums to one exactly because
    Q̄_{t-1} Q_t = Q̄_t.
    """
    hit = (x_t[..., None] == torch.arange(K, device=x_t.device)).float()
    r = step / T
    previous_r = (step - 1) / T
    b = 1 / (T - step + 1)
    f = logits.float().softmax(dim=-1)
    v = f / (r / K + (1 - r) * hit)
    return (b / K + (1 - b) * hit) * ((1 - previous_r) * v + (previous_r / K) * v.sum(dim=-1, keepdim=True))


def sample_categorical(probs, generator=None):
    """Inverse-CDF draw along the last axis with an explicit generator."""
    u = torch.rand(probs.shape[:-1] + (1,), device=probs.device, generator=generator)
    cdf = probs.cumsum(dim=-1)
    cdf = cdf / cdf[..., -1:]
    return (cdf < u).sum(dim=-1).clamp_(max=probs.shape[-1] - 1)


# ----------------------------------------------------------------- the network

class Denoiser(nn.Module):
    """Bidirectional Transformer predicting x_0 logits from (x_t, t)."""

    def __init__(self, width=64, layers=2, heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.token = nn.Embedding(K, width)
        self.position = nn.Embedding(LENGTH, width)
        self.time = nn.Embedding(T + 1, width)
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=ff_mult * width,
            batch_first=True, activation="gelu", dropout=dropout,
        ) for _ in range(layers))
        self.output = nn.Linear(width, K)

    def forward(self, x_t, t):
        positions = torch.arange(x_t.shape[1], device=x_t.device)
        h = self.token(x_t) + self.position(positions)[None] + self.time(t)[:, None]
        for layer in self.layers:
            h = layer(h)
        return self.output(h)


MODEL_KEYS = ("width", "layers", "heads", "ff_mult", "dropout")


def build_model(config):
    return Denoiser(**{key: config[key] for key in MODEL_KEYS})


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path, device):
    # weights_only loading: tensors and plain containers only; the saved config
    # also holds the PyTorch version object, which is allowed explicitly.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        state = torch.load(path, map_location=device, weights_only=True)
    config = state["config"]
    if (config["T"], config["K"], config["length"]) != (T, K, LENGTH):
        raise ValueError("Checkpoint uses a different diffusion setting")
    model = build_model(config["model"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


# ------------------------------------------------------------------- sampling

@torch.no_grad()
def reverse_chain(model, x_t, t, generator, keep_states=False):
    """Run p_theta from per-row step t (tensor [B]) down to 0.

    Rows whose t is already 0 are left unchanged. Returns the final tokens and,
    if requested, the list of states indexed by the global step.
    """
    x = x_t.clone()
    t = t.to(x.device)
    states = {}
    start = int(t.max().item()) if len(t) else 0
    if keep_states:
        states[start] = x.cpu().clone()
    for step in range(start, 0, -1):
        active = t >= step
        if active.any():
            rows = active.nonzero(as_tuple=True)[0]
            steps = torch.full((len(rows),), step, dtype=torch.long, device=x.device)
            probs = reverse_probs(model(x[rows], steps), x[rows], step)
            x[rows] = sample_categorical(probs, generator)
        if keep_states:
            states[step - 1] = x.cpu().clone()
    return x, states


@torch.no_grad()
def one_shot(model, x_t, t, generator=None, mode="argmax"):
    """Read x_0 directly from f_theta(x_0 | x_t); rows with t = 0 are copied."""
    logits = model(x_t, t)
    if mode == "argmax":
        out = logits.argmax(dim=-1)
    else:
        out = sample_categorical(logits.float().softmax(dim=-1), generator)
    return torch.where((t == 0)[:, None], x_t, out)


# -------------------------------------------------------- decoding and export

def _scripts():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))


def repair_leading_hold(tokens):
    tokens = list(tokens)
    repaired = tokens[0] == HOLD
    if repaired:
        tokens[0] = REST
    return tokens, repaired


def decode_tokens(tokens):
    """Spans (start, midi or None, duration) using the repository decoder."""
    _scripts()
    from encode_folk_windows import decode

    return decode(list(tokens), MIDI_MIN, MIDI_MAX)


def describe(raw):
    """Fields shared by every sampled output row."""
    raw = [int(token) for token in raw]
    decoded, repaired = repair_leading_hold(raw)
    spans = decode_tokens(decoded)
    return dict(raw_generated_tokens=raw, decoded_tokens=decoded, leading_hold_repaired=repaired,
                has_sounding_note=any(pitch is not None for _, pitch, _ in spans))


def export_score(prefix, tokens):
    """Write <prefix>.mid and <prefix>.musicxml with the repository exporters."""
    _scripts()
    from folk_window_samples import export_midi, write_musicxml

    decoded, repaired = repair_leading_hold(tokens)
    spans = decode_tokens(decoded)
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    export_midi(prefix.with_suffix(".mid"), spans)
    write_musicxml(prefix.with_suffix(".musicxml"), spans)
    return repaired
