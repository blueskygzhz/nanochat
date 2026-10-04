"""
Checkpoint save/load. The from-scratch counterpart of `nanochat/checkpoint_manager.py`.

Format: a directory per step, holding `model.npz` (parameters), `optim.npz` (optimizer
state, optional) and `meta.json` (config + step + whatever the training script wants
to record). Writes go to a temporary file first and are then renamed, because rename
is atomic on POSIX -- an interrupted save can never leave a half-written checkpoint
that loads without complaint.

numpy's `.npz` is used as a container only; it holds plain arrays, so there is no
pickle involved and no code executes on load.
"""

import glob
import json
import os
import re

import numpy as np

from nanochat.scratch.model import GPT, GPTConfig

__all__ = [
    "save_checkpoint", "load_checkpoint", "load_meta", "build_model",
    "find_last_step", "list_steps", "load_model",
]


def _atomic_write(path, write_fn):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic on POSIX


def checkpoint_dir(base_dir, step):
    return os.path.join(base_dir, f"step_{step:06d}")


def save_checkpoint(base_dir, step, model, optimizer=None, meta=None):
    """Write parameters, optimizer state and metadata for one step."""
    out = checkpoint_dir(base_dir, step)
    os.makedirs(out, exist_ok=True)

    _atomic_write(os.path.join(out, "model.npz"),
                  lambda f: np.savez(f, **model.state_dict()))

    if optimizer is not None:
        flat = {}
        for which in ("muon", "adamw"):
            sub = getattr(optimizer, which, None)
            if sub is None:
                continue
            # Optimizer state is keyed by id(param), which is not stable across
            # processes, so re-key it by parameter *name* before writing.
            names = {id(p): n for n, p in model.named_parameters()}
            for pid, state in sub.state.items():
                if pid not in names:
                    continue
                for key, value in state.items():
                    flat[f"{which}|{names[pid]}|{key}"] = np.asarray(value)
        _atomic_write(os.path.join(out, "optim.npz"), lambda f: np.savez(f, **flat))

    payload = {"step": step, "config": vars(model.config), **(meta or {})}
    _atomic_write(os.path.join(out, "meta.json"),
                  lambda f: f.write(json.dumps(payload, indent=2).encode()))
    return out


def load_meta(base_dir, step):
    """Just the metadata, without building the model.

    Needed before the model exists: the tokenizer recorded here determines the
    vocabulary size, and so the shape of the embedding table.
    """
    path = os.path.join(checkpoint_dir(base_dir, step), "meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint at {checkpoint_dir(base_dir, step)}")
    with open(path, "rb") as f:
        return json.loads(f.read())


def load_checkpoint(base_dir, step, model=None, optimizer=None):
    """Load a checkpoint. Builds the model from the saved config if none is given."""
    src = checkpoint_dir(base_dir, step)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"no checkpoint at {src}")

    with open(os.path.join(src, "meta.json"), "rb") as f:
        meta = json.loads(f.read())

    if model is None:
        cfg = dict(meta["config"])
        # Checkpoints from before `moe_hidden_act` existed have relu^2 experts
        cfg.setdefault("moe_hidden_act", "relu2")
        model = GPT(GPTConfig(**cfg))
    with np.load(os.path.join(src, "model.npz")) as z:
        model.load_state_dict({k: z[k] for k in z.files})

    optim_path = os.path.join(src, "optim.npz")
    if optimizer is not None and os.path.exists(optim_path):
        by_name = dict(model.named_parameters())
        with np.load(optim_path) as z:
            for flat_key in z.files:
                which, name, key = flat_key.split("|", 2)
                sub = getattr(optimizer, which, None)
                if sub is None or name not in by_name:
                    continue
                sub.state.setdefault(id(by_name[name]), {})[key] = z[flat_key]
        # `step` is stored as a 0-d array by savez; Adam needs it back as an int
        for which in ("muon", "adamw"):
            sub = getattr(optimizer, which, None)
            for state in (sub.state.values() if sub else ()):
                if "step" in state:
                    state["step"] = int(state["step"])

    return model, meta


def build_model(base_dir, step):
    """Load just the model, in eval mode, for inference or evaluation."""
    model, meta = load_checkpoint(base_dir, step)
    model.eval()
    return model, meta


def list_steps(base_dir):
    """Every step number that has a complete checkpoint, ascending."""
    steps = []
    for path in glob.glob(os.path.join(base_dir, "step_*")):
        m = re.fullmatch(r"step_(\d+)", os.path.basename(path))
        if m and os.path.exists(os.path.join(path, "meta.json")):
            steps.append(int(m.group(1)))
    return sorted(steps)


def find_last_step(base_dir):
    steps = list_steps(base_dir)
    return steps[-1] if steps else None


def load_model(base_dir, step=None):
    """Load a checkpoint by step, or the latest one if step is None."""
    if step is None:
        step = find_last_step(base_dir)
        if step is None:
            raise FileNotFoundError(f"no checkpoints in {base_dir}")
    return build_model(base_dir, step)
