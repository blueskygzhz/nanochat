"""
The from-scratch replacement for `torch.nn`: a module/parameter registration
system plus the handful of layers a transformer needs.

`Module` reimplements the part of PyTorch that is easy to take for granted -- the
bookkeeping that lets you write `self.attn = CausalSelfAttention(...)` and later
get a flat list of every parameter in the tree, save it, reload it, or hand it to
an optimizer. It is about 60 lines, and seeing it written out is most of the point
of this package.
"""

import math

import numpy as np

from nanochat.scratch.tensor import (
    Tensor, cat, cross_entropy, index_add, masked_fill, no_grad, rms_norm, softmax,
)

__all__ = [
    "Parameter", "Module", "ModuleList", "ModuleDict",
    "Linear", "Embedding", "norm", "relu_squared",
    "attention", "cross_entropy", "no_grad",
]


class Parameter(Tensor):
    """A Tensor that is a leaf of the graph and is owned by a Module."""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    """Base class holding named parameters and named child modules."""

    def __init__(self):
        self._params = {}
        self._modules = {}
        self.training = True

    # Registration happens by attribute assignment, exactly like torch. This is the
    # whole trick: __setattr__ notices Parameters and Modules and files them away in
    # ordered dicts, which is what makes .parameters() recursion possible.
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.__dict__.setdefault("_params", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        else:
            # Re-assigning a name that used to hold a param/module should unregister it
            self.__dict__.get("_params", {}).pop(name, None)
            self.__dict__.get("_modules", {}).pop(name, None)
        object.__setattr__(self, name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_parameters(self, prefix=""):
        for name, p in self._params.items():
            yield prefix + name, p
        for name, m in self._modules.items():
            yield from m.named_parameters(prefix + name + ".")

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def modules(self):
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def state_dict(self):
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, sd):
        own = dict(self.named_parameters())
        missing = set(own) - set(sd)
        if missing:
            raise KeyError(f"missing keys in state_dict: {sorted(missing)}")
        for name, value in sd.items():
            if name not in own:
                raise KeyError(f"unexpected key in state_dict: {name}")
            value = np.asarray(value, dtype=np.float32)
            if value.shape != own[name].data.shape:
                raise ValueError(f"shape mismatch for {name}: {value.shape} vs {own[name].data.shape}")
            own[name].data = value.copy()


class ModuleList(Module):
    def __init__(self, modules=()):
        super().__init__()
        self._items = list(modules)
        for i, m in enumerate(self._items):
            self._modules[str(i)] = m

    def append(self, m):
        self._modules[str(len(self._items))] = m
        self._items.append(m)
        return self

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self):
        return len(self._items)


class ModuleDict(Module):
    def __init__(self, modules=None):
        super().__init__()
        for k, v in (modules or {}).items():
            self._modules[k] = v

    def __getitem__(self, k):
        return self._modules[k]

    def __setitem__(self, k, v):
        self._modules[k] = v

    def __contains__(self, k):
        return k in self._modules

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()


# ----------------------------------------------------------------------------
# layers

class Linear(Module):
    """y = x @ W^T (bias optional). Weight is (out_features, in_features), matching torch."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bound = 1.0 / math.sqrt(in_features)
        self.weight = Parameter(np.random.uniform(-bound, bound, (out_features, in_features)))
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        y = x @ self.weight.mT
        return y + self.bias if self.bias is not None else y


class Embedding(Module):
    """A lookup table. The forward is integer-array indexing; the backward is the
    scatter-add that `Tensor.__getitem__` already implements via `np.add.at`, which
    is what makes repeated tokens accumulate gradient correctly."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(np.random.normal(0.0, 1.0, (num_embeddings, embedding_dim)))

    def forward(self, idx):
        return self.weight[np.asarray(idx, dtype=np.int64)]


def norm(x, eps=1e-6):
    """RMSNorm with no learnable gain, matching nanochat's `norm()`."""
    return rms_norm(x, eps=eps)


def relu_squared(x):
    """ReLU squared, the activation nanochat uses in every FFN."""
    return x.relu().square()


# ----------------------------------------------------------------------------
# attention

def causal_window_mask(q_len, kv_len, window=-1, offset=0):
    """Boolean mask, True where attention is *blocked*.

    `offset` is the absolute position of the first query (nonzero when decoding with
    a cache). `window` mirrors FA3's `window_size=(left, 0)`: a query at position i
    attends to keys j with i - window <= j <= i. window < 0 means unlimited.
    """
    i = np.arange(q_len, dtype=np.int64)[:, None] + offset
    j = np.arange(kv_len, dtype=np.int64)[None, :]
    blocked = j > i
    if window >= 0:
        blocked |= (i - j) > window
    return blocked


def attention(q, k, v, window=-1, offset=0):
    """Naive scaled dot-product attention with GQA support.

    q is (B, T, H, D) and k/v are (B, S, H_kv, D) -- the same layout nanochat feeds
    to FlashAttention. This materialises the full (B, H, T, S) score matrix, which is
    exactly what FlashAttention exists to avoid; at the scale this package runs at,
    that is a fine trade for being readable.
    """
    B, T, H, D = q.shape
    S, H_kv = k.shape[1], k.shape[2]
    if H % H_kv:
        raise ValueError(f"n_head {H} must be divisible by n_kv_head {H_kv}")
    rep = H // H_kv
    if rep > 1:  # GQA: query head h reads kv head h // rep
        k = k.reshape(B, S, H_kv, 1, D).expand(B, S, H_kv, rep, D).reshape(B, S, H, D)
        v = v.reshape(B, S, H_kv, 1, D).expand(B, S, H_kv, rep, D).reshape(B, S, H, D)

    q = q.permute(0, 2, 1, 3)  # (B, H, T, D)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    scores = (q @ k.mT) * (1.0 / math.sqrt(D))          # (B, H, T, S)
    scores = masked_fill(scores, causal_window_mask(T, S, window, offset), -1e9)
    y = softmax(scores, axis=-1) @ v                    # (B, H, T, D)
    return y.permute(0, 2, 1, 3)                        # back to (B, T, H, D)


def apply_rotary_emb(x, cos, sin):
    """Rotate pairs of channels. Mirrors nanochat's convention (rotation by -theta,
    which is functionally equivalent since only the relative q/k rotation matters)."""
    if x.ndim != 4:
        raise ValueError("apply_rotary_emb expects (B, T, H, D)")
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return cat([y1, y2], axis=3)


__all__ += ["causal_window_mask", "apply_rotary_emb", "index_add", "cat", "softmax"]
