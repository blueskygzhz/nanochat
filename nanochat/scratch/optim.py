"""
Optimizers, from scratch.

`AdamW` and `Muon`, written against the from-scratch `Tensor`. nanochat trains with
both at once: AdamW for things that are not matrices (embeddings, the output head,
per-layer scalars) and Muon for the 2-D hidden weights.

What Muon does, in one sentence: instead of stepping along the gradient, it steps
along the *nearest orthogonal matrix* to the momentum buffer, so every singular
direction of the update has the same magnitude and no single direction dominates.
The orthogonalisation is done by a matrix-only polynomial iteration (Polar Express),
so it needs nothing but matmuls -- no SVD.
"""

import numpy as np

__all__ = ["Optimizer", "AdamW", "Muon", "MuonAdamW", "polar_express", "setup_optimizer"]


# Coefficients for Polar Express, num_iters=5 (https://arxiv.org/pdf/2505.16932).
# Same values nanochat/optim.py uses.
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def polar_express(G, steps=5):
    """Return (approximately) the nearest orthogonal matrix to the 2-D matrix `G`.

    Three stages:
      1. MuonEq row equilibration -- rescale every row to the mean row norm, so the
         spectrum entering the iteration is better conditioned.
      2. The Polar Express iteration X <- aX + (bA + cA^2)X with A = XX^T. Each step
         pushes the singular values of X towards 1; the coefficients are tuned to
         maximise the slope at zero, so small singular values catch up fast.
      3. Muon+ renormalisation -- snap the Frobenius norm to sqrt(min(m, n)), which
         is the norm an exactly orthogonal matrix of this shape would have.
    """
    X = np.asarray(G, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"polar_express expects a 2-D matrix, got {X.shape}")
    m, n = X.shape

    fro = np.linalg.norm(X)
    if fro < 1e-12:
        return np.zeros_like(X)

    target = fro / np.sqrt(m)
    row_norm = np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-6)
    X = X * (target / row_norm)

    X = X / (np.linalg.norm(X) * 1.01 + 1e-6)
    tall = m > n
    for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
        if tall:
            A = X.T @ X
            X = a * X + X @ (b * A + c * (A @ A))
        else:
            A = X @ X.T
            X = a * X + (b * A + c * (A @ A)) @ X

    current = max(float(np.linalg.norm(X)), 1e-6)
    return X * (np.sqrt(min(m, n)) / current)


# ----------------------------------------------------------------------------

class Optimizer:
    """Holds parameter groups and per-parameter state. Mirrors torch's interface."""

    def __init__(self, param_groups, defaults):
        if not param_groups:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": list(param_groups)}]
        self.param_groups = []
        for g in param_groups:
            group = dict(defaults)
            group.update(g)
            group["params"] = list(group["params"])
            self.param_groups.append(group)
        self.state = {}

    def _state(self, p):
        return self.state.setdefault(id(p), {})

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None

    def step(self):
        raise NotImplementedError


class AdamW(Optimizer):
    """Adam with *decoupled* weight decay: the decay is applied straight to the
    parameter instead of being folded into the gradient, so it does not get divided
    by the second-moment estimate."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    def step(self):
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self._state(p)
                if not st:
                    st["step"] = 0
                    st["m"] = np.zeros_like(p.data)
                    st["v"] = np.zeros_like(p.data)
                st["step"] += 1
                t = st["step"]
                st["m"] += (1 - b1) * (g - st["m"])
                st["v"] += (1 - b2) * (g * g - st["v"])
                m_hat = st["m"] / (1 - b1 ** t)
                v_hat = st["v"] / (1 - b2 ** t)
                if wd:
                    p.data -= lr * wd * p.data
                p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)


class Muon(Optimizer):
    """Momentum Orthogonalised by Newton-Schulz, for 2-D hidden weights only.

    The learning rate is scale-free thanks to the orthogonalisation, but the useful
    step size still depends on the matrix shape, so we apply the standard
    `sqrt(max(1, out/in))` correction.

    Not mirrored from `nanochat/optim.py`: the factored second-moment variance
    reduction and the cautious-update mask. Those are refinements on top of the
    algorithm below, not part of it.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, ns_steps=5, nesterov=True):
        super().__init__(params, {"lr": lr, "momentum": momentum, "weight_decay": weight_decay,
                                  "ns_steps": ns_steps, "nesterov": nesterov})
        for group in self.param_groups:
            for p in group["params"]:
                if p.data.ndim != 2:
                    raise ValueError(f"Muon only accepts 2-D parameters, got {p.data.shape}")

    def step(self):
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                if not st:
                    st["momentum"] = np.zeros_like(p.data)
                buf = st["momentum"]
                buf += (1 - mu) * (p.grad - buf)
                g = p.grad + mu * (buf - p.grad) if group["nesterov"] else buf
                update = polar_express(g, group["ns_steps"])
                out_dim, in_dim = p.data.shape
                scale = max(1.0, out_dim / in_dim) ** 0.5
                if wd:
                    p.data -= lr * wd * p.data
                p.data -= lr * scale * update


class MuonAdamW:
    """Runs a Muon and an AdamW together, so the training loop sees one object."""

    def __init__(self, muon, adamw):
        self.muon, self.adamw = muon, adamw

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def step(self):
        self.muon.step()
        self.adamw.step()

    def zero_grad(self):
        self.muon.zero_grad()
        self.adamw.zero_grad()


def setup_optimizer(model, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                    scalar_lr=0.5, weight_decay=0.0):
    """Split the model's parameters the way nanochat does and build the optimizer.

    The split is the interesting part:
      - embeddings and the output head are lookup tables, not linear maps between
        feature spaces, so orthogonalising them is meaningless -> AdamW
      - 2-D hidden weights -> Muon
      - per-layer scalars (resid/x0/smear/backout lambdas) are 1-D and get a much
        higher learning rate, since they are few and start near their neutral value
    """
    named = dict(model.named_parameters())
    embedding_names = {n for n in named if n.startswith("wte.") or n.startswith("value_embeds.")}
    head_names = {n for n in named if n.startswith("lm_head.")}

    embeddings, unembeddings, matrices, scalars = [], [], [], []
    for n, p in named.items():
        if n in embedding_names:
            embeddings.append(p)
        elif n in head_names:
            unembeddings.append(p)
        elif p.data.ndim == 2:
            matrices.append(p)
        else:
            scalars.append(p)

    adamw_groups = []
    if unembeddings:
        adamw_groups.append({"params": unembeddings, "lr": unembedding_lr})
    if embeddings:
        adamw_groups.append({"params": embeddings, "lr": embedding_lr})
    if scalars:
        adamw_groups.append({"params": scalars, "lr": scalar_lr})

    muon = Muon(matrices, lr=matrix_lr, weight_decay=weight_decay)
    adamw = AdamW(adamw_groups, lr=unembedding_lr, weight_decay=weight_decay)
    return MuonAdamW(muon, adamw)
