"""
Reverse-mode automatic differentiation, written from scratch on top of numpy.

This is the from-scratch replacement for `torch.Tensor` + `torch.autograd` used by
the `nanochat.scratch` package. Nothing in this package imports torch.

numpy is used purely as an n-dimensional array with BLAS-backed matmul. It provides
no autograd, no layers, no optimizers and no model -- all of that is written here.

Design
------
Every `Tensor` produced by an operation remembers (a) its parent tensors and (b) a
closure that knows how to push a gradient from the output back to those parents.
`Tensor.backward()` walks the graph in reverse topological order and calls those
closures. This is the classic micrograd design, with three additions that it takes
to actually train a transformer:

  1. Broadcasting is handled properly (`_unbroadcast`), so `(B,T,C) + (C,)` works
     and the gradient is summed back down to `(C,)`.
  2. Operations whose naive composition is numerically unstable are *fused* with a
     hand-derived analytic backward: `softmax`, `cross_entropy`, `rms_norm`.
  3. The scatter/gather ops a MoE router needs (`topk`, `index_add`, integer-array
     indexing) propagate gradients correctly through duplicate indices.

Everything is float32. There is no device concept: it is always CPU.
"""

import numpy as np

__all__ = [
    "Tensor", "no_grad", "is_grad_enabled", "set_dtype", "get_dtype",
    "tensor", "zeros", "ones", "arange",
    "cat", "stack", "softmax", "relu_squared", "swiglu", "rms_norm", "cross_entropy",
    "masked_fill", "topk", "index_add", "where", "add_aux_loss", "linear", "rotary",
]


# ----------------------------------------------------------------------------
# global grad-enabled flag (the equivalent of torch.no_grad / torch.is_grad_enabled)

_grad_enabled = True


def is_grad_enabled():
    return _grad_enabled


class no_grad:
    """Context manager *and* decorator that disables graph construction."""

    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self._prev
        return False

    def __call__(self, fn):
        import inspect
        if inspect.isgeneratorfunction(fn):
            # Calling a generator function only creates the generator; its body runs
            # later, on each next(). Wrapping the call alone would leave the body with
            # grad enabled, so grad is switched off around every resumption instead --
            # and back on while the caller holds control between items.
            def wrapper(*args, **kwargs):
                gen = fn(*args, **kwargs)
                try:
                    while True:
                        with no_grad():
                            try:
                                item = next(gen)
                            except StopIteration as stop:
                                return stop.value
                        yield item
                finally:
                    gen.close()
        else:
            def wrapper(*args, **kwargs):
                with no_grad():
                    return fn(*args, **kwargs)
        wrapper.__name__ = getattr(fn, "__name__", "wrapped")
        wrapper.__doc__ = fn.__doc__
        return wrapper


# ----------------------------------------------------------------------------
# element type
#
# float32 is the training dtype. The switch exists because finite-difference
# gradient checking is meaningless in float32: the roundoff floor
# (eps * |f| / h) swamps the signal. Tests flip the engine to float64.

_DTYPE = np.dtype(np.float32)


def get_dtype():
    return _DTYPE


def set_dtype(dtype):
    """Set the element type for all tensors created from here on."""
    global _DTYPE
    _DTYPE = np.dtype(dtype)


# ----------------------------------------------------------------------------
# helpers

def _as_array(x):
    if isinstance(x, np.ndarray):
        # Every op output lands here; skipping the astype call when the dtype already
        # matches is measurable when a decode step builds hundreds of tiny tensors.
        return x if x.dtype is _DTYPE or x.dtype == _DTYPE else x.astype(_DTYPE)
    return np.asarray(x, dtype=_DTYPE)


def _unbroadcast(grad, shape):
    """Sum `grad` back down to `shape`, undoing numpy's broadcasting.

    Broadcasting prepends length-1 axes and stretches existing length-1 axes, so the
    adjoint is to sum over exactly those axes.
    """
    if grad.shape == shape:
        return grad
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


def _accumulate(t, g):
    """Accumulate gradient `g` into tensor `t` (gradients add across fan-out).

    The first write copies instead of zero-filling and then adding: one pass over
    memory rather than two. It must be a copy, not an alias -- the same `g` is often
    handed to several parents (e.g. both sides of an add), and `g` may be a read-only
    broadcast view.
    """
    if not t.requires_grad:
        return
    g = _unbroadcast(_as_array(g), t.data.shape)
    if t.grad is None:
        t.grad = np.array(g, dtype=_DTYPE, copy=True)
    else:
        t.grad += g


def _freed_backward(_g):
    """Stands in for the backward closure of a node whose graph has been released.

    A sentinel rather than None, so a freed interior node is never mistaken for a
    leaf: that would silently park gradient on it instead of failing loudly.
    """
    raise RuntimeError("Trying to backward through the graph a second time. "
                       "Pass retain_graph=True to the first backward() if you need to.")


def _make(data, parents, op, backward):
    """Build an output tensor, wiring the backward closure only if grad is needed."""
    requires_grad = _grad_enabled and any(p.requires_grad for p in parents)
    out = Tensor(data, requires_grad=requires_grad)
    if requires_grad:
        out._parents = parents
        out._op = op
        out._backward = backward
    return out


def _wrap(x):
    return x if isinstance(x, Tensor) else Tensor(x)


# ----------------------------------------------------------------------------

class Tensor:
    """A float32 n-d array that records how it was computed."""

    def __init__(self, data, requires_grad=False):
        self.data = _as_array(data)
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._backward = None
        self._parents = ()
        self._op = "leaf"

    # -- plumbing ------------------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def size(self, dim=None):
        return self.data.shape if dim is None else self.data.shape[dim]

    def numel(self):
        return self.data.size

    def item(self):
        return float(self.data.reshape(-1)[0])

    def numpy(self):
        return self.data

    def detach(self):
        """A tensor sharing the same buffer but cut off from the graph."""
        return Tensor(self.data, requires_grad=False)

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad}, op={self._op})"

    def __len__(self):
        return self.data.shape[0]

    # -- backward pass -------------------------------------------------------

    def backward(self, grad=None, retain_graph=False):
        """Backpropagate from this tensor to every leaf that requires grad.

        Memory, which is what bounds model size here, is managed the way torch does it:
          - an interior node's gradient is dropped as soon as it has been pushed to its
            parents, so at most a frontier of gradients is alive at once;
          - unless `retain_graph=True`, each interior node also drops its parents and
            its backward closure. The closures are what hold the saved activations, so
            without this a `loss` still in scope keeps the entire forward pass alive --
            in a training loop, across the *next* forward too, doubling peak memory.
        Only leaves (parameters, inputs) keep `.grad`.
        """
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() on a non-scalar requires an explicit grad")
            grad = np.ones_like(self.data)
        if not self.requires_grad:
            return

        # Reverse topological order, built with an explicit stack so that deep graphs
        # (many layers x many ops) cannot blow the Python recursion limit.
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in visited:
                    stack.append((p, False))

        _accumulate(self, grad)
        for node in reversed(topo):
            if node._backward is None:
                continue  # a leaf: its .grad is the result
            if node.grad is not None:
                node._backward(node.grad)
            node.grad = None
            if not retain_graph:
                node._backward = _freed_backward
                node._parents = ()

    # -- elementwise binary --------------------------------------------------

    def __add__(self, other):
        other = _wrap(other)

        def backward(g):
            _accumulate(self, g)
            _accumulate(other, g)
        return _make(self.data + other.data, (self, other), "add", backward)

    def __sub__(self, other):
        other = _wrap(other)

        def backward(g):
            _accumulate(self, g)
            _accumulate(other, -g)
        return _make(self.data - other.data, (self, other), "sub", backward)

    # The backwards below check `requires_grad` before computing a parent's gradient,
    # rather than computing it and letting `_accumulate` discard it. Constants such as
    # the rotary cos/sin tables appear in every layer, so that waste adds up.

    def __mul__(self, other):
        other = _wrap(other)

        def backward(g):
            if self.requires_grad:
                _accumulate(self, g * other.data)
            if other.requires_grad:
                _accumulate(other, g * self.data)
        return _make(self.data * other.data, (self, other), "mul", backward)

    def __truediv__(self, other):
        other = _wrap(other)

        def backward(g):
            if self.requires_grad:
                _accumulate(self, g / other.data)
            if other.requires_grad:
                _accumulate(other, -g * self.data / (other.data * other.data))
        return _make(self.data / other.data, (self, other), "div", backward)

    def __pow__(self, p):
        if isinstance(p, Tensor):
            raise NotImplementedError("tensor exponents are not supported")

        def backward(g):
            _accumulate(self, g * p * self.data ** (p - 1))
        return _make(self.data ** p, (self,), "pow", backward)

    def __neg__(self):
        def backward(g):
            _accumulate(self, -g)
        return _make(-self.data, (self,), "neg", backward)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return _wrap(other) - self

    def __rtruediv__(self, other):
        return _wrap(other) / self

    def __matmul__(self, other):
        """Batched matmul. Both operands must be at least 2-D.

        The common case in a transformer is activations (B, T, K) times a weight
        (K, N). Done naively, the weight gradient is B separate (K, T) @ (T, N)
        products followed by a sum over B. Folding the batch into the rows instead
        makes it a single (K, B*T) @ (B*T, N) GEMM -- the same arithmetic, one BLAS
        call, and no (B, K, N) temporary.
        """
        other = _wrap(other)
        a, b = self.data, other.data
        if a.ndim < 2 or b.ndim < 2:
            raise NotImplementedError("matmul requires both operands to be >= 2-D")
        fold = a.ndim > 2 and b.ndim == 2
        if fold:
            out = (a.reshape(-1, a.shape[-1]) @ b).reshape(*a.shape[:-1], b.shape[-1])
        else:
            out = a @ b

        def backward(g):
            if fold:
                g2 = g.reshape(-1, g.shape[-1])
                if self.requires_grad:
                    _accumulate(self, (g2 @ b.T).reshape(a.shape))
                if other.requires_grad:
                    _accumulate(other, a.reshape(-1, a.shape[-1]).T @ g2)
                return
            if self.requires_grad:
                _accumulate(self, g @ np.swapaxes(b, -1, -2))
            if other.requires_grad:
                _accumulate(other, np.swapaxes(a, -1, -2) @ g)
        return _make(out, (self, other), "matmul", backward)

    # -- elementwise unary ---------------------------------------------------

    def exp(self):
        y = np.exp(self.data)

        def backward(g):
            _accumulate(self, g * y)
        return _make(y, (self,), "exp", backward)

    def log(self):
        def backward(g):
            _accumulate(self, g / self.data)
        return _make(np.log(self.data), (self,), "log", backward)

    def sqrt(self):
        y = np.sqrt(self.data)

        def backward(g):
            _accumulate(self, g * 0.5 / y)
        return _make(y, (self,), "sqrt", backward)

    def rsqrt(self):
        y = 1.0 / np.sqrt(self.data)

        def backward(g):
            _accumulate(self, g * -0.5 * y ** 3)
        return _make(y, (self,), "rsqrt", backward)

    def tanh(self):
        y = np.tanh(self.data)

        def backward(g):
            _accumulate(self, g * (1.0 - y * y))
        return _make(y, (self,), "tanh", backward)

    def sigmoid(self):
        # 0.5*(tanh(x/2)+1) == 1/(1+exp(-x)) but without the overflow for large |x|
        y = 0.5 * (np.tanh(0.5 * self.data) + 1.0)

        def backward(g):
            _accumulate(self, g * y * (1.0 - y))
        return _make(y, (self,), "sigmoid", backward)

    def relu(self):
        mask = self.data > 0

        def backward(g):
            _accumulate(self, g * mask)
        return _make(np.where(mask, self.data, np.float32(0.0)), (self,), "relu", backward)

    def square(self):
        def backward(g):
            _accumulate(self, g * 2.0 * self.data)
        return _make(self.data * self.data, (self,), "square", backward)

    def clamp_min(self, lo):
        mask = self.data > lo

        def backward(g):
            _accumulate(self, g * mask)
        return _make(np.maximum(self.data, np.float32(lo)), (self,), "clamp_min", backward)

    # -- reductions ----------------------------------------------------------

    def sum(self, axis=None, keepdims=False):
        shape = self.data.shape

        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            _accumulate(self, np.broadcast_to(g, shape))
        return _make(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum", backward)

    def mean(self, axis=None, keepdims=False):
        shape = self.data.shape
        n = self.data.size if axis is None else int(np.prod([shape[a] for a in np.atleast_1d(axis)]))

        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            _accumulate(self, np.broadcast_to(g / n, shape))
        return _make(self.data.mean(axis=axis, keepdims=keepdims), (self,), "mean", backward)

    def max(self, axis=None, keepdims=False):
        """Maximum over `axis`. Ties split the gradient evenly, matching torch.amax."""
        m = self.data.max(axis=axis, keepdims=True)
        mask = (self.data == m).astype(np.float32)
        count = mask.sum(axis=axis, keepdims=True)

        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            _accumulate(self, mask * g / count)
        return _make(self.data.max(axis=axis, keepdims=keepdims), (self,), "max", backward)

    def var(self, axis=None, keepdims=False, unbiased=False):
        mu = self.mean(axis=axis, keepdims=True)
        d = self - mu
        n = self.data.size if axis is None else int(np.prod([self.data.shape[a] for a in np.atleast_1d(axis)]))
        s = (d * d).sum(axis=axis, keepdims=keepdims)
        return s / float(n - 1 if unbiased else n)

    def norm(self):
        return (self * self).sum().sqrt()

    # -- shape ---------------------------------------------------------------

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        old = self.data.shape

        def backward(g):
            _accumulate(self, g.reshape(old))
        return _make(self.data.reshape(shape), (self,), "reshape", backward)

    view = reshape

    def permute(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        inverse = tuple(np.argsort(axes))

        def backward(g):
            _accumulate(self, np.transpose(g, inverse))
        return _make(np.transpose(self.data, axes), (self,), "permute", backward)

    def swapaxes(self, a, b):
        def backward(g):
            _accumulate(self, np.swapaxes(g, a, b))
        return _make(np.swapaxes(self.data, a, b), (self,), "swapaxes", backward)

    transpose = swapaxes

    @property
    def mT(self):
        """Transpose of the last two dims (the torch `.mT` spelling)."""
        return self.swapaxes(-1, -2)

    def unsqueeze(self, axis):
        old = self.data.shape

        def backward(g):
            _accumulate(self, g.reshape(old))
        return _make(np.expand_dims(self.data, axis), (self,), "unsqueeze", backward)

    def squeeze(self, axis=None):
        old = self.data.shape

        def backward(g):
            _accumulate(self, g.reshape(old))
        return _make(np.squeeze(self.data, axis=axis), (self,), "squeeze", backward)

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        old = self.data.shape

        def backward(g):
            _accumulate(self, _unbroadcast(g, old))
        return _make(np.broadcast_to(self.data, shape), (self,), "expand", backward)

    def contiguous(self):
        return self

    def __getitem__(self, key):
        old = self.data.shape
        basic = all(isinstance(k, (int, slice, type(None), type(Ellipsis)))
                    for k in (key if isinstance(key, tuple) else (key,)))

        def backward(g):
            z = np.zeros(old, dtype=_DTYPE)
            if basic:
                z[key] = g          # slices never repeat an element
            else:
                np.add.at(z, key, g)  # integer/boolean indexing can, so accumulate
            _accumulate(self, z)
        return _make(self.data[key], (self,), "getitem", backward)

    # -- method spellings of the free functions ------------------------------

    def softmax(self, axis=-1):
        return softmax(self, axis)

    def cat(self, others, axis=0):
        return cat([self] + list(others), axis)


# ----------------------------------------------------------------------------
# constructors

def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.zeros(shape, dtype=_DTYPE), requires_grad=requires_grad)


def ones(*shape, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.ones(shape, dtype=_DTYPE), requires_grad=requires_grad)


def arange(n, requires_grad=False):
    return Tensor(np.arange(n, dtype=_DTYPE), requires_grad=requires_grad)


# ----------------------------------------------------------------------------
# multi-tensor ops

def cat(tensors, axis=0):
    tensors = list(tensors)
    sizes = [t.data.shape[axis] for t in tensors]

    def backward(g):
        offset = 0
        for t, s in zip(tensors, sizes):
            idx = [slice(None)] * g.ndim
            idx[axis] = slice(offset, offset + s)
            _accumulate(t, g[tuple(idx)])
            offset += s
    out = np.concatenate([t.data for t in tensors], axis=axis)
    return _make(out, tuple(tensors), "cat", backward)


def stack(tensors, axis=0):
    tensors = list(tensors)

    def backward(g):
        for i, t in enumerate(tensors):
            _accumulate(t, np.take(g, i, axis=axis))
    out = np.stack([t.data for t in tensors], axis=axis)
    return _make(out, tuple(tensors), "stack", backward)


def where(cond, a, b):
    """Elementwise select. `cond` is a plain boolean ndarray (not differentiable)."""
    a, b = _wrap(a), _wrap(b)
    cond = np.asarray(cond, dtype=bool)

    def backward(g):
        _accumulate(a, g * cond)
        _accumulate(b, g * ~cond)
    return _make(np.where(cond, a.data, b.data), (a, b), "where", backward)


def masked_fill(t, mask, value):
    """Set entries where `mask` is True to `value`; gradient is blocked there."""
    mask = np.asarray(mask, dtype=bool)

    def backward(g):
        _accumulate(t, g * ~mask)
    return _make(np.where(mask, np.float32(value), t.data), (t,), "masked_fill", backward)


# ----------------------------------------------------------------------------
# fused ops with hand-derived backwards
#
# These three could be composed from the primitives above, but the composition
# either overflows (softmax/cross_entropy on large logits) or materialises a lot
# of intermediate graph for no reason (rms_norm). The analytic forms below are
# both stable and much cheaper.

def softmax(t, axis=-1, mask=None):
    """p = exp(x - max) / sum(exp(x - max));  dx = p * (g - sum(g*p))

    `mask` (boolean, True = blocked, broadcastable to x) fuses attention's
    `masked_fill(-inf)` into the softmax. Blocked entries get probability exactly 0,
    and because dx is proportional to p, they also get gradient exactly 0 -- the same
    result as a separate masked_fill, without materialising and back-propagating
    through another full-size score tensor. Every row must keep at least one
    unblocked entry, which a causal mask guarantees (the diagonal).
    """
    # One full-size allocation, then everything in place: on attention scores this is
    # the largest tensor in the model, and the naive chain allocates four of them.
    x = t.data
    p = np.where(mask, -np.inf, x) if mask is not None else x.copy()
    p -= p.max(axis=axis, keepdims=True)
    np.exp(p, out=p)
    p /= p.sum(axis=axis, keepdims=True)

    def backward(g):
        gp = g * p
        gp -= p * gp.sum(axis=axis, keepdims=True)
        _accumulate(t, gp)
    return _make(p, (t,), "softmax", backward)


def relu_squared(t):
    """relu(x)^2 in one op. dx = 2 * relu(x) * g.

    Composing `relu().square()` stores two full-size activations and walks the graph
    twice; this is the FFN's activation, i.e. the widest tensor in the model.
    """
    r = np.maximum(t.data, _DTYPE.type(0.0))

    def backward(g):
        _accumulate(t, (2.0 * g) * r)
    return _make(r * r, (t,), "relu_squared", backward)


def swiglu(gate, up):
    """silu(gate) * up in one op -- the SwiGLU FFN body DeepSeek/LLaMA use.

    With s = sigmoid(a) and silu(a) = a * s:
        d/da = g * up * s * (1 + a * (1 - s))      d/dup = g * silu(a)
    """
    a, b = gate.data, up.data
    s = 0.5 * (np.tanh(0.5 * a) + 1.0)  # overflow-free sigmoid
    act = a * s

    def backward(g):
        if gate.requires_grad:
            _accumulate(gate, g * b * s * (1.0 + a * (1.0 - s)))
        if up.requires_grad:
            _accumulate(up, g * act)
    return _make(act * b, (gate, up), "swiglu", backward)


def add_aux_loss(x, loss):
    """DeepSeek's `AddAuxiliaryLoss`: identity on `x` in the forward, and in the
    backward an incoming gradient of exactly 1.0 for the scalar `loss`.

    This ties the auxiliary loss to the activations instead of to the objective, so
    it is optimised by *any* backward pass that reaches `x`, and its gradient does not
    scale with whatever the main loss is multiplied by (e.g. 1/grad_accum_steps).
    """
    if loss.data.size != 1:
        raise ValueError("add_aux_loss expects a scalar loss")

    def backward(g):
        _accumulate(x, g)
        if loss.requires_grad:
            _accumulate(loss, np.ones_like(loss.data))
    return _make(x.data, (x, loss), "add_aux_loss", backward)


def rms_norm(t, eps=1e-6, scale=1.0):
    """y = scale * x / sqrt(mean(x^2) + eps)

    With r = (mean(x^2) + eps)^(-1/2) and d = x.shape[-1]:
        dL/dx = r*g - (r^3 / d) * x * sum(g * x)        (g already multiplied by scale)

    `scale` folds a constant gain (the attention's QK-norm x1.2) into the same op. It
    is applied after the normalisation, exactly like a separate `* scale`, so the
    result is bit-identical to the unfused form.
    """
    x = t.data
    d = x.shape[-1]
    # add.reduce / d is what ndarray.mean does internally, minus its Python overhead
    r = 1.0 / np.sqrt(np.add.reduce(x * x, axis=-1, keepdims=True) / d + eps)
    y = x * r
    if scale != 1.0:
        y *= scale

    def backward(g):
        if scale != 1.0:
            g = g * scale
        s = (g * x).sum(axis=-1, keepdims=True)
        _accumulate(t, r * g - (r ** 3) * x * s / d)
    return _make(y, (t,), "rms_norm", backward)


def linear(x, weight):
    """y = x @ weight.T for weight (out, in), in one op.

    Spelled `x @ weight.mT` this is two graph nodes (a transpose view, then a matmul)
    and the weight gradient is computed transposed and then transposed back. Fused,
    leading dims are folded into the rows so both directions are single GEMMs:
        dx = g @ W        dW = g^T @ x
    """
    a, w = x.data, weight.data
    lead = a.shape[:-1]
    a2 = a.reshape(-1, a.shape[-1])
    out = (a2 @ w.T).reshape(*lead, w.shape[0])

    def backward(g):
        g2 = g.reshape(-1, g.shape[-1])
        if x.requires_grad:
            _accumulate(x, (g2 @ w).reshape(a.shape))
        if weight.requires_grad:
            _accumulate(weight, g2.T @ a2)
    return _make(out, (x, weight), "linear", backward)


def rotary(x, cos, sin):
    """Rotary embedding on (B, T, H, D) in one op; `cos`/`sin` are constant arrays
    broadcastable to (B, T, H, D/2).

        y1 = x1*cos + x2*sin          y2 = -x1*sin + x2*cos

    The backward is the inverse rotation (a rotation's adjoint is its transpose):
        dx1 = g1*cos - g2*sin         dx2 = g1*sin + g2*cos

    Composed from primitives this is ten graph nodes per call (two slices, four
    multiplies, a negation, two adds and a concat), called twice per layer.
    """
    a = x.data
    d = a.shape[-1] // 2
    x1, x2 = a[..., :d], a[..., d:]
    out = np.empty_like(a)
    out[..., :d] = x1 * cos + x2 * sin
    out[..., d:] = x1 * (-sin) + x2 * cos

    def backward(g):
        g1, g2 = g[..., :d], g[..., d:]
        dx = np.empty_like(g)
        dx[..., :d] = g1 * cos - g2 * sin
        dx[..., d:] = g1 * sin + g2 * cos
        _accumulate(x, dx)
    return _make(out, (x,), "rotary", backward)


def cross_entropy(logits, targets, ignore_index=-1, reduction="mean"):
    """Softmax cross-entropy straight from logits (never materialises probabilities
    in the forward), with `ignore_index` masking. dlogits = (softmax - onehot)."""
    x = logits.data
    if x.ndim != 2:
        raise ValueError(f"cross_entropy expects (N, V) logits, got {x.shape}")
    n, _ = x.shape
    t = np.asarray(targets).reshape(-1).astype(np.int64)
    if t.shape[0] != n:
        raise ValueError("logits and targets disagree on N")

    z = x - x.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    valid = t != ignore_index
    safe_t = np.where(valid, t, 0)
    rows = np.arange(n)
    nll = np.where(valid, -logp[rows, safe_t], _DTYPE.type(0.0))
    count = max(int(valid.sum()), 1)

    if reduction == "mean":
        out = nll.sum() / count
    elif reduction == "sum":
        out = nll.sum()
    elif reduction == "none":
        out = nll
    else:
        raise ValueError(f"unknown reduction: {reduction}")

    def backward(g):
        p = np.exp(logp)
        p[rows, safe_t] -= 1.0
        p *= valid[:, None]
        if reduction == "mean":
            p *= _DTYPE.type(g) / count
        elif reduction == "sum":
            p *= _DTYPE.type(g)
        else:
            p *= np.asarray(g, dtype=_DTYPE).reshape(n, 1)
        _accumulate(logits, p)
    return _make(out, (logits,), "cross_entropy", backward)


# ----------------------------------------------------------------------------
# scatter / gather, needed by the MoE router

def topk(t, k, axis=-1):
    """Top-k values along `axis`, unsorted (equivalent to torch.topk(sorted=False)).

    Returns `(values_tensor, indices_ndarray)`. Indices are discrete, so only the
    values carry gradient -- exactly like torch.
    """
    x = t.data
    idx = np.argpartition(-x, k - 1, axis=axis)
    idx = np.take(idx, np.arange(k), axis=axis)
    vals = np.take_along_axis(x, idx, axis=axis)
    old = x.shape

    def backward(g):
        z = np.zeros(old, dtype=_DTYPE)
        np.put_along_axis(z, idx, g, axis=axis)  # indices within a row are unique
        _accumulate(t, z)
    return _make(vals, (t,), "topk", backward), idx


def index_add(shape, index, src):
    """`zeros(shape).index_add_(0, index, src)`.

    The adjoint of a scatter-add is a gather, which is what makes this the natural
    way to combine expert outputs back into the token stream: a token that was sent
    to k experts receives gradient from all k of them.
    """
    index = np.asarray(index)
    out = np.zeros(shape, dtype=_DTYPE)
    np.add.at(out, index, src.data)

    def backward(g):
        _accumulate(src, g[index])
    return _make(out, (src,), "index_add", backward)
