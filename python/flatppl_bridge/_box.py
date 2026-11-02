"""Boxes of Python callables, and the dispatcher the native server calls."""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

__all__ = ["Box", "InvalidInput"]

# Status codes of the plugin ABI (docs/plugin-abi.md).
_INVALID = 1
_UNSUPPORTED = 2
_FAILURE = 3

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DTYPES = {"f8": np.float64, "f4": np.float32, "i8": np.int64, "b1": np.bool_}
_RNG_TAG = b"FPBNP1"


class InvalidInput(ValueError):
    """Raise from a box callable to reject its input. The client gets HTTP 400."""


@dataclass
class _Member:
    kind: str
    input: str = "anything"
    output: str = "anything"
    variate: str = "anything"
    normalized: bool = False
    apply: Callable | None = None
    vjp: Callable | None = None
    logdensity: Callable | None = None
    gradient: Callable | None = None
    sample: Callable | None = None
    queries: list[str] = field(default_factory=list)

    def describe(self, name: str) -> str:
        fields = [f'kind = "{self.kind}"']
        fields.append("queries = [" + ", ".join(f'"{q}"' for q in self.queries) + "]")
        # A function's gradient is its VJP with a cotangent of ones.
        gradient = self.vjp if self.kind == "function" else self.gradient
        fields.append(f"gradient = {_bool(gradient is not None)}")
        fields.append(f"vjp = {_bool(self.vjp is not None)}")
        if self.kind == "function":
            fields += [f"input = {self.input}", f"output = {self.output}"]
        else:
            if self.kind == "kernel":
                fields.append(f"input = {self.input}")
            fields += [f"variate = {self.variate}", f"normalized = {_bool(self.normalized)}"]
        return f"{name} = record({', '.join(fields)})"


def _bool(value: bool) -> str:
    return "true" if value else "false"


class Box:
    """A black box whose members are Python callables.

    Every callable works on a batch of rows. An array argument has the row axis
    first, and a record argument is a dict of such arrays. A callable returns
    its results with the same row axis. A sampler is the exception: it sees one
    input without a row axis.

    Value sets are FlatPPL text, for example ``"cartpow(reals, 3)"``. The
    default ``"anything"`` leaves the checks to the callable.
    """

    def __init__(self) -> None:
        self._members: dict[str, _Member] = {}

    def function(
        self,
        name: str,
        fn: Callable[[Any], Any],
        *,
        input: str = "anything",
        output: str = "anything",
        vjp: Callable[[Any, np.ndarray], Any] | None = None,
    ) -> None:
        """Add a function member ``name``, answered by ``fn(x)``.

        ``vjp(x, cotangent)`` returns the vector-Jacobian product, shaped like
        ``x``. With it, the member also answers gradients, which are VJPs with
        a cotangent of ones.
        """
        self._add(name, _Member("function", input=input, output=output, apply=fn, vjp=vjp, queries=["apply"]))

    def kernel(
        self,
        name: str,
        *,
        logdensity: Callable[[Any, np.ndarray], np.ndarray] | None = None,
        gradient: Callable[[Any, np.ndarray], tuple[Any, np.ndarray]] | None = None,
        sample: Callable[[Any, tuple[int, ...], np.random.Generator], np.ndarray] | None = None,
        input: str = "anything",
        variate: str = "anything",
        normalized: bool = False,
    ) -> None:
        """Add a kernel member ``name``: a measure for each input.

        - ``logdensity(input, x)`` returns the log density of each row.
        - ``gradient(input, x)`` returns ``(d_input, d_x)``, the gradients of
          each row's log density.
        - ``sample(input, size, rng)`` returns an array of shape ``size`` plus
          the variate shape. It needs ``normalized=True``.
        """
        self._add(name, self._measure("kernel", logdensity, gradient, sample, input, variate, normalized))

    def measure(
        self,
        name: str,
        *,
        logdensity: Callable[[np.ndarray], np.ndarray] | None = None,
        gradient: Callable[[np.ndarray], np.ndarray] | None = None,
        sample: Callable[[tuple[int, ...], np.random.Generator], np.ndarray] | None = None,
        variate: str = "anything",
        normalized: bool = False,
    ) -> None:
        """Add a measure member ``name``, with the callables of :meth:`kernel` minus the input."""
        self._add(name, self._measure("measure", logdensity, gradient, sample, "anything", variate, normalized))

    @staticmethod
    def _measure(kind, logdensity, gradient, sample, input, variate, normalized) -> _Member:
        if sample is not None and not normalized:
            raise ValueError("sampling needs a normalized measure (normalized=True)")
        if gradient is not None and logdensity is None:
            raise ValueError("a gradient needs a log density")
        queries = []
        if logdensity is not None:
            queries.append("builtin_logdensityof")
        if sample is not None:
            queries.append("builtin_sample")
        if not queries:
            raise ValueError("give a log density, a sampler, or both")
        return _Member(
            kind,
            input=input,
            variate=variate,
            normalized=normalized,
            logdensity=logdensity,
            gradient=gradient,
            sample=sample,
            queries=queries,
        )

    def _add(self, name: str, member: _Member) -> None:
        if not _NAME.match(name):
            raise ValueError(f"member name {name!r} is not a FlatPPL name")
        if name in self._members:
            raise ValueError(f"member {name!r} exists already")
        self._members[name] = member

    def describe(self) -> str:
        """The describe text of this box (see docs/protocol.md)."""
        return "".join(m.describe(n) + "\n" for n, m in self._members.items())

    def _dispatch(self, op, member, rows, input, point, sizes, rng, derivative):
        try:
            return self._call(op, member, rows, input, point, sizes, rng, derivative)
        except InvalidInput as error:
            return ("error", _INVALID, str(error))
        except NotImplementedError as error:
            return ("error", _UNSUPPORTED, str(error))
        except Exception:
            return ("error", _FAILURE, traceback.format_exc(limit=8))

    def _call(self, op, name, rows, input, point, sizes, rng, derivative):
        m = self._members.get(name)
        if m is None:
            raise NotImplementedError(f"no member {name!r}")
        x = _value(input)
        p = _value(point)
        vjp = derivative is not None and derivative[0] == "vjp"
        wrt_input = derivative is not None and derivative[1]
        wrt_point = derivative is not None and derivative[2]
        d_input = d_point = new_rng = None

        if op == "apply" and m.apply is not None:
            value = _rows(m.apply(x), rows, "the function")
            if derivative is not None:
                if m.vjp is None:
                    raise NotImplementedError(f"{name!r} has no {derivative[0]}")
                cotangent = _value(derivative[3]) if vjp else np.ones_like(value, dtype=np.float64)
                d_input = _rows(m.vjp(x, cotangent), rows, "the vjp")
        elif op == "logdensity" and m.logdensity is not None:
            args = (x, p) if m.kind == "kernel" else (p,)
            value = _rows(np.asarray(m.logdensity(*args), dtype=np.float64), rows, "the log density")
            if derivative is not None:
                if vjp or m.gradient is None:
                    raise NotImplementedError(f"{name!r} has no {derivative[0]}")
                grads = m.gradient(*args)
                if m.kind == "kernel":
                    d_in, d_x = grads
                    d_input = _rows(d_in, rows, "the gradient") if wrt_input else None
                    d_point = _rows(d_x, rows, "the gradient") if wrt_point else None
                else:
                    d_point = _rows(grads, rows, "the gradient") if wrt_point else None
        elif op == "sample" and m.sample is not None:
            generator = _generator(rng)
            size = tuple(int(n) for n in sizes)
            single = _unrow(x) if m.kind == "kernel" else None
            args = (single, size, generator) if m.kind == "kernel" else (size, generator)
            draws = np.asarray(m.sample(*args))
            count = int(np.prod(size)) if size else 1
            if draws.shape[: len(size)] != size:
                raise ValueError(f"the sampler returned shape {draws.shape} for size {size}")
            value = draws.reshape((count,) + draws.shape[len(size) :])
            new_rng = _state(generator)
        else:
            raise NotImplementedError(f"{name!r} does not answer {op}")
        return ("ok", _tree(value), _tree(d_input), _tree(d_point), new_rng)


def _value(tree):
    """A value tree as a NumPy array, or a dict of them for a record."""
    if tree is None:
        return None
    if tree[0] == "record":
        return {name: _value(t) for name, t in tree[1]}
    _, dtype, shape, data = tree
    return np.frombuffer(data, dtype=np.dtype(_DTYPES[dtype]).newbyteorder("<")).reshape(shape)


def _tree(value):
    """A NumPy array, or a dict of them, as a value tree."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return ("record", [(str(k), _tree(v)) for k, v in value.items()])
    array = np.asarray(value)
    for code, dtype in _DTYPES.items():
        if array.dtype == dtype:
            break
    else:
        if np.issubdtype(array.dtype, np.integer):
            code, array = "i8", array.astype(np.int64)
        elif np.issubdtype(array.dtype, np.floating):
            code, array = "f8", array.astype(np.float64)
        else:
            raise TypeError(f"a box returned dtype {array.dtype}, which holds no FlatPPL value")
    data = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<")).tobytes()
    return ("array", code, tuple(array.shape), data)


def _rows(value, rows, what):
    """Check that every array in ``value`` has ``rows`` rows."""
    arrays = value.values() if isinstance(value, Mapping) else [value]
    for array in arrays:
        shape = np.shape(array)
        if not shape or shape[0] != rows:
            raise ValueError(f"{what} returned shape {shape}, expected {rows} rows first")
    return value


def _unrow(value):
    """Drop the row axis of a one-row value."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {k: _unrow(v) for k, v in value.items()}
    array = value[0]
    return array.item() if array.ndim == 0 else array


def _generator(rng) -> np.random.Generator:
    if rng is None:
        raise InvalidInput("sampling needs an RNG state")
    kind, data = rng
    if kind == "seed":
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(data))))
    if len(data) != len(_RNG_TAG) + 32 + 5 or not data.startswith(_RNG_TAG):
        raise NotImplementedError("a foreign RNG state")
    body = data[len(_RNG_TAG) :]
    bit_generator = np.random.PCG64()
    bit_generator.state = {
        "bit_generator": "PCG64",
        "state": {
            "state": int.from_bytes(body[:16], "little"),
            "inc": int.from_bytes(body[16:32], "little"),
        },
        "has_uint32": body[32],
        "uinteger": int.from_bytes(body[33:37], "little"),
    }
    return np.random.Generator(bit_generator)


def _state(generator: np.random.Generator) -> bytes:
    """The PCG64 state, tagged so that a foreign state is refused."""
    state = generator.bit_generator.state
    inner = state["state"]
    return (
        _RNG_TAG
        + int(inner["state"]).to_bytes(16, "little")
        + int(inner["inc"]).to_bytes(16, "little")
        + bytes([state["has_uint32"]])
        + int(state["uinteger"]).to_bytes(4, "little")
    )
