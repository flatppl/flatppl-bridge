# flatppl-bridge for Python

Serve Python functions, log densities and samplers as FlatPPL black boxes.

A FlatPPL model loads a black box as an ordinary module of `external(anything)` members and uses the usual `logdensityof`, `rand`, `~` and function application. The FlatPPL engine sends the box primitive queries over HTTP, with data as Arrow IPC. This package runs that server, with members written in Python. The protocol is in `docs/protocol.md` of the repository.

## Install

```sh
pip install flatppl-bridge
```

Building from source needs a Rust toolchain and [maturin](https://www.maturin.rs): run `maturin develop --release` in this directory.

## Serve a box

```python
import numpy as np
from scipy import stats
import flatppl_bridge as fb

box = fb.Box()

# An expensive function, with a vector-Jacobian product.
box.function(
    "energy",
    lambda x: np.sum(x**2, axis=-1),
    input="cartpow(reals, 3)",
    output="reals",
    vjp=lambda x, ct: 2 * x * ct[:, None],
)

# A kernel: a normal distribution for each input (mu, sigma).
box.kernel(
    "normal",
    logdensity=lambda p, x: stats.norm.logpdf(x, p["mu"], p["sigma"]),
    sample=lambda p, size, rng: rng.normal(p["mu"], p["sigma"], size),
    input="cartprod(mu = reals, sigma = posreals)",
    variate="reals",
    normalized=True,
)

fb.serve({"physics": box}, port=8737)
```

`fb.serve` blocks until interrupted. `fb.Server` serves in background threads, for tests and notebooks:

```python
with fb.Server({"physics": box}) as server:
    print(server.url)  # http://127.0.0.1:<free port>
```

## Conventions

- **Batches.** A callable gets a batch of rows: an array argument has the row axis first, and a record argument is a dict of such arrays. It returns its results with the same row axis. A FlatPPL `broadcast` over many points reaches the callable once.
- **Samplers** get one input without a row axis, a size tuple and a `numpy.random.Generator`. They return an array of shape `size` plus the variate shape. Only a normalized member can sample, because only a probability measure has samples.
- **Value sets** are FlatPPL text, such as `"cartpow(reals, 3)"`. The default `"anything"` leaves the checks to the callable.
- **Derivatives.** A function's `vjp(x, cotangent)` returns the vector-Jacobian product, shaped like `x`. The member then also answers gradients. A kernel's `gradient(input, x)` returns `(d_input, d_x)`. A measure's `gradient(x)` returns `d_x`.
- **Random numbers.** The box uses NumPy's PCG64. A query seeds it with `rnginit(seed)`, and the reply carries the new state, which the next query passes back with `rngstate(...)`. Equal states give equal samples.
- **Errors.** Raise `fb.InvalidInput` to reject an input (HTTP 400). Raise `NotImplementedError` for an unsupported query (HTTP 422). Any other exception is a box failure (HTTP 500). A degenerate result is a value: return `-inf` for a log density outside the support.

Input arrays are read-only views. Copy one before changing it.

## Plugin libraries

A server can also host compiled plugins, for example a Stan model through the BridgeStan plugin of this repository:

```python
config = '{"model": "models/model_model.so", "data": "data.json"}'
with fb.Server({"physics": box}, libraries={"stan": ("libfpb_bridgestan.so", config)}) as server:
    ...
```

## Security

`Server` listens on `127.0.0.1` by default. Pass `token=...` to require `Authorization: Bearer <token>`. Serve over HTTPS outside a trusted network, behind a reverse proxy or with the `flatppl-bridge-server` binary. The server checks every upload against its hash, and a query reads only uploaded objects and its own parts.

## Test

```sh
pixi run -e test test
```
