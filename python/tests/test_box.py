"""Python boxes over HTTP. Expected values come from scipy and NumPy, not from the box."""

from __future__ import annotations

import numpy as np
import pytest
import requests
from scipy import stats

import flatppl_bridge as fb
from client import query

HEAD = 'py = load_module("box:py")\n'


def _normal_box(calls: list | None = None) -> fb.Box:
    box = fb.Box()

    def norm2(x):
        if calls is not None:
            calls.append(len(x))
        return np.sum(x**2, axis=-1)

    box.function("norm2", norm2, input="cartpow(reals, 3)", output="reals", vjp=lambda x, ct: 2 * x * ct[:, None])
    def expm(x):
        if calls is not None:
            calls.append(len(x))
        return np.exp(x)

    box.function("expm", expm, vjp=lambda x, ct: np.exp(x) * ct)
    box.kernel(
        "normal",
        logdensity=lambda p, x: stats.norm.logpdf(x, p["mu"], p["sigma"]),
        gradient=lambda p, x: (
            {"mu": (x - p["mu"]) / p["sigma"] ** 2, "sigma": ((x - p["mu"]) ** 2 / p["sigma"] ** 2 - 1) / p["sigma"]},
            -(x - p["mu"]) / p["sigma"] ** 2,
        ),
        sample=lambda p, size, rng: rng.normal(p["mu"], p["sigma"], size),
        input="cartprod(mu = reals, sigma = posreals)",
        variate="reals",
        normalized=True,
    )
    box.measure("std", sample=lambda size, rng: rng.standard_normal(size + (2,)), variate="cartpow(reals, 2)", normalized=True)

    def picky(x):
        if np.any(x < 0):
            raise fb.InvalidInput("negative input")
        return x

    box.function("picky", picky)
    box.function("broken", lambda x: 1 / 0)
    return box


@pytest.fixture
def server():
    with fb.Server({"py": _normal_box()}) as s:
        yield s


def test_describe_lists_the_members(server):
    text = requests.get(f"{server.url}/v1/boxes/py/describe", timeout=10).text
    assert 'norm2 = record(kind = "function", queries = ["apply"], gradient = true, vjp = true' in text
    assert 'std = record(kind = "measure", queries = ["builtin_sample"]' in text


def test_application_gradient_and_vjp(server):
    text = HEAD + 'x = load_data("part:x", cartpow(reals, 3))\nresult = py.norm2(x)\n'
    x = np.array([1.0, 2.0, 3.0])
    parts = query(server.url, text, {"x": x}, {"derivative": {"kind": "gradient", "wrt": ["x"]}}).parts()
    assert parts["value"] == 14.0
    np.testing.assert_array_equal(parts["d_x"], 2 * x)
    text = HEAD + 'x = load_data("part:x", cartpow(reals, 2))\nct = load_data("part:ct", cartpow(reals, 2))\nresult = py.expm(x)\n'
    parts = query(
        server.url,
        text,
        {"x": np.array([0.0, 1.0]), "ct": np.array([2.0, 3.0])},
        {"derivative": {"kind": "vjp", "wrt": ["x"], "cotangent": "ct"}},
    ).parts()
    np.testing.assert_allclose(parts["value"], [1.0, np.e], rtol=1e-15)
    np.testing.assert_allclose(parts["d_x"], [2.0, 3 * np.e], rtol=1e-15)


def test_broadcast_reaches_the_callable_once():
    calls = []
    with fb.Server({"py": _normal_box(calls)}) as s:
        text = HEAD + 'x = load_data("part:x", cartpow(reals, 1000))\nresult = py.expm.(x)\n'
        x = np.linspace(-1, 1, 1000)
        np.testing.assert_allclose(query(s.url, text, {"x": x}).parts()["value"], np.exp(x), rtol=1e-15)
        text = HEAD + 'x = load_data("part:x", cartpow(reals, 3))\nresult = py.norm2(x)\n'
        query(s.url, text, {"x": np.ones(3)})
        assert calls == [1000, 1]


def test_kernel_log_density_matches_scipy(server):
    # Spec §04: a record is not a broadcast operand, so batch over a table instead.
    text = HEAD + 'x = load_data("part:x", cartpow(reals, 4))\nresult = builtin_logdensityof.(py.normal, (mu = 1.5, sigma = 2.0), x)\n'
    x = np.array([-1.0, 0.3, 1.5, 4.0])
    reply = query(server.url, text, {"x": x})
    assert reply.status == 400
    text = HEAD + 'mu = 1.5\nx = 0.3\nresult = builtin_logdensityof(py.normal, (mu = mu, sigma = 2.0), x)\n'
    parts = query(server.url, text, options={"derivative": {"kind": "gradient", "wrt": ["mu", "x"]}}).parts()
    assert parts["value"] == pytest.approx(stats.norm.logpdf(0.3, 1.5, 2.0), rel=1e-15)
    assert parts["d_mu"] == pytest.approx(-0.3, rel=1e-14)
    assert parts["d_x"] == pytest.approx(0.3, rel=1e-14)


def test_sampling_follows_numpy_and_threads_the_state(server):
    text = HEAD + "result = builtin_sample(rnginit(5), py.normal, (mu = 10.0, sigma = 0.5), 3, 2)\n"
    parts = query(server.url, text).parts()
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(list((5).to_bytes(8, "little")))))
    expected = rng.normal(10.0, 0.5, (3, 2))
    np.testing.assert_array_equal(parts["value"], expected)
    state = ", ".join(str(int(b)) for b in parts["rngstate"])
    resume = HEAD + f"result = builtin_sample(rngstate([{state}]), py.normal, (mu = 10.0, sigma = 0.5), 3, 2)\n"
    np.testing.assert_array_equal(query(server.url, resume).parts()["value"], rng.normal(10.0, 0.5, (3, 2)))
    text = HEAD + "result = builtin_sample(rnginit(1), py.std, 4)\n"
    assert query(server.url, text).parts()["value"].shape == (4, 2)
    text = HEAD + "result = builtin_sample(rnginit(1), py.std)\n"
    assert query(server.url, text).parts()["value"].shape == (2,)


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("result = builtin_sample(rngstate([1, 2, 3]), py.std)\n", 422),
        ("result = py.picky([-1.0])\n", 400),
        ("result = py.broken([1.0])\n", 500),
        ("result = builtin_logdensityof(py.std, [1.0, 2.0])\n", 422),
        ("result = py.missing(1.0)\n", 422),
    ],
)
def test_refusals(server, text, status):
    reply = query(server.url, HEAD + text)
    assert reply.status == status, reply.body
    assert reply.content_type == "application/problem+json"


def test_box_rules():
    box = fb.Box()
    with pytest.raises(ValueError, match="normalized"):
        box.measure("m", sample=lambda size, rng: rng.random(size))
    with pytest.raises(ValueError, match="FlatPPL name"):
        box.function("1f", abs)
    with pytest.raises(ValueError, match="at least one"):
        fb.Server({})


def test_token():
    with fb.Server({"py": _normal_box()}, token="s3cret") as s:
        assert requests.get(f"{s.url}/v1/boxes", timeout=10).status_code == 401
        assert query(s.url, HEAD + "result = py.expm(1.0)\n", token="s3cret").status == 200
