"""Python boxes over HTTP. Expected values come from scipy and NumPy, not from the box."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
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
    box.kernel(
        "shifted",
        sample=lambda mu, size, rng: mu + rng.standard_normal(size + mu.shape),
        input="cartpow(reals, 2)",
        variate="cartpow(reals, 2)",
        normalized=True,
    )

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
    response = requests.get(f"{server.url}/v1/boxes/py/describe", timeout=10)
    assert response.headers["content-type"] == "application/json"
    members = response.json()
    assert members["norm2"] == {
        "kind": "function",
        "queries": ["apply"],
        "gradient": True,
        "vjp": True,
        "input": "cartpow(reals, 3)",
        "output": "reals",
    }
    assert members["std"]["queries"] == ["builtin_sample"]
    assert members["std"]["variate"] == "cartpow(reals, 2)"
    assert json.loads(fb.Box().describe()) == {}


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


def _numpy_rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(seed.to_bytes(8, "little")))))


def _literal(state) -> str:
    return "[" + ", ".join(str(int(b)) for b in state) + "]"


def _reals(vector) -> str:
    return "[" + ", ".join(repr(float(x)) for x in vector) + "]"


def test_batched_sampling_equals_unbatched_draws_per_row(server):
    mus = np.array([[0.0, 1.0], [10.0, 20.0], [-5.0, 5.0]])
    head = HEAD + 'mus = load_data("part:mus", cartpow(cartpow(reals, 2), 3))\n'
    text = head + "result = broadcast(builtin_sample, rnginit.([11, 12, 13]), py.shifted, mus, 4)\n"
    parts = query(server.url, text, {"mus": mus}).parts()
    assert parts["value"].shape == (3, 4, 2)
    states = parts["rngstate"]
    assert states.shape[0] == 3
    for row, seed in enumerate([11, 12, 13]):
        expected = mus[row] + _numpy_rng(seed).standard_normal((4, 2))
        np.testing.assert_array_equal(parts["value"][row], expected)
        single = HEAD + f"result = builtin_sample(rnginit({seed}), py.shifted, {_reals(mus[row])}, 4)\n"
        one = query(server.url, single).parts()
        np.testing.assert_array_equal(one["value"], parts["value"][row])
        np.testing.assert_array_equal(one["rngstate"], states[row])

    # Resume every row from the state the batched reply returned.
    text = head + f's = load_data("part:s", cartpow(cartpow(integers, {states.shape[1]}), 3))\n'
    text += "result = broadcast(builtin_sample, rngstate.(s), py.shifted, mus, 4)\n"
    resumed = query(server.url, text, {"mus": mus, "s": states}).parts()["value"]
    for row in range(3):
        single = HEAD + f"result = builtin_sample(rngstate({_literal(states[row])}), py.shifted, {_reals(mus[row])}, 4)\n"
        np.testing.assert_array_equal(query(server.url, single).parts()["value"], resumed[row])

    # A measure member: one draw per row.
    text = HEAD + "result = broadcast(builtin_sample, rnginit.([1, 2]), py.std)\n"
    value = query(server.url, text).parts()["value"]
    np.testing.assert_array_equal(value, [_numpy_rng(1).standard_normal(2), _numpy_rng(2).standard_normal(2)])


@pytest.mark.parametrize(
    "rng",
    ["rnginit(1)", "rnginit.([1])"],
)
def test_batched_sampling_refuses_a_shared_rng_state(server, rng):
    text = HEAD + f'mus = load_data("part:mus", cartpow(cartpow(reals, 2), 3))\nresult = broadcast(builtin_sample, {rng}, py.shifted, mus)\n'
    reply = query(server.url, text, {"mus": np.zeros((3, 2))})
    assert reply.status == 422, reply.body


def _list_part(rows: np.ndarray) -> bytes:
    column = pa.array([list(r) for r in rows], type=pa.list_(pa.float64()))
    table = pa.table({"xs": column})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def test_a_batch_of_vectors_reaches_the_callable_one_vector_per_row():
    calls = []
    xs = np.arange(12.0).reshape(4, 3)
    with fb.Server({"py": _normal_box(calls)}) as s:
        text = HEAD + 'xs = load_data("part:xs", cartpow(cartpow(reals, 3), 4))\nresult = py.norm2.(xs)\n'
        np.testing.assert_array_equal(query(s.url, text, {"xs": xs}).parts()["value"], np.sum(xs**2, axis=1))
        np.testing.assert_array_equal(query(s.url, text, {"xs": _list_part(xs)}).parts()["value"], np.sum(xs**2, axis=1))
        assert calls == [4, 4]
        # A matrix broadcasts to scalars, which are not in cartpow(reals, 3).
        text = HEAD + 'xs = load_data("part:xs", cartpow(reals, [4, 3]))\nresult = py.norm2.(xs)\n'
        assert query(s.url, text, {"xs": xs}).status == 400
        assert calls == [4, 4]


def test_the_server_checks_inputs_before_the_callable():
    calls = []
    with fb.Server({"py": _normal_box(calls)}) as s:
        assert query(s.url, HEAD + "result = py.norm2([1.0, 2.0])\n").status == 400
        assert query(s.url, HEAD + "result = py.norm2([true, false, true])\n").status == 400
        assert calls == []
        text = HEAD + "result = builtin_logdensityof(py.normal, (mu = 0.0, sigma = -1.0), 0.3)\n"
        reply = query(s.url, text)
        assert reply.status == 400
        assert "posreals" in reply.json()["detail"]
