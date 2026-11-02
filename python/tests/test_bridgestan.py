"""The BridgeStan plugin, served from Python, against BridgeStan's own Python API.

Needs the plugin library and compiled models: set FPB_BRIDGESTAN_PLUGIN to the
plugin library and FPB_BRIDGESTAN_MODELS to the model directory (see README).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import flatppl_bridge as fb
from client import query

PLUGIN = os.environ.get("FPB_BRIDGESTAN_PLUGIN")
MODELS = os.environ.get("FPB_BRIDGESTAN_MODELS")
pytestmark = pytest.mark.skipif(not (PLUGIN and MODELS), reason="needs FPB_BRIDGESTAN_PLUGIN and FPB_BRIDGESTAN_MODELS")


def test_log_density_and_gradient_match_bridgestan():
    bridgestan = pytest.importorskip("bridgestan")
    model = f"{MODELS}/normal_model.so"
    config = json.dumps({"model": model})
    with fb.Server(libraries={"stan": (PLUGIN, config)}) as server:
        text = 'stan = load_module("box:stan")\nu = load_data("part:u", cartpow(reals, 2))\nresult = builtin_logdensityof(stan.posterior, u)\n'
        u = np.array([0.3, 0.2])
        parts = query(server.url, text, {"u": u}, {"derivative": {"kind": "gradient", "wrt": ["u"]}}).parts()
    direct = bridgestan.StanModel(model)
    value, grad = direct.log_density_gradient(u, propto=False, jacobian=True)
    assert parts["value"] == value
    np.testing.assert_array_equal(parts["d_u"], grad)


def test_a_batch_of_theta_vectors_matches_bridgestan_per_row():
    bridgestan = pytest.importorskip("bridgestan")
    model = f"{MODELS}/normal_model.so"
    thetas = np.array([[0.3, 0.2], [-1.0, 0.5], [2.0, -0.7]])
    with fb.Server(libraries={"stan": (PLUGIN, json.dumps({"model": model}))}) as server:
        text = 'stan = load_module("box:stan")\nthetas = load_data("part:thetas", cartpow(cartpow(reals, 2), 3))\nresult = builtin_logdensityof.(stan.posterior, thetas)\n'
        parts = query(server.url, text, {"thetas": thetas}, {"derivative": {"kind": "gradient", "wrt": ["thetas"]}}).parts()
    direct = bridgestan.StanModel(model)
    for row, theta in enumerate(thetas):
        value, grad = direct.log_density_gradient(theta, propto=False, jacobian=True)
        assert parts["value"][row] == value
        np.testing.assert_array_equal(parts["d_thetas"][row], grad)
