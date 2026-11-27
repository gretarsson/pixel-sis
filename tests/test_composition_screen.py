"""Check the aggregated local susceptibility operator by explicit enumeration."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
spec = importlib.util.spec_from_file_location("screen_sir_rates", EXAMPLES / "screen_sir_rates.py")
rates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rates)
sys.modules["screen_sir_rates"] = rates
spec = importlib.util.spec_from_file_location(
    "screen_sir_composition", EXAMPLES / "screen_sir_composition.py"
)
composition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(composition)


def test_coarse_local_composition_basis():
    N = np.array([10, 25, 30, 50])
    labels = np.array([0, 1, 0, 1])
    population = np.bincount(labels, weights=N)
    coords = np.c_[np.arange(4), np.zeros((4, 2), dtype=int)].astype(np.int16)
    index = np.arange(4, dtype=np.int32).reshape(4, 1, 1)
    fields = np.array([[0.2, 0.3, 0.4, 0.7], [0.5, 0.4, 0.1, 0.2], [0.3, 0.3, 0.5, 0.1]])
    offsets = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int16)
    probability = np.array([0.2, 0.6, 0.2])
    K = np.diag(np.full(4, 0.6)) + np.diag(np.full(3, 0.2), 1) + np.diag(np.full(3, 0.2), -1)
    Z = K @ N
    L = K * N[None, :] / Z[:, None]
    actual = composition.local_basis(
        N, population, labels, fields, coords, index, offsets, probability, Z
    )
    G = np.eye(2)[labels]
    for c in range(3):
        expected = (G.T @ np.diag(N) @ L @ np.diag(fields[c]) @ G) / population[:, None]
        np.testing.assert_allclose(actual[c], expected.T, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual.sum(axis=(0, 1)), 1)
