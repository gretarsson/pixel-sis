"""Independent numerical checks for the SIR search surrogate integrator."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

spec = importlib.util.spec_from_file_location(
    "screen_sir_rates", Path(__file__).resolve().parents[1] / "examples/screen_sir_rates.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "beta,gamma", [(0.0, 0.5), (0.5, 0.0), (0.5, 0.5), (3.0, 5.0), (10.0, 0.3)]
)
def test_sir_surrogate_matches_independent_integrator(beta, gamma):
    rng = np.random.default_rng(27)
    Q = rng.random((5, 5))
    Q /= Q.sum(axis=1)[:, None]
    N = np.array([100.0, 230.0, 800.0, 50.0, 110.0])
    initial = np.array([20.0, 0.0, 0.0, 10.0, 0.0])
    times = np.array([0.1, 0.2, 1.0, 3.0, 9.0])

    def force(I):
        return beta * (Q.T @ I)

    I, R, _ = module.integrate_sir(force, N, initial, times, gamma, rtol=1e-7)

    def rhs(t, state):
        infected, removed = state[:5], state[5:]
        clearances = gamma * infected
        return np.r_[(1 - (infected + removed) / N) * force(infected) - clearances, clearances]

    reference = solve_ivp(
        rhs,
        [0, 9],
        np.r_[initial, np.zeros(5)],
        t_eval=times,
        rtol=1e-11,
        atol=1e-11,
        method="DOP853",
    ).y
    np.testing.assert_allclose(I, reference[:5].T, rtol=3e-5, atol=2e-5)
    np.testing.assert_allclose(R, reference[5:].T, rtol=3e-5, atol=2e-5)
    assert min(I.min(), R.min(), (N - I - R).min()) > -1e-8
    assert np.diff(R, axis=0).min() > -1e-8
