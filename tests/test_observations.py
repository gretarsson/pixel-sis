import numpy as np
import pytest
from scipy import sparse
from test_process import generator

from pixel_sis import Rates, from_contacts, simulate_ensemble
from pixel_sis.fitting import fit_rates
from pixel_sis.observations import RegionalObservations


def test_analytic_global_amplitude():
    obs = RegionalObservations(
        np.array([0.0, 1.0]),
        sparse.eye(2),
        np.array([4.0, 8.0, 6.0, 10.0]),
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 1]),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array(["a", "b", "a", "b"]),
        np.array([0, 0, 1, 1]),
        factor=1.0,
    )
    loss, scale = obs.loss_and_scale(np.array([[2.0, 4.0], [3.0, 5.0]]))
    assert scale == pytest.approx(2.0) and loss == pytest.approx(0.0)
    assert obs.loss_and_scale(np.zeros((2, 2)))[0] == pytest.approx(1.0)


def test_missing_measurements_are_not_zeros():
    obs = RegionalObservations(
        np.array([1.0]),
        sparse.csr_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([0.0, 4.0, 6.0]),
        np.array([0, 0, 0]),
        np.array([0, 1, 1]),
        np.ones(3),
        np.array(["a", "b", "b"]),
        np.array([0, 0, 1]),
    )
    painted = obs.atlas_means()[0]
    assert painted[0] == 0 and painted[1] == 5 and np.isnan(painted[2])


@pytest.mark.parametrize("fixed_scale", [None, 2.0])
def test_fit_reduces_error_on_an_independent_synthetic_target(fixed_scale):
    from scipy.linalg import expm

    N = np.array([4])
    initial = np.array([2])
    times = np.array([0.0, 0.4, 1.0, 2.0])
    states, index, G = generator(N, np.array([[0.8]]), 0.2)
    mean = np.array([expm(t * G)[index[tuple(initial)]] @ states for t in times])
    obs = RegionalObservations(
        times,
        sparse.eye(1),
        2 * mean[:, 0],
        np.arange(4),
        np.zeros(4, dtype=int),
        np.ones(4),
        np.repeat("a", 4),
        np.arange(4),
        factor=1.0,
    )
    model = from_contacts(N, initial, [[1.0]])
    start = Rates(0.05, 0.0, 0.7)
    e = simulate_ensemble(
        model, times, start, runs=128, seed=1234, observation=obs.operator, maps=False
    )
    initial_loss, _ = obs.loss_and_scale(e["density_samples"].mean(0), scale=fixed_scale)
    fit = fit_rates(
        model,
        obs,
        [start],
        tracer_only=True,
        runs=128,
        selection_runs=256,
        seed=1234,
        max_evaluations=45,
        observation_scale_um=fixed_scale,
    )
    assert fit["selected"]["relative_rms"] ** 2 < initial_loss
    assert fit["selected"]["rates"]["beta_local"] == 0
    assert "converged" in fit["selected"]
    if fixed_scale is not None:
        assert all(c["scale_um"] == fixed_scale for c in fit["candidates"])


def test_fit_respects_small_rate_bounds():
    model = from_contacts([4], [2], [[1.0]])
    obs = RegionalObservations(
        np.array([0.0, 1.0]),
        sparse.eye(1),
        np.array([2.0, 2.0]),
        np.array([0, 1]),
        np.zeros(2, dtype=int),
        np.ones(2),
        np.array(["a", "a"]),
        np.array([0, 1]),
        factor=1.0,
    )
    fit = fit_rates(
        model,
        obs,
        [Rates(0.005, 0, 0.005)],
        tracer_only=True,
        upper_rate=0.01,
        observation_scale_um=1.0,
        runs=2,
        selection_runs=2,
        max_evaluations=4,
    )
    assert all(0 <= x <= 0.01 for x in fit["selected"]["rates"].values())
