"""Initial-state uncertainty is explicit and has the declared count law."""

import numpy as np
import pytest

from pixel_sis import Rates, simulate_ensemble
from pixel_sis.model import from_contacts


def test_binomial_initial_law_and_reproducibility():
    N = np.array([100, 200, 300])
    p = np.array([0.1, 0.7, 0.0])
    model = from_contacts(N, np.zeros(3), np.eye(3))
    options = dict(runs=1000, seed=21, dynamics="sir", initial_probabilities=p)
    result = simulate_ensemble(model, [0.0, 1.0], Rates(0, 0, 0), **options)
    repeat = simulate_ensemble(model, [0.0, 1.0], Rates(0, 0, 0), **options)
    np.testing.assert_array_equal(result["total_samples"], repeat["total_samples"])
    np.testing.assert_array_equal(result["total_samples"][:, 0], result["initial_total_samples"])
    np.testing.assert_array_equal(result["count_mean"][0], result["count_mean"][1])
    np.testing.assert_array_equal(result["removed_mean"], 0)
    error = abs(result["count_mean"][0] - N * p)
    assert np.all(error <= 5 * np.sqrt(N * p * (1 - p) / 1000) + 1e-12)
    assert abs(result["initial_total_samples"].var(ddof=1) - np.sum(N * p * (1 - p))) < 10
    np.testing.assert_array_equal(model.initial, 0)


@pytest.mark.parametrize("p", [[0.2], [-0.1, 0.5], [0, 1.1], [0, np.nan]])
def test_invalid_initial_probabilities(p):
    model = from_contacts([10, 10], [1, 0], np.eye(2))
    with pytest.raises(ValueError, match="Initial probabilities"):
        simulate_ensemble(model, [0, 1], Rates(0, 0, 0), runs=2, initial_probabilities=p)


def test_joint_initial_law_moments_and_conservation():
    N = np.array([100, 200])
    p = np.array([0.2, 0.0])
    pr = np.array([0.5, 1.0])
    model = from_contacts(N, [0, 0], np.eye(2))
    result = simulate_ensemble(
        model,
        [0, 1],
        Rates(0, 0, 0),
        runs=2000,
        seed=41,
        dynamics="sir",
        initial_probabilities=p,
        initial_removed_probabilities=pr,
    )
    np.testing.assert_allclose(result["count_mean"][0], N * p, atol=0.5)
    np.testing.assert_allclose(result["initial_removed_mean"], N * pr, atol=0.6)
    infected = result["initial_total_samples"]
    removed = result["initial_removed_total_samples"]
    np.testing.assert_allclose(infected.var(ddof=1), np.sum(N * p * (1 - p)), rtol=0.12)
    np.testing.assert_allclose(removed.var(ddof=1), np.sum(N * pr * (1 - pr)), rtol=0.12)
    np.testing.assert_allclose(np.cov(infected, removed)[0, 1], -np.sum(N * p * pr), atol=2)
    np.testing.assert_array_equal(result["newly_removed_total_samples"], 0)
    np.testing.assert_array_equal(result["newly_removed_mean"], 0)
    assert np.all(result["susceptible_total_samples"] >= 0)
    np.testing.assert_array_equal(
        result["susceptible_total_samples"]
        + result["total_samples"]
        + result["removed_total_samples"],
        N.sum(),
    )


def test_initial_resistant_cells_never_reinfect():
    model = from_contacts([100], [0], [[1.0]])
    result = simulate_ensemble(
        model,
        [0, 1, 3],
        Rates(10, 10, 1),
        runs=1000,
        seed=42,
        dynamics="sir",
        initial_probabilities=[0.4],
        initial_removed_probabilities=[0.6],
    )
    np.testing.assert_array_equal(result["susceptible_total_samples"], 0)
    expected = 40 * np.exp(-result["times"])
    assert np.all(abs(result["count_mean"][:, 0] - expected) < 5 * result["count_mc_se"][:, 0])
    assert np.all(np.diff(result["newly_removed_total_samples"], axis=1) >= 0)


@pytest.mark.parametrize("pr", [[0.1], [-0.1, 0.0], [0.9, 0.0], [0.1, np.nan]])
def test_invalid_joint_initial_probabilities(pr):
    model = from_contacts([10, 10], [1, 0], np.eye(2))
    with pytest.raises(ValueError, match="joint initial"):
        simulate_ensemble(
            model,
            [0],
            Rates(0, 0, 0),
            runs=2,
            dynamics="sir",
            initial_probabilities=[0.2, 0.0],
            initial_removed_probabilities=pr,
        )
