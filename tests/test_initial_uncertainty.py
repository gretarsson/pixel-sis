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
