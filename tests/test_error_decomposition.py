"""The common-trajectory SSE decomposition must hold with unequal weights."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "decompose_sir_error", Path(__file__).resolve().parents[1] / "examples/decompose_sir_error.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_weighted_decomposition_and_zero_weight_records():
    y = np.array([1.0, 3.0, 9.0, 7.0, 1000.0])
    w = np.array([1.0, 3.0, 2.0, 1.0, 0.0])
    cell = np.array([10, 10, 20, 30, 30])
    prediction = np.array([2.0, 2.0, 8.0, 7.0, 0.0])
    scatter, mismatch, mean = module.decompose(y, w, cell, prediction)
    np.testing.assert_allclose(mean, [2.5, 2.5, 9.0, 7.0, 7.0])
    np.testing.assert_allclose(scatter.sum(), 3.0)
    np.testing.assert_allclose(mismatch.sum(), 3.0)
    assert scatter.sum() + mismatch.sum() == np.sum(w * (y - prediction) ** 2)


def test_individual_predictions_cannot_use_common_curve_identity():
    with pytest.raises(ValueError, match="one prediction"):
        module.decompose([1.0, 2.0], [1.0, 1.0], [0, 0], [1.0, 2.0])
