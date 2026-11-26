"""Current regional observation model, separate from the SIS dynamics.

Section-level sampling/detection reconstruction is future work. Positive point
exports alone are not treated as complete tissue coverage or a cell census.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass
class RegionalObservations:
    times: np.ndarray
    operator: sparse.csr_matrix
    values: np.ndarray
    time_index: np.ndarray
    group: np.ndarray
    weights: np.ndarray
    labels: np.ndarray
    mouse_index: np.ndarray
    factor: float = 1e6

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.operator = sparse.csr_matrix(self.operator, dtype=float)
        for name in ("values", "time_index", "group", "weights", "labels", "mouse_index"):
            setattr(self, name, np.asarray(getattr(self, name)))
        if (
            self.times.ndim != 1
            or not len(self.times)
            or not np.isfinite(self.times).all()
            or np.any(self.times < 0)
            or np.any(np.diff(self.times) <= 0)
        ):
            raise ValueError("Observation times must be finite, distinct, ordered and nonnegative")
        if self.values.ndim != 1 or not len(self.values):
            raise ValueError("Empty observation set")
        if not all(
            getattr(self, name).shape == self.values.shape
            for name in ("time_index", "group", "weights", "labels", "mouse_index")
        ):
            raise ValueError("Observation record shapes differ")
        for name, upper in (("time_index", len(self.times)), ("group", self.operator.shape[0])):
            a = getattr(self, name)
            if not np.issubdtype(a.dtype, np.integer) or np.any(a < 0) or np.any(a >= upper):
                raise ValueError("Invalid observation " + name)
        if not np.isfinite(self.values).all() or np.any(self.values < 0):
            raise ValueError("Observation values must be finite and nonnegative")
        if (
            not np.isfinite(self.weights).all()
            or np.any(self.weights < 0)
            or self.weights.sum() <= 0
        ):
            raise ValueError("Invalid observation weights")
        if np.any(self.operator.data < 0) or not np.isfinite(self.operator.data).all():
            raise ValueError("Invalid observation operator")
        if not np.isfinite(self.factor) or self.factor <= 0:
            raise ValueError("Observation conversion factor must be positive")
        self.weights = self.weights.astype(float) / self.weights.sum()

    @classmethod
    def load(cls, data_dir):
        folder = Path(data_dir)
        with np.load(folder / "observations.npz", allow_pickle=False) as f:
            base = {k: f[k] for k in f.files}
        with np.load(folder / "cell_body_observations.npz", allow_pickle=False) as f:
            d = {k: f[k] for k in f.files}
        index = d["base_record_index"]
        np.testing.assert_array_equal(d["times"], base["times"])
        for key in ("time_index", "group", "mouse_index", "labels", "area"):
            np.testing.assert_array_equal(d[key], base[key][index])
        return cls(
            d["times"],
            sparse.load_npz(folder / "observation_density_fine.npz"),
            d["values"],
            d["time_index"],
            d["group"],
            base["weights"][index],
            d["labels"],
            d["mouse_index"],
        )

    def predictions(self, density_mean):
        density_mean = np.asarray(density_mean)
        if density_mean.shape != (len(self.times), self.operator.shape[0]):
            raise ValueError("Expected one regional prediction per observation time")
        return self.factor * density_mean[self.time_index, self.group]

    def loss_and_scale(self, density_mean, scale=None):
        """Weighted normalized SSE and one nonnegative global amplitude."""
        p = self.predictions(density_mean)
        if not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("Predictions must be finite and nonnegative")
        w, y = self.weights, self.values
        norm = float(np.dot(w, y * y))
        if norm <= 0:
            raise ValueError(
                "Relative error requires at least one positive-weight nonzero observation"
            )
        if scale is None:
            denominator = float(np.dot(w, p * p))
            scale = max(0.0, float(np.dot(w * y, p)) / denominator) if denominator else 0.0
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("Observation scale must be finite and nonnegative")
        return float(np.dot(w, (scale * p - y) ** 2) / norm), float(scale)

    def atlas_means(self):
        """Equal-mouse regional means painted on the observation support.

        These are regional values, not independently observed voxel measurements.
        Missing regional coverage remains NaN, rather than becoming zero.
        """
        sums = np.zeros((len(self.times), self.operator.shape[1]))
        numbers = np.zeros_like(sums, dtype=np.int16)
        O = self.operator
        for mouse in np.unique(self.mouse_index):
            occupied = np.zeros_like(sums, dtype=bool)
            for r in np.flatnonzero(self.mouse_index == mouse):
                t, g = self.time_index[r], self.group[r]
                ids = O.indices[O.indptr[g] : O.indptr[g + 1]]
                if occupied[t, ids].any():
                    raise ValueError("Overlapping regional observations within a mouse and time")
                occupied[t, ids] = True
                sums[t, ids] += self.values[r]
                numbers[t, ids] += 1
        return np.divide(sums, numbers, out=np.full_like(sums, np.nan), where=numbers > 0)
