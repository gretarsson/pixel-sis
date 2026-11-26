"""Reproducible ensembles, with Monte Carlo uncertainty and explicit run limits."""

from dataclasses import asdict, dataclass
from numbers import Integral
from time import perf_counter

import numpy as np

from ._kernels import run_strength

SEED_STRIDE = 100003


@dataclass(frozen=True)
class Rates:
    beta_tracer: float
    beta_local: float
    gamma: float

    def __post_init__(self):
        a = np.array([self.beta_tracer, self.beta_local, self.gamma])
        if not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError("Rates must be finite and nonnegative")

    def values(self):
        return (self.beta_tracer, self.beta_local, self.gamma)

    def to_dict(self):
        return asdict(self)


def simulate_ensemble(
    model,
    times,
    rates,
    *,
    runs=256,
    seed=112500100,
    max_events=2_000_000,
    observation=None,
    maps=True,
    progress=None,
):
    """Simulate independent paths without retaining all voxel trajectories.

    A path reaching the attempt limit raises; incomplete paths are never scored.
    If observation is supplied, its rows map counts to the desired readouts.
    """
    times = np.asarray(times, dtype=np.float64)
    if (
        times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or np.any(times < 0)
        or np.any(np.diff(times) < 0)
    ):
        raise ValueError("Times must be a nonempty, finite, ordered, nonnegative vector")
    if not isinstance(runs, Integral) or runs < 2:
        raise ValueError("At least two integer ensemble runs are required for a Monte Carlo SE")
    if not isinstance(max_events, Integral) or max_events < 1:
        raise ValueError("max_events must be a positive integer")
    if not isinstance(seed, Integral) or seed < 0 or seed + SEED_STRIDE * (runs - 1) > 2**32 - 1:
        raise ValueError("The complete seed sequence must fit uint32")
    if not isinstance(rates, Rates):
        raise TypeError("Pass a Rates instance")
    if observation is not None and observation.shape[1] != len(model.population):
        raise ValueError("Observation operator does not match the model")
    # Compile before recording simulation timings.
    run_strength(*model.arguments, np.array([0.0]), *rates.values(), seed, max_events)
    mean = np.zeros((len(times), len(model.population))) if maps else None
    m2 = np.zeros_like(mean) if maps else None
    totals, densities, seconds, events = [], [], [], []
    for r in range(runs):
        before = perf_counter()
        counts, attempts, changes = run_strength(
            *model.arguments, times, *rates.values(), seed + SEED_STRIDE * r, max_events
        )
        seconds.append(perf_counter() - before)
        totals.append(counts.sum(axis=1))
        events.append([attempts, changes])
        if observation is not None:
            densities.append((observation @ counts.T).T)
        if maps:
            delta = counts - mean
            mean += delta / (r + 1)
            m2 += delta * (counts - mean)
        if progress is not None:
            progress(r + 1, runs)
    result = dict(
        times=times,
        rates=np.array(rates.values()),
        total_samples=np.asarray(totals),
        seconds=np.asarray(seconds),
        events=np.asarray(events),
        seed_base=np.array(seed),
    )
    if maps:
        result.update(
            count_mean=mean,
            count_mc_se=np.sqrt(np.maximum(m2, 0) / (runs * (runs - 1))),
            population=model.population,
        )
    if observation is not None:
        result["density_samples"] = np.asarray(densities)
    return result
