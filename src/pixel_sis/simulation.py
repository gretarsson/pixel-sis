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
    dynamics="sis",
    initial_probabilities=None,
    susceptibility=None,
    clearance_multiplier=None,
):
    """Simulate independent paths without retaining all voxel trajectories.

    A path reaching the attempt limit raises; incomplete paths are never scored.
    If observation is supplied, its rows map infected counts to the readouts.
    SIR starts with R=0 and keeps removed cells out of both infection and readout.
    Optional initial probabilities define independent Binomial(N_i,p_i) counts
    for each path. Omitting them preserves the fixed initial count vector.
    Optional nonnegative voxel fields multiply target infection and removal
    intensities. Event thinning samples these modified rates exactly.
    """
    times = np.asarray(times, dtype=np.float64)
    if dynamics not in ("sis", "sir"):
        raise ValueError("Dynamics must be 'sis' or 'sir'")
    sir = dynamics == "sir"
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
    if initial_probabilities is not None:
        initial_probabilities = np.asarray(initial_probabilities, dtype=float)
        if (
            initial_probabilities.shape != model.population.shape
            or not np.isfinite(initial_probabilities).all()
            or np.any(initial_probabilities < 0)
            or np.any(initial_probabilities > 1)
        ):
            raise ValueError("Initial probabilities must match population and lie in [0,1]")
    if observation is not None and observation.shape[1] != len(model.population):
        raise ValueError("Observation operator does not match the model")
    for name, field in (
        ("susceptibility", susceptibility),
        ("clearance_multiplier", clearance_multiplier),
    ):
        if field is not None:
            field = np.asarray(field, dtype=float)
            if (
                field.shape != model.population.shape
                or not np.isfinite(field).all()
                or np.any(field < 0)
            ):
                raise ValueError(name + " must be a finite nonnegative population-shaped field")
            if name == "susceptibility":
                susceptibility = field
            else:
                clearance_multiplier = field
    # Compile before recording simulation timings.
    removed_counts = np.zeros((len(times), len(model.population)), dtype=np.int32) if sir else None
    run_strength(
        *model.arguments,
        np.array([0.0]),
        *rates.values(),
        seed,
        max_events,
        sir,
        removed_counts,
        susceptibility,
        clearance_multiplier,
    )
    mean = np.zeros((len(times), len(model.population))) if maps else None
    m2 = np.zeros_like(mean) if maps else None
    removed_mean = np.zeros_like(mean) if maps and sir else None
    removed_m2 = np.zeros_like(mean) if maps and sir else None
    totals, densities, seconds, events = [], [], [], []
    removed_totals = []
    initial_totals = []
    for r in range(runs):
        before = perf_counter()
        arguments = model.arguments
        if initial_probabilities is not None:
            # Separate reproducible stream from the CTMC event generator.
            rng = np.random.default_rng(np.random.SeedSequence([seed, r, 731901]))
            initial = rng.binomial(model.population, initial_probabilities)
            arguments = (initial, *model.arguments[1:])
        initial_totals.append(int(arguments[0].sum()))
        counts, attempts, changes = run_strength(
            *arguments,
            times,
            *rates.values(),
            seed + SEED_STRIDE * r,
            max_events,
            sir,
            removed_counts,
            susceptibility,
            clearance_multiplier,
        )
        seconds.append(perf_counter() - before)
        totals.append(counts.sum(axis=1))
        if sir:
            removed_totals.append(removed_counts.sum(axis=1))
        events.append([attempts, changes])
        if observation is not None:
            densities.append((observation @ counts.T).T)
        if maps:
            delta = counts - mean
            mean += delta / (r + 1)
            m2 += delta * (counts - mean)
            if sir:
                delta_r = removed_counts - removed_mean
                removed_mean += delta_r / (r + 1)
                removed_m2 += delta_r * (removed_counts - removed_mean)
        if progress is not None:
            progress(r + 1, runs)
    result = dict(
        times=times,
        rates=np.array(rates.values()),
        total_samples=np.asarray(totals),
        seconds=np.asarray(seconds),
        events=np.asarray(events),
        seed_base=np.array(seed),
        dynamics=np.array(dynamics),
    )
    if maps:
        result.update(
            count_mean=mean,
            count_mc_se=np.sqrt(np.maximum(m2, 0) / (runs * (runs - 1))),
            population=model.population,
        )
    if initial_probabilities is not None:
        result["initial_total_samples"] = np.array(initial_totals)
        result["initial_probabilities"] = initial_probabilities.copy()
    if susceptibility is not None:
        result["susceptibility"] = susceptibility.copy()
    if clearance_multiplier is not None:
        result["clearance_multiplier"] = clearance_multiplier.copy()
    if observation is not None:
        result["density_samples"] = np.asarray(densities)
    if sir:
        result["removed_total_samples"] = np.asarray(removed_totals)
        result["susceptible_total_samples"] = (
            model.population.sum() - result["total_samples"] - result["removed_total_samples"]
        )
        if maps:
            result["removed_mean"] = removed_mean
            result["removed_mc_se"] = np.sqrt(np.maximum(removed_m2, 0) / (runs * (runs - 1)))
    return result
