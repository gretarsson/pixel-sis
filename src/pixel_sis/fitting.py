"""A small multistart fitter for the immediate regional readout."""

from dataclasses import asdict

import numpy as np
from scipy.optimize import minimize

from .simulation import Rates, simulate_ensemble


def fit_rates(
    model,
    observations,
    starts,
    *,
    tracer_only=False,
    runs=64,
    selection_runs=256,
    seed=112600001,
    max_evaluations=100,
    max_events=500000,
    upper_rate=2.0,
    observation_scale_um=None,
    progress=None,
    dynamics="sis",
):
    """Bounded Nelder-Mead with fixed search seeds and fresh finalist ensembles.

    Profiles the global amplitude unless observation_scale_um fixes it. The
    returned searches include convergence flags; this is not a global guarantee.
    The historical fitted parameters remain separate from this portable fitter.
    """
    if not starts or max_evaluations < 1 or not np.isfinite(upper_rate) or upper_rate <= 0:
        raise ValueError("Provide starting rates, a positive budget and positive rate bound")
    if dynamics not in ("sis", "sir"):
        raise ValueError("Dynamics must be 'sis' or 'sir'")
    active = [0, 2] if tracer_only else [0, 1, 2]
    starts = [s if isinstance(s, Rates) else Rates(**s) for s in starts]
    for s in starts:
        if max(s.values()) > upper_rate:
            raise ValueError("A starting rate exceeds upper_rate")
        if tracer_only and s.beta_local != 0:
            raise ValueError("Tracer-only starts must have beta_local=0")
    # Validate the observation normalization before starting an expensive search.
    observations.loss_and_scale(
        np.zeros((len(observations.times), observations.operator.shape[0])),
        scale=observation_scale_um,
    )
    finalists = []
    for start_index, start in enumerate(starts):
        best = [np.inf, None, None]
        capped = [0]
        calls = [0]

        def unpack(x):
            values = np.zeros(3)
            values[active] = x
            return Rates(*values)

        def objective(x):
            calls[0] += 1
            rates = unpack(x)
            try:
                e = simulate_ensemble(
                    model,
                    observations.times,
                    rates,
                    runs=runs,
                    seed=seed,
                    max_events=max_events,
                    observation=observations.operator,
                    maps=False,
                    dynamics=dynamics,
                )
            except RuntimeError as exc:
                if "event limit" not in str(exc):
                    raise
                capped[0] += 1
                return 1e12
            loss, scale = observations.loss_and_scale(
                e["density_samples"].mean(axis=0), scale=observation_scale_um
            )
            if loss < best[0]:
                best[:] = [loss, rates, scale]
            if progress is not None:
                progress(
                    {
                        "start": start_index,
                        "evaluation": calls[0],
                        "best_relative_rms": float(np.sqrt(best[0])),
                    }
                )
            return loss

        x0 = np.array(start.values())[active]
        simplex = np.tile(x0, (len(active) + 1, 1))
        step = min(0.05, upper_rate / 2)
        for j in range(len(active)):
            simplex[j + 1, j] = x0[j] + (step if x0[j] + step <= upper_rate else -step)
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            bounds=[(0.0, upper_rate)] * len(active),
            options={
                "maxfev": max_evaluations,
                "initial_simplex": simplex,
                "xatol": 0.002,
                "fatol": 1e-5,
            },
        )
        if best[1] is None:
            raise RuntimeError("Every evaluation reached the event limit; no valid candidate")
        e = simulate_ensemble(
            model,
            observations.times,
            best[1],
            runs=selection_runs,
            seed=seed + 100000000,
            max_events=max_events,
            observation=observations.operator,
            maps=False,
            dynamics=dynamics,
        )
        loss, scale = observations.loss_and_scale(
            e["density_samples"].mean(axis=0), scale=observation_scale_um
        )
        finalists.append(
            {
                "start": start_index,
                "rates": asdict(best[1]),
                "scale_um": scale,
                "relative_rms": float(np.sqrt(loss)),
                "search_relative_rms": float(np.sqrt(best[0])),
                "converged": bool(result.success),
                "message": str(result.message),
                "evaluations": calls[0],
                "capped_evaluations": capped[0],
            }
        )
    return {
        "candidates": finalists,
        "selected": min(finalists, key=lambda x: x["relative_rms"]),
        "tracer_only": tracer_only,
        "search_runs": runs,
        "selection_runs": selection_runs,
        "seed": seed,
        "max_evaluations": max_evaluations,
        "upper_rate": upper_rate,
        "max_events": max_events,
        "observation_scale_um": observation_scale_um,
        "dynamics": dynamics,
        "scope": "In-sample fit; fresh finalist paths check simulation noise",
    }
