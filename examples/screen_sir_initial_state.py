"""Test a tracer-defined initial exposure pattern without using outcome weights.

This changes the initial-state hypothesis explicitly: some initial infected cells
may lie in projecting somata, representing unresolved early uptake/transport.
It is not a measured initial condition. Rates and seed burden are fitted jointly.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import differential_evolution, minimize
from screen_sir_rates import CoarseSIR


def initial_basis(data, output):
    N = np.load(data / "merfish_population.npy")
    seed = np.load(data / "fitting_initial_fine.npy")
    U = sparse.load_npz(data / "raw_source_factors.npz")
    V = sparse.load_npz(data / "raw_target_factors.npz")
    projected = N * (U @ (V @ (seed / N)))
    projected /= projected.sum()
    local = seed / seed.sum()
    np.savez_compressed(output, local=local, projected=projected, population=N)
    return local, projected, N


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=40)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model = CoarseSIR(a.data)
    local, projected, N = initial_basis(a.data, a.out / "initial_basis.npz")
    coarse_local = np.bincount(model.labels, weights=local, minlength=len(model.N))
    coarse_projected = np.bincount(model.labels, weights=projected, minlength=len(model.N))
    local_support = local > 0
    outside_max = np.max(projected[~local_support] / N[~local_support])
    started = time.perf_counter()
    rows = []
    best = [np.inf, None]

    def objective(x):
        rates = np.exp(x[:3])
        B = float(np.exp(x[3]))
        f = float(x[4])
        max_fraction = B * max(
            np.max(
                ((1 - f) * local[local_support] + f * projected[local_support]) / N[local_support]
            ),
            f * outside_max,
        )
        if max_fraction > 1:
            return 10 + max_fraction
        initial = B * ((1 - f) * coarse_local + f * coarse_projected)
        loss, *_ = model.predict(rates, initial=initial)
        row = dict(
            rates=rates.tolist(),
            initial_total=B,
            projection_fraction=f,
            rms=float(np.sqrt(loss)),
            evaluation=len(rows) + 1,
            seconds=time.perf_counter() - started,
        )
        rows.append(row)
        if loss < best[0]:
            best[:] = [loss, x.copy()]
            (a.out / "best.json").write_text(json.dumps(row, indent=2))
        if len(rows) % 40 == 0:
            print(json.dumps({**row, "best_rms": float(np.sqrt(best[0]))}), flush=True)
        return loss

    bounds = [(np.log(1e-4), np.log(30))] * 3 + [(np.log(1), np.log(100000)), (0, 1)]
    baseline = np.r_[
        np.log([0.13384374064254917, 0.4652763918378955, 0.0037238148823040313, 1000]), 0.0
    ]
    objective(baseline)
    result = differential_evolution(
        objective,
        bounds,
        seed=112704101,
        popsize=6,
        maxiter=a.iterations,
        polish=False,
        tol=5e-4,
        x0=baseline,
        workers=1,
    )
    refined = minimize(
        objective,
        best[1],
        method="Nelder-Mead",
        bounds=bounds,
        options=dict(maxfev=300, xatol=0.002, fatol=1e-7),
    )
    x = best[1]
    rates = np.exp(x[:3])
    B = float(np.exp(x[3]))
    f = float(x[4])
    initial_fine = B * ((1 - f) * local + f * projected)
    initial = np.bincount(model.labels, weights=initial_fine, minlength=len(model.N))
    loss, I, R, density, _ = model.predict(rates, rtol=2e-8, initial=initial)
    np.savez_compressed(
        a.out / "surrogate.npz",
        rates=rates,
        initial_fine_mean=initial_fine,
        counts=I,
        removed=R,
        density=density,
        times=model.obs.times,
        population=model.N,
    )
    summary = dict(
        rates=rates.tolist(),
        initial_total=B,
        projection_fraction=f,
        relative_rms=float(np.sqrt(loss)),
        width_um=800,
        readout_um=6,
        initial_hypothesis="B*((1-f)*local+f*normalized(Q.T@local)); fixed tracer and population",
        initial_fraction_max=float(np.max(initial_fine / N)),
        seed=112704101,
        search="Differential evolution followed by Nelder-Mead; not a global guarantee",
        iterations=a.iterations,
        evaluations=len(rows),
        seconds=time.perf_counter() - started,
        bounds=bounds,
        full_reference_validated=False,
        differential_evolution_success=bool(result.success),
        refinement_success=bool(refined.success),
    )
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (a.out / "trace.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
