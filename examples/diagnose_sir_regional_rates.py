"""Ask whether SIR can reproduce the two prominent early-peaking regions.

This is an explicitly outcome-selected diagnostic, not independent validation.
The two fitted susceptibility/removal pairs apply bilaterally. Other voxels
retain the previously fitted composition field; all network feedback remains.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import differential_evolution, minimize
from screen_sir_rates import integrate_sir

from pixel_sis.observations import RegionalObservations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--basis", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=35)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    obs = RegionalObservations.load(a.data)
    O = sparse.load_npz(a.basis / "observation.npz")
    with np.load(a.basis / "basis.npz") as f:
        QT, LT, N, fields, initial, labels = [
            f[k] for k in ["tracer", "local", "population", "fields", "initial", "labels"]
        ]
    parameters = json.loads((a.baseline / "summary.json").read_text())["parameters"]
    bt, bl, go, qg, qd, gg, gd = parameters
    q = np.array([1, qg, qd])
    base_K = bt * np.einsum("c,cij->ij", q, QT) + bl * np.einsum("c,cij->ij", q, LT)
    unweighted_K = bt * QT.sum(0) + bl * LT.sum(0)
    base_gamma = np.array([go, gg, gd]) @ fields
    masks = []
    fine_masks = []
    for region in ["BLA", "SNc"]:
        groups = np.unique(obs.group[np.isin(obs.labels, ["i" + region, "c" + region])])
        fine = np.asarray(obs.operator[groups].sum(axis=0)).ravel() > 0
        mask = np.asarray(O[groups].sum(axis=0)).ravel() > 0
        # Every target compartment must be wholly inside/outside this region.
        np.testing.assert_array_equal(mask[labels], fine)
        masks.append(mask)
        fine_masks.append(fine)
    # Each target region contributes equally, then each time and mouse equally.
    w = np.zeros(len(obs.values))
    for label in ["iBLA", "iSNc"]:
        for t in range(len(obs.times)):
            k = (obs.labels == label) & (obs.time_index == t)
            if k.any():
                w[k] = 1 / (2 * len(obs.times) * k.sum())
    norm = float(w @ (obs.values**2))
    original_norm = float(obs.weights @ (obs.values**2))
    rows = []
    started = time.perf_counter()
    best = [np.inf, None]

    def predict(x, rtol=2e-5):
        regional = np.exp(x).reshape(2, 2)
        K = base_K.copy()
        gamma = base_gamma.copy()
        for mask, (chi, removal) in zip(masks, regional):
            K[mask] = chi * unweighted_K[mask]
            gamma[mask] = removal
        I, R, _ = integrate_sir(lambda I: K @ I, N, initial, obs.times, gamma, rtol=rtol)
        density = (O @ I.T).T
        yhat = 6 * obs.predictions(density)
        loss = float(w @ ((yhat - obs.values) ** 2) / norm)
        total = float(obs.weights @ ((yhat - obs.values) ** 2) / original_norm)
        return loss, total, I, R, density

    def objective(x):
        loss, total, *_ = predict(x)
        row = dict(
            evaluation=len(rows) + 1,
            regional_parameters=np.exp(x).reshape(2, 2).tolist(),
            target_rms=float(np.sqrt(loss)),
            original_area_rms=float(np.sqrt(total)),
            seconds=time.perf_counter() - started,
        )
        rows.append(row)
        if loss < best[0]:
            best[:] = [loss, x.copy()]
            (a.out / "best.json").write_text(json.dumps(row, indent=2))
        if len(rows) % 40 == 0:
            print(json.dumps({**row, "best_target_rms": float(np.sqrt(best[0]))}), flush=True)
        return loss

    bounds = [(np.log(0.01), np.log(1000)), (np.log(0.001), np.log(10))] * 2
    configuration = dict(
        regions=["BLA", "SNc"],
        parameter_order=["susceptibility", "removal_per_month"],
        bilateral=True,
        readout_um=6,
        base_parameters=parameters,
        base_model=str(a.baseline),
        basis=str(a.basis),
        objective="Equal target-region/time/mouse squared error; right BLA and SNc only",
        bounds=bounds,
        seed=112708101,
        iterations=a.iterations,
        interpretation="Outcome-selected diagnostic; other-region error is reported separately",
    )
    (a.out / "configuration.json").write_text(json.dumps(configuration, indent=2))
    differential_evolution(
        objective,
        bounds,
        seed=112708101,
        popsize=6,
        maxiter=a.iterations,
        tol=5e-4,
        polish=False,
        workers=1,
    )
    result = minimize(
        objective,
        best[1],
        method="Nelder-Mead",
        bounds=bounds,
        options=dict(maxfev=300, xatol=0.002, fatol=1e-7),
    )
    x = best[1]
    loss, total, I, R, density = predict(x, rtol=2e-8)
    np.savez_compressed(
        a.out / "surrogate.npz",
        regional_parameters=np.exp(x).reshape(2, 2),
        density=density,
        counts=I,
        removed=R,
        times=obs.times,
        population=N,
        fine_region_masks=np.array(fine_masks),
    )
    result = dict(
        target_rms=float(np.sqrt(loss)),
        original_area_rms=float(np.sqrt(total)),
        regional_parameters=np.exp(x).reshape(2, 2).tolist(),
        evaluations=len(rows),
        elapsed_seconds=time.perf_counter() - started,
        full_reference_validated=False,
        scope=configuration["interpretation"],
    )
    (a.out / "summary.json").write_text(json.dumps(result, indent=2))
    (a.out / "trace.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
