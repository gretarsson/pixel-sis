"""Joint whole-cohort fit with explicit bilateral BLA/SNc rate differences.

The regions are outcome-selected from the diagnostic. This is exploratory
model fitting; it must not be presented as independent biological validation.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from screen_sir_rates import integrate_sir

from pixel_sis.observations import RegionalObservations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--basis", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--diagnostic", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-evaluations", type=int, default=700)
    p.add_argument("--at-risk", action="store_true")
    p.add_argument("--continue-from", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    obs = RegionalObservations.load(a.data)
    original_weights = obs.weights.copy()
    w = np.zeros(len(obs.values))
    for t in range(len(obs.times)):
        mice = np.unique(obs.mouse_index[obs.time_index == t])
        for mouse in mice:
            k = (obs.time_index == t) & (obs.mouse_index == mouse)
            w[k] = 1 / (len(obs.times) * len(mice) * k.sum())
    O = sparse.load_npz(a.basis / "observation.npz")
    with np.load(a.basis / "basis.npz") as f:
        QT, LT, N, fields, initial, labels = [
            f[k] for k in ["tracer", "local", "population", "fields", "initial", "labels"]
        ]
    with np.load(a.diagnostic / "surrogate.npz") as f:
        fine_masks = f["fine_region_masks"]
        regional = f["regional_parameters"].ravel()
    masks = np.array(
        [np.bincount(labels, weights=mask, minlength=len(N)) > 0 for mask in fine_masks]
    )
    for fine, coarse in zip(fine_masks, masks):
        np.testing.assert_array_equal(coarse[labels], fine)
    baseline = json.loads((a.baseline / "summary.json").read_text())["parameters"]
    base = np.r_[baseline, regional]
    starts = [base.copy(), base.copy(), base.copy()]
    starts[1][0] *= 0.1
    starts[1][3] = 0.5
    starts[2][0] = 0.01
    starts[2][1] = 0.4
    starts[2][2] = 0.05
    starts[2][3] = 1.0
    starts[2][5] = 0.15
    if a.continue_from is not None:
        previous = json.loads((a.continue_from / "summary.json").read_text())["parameters"]
        starts = [np.array(previous, dtype=float)]
    if a.at_risk:
        base = np.r_[starts[0][:11], 1.0, 1.0]
        starts = [base.copy(), base.copy(), base.copy()]
        for start, fractions, removals in zip(
            starts[1:], [[0.15, 0.5], [0.3, 0.4]], [[0.1, 0.35], [0.3, 0.5]]
        ):
            for k in range(2):
                start[11 + k] = fractions[k]
                start[7 + 2 * k] /= fractions[k]
                start[8 + 2 * k] = removals[k]
    total_Q = QT.sum(0)
    total_L = LT.sum(0)
    norm = float(w @ (obs.values**2))
    old_norm = float(original_weights @ (obs.values**2))
    rows = []
    completed = []
    best = [np.inf, None]
    begun = time.perf_counter()

    def predict(parameters, rtol=2e-5):
        bt, bl, go, qg, qd, gg, gd = parameters[:7]
        q = np.array([1, qg, qd])
        K = bt * np.einsum("c,cij->ij", q, QT) + bl * np.einsum("c,cij->ij", q, LT)
        gamma = np.array([go, gg, gd]) @ fields
        R0 = np.zeros(len(N))
        for j, (mask, (chi, removal)) in enumerate(zip(masks, parameters[7:11].reshape(2, 2))):
            K[mask] = chi * (bt * total_Q[mask] + bl * total_L[mask])
            gamma[mask] = removal
            if a.at_risk:
                R0[mask] = (1 - parameters[11 + j]) * N[mask]
        if np.any(initial + R0 > N):
            return 100.0, 100.0, None, None, None
        I, R, _ = integrate_sir(
            lambda I: K @ I, N, initial, obs.times, gamma, rtol=rtol, initial_removed=R0
        )
        density = (O @ I.T).T
        prediction = 6 * obs.predictions(density)
        loss = float(w @ ((prediction - obs.values) ** 2) / norm)
        area = float(original_weights @ ((prediction - obs.values) ** 2) / old_norm)
        return loss, area, I, R, density

    def objective(x):
        parameters = np.exp(x)
        loss, area, *_ = predict(parameters)
        row = dict(
            parameters=parameters.tolist(),
            relative_rms=float(np.sqrt(loss)),
            original_area_rms=float(np.sqrt(area)),
            evaluation=len(rows) + 1,
            seconds=time.perf_counter() - begun,
        )
        rows.append(row)
        if loss < best[0]:
            best[:] = [loss, x.copy()]
            (a.out / "best.json").write_text(json.dumps(row, indent=2))
        if len(rows) % 50 == 0:
            print(json.dumps({**row, "best_rms": float(np.sqrt(best[0]))}), flush=True)
        return loss

    bounds = (
        [(np.log(1e-5), np.log(30))] * 3
        + [(np.log(0.01), np.log(200))] * 2
        + [(np.log(1e-5), np.log(30))] * 2
    )
    bounds += [(np.log(0.01), np.log(1000)), (np.log(0.001), np.log(10))] * 2
    if a.at_risk:
        bounds += [(np.log(0.01), 0.0)] * 2
    configuration = dict(
        starts=[s.tolist() for s in starts],
        bounds=bounds,
        parameter_order=[
            "beta_T",
            "beta_L",
            "gamma_O",
            "q_G",
            "q_D",
            "gamma_G",
            "gamma_D",
            "chi_BLA",
            "gamma_BLA",
            "chi_SNc",
            "gamma_SNc",
        ],
        objective="Equal time, mouse and regional record weights across the whole cohort",
        bilateral_overrides=True,
        region_selection="Outcome-selected diagnostic; exploratory",
        max_evaluations_per_start=a.max_evaluations,
        full_reference_validated=False,
        initial_state="Same infected marginal; additional initial resistance"
        if a.at_risk
        else "Same fixed binomial law from the previous initial-state fit",
        readout_um=6,
        initially_susceptible_fraction_fitted=a.at_risk,
        continue_from=None if a.continue_from is None else str(a.continue_from),
    )
    if a.at_risk:
        configuration["parameter_order"] += ["eligible_fraction_BLA", "eligible_fraction_SNc"]
    (a.out / "configuration.json").write_text(json.dumps(configuration, indent=2))
    for j, start in enumerate(starts):
        x = np.log(start)
        simplex = np.tile(x, (len(x) + 1, 1))
        for k in range(len(x)):
            simplex[k + 1, k] += 0.25 if x[k] + 0.25 <= bounds[k][1] else -0.25
        result = minimize(
            objective,
            x,
            method="Nelder-Mead",
            bounds=bounds,
            options=dict(
                maxfev=a.max_evaluations, initial_simplex=simplex, xatol=0.003, fatol=1e-7
            ),
        )
        row = dict(
            start=j,
            parameters=np.exp(result.x).tolist(),
            relative_rms=float(np.sqrt(result.fun)),
            optimizer_success=bool(result.success),
        )
        completed.append(row)
        (a.out / "starts.json").write_text(json.dumps(completed, indent=2))
        (a.out / "trace.json").write_text(json.dumps(rows, indent=2))
        parameters = np.exp(best[1])
        loss, area, I, R, density = predict(parameters, rtol=2e-8)
        np.savez_compressed(
            a.out / "surrogate.npz",
            parameters=parameters,
            density=density,
            counts=I,
            removed=R,
            times=obs.times,
            population=N,
            fine_region_masks=fine_masks,
        )
        (a.out / "summary.json").write_text(
            json.dumps(
                dict(
                    parameters=parameters.tolist(),
                    relative_rms=float(np.sqrt(loss)),
                    original_area_rms=float(np.sqrt(area)),
                    completed_starts=j + 1,
                    evaluations=len(rows),
                    elapsed_seconds=time.perf_counter() - begun,
                    full_reference_validated=False,
                ),
                indent=2,
            )
        )
        print("Completed start", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
