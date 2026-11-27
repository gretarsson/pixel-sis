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
    p.add_argument(
        "--fit-initial",
        type=Path,
        help="Initial local/projected basis and summary to refit jointly",
    )
    a = p.parse_args()
    if a.fit_initial is not None and a.at_risk:
        p.error(
            "Joint initial-burden fitting and initial-resistance fitting are separate experiments"
        )
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
    fine_initial_basis = None
    if a.fit_initial is not None:
        if any(len(s) != 11 for s in starts):
            raise ValueError(
                "Joint initial fitting starts from the eleven-parameter regional model"
            )
        with np.load(a.fit_initial / "initial_basis.npz") as f:
            fine_initial_basis = np.stack([f["local"], f["projected"]])
            fine_population = f["population"]
        np.testing.assert_array_equal(fine_population, np.load(a.data / "merfish_population.npy"))
        fine_probability_basis = fine_initial_basis / fine_population[None, :]
        coarse_initial_basis = np.stack(
            [np.bincount(labels, weights=v, minlength=len(N)) for v in fine_initial_basis]
        )
        state = json.loads((a.fit_initial / "summary.json").read_text())
        burden, fraction = state["initial_total"], state["projection_fraction"]
        initial_counts = [burden * (1 - fraction), burden * fraction]
        if a.continue_from is not None:
            previous_state = json.loads((a.continue_from / "summary.json").read_text())
            initial_counts = previous_state.get("initial_component_counts", initial_counts)
        starts = [np.r_[s[:11], initial_counts] for s in starts]
    total_Q = QT.sum(0)
    total_L = LT.sum(0)
    norm = float(w @ (obs.values**2))
    old_norm = float(original_weights @ (obs.values**2))
    rows = []
    completed = []
    best = [np.inf, None]
    begun = time.perf_counter()

    def predict(parameters, rtol=2e-5):
        state_initial = initial
        if a.fit_initial is not None:
            component_counts = parameters[11:]
            if (
                component_counts.sum() > 100000
                or np.max(component_counts @ fine_probability_basis) > 1
            ):
                return 100.0, 100.0, None, None, None
            state_initial = component_counts @ coarse_initial_basis
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
        if np.any(state_initial + R0 > N):
            return 100.0, 100.0, None, None, None
        I, R, _ = integrate_sir(
            lambda I: K @ I, N, state_initial, obs.times, gamma, rtol=rtol, initial_removed=R0
        )
        density = (O @ I.T).T
        prediction = 6 * obs.predictions(density)
        loss = float(w @ ((prediction - obs.values) ** 2) / norm)
        area = float(original_weights @ ((prediction - obs.values) ** 2) / old_norm)
        return loss, area, I, R, density

    def record_parameters(parameters):
        if a.fit_initial is None:
            return dict(parameters=parameters.tolist())
        counts = parameters[11:]
        return dict(
            parameters=parameters[:11].tolist(),
            initial_component_counts=counts.tolist(),
            initial_total=float(counts.sum()),
            projection_fraction=float(counts[1] / counts.sum()),
        )

    def objective(x):
        parameters = np.exp(x)
        loss, area, predicted_counts, *_ = predict(parameters)
        row = dict(
            **record_parameters(parameters),
            relative_rms=float(np.sqrt(loss)),
            original_area_rms=float(np.sqrt(area)),
            evaluation=len(rows) + 1,
            seconds=time.perf_counter() - begun,
            feasible_initial_state=predicted_counts is not None,
        )
        rows.append(row)
        if predicted_counts is not None and loss < best[0]:
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
    if a.fit_initial is not None:
        bounds += [(np.log(0.001), np.log(100000))] * 2
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
    if a.fit_initial is not None:
        configuration["parameter_order"] += ["initial_local_count", "initial_projected_count"]
        configuration["initial_state"] = (
            "Jointly fitted local/projected expected counts; independent binomial initial law"
        )
        configuration["initial_basis_source"] = str(a.fit_initial)
        configuration["initial_total_upper_bound"] = 100000
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
            **record_parameters(np.exp(result.x)),
            relative_rms=float(np.sqrt(result.fun)),
            optimizer_success=bool(result.success),
        )
        completed.append(row)
        (a.out / "starts.json").write_text(json.dumps(completed, indent=2))
        (a.out / "trace.json").write_text(json.dumps(rows, indent=2))
        if best[1] is None:
            raise ValueError("No feasible initial state evaluated")
        parameters = np.exp(best[1])
        loss, area, I, R, density = predict(parameters, rtol=2e-8)
        extra_arrays = {}
        if a.fit_initial is not None:
            extra_arrays["initial_fine_mean"] = parameters[11:] @ fine_initial_basis
        np.savez_compressed(
            a.out / "surrogate.npz",
            parameters=parameters if a.fit_initial is None else parameters[:11],
            density=density,
            counts=I,
            removed=R,
            times=obs.times,
            population=N,
            fine_region_masks=fine_masks,
            **extra_arrays,
        )
        (a.out / "summary.json").write_text(
            json.dumps(
                dict(
                    **record_parameters(parameters),
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
