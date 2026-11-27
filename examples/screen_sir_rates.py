"""Broad SIR rate search using a clearly labelled coarse ODE surrogate.

The frozen cell-reversed contact graph, original initial count and 6-um readout
are retained. This is a search aid, not a replacement for the 100-um CTMC.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import differential_evolution, minimize

from pixel_sis.observations import RegionalObservations


def integrate_sir(
    force, population, initial, times, gamma, rtol=2e-5, atol=1e-7, initial_removed=None
):
    """Positive exponential midpoint with step doubling and frozen hazard.

    Each step solves S->I->R exactly for a frozen per-susceptible hazard.
    The hazard is evaluated at a midpoint predictor. R=N-S-I throughout.
    """
    N = np.asarray(population, dtype=float)
    R0 = np.zeros_like(N) if initial_removed is None else np.asarray(initial_removed, dtype=float)
    S, I = N - initial - R0, np.array(initial, dtype=float)
    if R0.shape != N.shape or np.any(R0 < 0) or np.any(S < 0):
        raise ValueError("Invalid initially removed population")

    def frozen(S, I, hazard, dt):
        source = np.maximum(hazard, 0)
        difference = abs(gamma - source)
        factor = np.divide(
            -np.expm1(-difference * dt),
            difference,
            out=np.full_like(source, dt),
            where=difference > 1e-12,
        )
        survival_integral = np.exp(-np.minimum(source, gamma) * dt) * factor
        return S * np.exp(-source * dt), I * np.exp(-gamma * dt) + S * source * survival_integral

    def step(S, I, dt):
        _, half_I = frozen(S, I, force(I) / N, dt / 2)
        return frozen(S, I, force(half_I) / N, dt)

    outputs, removed = [], []
    t, dt, attempts = 0.0, 0.01, 0
    for target in times:
        while t < target:
            dt = min(dt, target - t, 0.25)
            attempts += 1
            if attempts > 100000 or dt < 1e-12:
                raise RuntimeError("SIR mean integration did not converge")
            s1, i1 = step(S, I, dt)
            sh, ih = step(S, I, dt / 2)
            s2, i2 = step(sh, ih, dt / 2)
            error_i = (i2 - i1) / (3 * (atol + rtol * np.maximum(I, i2)))
            error_r = (s1 + i1 - s2 - i2) / (
                3 * (atol + rtol * np.maximum(N - S - I - R0, N - s2 - i2 - R0))
            )
            error = np.sqrt(np.mean(error_i**2 + error_r**2) / 2)
            if not np.isfinite(error):
                raise RuntimeError("Nonfinite SIR integration error")
            if error <= 1:
                S, I = s2, i2
                t += dt
            dt *= 5 if error == 0 else np.clip(0.9 * error ** (-1 / 3), 0.2, 5)
        outputs.append(I.copy())
        removed.append(N - S - I)
    return np.array(outputs), np.array(removed), attempts * 6


class CoarseSIR:
    def __init__(self, data, width=800):
        self.obs = RegionalObservations.load(data)
        folder = data / f"merfish_cell_reverse_operator_{width}um"
        meta = json.loads((folder / "metadata.json").read_text())
        fine_population = np.load(data / "merfish_population.npy")
        if meta["population_sha256"] != hashlib.sha256(fine_population.tobytes()).hexdigest():
            raise ValueError("Population changed since operator construction")
        if meta["definition"] != "D^-1 W^T D / c_forward":
            raise ValueError("Wrong tracer operator")
        with np.load(folder / "geometry.npz") as f:
            self.N, self.labels = f["population"], f["labels"]
        np.testing.assert_allclose(
            self.N, np.bincount(self.labels, weights=fine_population), rtol=0, atol=0
        )
        H = sparse.load_npz(folder / "H.npz")
        T = sparse.load_npz(folder / "T.npz")
        self.HT, self.TT = H.T.tocsr(), T.T.tocsr()
        if len(self.N) <= 2000:
            self.QT = np.asarray((H @ T).T.toarray(), order="C")
        else:
            self.QT = None
        self.LT = sparse.load_npz(data / f"merfish_operator_reverse_{width}um" / "L.npz").T.tocsr()
        self.initial = np.bincount(
            self.labels, weights=np.load(data / "fitting_initial_fine.npy"), minlength=len(self.N)
        )
        G = sparse.csr_matrix(
            (np.ones(len(self.labels)), (np.arange(len(self.labels)), self.labels)),
            shape=(len(self.labels), len(self.N)),
        )
        self.O = (
            (self.obs.operator.multiply(fine_population[None, :]) @ G)
            .multiply((1 / self.N)[None, :])
            .tocsr()
        )
        self.width = width

    def predict(self, rates, rtol=2e-5, initial=None):
        bT, bL, gamma = rates

        def force(infected):
            tracer = self.QT @ infected if self.QT is not None else self.TT @ (self.HT @ infected)
            return bT * tracer + bL * (self.LT @ infected)

        infected, removed, nfev = integrate_sir(
            force,
            self.N,
            self.initial if initial is None else initial,
            self.obs.times,
            gamma,
            rtol=rtol,
        )
        tolerance = 1e-3
        if min(infected.min(), removed.min(), (self.N - infected - removed).min()) < -tolerance:
            raise RuntimeError("SIR mean integration violated count bounds")
        density = (self.O @ infected.T).T
        # Numerical roundoff at zero is not a biological negative density.
        loss, _ = self.obs.loss_and_scale(np.maximum(density, 0), scale=6.0)
        return loss, infected, removed, density, nfev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model = CoarseSIR(a.data, a.width)
    started = time.perf_counter()
    evaluations = 0
    best = np.inf

    def objective(logrates):
        nonlocal evaluations, best
        evaluations += 1
        rates = np.exp(logrates)
        loss, _, _, _, nfev = model.predict(rates)
        row = dict(
            evaluation=evaluations,
            rates=rates.tolist(),
            rms=float(np.sqrt(loss)),
            ode_evaluations=nfev,
            seconds=time.perf_counter() - started,
        )
        with (a.out / "trace.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if loss < best:
            best = loss
            (a.out / "best.json").write_text(json.dumps(row, indent=2))
        if evaluations % 20 == 0 or a.smoke:
            print(json.dumps({**row, "best_rms": float(np.sqrt(best))}), flush=True)
        return loss

    baseline = [0.13384374064254917, 0.4652763918378955, 0.0037238148823040313]
    if a.smoke:
        for rates in [baseline, [1, 1, 1], [3, 5, 5], [10, 10, 10]]:
            objective(np.log(rates))
        return
    settings = dict(
        width_um=a.width,
        rate_bounds_per_month=[1e-4, 30],
        observation_depth_um=6,
        seed_cells=int(model.initial.sum()),
        objective="same weighted normalized SSE on all 18292 records",
        search="differential evolution + Nelder-Mead, not a global guarantee",
        seed=112703101,
        iterations=a.iterations,
        surrogate="deterministic SIR on population-weighted coarse contacts",
        full_reference_validated=False,
    )
    (a.out / "configuration.json").write_text(json.dumps(settings, indent=2))
    objective(np.log(baseline))
    bounds = [(np.log(1e-4), np.log(30))] * 3
    result = differential_evolution(
        objective,
        bounds,
        seed=settings["seed"],
        popsize=8,
        maxiter=a.iterations,
        polish=False,
        tol=1e-4,
        x0=np.log(baseline),
        workers=1,
    )
    refined = minimize(
        objective,
        result.x,
        method="Nelder-Mead",
        bounds=bounds,
        options=dict(maxfev=180, xatol=0.002, fatol=1e-7),
    )
    optimum = refined if refined.fun < result.fun else result
    rates = np.exp(optimum.x)
    loss, I, R, density, nfev = model.predict(rates, rtol=2e-8)
    np.savez_compressed(
        a.out / "surrogate.npz",
        times=model.obs.times,
        counts=I,
        removed=R,
        density=density,
        rates=rates,
        population=model.N,
    )
    (a.out / "summary.json").write_text(
        json.dumps(
            dict(
                **settings,
                rates=rates.tolist(),
                relative_rms=float(np.sqrt(loss)),
                evaluations=evaluations,
                seconds=time.perf_counter() - started,
                solver="positive exponential midpoint with step doubling",
                final_rtol=2e-8,
                final_rhs_evaluations=nfev,
                optimization_success=bool(optimum.success),
            ),
            indent=2,
        )
    )
    print((a.out / "summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
