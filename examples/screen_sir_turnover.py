"""Check high-turnover SIR branches missed by independent log-rate searches."""

import argparse
import json
import time

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, eigs
from screen_sir_rates import CoarseSIR


def main():
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model = CoarseSIR(a.data)
    cache = {}
    rows = []
    start = time.perf_counter()

    def root(f):
        if f not in cache:

            def matvec(x):
                return f * (model.QT @ x) + (1 - f) * (model.LT @ x)

            op = LinearOperator((len(model.N),) * 2, matvec=matvec)
            vals, vectors = eigs(op, k=1, which="LR", v0=np.ones(len(model.N)), tol=1e-9)
            rho = float(vals[0].real)
            residual = (
                np.linalg.norm(matvec(vectors[:, 0]) - rho * vectors[:, 0])
                / np.linalg.norm(vectors[:, 0])
                / rho
            )
            if residual > 1e-7 or abs(vals[0].imag) > 1e-9 or rho <= 0:
                raise RuntimeError("Unverified dominant eigenvalue")
            cache[f] = rho
        return cache[f]

    def evaluate(g, f, gamma):
        rate = (gamma + g) / root(float(f))
        rates = [f * rate, (1 - f) * rate, gamma]
        loss, *rest = model.predict(rates)
        row = dict(
            growth=float(g), fraction=float(f), gamma=gamma, rates=rates, rms=float(np.sqrt(loss))
        )
        rows.append(row)
        return row

    choices = []
    for gamma in [0.1, 1.0, 5.0, 20.0]:
        grid = []
        for f in [0.0, 0.03, 0.1, 0.3, 0.6, 1.0]:
            for g in [0.2, 0.7, 1.2, 2.0, 4.0]:
                grid.append(evaluate(g, f, gamma))
        for kind, available in [
            ("best", grid),
            ("tracer_heavy", [r for r in grid if r["fraction"] >= 0.6]),
        ]:
            initial = min(available, key=lambda x: x["rms"])

            def objective(x):
                return evaluate(x[0], x[1], gamma)["rms"] ** 2

            result = minimize(
                objective,
                [initial["growth"], initial["fraction"]],
                method="Nelder-Mead",
                bounds=[(0.0, 6.0), (0.0, 1.0)],
                options=dict(maxfev=100, xatol=0.002, fatol=1e-6),
            )
            row = evaluate(*result.x, gamma)
            row["start_kind"] = kind
            choices.append(row)
            print(json.dumps({**row, "elapsed": time.perf_counter() - start}), flush=True)
            (a.out / "candidates.json").write_text(json.dumps(choices, indent=2))
        (a.out / "trace.json").write_text(json.dumps(rows, indent=2))
    (a.out / "settings.json").write_text(
        json.dumps(
            dict(
                width_um=800,
                readout_um=6,
                seed_cells=1000,
                clearances=[0.1, 1, 5, 20],
                fractions=[0, 0.03, 0.1, 0.3, 0.6, 1],
                growth_grid=[0.2, 0.7, 1.2, 2, 4],
                scope="Coarse deterministic search only; growth is its dilute dominant eigenvalue.",
                elapsed_seconds=time.perf_counter() - start,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
