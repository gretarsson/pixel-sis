"""Test composition-dependent SIR rates with an anatomical ODE search aid.

Fine-voxel susceptibility is f_other+q_G*f_G+q_DS*f_DS; removal is
gamma_other*f_other+gamma_G*f_G+gamma_DS*f_DS. These are environment-dependent
voxel rates, not an assertion of individual cell-type-specific contacts.
The original tracer and local contact operators are not renormalized.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy import sparse
from scipy.optimize import minimize
from screen_sir_rates import integrate_sir

from pixel_sis.model import local_offsets
from pixel_sis.observations import RegionalObservations


@njit(cache=True)
def local_basis(N, population, labels, fields, coords, index, offsets, probability, normalizer):
    n = len(population)
    matrices = np.zeros((3, n, n))
    for j in range(len(N)):
        A = labels[j]
        source = N[j] / (population[A] * normalizer[j])
        for k in range(len(offsets)):
            x, y, z = coords[j] + offsets[k]
            if (
                x < 0
                or y < 0
                or z < 0
                or x >= index.shape[0]
                or y >= index.shape[1]
                or z >= index.shape[2]
            ):
                continue
            i = index[x, y, z]
            if i < 0:
                continue
            B = labels[i]
            weight = source * probability[k] * N[i]
            for c in range(3):
                matrices[c, B, A] += weight * fields[c, i]
    return matrices


def prepare(data, out, initial_source):
    N = np.load(data / "merfish_population.npy")
    with np.load(data / "geometry.npz") as f:
        coords, index, left = f["coords"], f["index"], f["left"]
        ids = f["annotation"][tuple(coords.T)]
    initial = np.load(data / "fitting_initial_fine.npy")
    keys = np.stack([ids, left.astype(int)], axis=1)
    _, labels = np.unique(keys, axis=0, return_inverse=True)
    # Retain each originally seeded voxel rather than homogenizing the local seed.
    seed_ids = np.flatnonzero(initial)
    labels[seed_ids] = labels.max() + 1 + np.arange(len(seed_ids))
    _, labels = np.unique(labels, return_inverse=True)
    n = labels.max() + 1
    G = sparse.csr_matrix((np.ones(len(N)), (np.arange(len(N)), labels)), shape=(len(N), n))
    population = np.asarray(G.T @ N).ravel()
    with np.load(data / "merfish_cell_composition.npz") as f:
        fractions = f["fractions"].astype(float)
        names = f["names"]
        class_ids = f["class_ids"]
        np.testing.assert_array_equal(f["reference_population"], N)
    fractions /= fractions.sum(axis=1, keepdims=True)
    glut = np.array([str(name).endswith(" Glut") for name in names])
    ds = np.isin(class_ids, [20, 21])
    fields = np.stack(
        [fractions[:, ~(glut | ds)].sum(1), fractions[:, glut].sum(1), fractions[:, ds].sum(1)]
    )
    fields /= fields.sum(axis=0)
    del fractions
    U = sparse.load_npz(data / "raw_source_factors.npz").astype(float)
    V = sparse.load_npz(data / "raw_target_factors.npz").astype(float)
    normalizer = float(N @ (U @ (V @ np.ones(len(N)))) / N.sum())
    H = (G.T @ V.T).multiply((1 / (population * normalizer))[:, None]).tocsr()
    tracer = []
    for c in range(3):
        target = U.T.multiply((N * fields[c])[None, :]) @ G
        tracer.append((target.T @ H.T).toarray())
    tracer = np.asarray(tracer)
    del U, V, H, target
    with np.load(data / "composition_local_basis.npz") as f:
        local_normalizer = f["normalizer"]
    offsets, cdf = local_offsets()
    local = local_basis(
        N,
        population,
        labels,
        fields,
        coords,
        index,
        offsets,
        np.diff(np.r_[0.0, cdf]),
        local_normalizer,
    )
    np.testing.assert_allclose(local.sum(axis=(0, 1)), 1, rtol=2e-12, atol=2e-12)
    coarse_fields = np.stack(
        [np.asarray(G.T @ (N * field)).ravel() / population for field in fields]
    )
    with np.load(initial_source) as f:
        initial_mean = f["initial_fine_mean"]
    obs = RegionalObservations.load(data)
    O = (obs.operator.multiply(N[None, :]) @ G).multiply((1 / population)[None, :]).tocsr()
    np.savez_compressed(
        out / "basis.npz",
        tracer=tracer,
        local=local,
        population=population,
        fields=coarse_fields,
        labels=labels,
        initial=np.asarray(G.T @ initial_mean).ravel(),
        initial_fine_mean=initial_mean,
    )
    sparse.save_npz(out / "observation.npz", O)
    print("Prepared anatomical surrogate", n, "parcels", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--initial-source", type=Path, required=True)
    p.add_argument("--max-evaluations", type=int, default=220)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--basis-source", type=Path)
    p.add_argument("--weighting", choices=["area", "regions"], default="area")
    p.add_argument("--starts-from", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    basis_folder = a.out if a.basis_source is None else a.basis_source
    if not (basis_folder / "basis.npz").exists():
        if a.basis_source is not None:
            raise FileNotFoundError("Specified basis has not been prepared")
        prepare(a.data, a.out, a.initial_source)
    if a.prepare_only:
        return
    with np.load(basis_folder / "basis.npz") as f:
        QT, LT, N, fields, initial = [
            f[k] for k in ["tracer", "local", "population", "fields", "initial"]
        ]
    O = sparse.load_npz(basis_folder / "observation.npz")
    obs = RegionalObservations.load(a.data)
    original_weights = obs.weights.copy()
    if a.weighting == "regions":
        weights = np.zeros(len(obs.values))
        for t in range(len(obs.times)):
            mice = np.unique(obs.mouse_index[obs.time_index == t])
            for mouse in mice:
                mask = (obs.time_index == t) & (obs.mouse_index == mouse)
                weights[mask] = 1 / (len(obs.times) * len(mice) * mask.sum())
        obs.weights = weights / weights.sum()
    starts = [
        [0.000413504, 0.295354318, 0.010235546, 1.0, 1.0, 0.010235546, 0.010235546],
        [0.0004, 0.295, 0.01, 8.0, 12.0, 1.6, 1.3],
        [0.03, 0.3, 0.05, 4.0, 20.0, 0.5, 2.0],
        [0.2, 0.1, 0.02, 4.0, 10.0, 0.5, 0.8],
        [0.2, 0.05, 0.1, 1.0, 30.0, 0.1, 1.0],
    ]
    if a.starts_from is not None:
        starts = [row["parameters"] for row in json.loads(a.starts_from.read_text())]
    settings = dict(
        parameter_names=[
            "beta_tracer",
            "beta_local",
            "gamma_other",
            "q_glut",
            "q_dopa_sero",
            "gamma_glut",
            "gamma_dopa_sero",
        ],
        initial_source=str(a.initial_source),
        starts=starts,
        readout_um=6,
        max_evaluations_per_start=a.max_evaluations,
        compartments=len(N),
        fine_rate_definition=__doc__,
        full_reference_validated=False,
        weighting=a.weighting,
        basis_source=str(basis_folder),
        starts_from=None if a.starts_from is None else str(a.starts_from),
    )
    (a.out / "configuration.json").write_text(json.dumps(settings, indent=2))
    started = time.perf_counter()
    best = [np.inf, None]
    rows = []
    completed = []

    def predict(parameters, rtol=2e-5):
        bt, bl, go, qg, qd, gg, gd = parameters
        q = np.array([1, qg, qd])
        K = bt * np.einsum("c,cij->ij", q, QT) + bl * np.einsum("c,cij->ij", q, LT)
        gamma = np.array([go, gg, gd]) @ fields
        I, R, nfev = integrate_sir(lambda I: K @ I, N, initial, obs.times, gamma, rtol=rtol)
        density = (O @ I.T).T
        loss, _ = obs.loss_and_scale(np.maximum(density, 0), scale=6)
        return loss, I, R, density, nfev

    def objective(x):
        parameters = np.exp(x)
        loss, *_ = predict(parameters)
        row = dict(
            parameters=parameters.tolist(),
            rms=float(np.sqrt(loss)),
            evaluation=len(rows) + 1,
            seconds=time.perf_counter() - started,
        )
        rows.append(row)
        if loss < best[0]:
            best[:] = [loss, x.copy()]
            (a.out / "best.json").write_text(json.dumps(row, indent=2))
        if len(rows) % 30 == 0:
            print(json.dumps({**row, "best_rms": float(np.sqrt(best[0]))}), flush=True)
        return loss

    bounds = (
        [(np.log(1e-5), np.log(30))] * 3
        + [(np.log(0.01), np.log(200))] * 2
        + [(np.log(1e-5), np.log(30))] * 2
    )
    for j, start in enumerate(starts):
        x = np.log(start)
        simplex = np.tile(x, (8, 1))
        for k in range(7):
            simplex[k + 1, k] += 0.3
        result = minimize(
            objective,
            x,
            method="Nelder-Mead",
            bounds=bounds,
            options=dict(
                maxfev=a.max_evaluations, initial_simplex=simplex, xatol=0.003, fatol=1e-6
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
        loss, I, R, density, _ = predict(parameters, rtol=2e-8)
        prediction = 6 * obs.predictions(density)
        area_rms = float(
            np.sqrt(
                original_weights
                @ ((prediction - obs.values) ** 2)
                / (original_weights @ (obs.values**2))
            )
        )
        np.savez_compressed(
            a.out / "surrogate.npz",
            parameters=parameters,
            counts=I,
            removed=R,
            density=density,
            times=obs.times,
            population=N,
        )
        (a.out / "summary.json").write_text(
            json.dumps(
                dict(
                    **settings,
                    parameters=parameters.tolist(),
                    relative_rms=float(np.sqrt(loss)),
                    completed_starts=j + 1,
                    evaluations=len(rows),
                    seconds=time.perf_counter() - started,
                    original_area_weighted_rms=area_rms,
                ),
                indent=2,
            )
        )
        print("Completed start", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
