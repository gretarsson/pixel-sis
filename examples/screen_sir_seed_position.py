"""Screen uncertain right-CP injection positions at fixed fitted SIR rates.

Candidate locations cover the anatomical interior and are chosen without reading
pathology outcomes. The initial burden, footprint mixture and seed shape stay
fixed. Search uses an anatomical ODE with a 200-um CP grid; final checking still
requires the original 100-um stochastic process.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse
from screen_sir_rates import integrate_sir

from pixel_sis.observations import RegionalObservations


def focal_seed(N, xyz, center):
    distance2 = np.sum((xyz - center) ** 2, axis=1)
    weight = N * np.exp(-distance2 / (2 * 0.15**2)) * (distance2 <= 0.3**2)
    expected = 1000 * weight / weight.sum()
    seed = np.floor(expected).astype(np.int64)
    residual = 1000 - int(seed.sum())
    if residual:
        seed[np.argsort(expected - seed)[-residual:]] += 1
    if np.any(seed > N):
        raise ValueError("Provisional seed exceeds local population")
    return seed


def covering_centers(xyz, count):
    """Deterministic farthest-point coverage, independent of the observations."""
    if not len(xyz) or count < 1:
        raise ValueError("At least one interior voxel and one candidate are required")
    chosen = [int(np.argmin(np.sum((xyz - xyz.mean(0)) ** 2, axis=1)))]
    distance2 = np.sum((xyz - xyz[chosen[0]]) ** 2, axis=1)
    for _ in range(1, min(count, len(xyz))):
        j = int(np.argmax(distance2))
        if distance2[j] == 0:
            break
        chosen.append(j)
        distance2 = np.minimum(distance2, np.sum((xyz - xyz[j]) ** 2, axis=1))
    return xyz[chosen]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--basis", type=Path, required=True)
    p.add_argument("--rates-source", type=Path, required=True)
    p.add_argument("--initial-source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--candidates", type=int, default=24)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    obs = RegionalObservations.load(a.data)
    with np.load(a.basis / "basis.npz") as f:
        QT, LT, N, fields, labels = [
            f[k] for k in ["tracer", "local", "population", "fields", "labels"]
        ]
    with np.load(a.rates_source / "surrogate.npz") as f:
        fine_masks = f["fine_region_masks"]
    parameters = np.array(json.loads((a.rates_source / "summary.json").read_text())["parameters"])
    if len(parameters) != 11:
        raise ValueError("This experiment fixes the eleven-parameter regional-rate model")
    bt, bl, go, qg, qd, gg, gd = parameters[:7]
    K = bt * np.einsum("c,cij->ij", [1, qg, qd], QT) + bl * np.einsum("c,cij->ij", [1, qg, qd], LT)
    gamma = np.array([go, gg, gd]) @ fields
    for mask, (chi, removal) in zip(fine_masks, parameters[7:].reshape(2, 2)):
        coarse = np.bincount(labels, weights=mask, minlength=len(N)) > 0
        np.testing.assert_array_equal(coarse[labels], mask)
        K[coarse] = chi * (bt * QT[:, coarse].sum(0) + bl * LT[:, coarse].sum(0))
        gamma[coarse] = removal
    del QT, LT
    O = sparse.load_npz(a.basis / "observation.npz")
    with np.load(a.data / "geometry.npz") as f:
        coords, index = f["coords"], f["index"]
    xyz = (coords + 0.5) * 0.1
    fine_N = np.load(a.data / "merfish_population.npy")
    cp_groups = np.unique(obs.group[obs.labels == "iCP"])
    cp = np.asarray(obs.operator[cp_groups].sum(axis=0)).ravel() > 0
    cp_grid = np.zeros(index.shape, dtype=bool)
    cp_grid[tuple(coords[cp].T)] = True
    distance = ndimage.distance_transform_edt(cp_grid, sampling=0.1)[tuple(coords.T)]
    centers = covering_centers(xyz[cp & (distance >= 0.3)], a.candidates)
    original_center = json.loads((a.data / "observation_audit.json").read_text())[
        "seed_center_ccf_mm"
    ]
    centers = np.vstack([original_center, centers])
    initial_info = json.loads((a.initial_source / "summary.json").read_text())
    burden, fraction = initial_info["initial_total"], initial_info["projection_fraction"]
    U = sparse.load_npz(a.data / "raw_source_factors.npz")
    V = sparse.load_npz(a.data / "raw_target_factors.npz")
    weights = np.zeros(len(obs.values))
    for t in range(len(obs.times)):
        mice = np.unique(obs.mouse_index[obs.time_index == t])
        for mouse in mice:
            mask = (obs.time_index == t) & (obs.mouse_index == mouse)
            weights[mask] = 1 / (len(obs.times) * len(mice) * mask.sum())
    norm = weights @ obs.values**2
    original_norm = obs.weights @ obs.values**2
    rows, best = [], [np.inf, None, None, None, None]
    before = time.perf_counter()

    def predict(initial_mean, tolerance=2e-5):
        initial = np.bincount(labels, weights=initial_mean, minlength=len(N))
        I, R, _ = integrate_sir(lambda I: K @ I, N, initial, obs.times, gamma, rtol=tolerance)
        density = (O @ I.T).T
        residual = 6 * obs.predictions(density) - obs.values
        return (
            float(weights @ residual**2 / norm),
            float(obs.weights @ residual**2 / original_norm),
            I,
            R,
            density,
        )

    configuration = dict(
        parameters=parameters.tolist(),
        rates_source=str(a.rates_source),
        basis=str(a.basis),
        initial_source=str(a.initial_source),
        initial_total=burden,
        projection_fraction=fraction,
        centers_ccf_mm=centers.tolist(),
        seed_sigma_mm=0.15,
        seed_cutoff_mm=0.3,
        coverage=f"{a.candidates} farthest-point positions in right CP at least 0.3 mm from its boundary, plus the original provisional center",
        selection="Equal-region whole-cohort error; original area score also reported",
        changed="Initial position and its tracer-defined projected footprint only",
        unchanged="Rates, total expected initial burden, footprint mixture, population, connectivity and 6-um immediate observation rule",
        full_reference_validated=False,
    )
    (a.out / "configuration.json").write_text(json.dumps(configuration, indent=2))
    for j, center in enumerate(centers):
        seed = (
            np.load(a.data / "fitting_initial_fine.npy")
            if j == 0
            else focal_seed(fine_N, xyz, center)
        )
        local = seed / seed.sum()
        projected = fine_N * (U @ (V @ (local / fine_N)))
        projected /= projected.sum()
        initial_mean = burden * ((1 - fraction) * local + fraction * projected)
        maximum = float(np.max(initial_mean / fine_N))
        row = dict(candidate=j, center_ccf_mm=center.tolist(), initial_fraction_max=maximum)
        if maximum > 1:
            row["status"] = "Rejected: initial probability exceeds one; burden unchanged"
        else:
            loss, area, I, R, density = predict(initial_mean)
            row.update(
                status="screened",
                relative_rms=float(np.sqrt(loss)),
                original_area_rms=float(np.sqrt(area)),
            )
            if loss < best[0]:
                best[:] = [loss, initial_mean.copy(), row.copy(), local.copy(), projected.copy()]
        rows.append(row)
        (a.out / "candidates.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
    loss, area, I, R, density = predict(best[1], tolerance=2e-8)
    np.savez_compressed(
        a.out / "surrogate.npz",
        parameters=parameters,
        initial_fine_mean=best[1],
        fine_region_masks=fine_masks,
        density=density,
        counts=I,
        removed=R,
        times=obs.times,
        population=N,
    )
    summary = dict(
        parameters=parameters.tolist(),
        initial_total=burden,
        projection_fraction=fraction,
        selected_candidate=best[2],
        relative_rms=float(np.sqrt(loss)),
        original_area_rms=float(np.sqrt(area)),
        candidates=len(rows),
        elapsed_seconds=time.perf_counter() - before,
        full_reference_validated=False,
    )
    np.savez_compressed(
        a.out / "initial_basis.npz", local=best[3], projected=best[4], population=fine_N
    )
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
