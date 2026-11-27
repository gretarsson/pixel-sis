"""Confirm a composition-rate candidate using exact 100-um SIR event thinning."""

import argparse
import json
from pathlib import Path

import numpy as np

from pixel_sis import Rates, simulate_ensemble
from pixel_sis.model import load_reference, with_susceptibility
from pixel_sis.observations import RegionalObservations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--screen", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--initial-source", type=Path, required=True)
    p.add_argument("--start", type=int)
    p.add_argument("--runs", type=int, default=64)
    p.add_argument("--regional-source", type=Path)
    p.add_argument("--prepare-susceptibility", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    source = json.loads((a.screen / "summary.json").read_text())
    parameters = source["parameters"]
    if a.start is not None:
        parameters = json.loads((a.screen / "starts.json").read_text())[a.start]["parameters"]
    if len(parameters) not in (7, 11, 13):
        raise ValueError("Expected seven composition or eleven/thirteen regional parameters")
    if len(parameters) > 7 and a.regional_source is None:
        raise ValueError("Regional parameters require --regional-source")
    bt, bl, go, qg, qd, gg, gd = parameters[:7]
    with np.load(a.initial_source) as f:
        initial_mean = f["initial_fine_mean"]
    with np.load(a.data / "merfish_cell_composition.npz") as f:
        fractions = f["fractions"].astype(float)
        names = f["names"]
        ids = f["class_ids"]
    fractions /= fractions.sum(axis=1, keepdims=True)
    glut = np.array([str(name).endswith(" Glut") for name in names])
    ds = np.isin(ids, [20, 21])
    fields = np.stack(
        [fractions[:, ~(glut | ds)].sum(1), fractions[:, glut].sum(1), fractions[:, ds].sum(1)]
    )
    fields /= fields.sum(axis=0)
    susceptibility = np.array([1, qg, qd]) @ fields
    clearance = np.array([1, gg / go, gd / go]) @ fields
    initial_removed_probability = None
    if a.regional_source is not None:
        if len(parameters) not in (11, 13):
            raise ValueError("Regional overrides require eleven or thirteen parameters")
        with np.load(a.regional_source / "surrogate.npz") as f:
            masks = f["fine_region_masks"]
        if masks.shape != (2, len(susceptibility)) or np.any(masks.sum(axis=0) > 1):
            raise ValueError("Invalid regional override masks")
        for mask, (chi, removal) in zip(masks, np.array(parameters[7:11]).reshape(2, 2)):
            susceptibility[mask] = chi
            clearance[mask] = removal / go
        if len(parameters) == 13:
            initial_removed_probability = np.zeros(len(susceptibility))
            for mask, eligible in zip(masks, parameters[11:]):
                initial_removed_probability[mask] = 1 - eligible
    del fractions, fields
    model = load_reference(a.data)
    if a.prepare_susceptibility:
        model = with_susceptibility(model, susceptibility)
    obs = RegionalObservations.load(a.data)
    probabilities = initial_mean / model.population
    metadata = dict(
        parameters=parameters,
        source_screen=str(a.screen),
        selected_start=a.start,
        runs=a.runs,
        seed_base=212706101,
        max_events=20000000,
        voxel_um=100,
        readout_um=6,
        initial_law=(
            "Joint multinomial S/I/R; fixed infected marginal and fitted resistant fraction"
            if initial_removed_probability is not None
            else "Independent binomial with screened mean"
        ),
        generator="Target susceptibility multiplies infection hazards; clearance multiplies gamma_other. Exact count CTMC.",
        susceptibility_sampler="prepared weighted contacts"
        if a.prepare_susceptibility
        else "global envelope",
        clearance_sampler="weighted event tree",
        regional_source=None if a.regional_source is None else str(a.regional_source),
        initial_resistance=len(parameters) == 13,
    )
    (a.out / "configuration.json").write_text(json.dumps(metadata, indent=2))
    e = simulate_ensemble(
        model,
        obs.times,
        Rates(bt, bl, go),
        runs=a.runs,
        seed=metadata["seed_base"],
        max_events=metadata["max_events"],
        observation=obs.operator,
        dynamics="sir",
        maps=True,
        initial_probabilities=probabilities,
        susceptibility=None if a.prepare_susceptibility else susceptibility,
        clearance_multiplier=clearance,
        initial_removed_probabilities=initial_removed_probability,
        progress=lambda i, n: (
            print("Reference paths", i, "/", n, flush=True) if i % 8 == 0 else None
        ),
    )
    e["observation_scale_um"] = np.array(6.0)
    e["susceptibility"] = susceptibility
    np.savez_compressed(a.out / "ensemble.npz", **e)
    samples = 6e6 * e["density_samples"][:, obs.time_index, obs.group]
    prediction = samples.mean(axis=0)
    norm = float(obs.weights @ (obs.values**2))
    rms = float(np.sqrt(obs.weights @ ((prediction - obs.values) ** 2) / norm))
    gradient = obs.weights * (prediction - obs.values) / (norm * rms)
    se = float(np.std(samples @ gradient, ddof=1) / np.sqrt(a.runs))
    result = dict(
        relative_rms=rms,
        mc_se_rms=se,
        parameters=parameters,
        runs=a.runs,
        final_infected=float(e["total_samples"][:, -1].mean()),
        final_removed=float(e["removed_total_samples"][:, -1].mean()),
        median_seconds=float(np.median(e["seconds"])),
        maximum_attempts=int(e["events"][:, 0].max()),
        scope="Fresh full-reference CTMC at fixed candidate; the measurement approximation and composition-rate law remain assumptions.",
    )
    if initial_removed_probability is not None:
        result["initial_resistant"] = float(e["initial_removed_total_samples"].mean())
        result["final_newly_removed"] = float(e["newly_removed_total_samples"][:, -1].mean())
    (a.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
