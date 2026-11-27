"""Independent full-resolution CTMC check of a screened initial-state candidate."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pixel_sis import Rates, simulate_ensemble
from pixel_sis.cli import _hashes
from pixel_sis.model import load_reference
from pixel_sis.observations import RegionalObservations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--screen", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runs", type=int, default=64)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with np.load(a.screen / "surrogate.npz") as f:
        rates = f["rates"]
        initial_mean = f["initial_fine_mean"]
    N = np.load(a.data / "merfish_population.npy")
    probabilities = initial_mean / N
    seed = 212705101
    metadata = dict(
        rates=rates.tolist(),
        expected_initial_total=float(initial_mean.sum()),
        initial_law="Independent Binomial(N_i,p_i), p_i=screened mean / N_i",
        initial_rng="PCG64 with SeedSequence([seed_base,replicate_index,731901])",
        seed_base=seed,
        runs=a.runs,
        voxel_um=100,
        readout_um=6,
        max_events=2000000,
        source_screen=json.loads((a.screen / "summary.json").read_text()),
        initial_probabilities_sha256=hashlib.sha256(probabilities.tobytes()).hexdigest(),
        inputs=_hashes(a.data, True),
    )
    (a.out / "configuration.json").write_text(json.dumps(metadata, indent=2))
    model = load_reference(a.data)
    obs = RegionalObservations.load(a.data)

    def progress(i, total):
        if i % 16 == 0:
            print("Reference paths", i, "/", total, flush=True)

    e = simulate_ensemble(
        model,
        obs.times,
        Rates(*rates),
        runs=a.runs,
        seed=seed,
        observation=obs.operator,
        dynamics="sir",
        initial_probabilities=probabilities,
        maps=True,
        max_events=metadata["max_events"],
        progress=progress,
    )
    e["observation_scale_um"] = np.array(6.0)
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
        rates=rates.tolist(),
        runs=a.runs,
        initial_total_mean=float(e["initial_total_samples"].mean()),
        final_infected=float(e["total_samples"][:, -1].mean()),
        final_removed=float(e["removed_total_samples"][:, -1].mean()),
        median_seconds=float(np.median(e["seconds"])),
        maximum_attempts=int(e["events"][:, 0].max()),
        scope="Fresh CTMC paths at fixed screened parameters; no new biological validation cohort",
    )
    (a.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
