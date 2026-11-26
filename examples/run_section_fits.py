"""Reproduce the fixed-depth CTMC fits in the November 2026 report.

Requires the prepared mouse bundle; raw experimental data are not distributed.
Run on one numerical CPU thread. Outputs resume only with matching configuration.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pixel_sis import Rates, simulate_ensemble
from pixel_sis.cli import _hashes
from pixel_sis.fitting import fit_rates
from pixel_sis.model import load_reference
from pixel_sis.observations import RegionalObservations


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(
        observation_scale_um=6.0,
        runs=32,
        selection_runs=256,
        seed=112600101,
        max_evaluations=80,
        max_events=2000000,
        upper_rate=1.0,
    )
    manifest = dict(
        fit_settings=config,
        final_runs=512,
        final_seed=142600101,
        physical_assumption="Ideal cell-center counting in a 6-um section; not calibrated profile stereology",
        inputs=_hashes(args.data, True),
        cases={
            "mixed": [[0.25, 0.35, 0.05], [0.35, 0.15, 0.1]],
            "tracer_only": [[0.55, 0.0, 0.1], [0.4, 0.0, 0.01]],
        },
    )
    manifest_file = args.out / "configuration.json"
    if manifest_file.exists():
        if json.loads(manifest_file.read_text()) != manifest:
            raise ValueError("Configuration or inputs changed; use a new output folder")
    else:
        dump(manifest_file, manifest)
    model = load_reference(args.data)
    obs = RegionalObservations.load(args.data)
    dump(args.out / "model.json", model.metadata)
    print("Loaded", model.metadata, flush=True)
    begun = time.perf_counter()
    for name, starts in manifest["cases"].items():
        folder = args.out / name
        folder.mkdir(exist_ok=True)

        def progress(row):
            if row["evaluation"] % 10 == 0:
                print(name, json.dumps(row), flush=True)

        if (folder / "fit.json").exists():
            fit = json.loads((folder / "fit.json").read_text())
        else:
            pilots = []
            for start in starts:
                e = simulate_ensemble(
                    model,
                    obs.times,
                    Rates(*start),
                    runs=4,
                    seed=152600101,
                    observation=obs.operator,
                    maps=False,
                    max_events=config["max_events"],
                )
                loss, _ = obs.loss_and_scale(e["density_samples"].mean(0), scale=6.0)
                pilots.append(
                    dict(
                        rates=start,
                        median_seconds=float(np.median(e["seconds"])),
                        relative_rms=float(np.sqrt(loss)),
                    )
                )
            dump(folder / "pilots.json", pilots)
            print(name, "pilots", pilots, flush=True)
            fit = fit_rates(
                model,
                obs,
                [Rates(*s) for s in starts],
                tracer_only=name == "tracer_only",
                progress=progress,
                **config,
            )
            dump(folder / "fit.json", fit)
            print(name, "selected", fit["selected"], flush=True)
        if not (folder / "ensemble.npz").exists():

            def final_progress(done, total):
                if done % 128 == 0:
                    print(name, "final", done, "/", total, flush=True)

            e = simulate_ensemble(
                model,
                obs.times,
                Rates(**fit["selected"]["rates"]),
                runs=manifest["final_runs"],
                seed=manifest["final_seed"],
                max_events=config["max_events"],
                observation=obs.operator,
                maps=True,
                progress=final_progress,
            )
            e["observation_scale_um"] = 6.0
            np.savez_compressed(folder / "ensemble.npz", **e)
    if _hashes(args.data, True) != manifest["inputs"]:
        raise ValueError("Scientific input changed during the experiment")
    dump(
        args.out / "run_complete.json",
        dict(elapsed_seconds=time.perf_counter() - begun, input_hashes_unchanged=True),
    )
    print("All fixed-depth fits finished", flush=True)


if __name__ == "__main__":
    main()
