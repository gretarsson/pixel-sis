"""Fit SIR to the same fixed-depth pathology records as the SIS comparison.

Keeps the 100-um reference, seed, rates' units and 6-um readout unchanged.
Each search is checkpointed separately. No experimental inputs are distributed.
"""

import argparse
import hashlib
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    code = sorted((root / "src/pixel_sis").glob("*.py")) + [Path(__file__).resolve()]
    config = dict(
        dynamics="sir",
        observation_scale_um=6.0,
        runs=32,
        selection_runs=256,
        seed=112700101,
        max_evaluations=80,
        max_events=2000000,
        upper_rate=1.0,
    )
    manifest = dict(
        fit_settings=config,
        final_runs=512,
        final_seed=142700101,
        pilot_seed=152700101,
        removed_initial=0,
        observation="Only I contributes; ideal cell-center counting in a 6-um section",
        inputs=_hashes(args.data, True),
        code_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code
        },
        cases={
            "mixed": [[0.25, 0.35, 0.05], [0.35, 0.15, 0.1], [0.5, 0.5, 0.3]],
            "tracer_only": [[0.55, 0.0, 0.1], [0.4, 0.0, 0.01], [0.9, 0.0, 0.3]],
        },
    )
    path = args.out / "configuration.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Configuration, code or inputs changed; use a new output folder")
    else:
        dump(path, manifest)
    model = load_reference(args.data)
    obs = RegionalObservations.load(args.data)
    dump(args.out / "model.json", model.metadata)
    print("Loaded", model.metadata, flush=True)
    begun = time.perf_counter()
    for name, starts in manifest["cases"].items():
        folder = args.out / name
        folder.mkdir(exist_ok=True)
        candidates = []
        for index, start in enumerate(starts):
            checkpoint = folder / f"start_{index}.json"
            if checkpoint.exists():
                fit = json.loads(checkpoint.read_text())
            else:
                e = simulate_ensemble(
                    model,
                    obs.times,
                    Rates(*start),
                    runs=4,
                    seed=manifest["pilot_seed"],
                    observation=obs.operator,
                    maps=False,
                    max_events=config["max_events"],
                    dynamics="sir",
                )
                loss, _ = obs.loss_and_scale(e["density_samples"].mean(0), scale=6.0)
                pilot = dict(
                    rates=start,
                    median_seconds=float(np.median(e["seconds"])),
                    relative_rms=float(np.sqrt(loss)),
                )
                dump(folder / f"pilot_{index}.json", pilot)
                print(name, "start", index, "pilot", pilot, flush=True)

                def progress(row):
                    row.update(case=name, start=index, elapsed_seconds=time.perf_counter() - begun)
                    with (folder / "search_trace.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    if row["evaluation"] % 10 == 0:
                        print(json.dumps(row), flush=True)

                fit = fit_rates(
                    model,
                    obs,
                    [Rates(*start)],
                    tracer_only=name == "tracer_only",
                    progress=progress,
                    **config,
                )
                fit["candidates"][0]["start"] = index
                fit["initial_rates"] = start
                dump(checkpoint, fit)
            candidates.extend(fit["candidates"])
            print(name, "completed start", index, fit["selected"], flush=True)
        combined = dict(fit)
        combined.pop("initial_rates", None)
        combined["candidates"] = candidates
        combined["selected"] = min(candidates, key=lambda c: c["relative_rms"])
        dump(folder / "fit.json", combined)
        if not (folder / "ensemble.npz").exists():

            def final_progress(done, total):
                if done % 128 == 0:
                    print(name, "final", done, "/", total, flush=True)

            e = simulate_ensemble(
                model,
                obs.times,
                Rates(**combined["selected"]["rates"]),
                runs=manifest["final_runs"],
                seed=manifest["final_seed"],
                max_events=config["max_events"],
                observation=obs.operator,
                maps=True,
                progress=final_progress,
                dynamics="sir",
            )
            e["observation_scale_um"] = 6.0
            np.savez_compressed(folder / "ensemble.npz", **e)
            loss, _ = obs.loss_and_scale(e["density_samples"].mean(0), scale=6.0)
            print(name, "final RMS", float(np.sqrt(loss)), flush=True)
    if _hashes(args.data, True) != manifest["inputs"]:
        raise ValueError("Scientific inputs changed during the experiment")
    if any(
        hashlib.sha256((root / p).read_bytes()).hexdigest() != h
        for p, h in manifest["code_sha256"].items()
    ):
        raise ValueError("Simulation source changed during the experiment")
    dump(
        args.out / "run_complete.json",
        dict(
            elapsed_seconds=time.perf_counter() - begun,
            input_hashes_unchanged=True,
            simulation_code_unchanged=True,
        ),
    )
    print("All SIR fits finished", flush=True)


if __name__ == "__main__":
    main()
