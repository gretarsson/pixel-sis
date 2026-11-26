"""Command-line interface for portable, explicitly configured experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import __version__
from .fitting import fit_rates
from .model import from_contacts, load_reference
from .observations import RegionalObservations
from .plotting import plot_slices, plot_totals
from .simulation import Rates, simulate_ensemble


def _dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def _folder(path):
    out = Path(path)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}. Choose a new directory.")
    out.mkdir(parents=True)
    return out


def _hashes(folder, observations=False):
    files = [
        "merfish_population.npy",
        "fitting_initial_fine.npy",
        "geometry.npz",
        "raw_source_factors.npz",
        "raw_target_factors.npz",
    ]
    if observations:
        files += ["observations.npz", "cell_body_observations.npz", "observation_density_fine.npz"]
    result = {}
    for filename in files:
        digest = hashlib.sha256()
        with (Path(folder) / filename).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        result[filename] = digest.hexdigest()
    return result


def _progress(done, total):
    if done % 64 == 0 or done == total:
        print(f"Completed {done}/{total} paths", flush=True)


def _demo(args):
    out = _folder(args.out)
    # Synthetic three-community example; no experimental data are required.
    groups = np.repeat(np.arange(3), 10)
    Q = np.where(groups[:, None] == groups[None, :], 1 / 9, 0.1 / 20)
    np.fill_diagonal(Q, 0)
    N = np.full(30, 100)
    initial = np.zeros(30, dtype=int)
    initial[:10] = 5
    model = from_contacts(N, initial, Q)
    rates = Rates(0.45, 0.1, 0.2)
    result = simulate_ensemble(
        model, np.linspace(0, 9, 19), rates, runs=args.runs, seed=20261126, progress=_progress
    )
    np.savez_compressed(out / "ensemble.npz", **result)
    _dump(
        out / "run.json",
        {
            "version": __version__,
            "model": model.metadata,
            "rates": rates.to_dict(),
            "runs": args.runs,
            "seed": 20261126,
            "description": "Synthetic example, not mouse data",
        },
    )
    plot_totals(result, out / "trajectory.png")
    print(f"Demo saved to {out}")


def _simulate(args):
    cfg = json.loads(Path(args.config).read_text())
    allowed = {"rates", "times", "runs", "seed", "max_events", "observation_scale_um"}
    if set(cfg) - allowed:
        raise ValueError("Unknown simulation settings: " + ", ".join(sorted(set(cfg) - allowed)))
    rates = Rates(**cfg["rates"])
    out = _folder(args.out)
    _dump(out / "configuration.json", cfg)
    model = load_reference(args.data)
    obs = RegionalObservations.load(args.data) if args.observations else None
    inputs = _hashes(args.data, args.observations)
    result = simulate_ensemble(
        model,
        cfg["times"],
        rates,
        runs=cfg.get("runs", 256),
        seed=cfg.get("seed", 112500100),
        max_events=cfg.get("max_events", 2000000),
        observation=obs.operator if obs is not None else None,
        progress=_progress,
    )
    if "observation_scale_um" in cfg:
        a = cfg["observation_scale_um"]
        if not np.isfinite(a) or a < 0:
            raise ValueError("Invalid observation scale")
        result["observation_scale_um"] = a
    np.savez_compressed(out / "ensemble.npz", **result)
    _dump(
        out / "run.json",
        {
            "version": __version__,
            "model": model.metadata,
            "settings": cfg,
            "input_sha256": inputs,
            "final_mean_affected": float(result["total_samples"][:, -1].mean()),
            "maximum_attempts": int(result["events"][:, 0].max()),
        },
    )
    plot_totals(result, out / "trajectory.png")
    print(f"Simulation saved to {out}")


def _plot(args):
    with np.load(args.simulation, allow_pickle=False) as f:
        result = {k: f[k] for k in f.files}
    scale = args.scale
    if scale is None:
        if "observation_scale_um" not in result:
            raise ValueError("Specify --scale, or use a simulation containing observation_scale_um")
        scale = float(result["observation_scale_um"])
    plot_slices(result, args.data, args.out, scale_um=scale, vmax=args.vmax, regional=args.regional)
    print(f"Saved {args.out}.png and .pdf")


def _fit(args):
    cfg = json.loads(Path(args.config).read_text())
    allowed = {
        "starts",
        "tracer_only",
        "runs",
        "selection_runs",
        "seed",
        "max_evaluations",
        "max_events",
        "upper_rate",
    }
    if set(cfg) - allowed:
        raise ValueError("Unknown fitting settings: " + ", ".join(sorted(set(cfg) - allowed)))
    out = _folder(args.out)
    _dump(out / "configuration.json", cfg)
    model = load_reference(args.data)
    obs = RegionalObservations.load(args.data)
    inputs = _hashes(args.data, True)

    def report(row):
        if row["evaluation"] % 10 == 0:
            print(json.dumps(row), flush=True)

    result = fit_rates(model, obs, progress=report, **cfg)
    result.update(version=__version__, input_sha256=inputs)
    _dump(out / "fit.json", result)
    print(json.dumps(result["selected"], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Finite-population voxel SIS")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a synthetic example without experimental data")
    demo.add_argument("--out", required=True)
    demo.add_argument("--runs", type=int, default=64)
    demo.set_defaults(func=_demo)
    simulate = commands.add_parser("simulate", help="Run the prepared 100-um reference model")
    simulate.add_argument("--data", required=True)
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--out", required=True)
    simulate.add_argument("--observations", action="store_true")
    simulate.set_defaults(func=_simulate)
    plot = commands.add_parser("plot", help="Plot simulated slices directly above regional data")
    plot.add_argument("--simulation", required=True)
    plot.add_argument("--data", required=True)
    plot.add_argument("--out", required=True)
    plot.add_argument("--scale", type=float)
    plot.add_argument("--vmax", type=float, default=1300.0)
    plot.add_argument("--regional", action="store_true")
    plot.set_defaults(func=_plot)
    fit = commands.add_parser("fit", help="Fit the immediate regional observation model")
    fit.add_argument("--data", required=True)
    fit.add_argument("--config", required=True)
    fit.add_argument("--out", required=True)
    fit.set_defaults(func=_fit)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
