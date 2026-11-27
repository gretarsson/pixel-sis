"""Assemble a concise report from completed ensembles; never score partial runs."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pixel_sis.observations import RegionalObservations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    a.report.mkdir(parents=True, exist_ok=True)
    figures = a.report / "figures"
    figures.mkdir(exist_ok=True)
    obs = RegionalObservations.load(a.data)
    region_weights = np.zeros(len(obs.values))
    for t in range(len(obs.times)):
        mice = np.unique(obs.mouse_index[obs.time_index == t])
        for mouse in mice:
            mask = (obs.time_index == t) & (obs.mouse_index == mouse)
            region_weights[mask] = 1 / (len(obs.times) * len(mice) * mask.sum())
    weights = {"area": obs.weights, "regions": region_weights}
    cases = [
        ("Uniform SIR", "sir_fit_20261127/mixed/ensemble.npz"),
        ("Revised initial state", "sir_initial_reference_20261127/ensemble.npz"),
        ("Composition rates, area fit", "sir_composition_reference_start2_20261127/ensemble.npz"),
        (
            "Composition rates, regional fit",
            "sir_composition_regions_reference_20261127/ensemble.npz",
        ),
    ]
    metrics = []
    for name, relative in cases:
        with np.load(a.outputs / relative) as f:
            samples = 6e6 * f["density_samples"][:, obs.time_index, obs.group]
            N = f["population"]
            counts = f["count_mean"]
            removed = f["removed_mean"]
            if np.min(N - counts - removed) < -1e-7 or np.diff(removed, axis=0).min() < -1e-7:
                raise ValueError("Invalid SIR ensemble conservation")
            row = dict(
                name=name,
                source=relative,
                runs=len(samples),
                rates=f["rates"].tolist(),
                total_infected=f["total_samples"].mean(0).tolist(),
                total_removed=f["removed_total_samples"].mean(0).tolist(),
                parameters_represent="rates are beta_T, beta_L, gamma_other; see corresponding configuration for rate fields",
            )
        prediction = samples.mean(0)
        config_path = (a.outputs / relative).parent / "configuration.json"
        if config_path.exists() and "parameters" in json.loads(config_path.read_text()):
            row["parameters"] = json.loads(config_path.read_text())["parameters"]
        else:
            bt, bl, gamma = row["rates"]
            row["parameters"] = [bt, bl, gamma, 1.0, 1.0, gamma, gamma]
        for kind, w in weights.items():
            norm = float(w @ (obs.values**2))
            rms = float(np.sqrt(w @ ((prediction - obs.values) ** 2) / norm))
            gradient = w * (prediction - obs.values) / (norm * rms)
            se = float(np.std(samples @ gradient, ddof=1) / np.sqrt(len(samples)))
            row[kind + "_rms"] = rms
            row[kind + "_mc_se"] = se
        metrics.append(row)
    selected = int(np.argmin([r["area_rms"] for r in metrics]))
    report = dict(
        cases=metrics,
        selected_for_figure=selected,
        figure_selection="Lowest original area-weighted error among completed reference checks; not a global optimum",
        interpretation="Exploratory in-sample improvement. Important regional timing failures remain.",
    )
    (a.report / "metrics.json").write_text(json.dumps(report, indent=2))
    table = [
        r"\begin{tabular}{lrrr}\toprule Model & Area-weighted RMS & Equal-region RMS & Paths\\\midrule"
    ]
    for row in metrics:
        table.append(
            f"{row['name']} & {row['area_rms']:.4f} & {row['regions_rms']:.4f} & {row['runs']}\\\\"
        )
    table.append(r"\bottomrule\end{tabular}")
    (figures / "results_table.tex").write_text("\n".join(table))
    parameter_table = [
        r"\begin{tabular}{lrrrrrrr}\toprule Model & $\beta_T$ & $\beta_L$ & $\gamma_O$ & $q_G$ & $q_D$ & $\gamma_G$ & $\gamma_D$\\\midrule"
    ]
    for name, row in zip(
        ["Uniform", "New start", "Composition (area)", "Composition (regions)"], metrics
    ):
        parameter_table.append(
            name + " & " + " & ".join(f"{v:.3g}" for v in row["parameters"]) + r"\\"
        )
    parameter_table.append(r"\bottomrule\end{tabular}")
    (figures / "parameters_table.tex").write_text("\n".join(parameter_table))
    initial = json.loads((a.outputs / "sir_initial_screen_20261127/summary.json").read_text())
    chosen = metrics[selected]
    difference = 1 - (chosen["area_rms"] / metrics[0]["area_rms"]) ** 2
    (a.report / "results.tex").write_text(
        f"The best completed reference check reduces the original relative RMS error from {metrics[0]['area_rms']:.4f} to {chosen['area_rms']:.4f}, a {100 * difference:.1f}\\% reduction in weighted squared error. "
        f"Its Monte Carlo standard error is {chosen['area_mc_se']:.5f}; this quantifies simulation noise only. "
        "The trajectories still miss important regional peaks. This is an improved fit, not a successful explanation of the whole dataset.\n\n"
        f"The revised initial state has expected burden $B={initial['initial_total']:.1f}$ and tracer-footprint fraction $f={initial['projection_fraction']:.4f}$. "
        "These are fitted assumptions about an unobserved starting state, not measured counts of initially affected cells.\n"
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("plot_sir_diagnostics.py")),
            "--data",
            str(a.data),
            "--baseline",
            str(a.outputs / cases[0][1]),
            "--revised",
            str(a.outputs / cases[selected][1]),
            "--out",
            str(figures),
            "--baseline-label",
            "Uniform SIR",
            "--revised-label",
            "Best checked extension",
        ],
        check=True,
    )
    for ext in ["pdf", "png"]:
        shutil.copy2(a.outputs / "sir_diagnostics_20261127" / ("error_diagnosis." + ext), figures)
    fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
    names = ["Uniform", "New start", "Composition\narea fit", "Composition\nregional fit"]
    x = np.arange(4)
    ax.bar(
        x - 0.17,
        [r["area_rms"] for r in metrics],
        width=0.34,
        label="Area weights",
        color="#16877e",
    )
    ax.bar(
        x + 0.17,
        [r["regions_rms"] for r in metrics],
        width=0.34,
        label="Equal regions",
        color="#88a8c2",
    )
    ax.set(xticks=x, xticklabels=names, ylabel="Relative RMS error", ylim=(0, 1.06))
    ax.axhline(1, ls=":", color=".5", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=2)
    for ext in ["pdf", "png"]:
        fig.savefig(figures / ("fit_errors." + ext), dpi=200)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
