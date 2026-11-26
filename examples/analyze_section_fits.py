"""Make report metrics and scientific figures from completed CTMC ensembles."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from pixel_sis.observations import RegionalObservations
from pixel_sis.plotting import plot_slices


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--fits", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    obs = RegionalObservations.load(a.data)
    population = np.load(a.data / "merfish_population.npy", allow_pickle=False)
    old = json.loads((a.previous / "summary.json").read_text())
    cases = {
        "mixed": (load(a.fits / "mixed/ensemble.npz"), 6.0, "Mixed, fixed depth"),
        "tracer_only": (load(a.fits / "tracer_only/ensemble.npz"), 6.0, "Tracer, fixed depth"),
        "previous": (
            load(a.previous / "reference_ensemble.npz"),
            old["observation_scale_um"],
            "Mixed, fitted scale",
        ),
    }
    plt.rcParams.update(
        {"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = {"mixed": "#24799a", "tracer_only": "#d88135", "previous": "#787878"}
    w, y = obs.weights, obs.values
    norm = w @ (y * y)
    denom = w @ ((y - w @ y) ** 2)
    metrics, predictions, aligned = {}, {}, {}
    checks = {}
    for name, (e, h, label) in cases.items():
        ids = np.searchsorted(e["times"], obs.times)
        np.testing.assert_array_equal(e["times"][ids], obs.times)
        d = e["density_samples"][:, ids]
        np.testing.assert_array_equal(e["population"], population)
        if not np.isfinite(e["count_mean"]).all() or np.any(e["count_mean"] < 0):
            raise ValueError("Invalid ensemble mean")
        if np.any(e["count_mean"] > population[None, :] + 1e-8):
            raise ValueError("Mean exceeds cell capacity")
        np.testing.assert_allclose(
            e["count_mean"].sum(1), e["total_samples"].mean(0), rtol=1e-11, atol=1e-8
        )
        projected = (obs.operator @ e["count_mean"][ids].T).T
        np.testing.assert_allclose(projected, d.mean(0), rtol=1e-11, atol=1e-14)
        if name != "previous" and np.max(e["events"][:, 0]) >= 2000000:
            raise ValueError("A final trajectory reached the attempt cap")
        checks[name] = dict(
            population_unchanged=True,
            means_within_capacity=True,
            voxel_totals_match_sample_totals=True,
            independently_accumulated_regional_means_match=True,
            max_projection_difference=float(np.abs(projected - d.mean(0)).max()),
        )
        aligned[name] = d
        pred = h * obs.predictions(d.mean(0))
        predictions[name] = pred
        sse = float(w @ ((pred - y) ** 2))
        # Delta-method MC SE of the reported RMS, holding rates and scale fixed.
        sample_predictions = h * obs.factor * d[:, obs.time_index, obs.group]
        influence = (sample_predictions - pred) @ (w * (pred - y)) / (np.sqrt(sse / norm) * norm)
        row = dict(
            rates=e["rates"].tolist(),
            scale_um=h,
            runs=len(d),
            relative_rms=float(np.sqrt(sse / norm)),
            relative_rms_mc_se=float(influence.std(ddof=1) / np.sqrt(len(d))),
            weighted_rmse=float(np.sqrt(sse)),
            weighted_r2=float(1 - sse / denom),
            sse_reduction_vs_zero=float(1 - sse / norm),
            final_affected_mean=float(e["total_samples"][:, -1].mean()),
            final_affected_mc_se=float(e["total_samples"][:, -1].std(ddof=1) / np.sqrt(len(d))),
            median_path_seconds=float(np.median(e["seconds"])),
            maximum_attempts=int(e["events"][:, 0].max()),
            months=[],
            per_time_rms=[],
            regional_rank_correlations=[],
        )
        for t, month in enumerate(obs.times):
            sel = obs.time_index == t
            wt = w[sel] / w[sel].sum()
            # Absolute RMS is meaningful even at near-zero early timepoints.
            row["months"].append(float(month))
            row["per_time_rms"].append(float(np.sqrt(wt @ ((pred[sel] - y[sel]) ** 2))))
            labels = np.unique(obs.labels[sel])
            true_mean = [y[sel & (obs.labels == lab)].mean() for lab in labels]
            model_mean = [pred[sel & (obs.labels == lab)].mean() for lab in labels]
            row["regional_rank_correlations"].append(
                float(spearmanr(true_mean, model_mean).statistic)
            )
        if name != "previous":
            fit = json.loads((a.fits / name / "fit.json").read_text())
            row["candidates"] = fit["candidates"]
        metrics[name] = row
    metrics["data"] = dict(
        records=len(y),
        mice=len(np.unique(obs.mouse_index)),
        times=obs.times.tolist(),
        windows=obs.operator.shape[0],
        zero_prediction_rmse=float(np.sqrt(norm)),
        constant_prediction_relative_rms=float(np.sqrt(denom / norm)),
        mice_per_time=[
            len(np.unique(obs.mouse_index[obs.time_index == t])) for t in range(len(obs.times))
        ],
    )
    free_prediction = np.empty_like(y)
    for t, g in set(zip(obs.time_index, obs.group)):
        rows = (obs.time_index == t) & (obs.group == g)
        free_prediction[rows] = np.average(y[rows], weights=w[rows])
    metrics["data"]["free_window_time_mean_relative_rms"] = float(
        np.sqrt((w @ ((free_prediction - y) ** 2)) / norm)
    )
    # Measurement sensitivity, with dynamics frozen: an illustration, not refits.
    densities = aligned["mixed"].mean(0)
    metrics["fixed_dynamics_depth_sensitivity"] = {
        str(h): float(np.sqrt(obs.loss_and_scale(densities, scale=h)[0])) for h in [6, 12, 20]
    }
    profiled_loss, profiled_scale = obs.loss_and_scale(densities)
    metrics["mixed_posthoc_profiled_scale"] = dict(
        scale_um=profiled_scale, relative_rms=float(np.sqrt(profiled_loss))
    )
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (a.out / "numerical_checks.json").write_text(json.dumps(checks, indent=2) + "\n")

    def save(fig, name):
        fig.savefig(a.out / (name + ".png"), dpi=190)
        fig.savefig(a.out / (name + ".pdf"))
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
    names = list(cases)
    axes[0].bar(
        range(3),
        [metrics[n]["relative_rms"] for n in names],
        color=[colors[n] for n in names],
        width=0.65,
    )
    axes[0].axhline(1, ls=":", color="black", lw=1)
    axes[0].set(
        xticks=range(3),
        xticklabels=["Mixed\n6 µm", "Tracer\n6 µm", "Mixed\nfitted scale"],
        ylabel="Relative RMS error",
        ylim=(0, 1.05),
    )
    for n, (e, h, label) in cases.items():
        axes[1].plot(obs.times, metrics[n]["per_time_rms"], marker="o", ms=3, color=colors[n])
        axes[2].plot(
            obs.times, metrics[n]["regional_rank_correlations"], marker="o", ms=3, color=colors[n]
        )
    axes[1].set(xlabel="Months", ylabel="RMS (profiles / mm²)", ylim=(0, None))
    axes[2].set(xlabel="Months", ylabel="Regional Spearman ρ", ylim=(-0.05, 1))
    save(fig, "fit_comparison")

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.7), layout="constrained")
    for ax, lab in zip(axes.flat, ["iCP", "iMOs", "iPL", "iBLA", "iSNc", "iCOAp"]):
        for t, month in enumerate(obs.times):
            sel = (obs.labels == lab) & (obs.time_index == t)
            vals = y[sel]
            if not len(vals):
                continue
            # Deterministic display jitter; no effect on calculations.
            ax.scatter(
                month + np.linspace(-0.06, 0.06, len(vals)),
                vals,
                s=9,
                color=".55",
                alpha=0.55,
                linewidth=0,
            )
            ax.plot(month, vals.mean(), "o", color="black", ms=4)
        for name, (e, h, title) in cases.items():
            ms, se = [], []
            for t in range(len(obs.times)):
                sel = (obs.labels == lab) & (obs.time_index == t)
                if not sel.any():
                    ms.append(np.nan)
                    se.append(np.nan)
                    continue
                samples = h * obs.factor * aligned[name][:, t, obs.group[sel]].mean(axis=1)
                ms.append(samples.mean())
                se.append(samples.std(ddof=1) / np.sqrt(len(samples)))
            ms, se = np.asarray(ms), np.asarray(se)
            ax.plot(
                obs.times,
                ms,
                color=colors[name],
                lw=1.6,
                ls="--" if name == "previous" else "-",
                label=title,
            )
            ax.fill_between(
                obs.times,
                np.maximum(0, ms - 1.96 * se),
                ms + 1.96 * se,
                color=colors[name],
                alpha=0.12,
                lw=0,
            )
        ax.set(title=lab[1:] + " (right)", xlabel="Months", ylim=(0, None))
    for ax in axes[:, 0]:
        ax.set_ylabel("Profiles / mm²")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=9)
    save(fig, "regional_trajectories")

    # Same anatomical averaging on both rows; fixed scale shared across all panels.
    for name in ["mixed", "tracer_only"]:
        e, h, _ = cases[name]
        plot_slices(
            e,
            a.data,
            a.out / (name + "_regional_slices"),
            scale_um=h,
            times=(0.1, 1, 3, 6, 9),
            planes=(55, 72, 83),
            vmax=250,
            regional=True,
            title="Mixed connectivity" if name == "mixed" else "Tracer only",
        )

    # Parameter rows and verified numbers for the LaTeX report.
    rows = []
    for name, (_, h, label) in cases.items():
        m = metrics[name]
        b, l, g = m["rates"]
        rows.append(
            f"{label} & {b:.4f} & {l:.4f} & {g:.4f} & {h:.2f} & {m['relative_rms']:.4f} & {m['weighted_r2']:.3f} \\\\"
        )
    table = (
        "\\begin{tabular}{lrrrrrr}\\toprule\n"
        "Model & $\\beta_T$ & $\\beta_L$ & $\\gamma$ & Depth $a$ & Rel. RMS & $R_w^2$\\\\\n"
        " & \\multicolumn{3}{c}{month$^{-1}$} & $\\um$ & &\\\\\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\\end{tabular}\n"
    )
    (a.out / "parameter_rows.tex").write_text(table)
    macros = []
    macros.append(
        "\\newcommand{\\FreeMeanError}{"
        + f"{metrics['data']['free_window_time_mean_relative_rms']:.4f}"
        + "}"
    )
    for name, prefix in [("mixed", "Mixed"), ("tracer_only", "Tracer"), ("previous", "Previous")]:
        m = metrics[name]
        for suffix, value in [
            ("Error", f"{m['relative_rms']:.4f}"),
            ("ErrorSE", f"{m['relative_rms_mc_se']:.5f}"),
            ("Improvement", f"{100 * m['sse_reduction_vs_zero']:.1f}"),
            ("Total", f"{m['final_affected_mean']:,.0f}"),
            ("RMSE", f"{m['weighted_rmse']:.2f}"),
            ("Seconds", f"{m['median_path_seconds']:.3f}"),
            ("Attempts", f"{m['maximum_attempts']:,}"),
        ]:
            macros.append("\\newcommand{\\" + prefix + suffix + "}{" + value + "}")
        if name != "previous":
            completed = sum(int(c["converged"]) for c in m["candidates"])
            capped = sum(c["capped_evaluations"] for c in m["candidates"])
            macros.append("\\newcommand{\\" + prefix + "Converged}{" + str(completed) + "}")
            macros.append("\\newcommand{\\" + prefix + "Capped}{" + str(capped) + "}")
    (a.out / "numbers.tex").write_text("\n".join(macros) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
