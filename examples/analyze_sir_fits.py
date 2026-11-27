"""Compare completed SIR and SIS fits without rerunning any trajectories."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pixel_sis.observations import RegionalObservations
from pixel_sis.plotting import plot_slices


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sir", type=Path, required=True)
    parser.add_argument("--sis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mixed-only",
        action="store_true",
        help="Analyze the mixed fits before tracer-only finishes",
    )
    args = parser.parse_args()
    kinds = ["mixed"] if args.mixed_only else ["mixed", "tracer_only"]
    args.out.mkdir(parents=True, exist_ok=True)
    obs = RegionalObservations.load(args.data)
    N = np.load(args.data / "merfish_population.npy", allow_pickle=False)
    w, y = obs.weights, obs.values
    norm = w @ (y * y)
    centered = w @ ((y - w @ y) ** 2)
    labels = ["iCP", "iMOs", "iPL", "iBLA", "iSNc", "iCOAp"]
    cases, metrics, checks, predictions = {}, {}, {}, {}
    colors = {
        "sir_mixed": "#16877b",
        "sir_tracer_only": "#ce7834",
        "sis_mixed": "#666666",
        "sis_tracer_only": "#b0b0b0",
    }
    titles = {
        "sir_mixed": "SIR, mixed",
        "sir_tracer_only": "SIR, tracer",
        "sis_mixed": "SIS, mixed",
        "sis_tracer_only": "SIS, tracer",
    }
    for dynamics, folder in [("sis", args.sis), ("sir", args.sir)]:
        manifest = json.loads((folder / "configuration.json").read_text())
        for kind in kinds:
            name = dynamics + "_" + kind
            e = load(folder / kind / "ensemble.npz")
            fit = json.loads((folder / kind / "fit.json").read_text())
            cases[name] = e
            np.testing.assert_array_equal(e["population"], N)
            np.testing.assert_array_equal(e["times"], obs.times)
            assert float(e["observation_scale_um"]) == 6.0
            assert np.isfinite(e["count_mean"]).all()
            assert np.all(e["count_mean"] >= 0)
            assert np.all(e["count_mean"] <= N + 1e-8)
            assert e["events"][:, 0].max() < manifest["fit_settings"]["max_events"]
            np.testing.assert_allclose(
                e["count_mean"].sum(1), e["total_samples"].mean(0), rtol=1e-11, atol=1e-8
            )
            projected = (obs.operator @ e["count_mean"].T).T
            np.testing.assert_allclose(
                projected, e["density_samples"].mean(0), rtol=1e-11, atol=1e-14
            )
            check = dict(
                population_unchanged=True,
                infected_means_within_capacity=True,
                voxel_and_sample_totals_agree=True,
                readout_uses_infected_only=True,
                maximum_projection_difference=float(
                    abs(projected - e["density_samples"].mean(0)).max()
                ),
                no_final_path_reached_attempt_limit=True,
            )
            if dynamics == "sir":
                assert str(e["dynamics"]) == "sir"
                R = e["removed_mean"]
                S = N - e["count_mean"] - R
                assert np.isfinite(R).all() and np.all(R >= 0)
                assert np.min(S) > -1e-8
                assert np.min(np.diff(R, axis=0)) > -1e-8
                assert np.max(np.diff(S, axis=0)) < 1e-8
                np.testing.assert_allclose(
                    R.sum(1), e["removed_total_samples"].mean(0), rtol=1e-11, atol=1e-8
                )
                np.testing.assert_array_equal(
                    e["total_samples"]
                    + e["removed_total_samples"]
                    + e["susceptible_total_samples"],
                    N.sum(),
                )
                assert np.min(e["susceptible_total_samples"]) >= 0
                assert np.min(np.diff(e["removed_total_samples"], axis=1)) >= 0
                assert np.max(np.diff(e["susceptible_total_samples"], axis=1)) <= 0
                check.update(
                    compartments_conserved=True,
                    susceptible_never_increases=True,
                    removed_never_decreases=True,
                )
            checks[name] = check
            d = e["density_samples"]
            pred = 6 * obs.predictions(d.mean(0))
            predictions[name] = pred
            sse = w @ ((pred - y) ** 2)
            rms = np.sqrt(sse / norm)
            sample_pred = 6 * obs.factor * d[:, obs.time_index, obs.group]
            influence = (sample_pred - pred) @ (w * (pred - y)) / (rms * norm)
            per_time = []
            for t in range(len(obs.times)):
                sel = obs.time_index == t
                wt = w[sel] / w[sel].sum()
                per_time.append(float(np.sqrt(wt @ ((pred[sel] - y[sel]) ** 2))))
            regional = {}
            for lab in labels:
                data, means, ses = [], [], []
                for t in range(len(obs.times)):
                    sel = (obs.labels == lab) & (obs.time_index == t)
                    if not sel.any():
                        data.append(None)
                        means.append(None)
                        ses.append(None)
                        continue
                    samples = 6 * obs.factor * d[:, t, obs.group[sel]].mean(axis=1)
                    data.append(float(y[sel].mean()))
                    means.append(float(samples.mean()))
                    ses.append(float(samples.std(ddof=1) / np.sqrt(len(samples))))
                regional[lab] = dict(data=data, model=means, mc_se=ses)
            row = dict(
                rates=e["rates"].tolist(),
                relative_rms=float(rms),
                relative_rms_mc_se=float(influence.std(ddof=1) / np.sqrt(len(d))),
                weighted_r2=float(1 - sse / centered),
                per_time_rmse=per_time,
                total_infected=e["total_samples"].mean(0).tolist(),
                total_infected_mc_se=(e["total_samples"].std(0, ddof=1) / np.sqrt(len(d))).tolist(),
                median_seconds=float(np.median(e["seconds"])),
                maximum_attempts=int(e["events"][:, 0].max()),
                runs=len(d),
                candidates=fit["candidates"],
                regional=regional,
            )
            if dynamics == "sir":
                row["total_removed"] = e["removed_total_samples"].mean(0).tolist()
                row["total_susceptible"] = e["susceptible_total_samples"].mean(0).tolist()
                row["removed_fraction_at_nine_months"] = row["total_removed"][-1] / N.sum()
                row["maximum_voxel_removed_fraction_at_nine_months"] = float(
                    np.max(e["removed_mean"][-1] / N)
                )
            metrics[name] = row
    # Ensure both experiments really used the same frozen inputs and measurement scale.
    sm = json.loads((args.sis / "configuration.json").read_text())
    rm = json.loads((args.sir / "configuration.json").read_text())
    assert sm["inputs"] == rm["inputs"]
    assert (
        sm["fit_settings"]["observation_scale_um"]
        == rm["fit_settings"]["observation_scale_um"]
        == 6
    )
    metrics["comparison"] = dict(
        times=obs.times.tolist(),
        records=len(y),
        mice=len(np.unique(obs.mouse_index)),
        identical_inputs=True,
        scale_um=6.0,
        relative_sse_improvement={
            kind: 1
            - (metrics["sir_" + kind]["relative_rms"] / metrics["sis_" + kind]["relative_rms"]) ** 2
            for kind in kinds
        },
        rms_difference={
            kind: metrics["sir_" + kind]["relative_rms"] - metrics["sis_" + kind]["relative_rms"]
            for kind in kinds
        },
        rms_difference_mc_se={
            kind: float(
                np.hypot(
                    metrics["sir_" + kind]["relative_rms_mc_se"],
                    metrics["sis_" + kind]["relative_rms_mc_se"],
                )
            )
            for kind in kinds
        },
    )
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.out / "numerical_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    plt.rcParams.update(
        {"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    )

    def save(fig, name):
        fig.savefig(args.out / (name + ".png"), dpi=190)
        fig.savefig(args.out / (name + ".pdf"))
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), layout="constrained")
    names = [d + "_" + k for k in kinds for d in ("sis", "sir")]
    axes[0].bar(
        range(len(names)),
        [metrics[n]["relative_rms"] for n in names],
        color=[colors[n] for n in names],
        width=0.65,
    )
    axes[0].axhline(1, ls=":", color="black", lw=1)
    axes[0].set(
        xticks=range(len(names)),
        xticklabels=[titles[n].replace(", ", "\n") for n in names],
        ylabel="Relative RMS error",
        ylim=(0, 1.05),
    )
    for name in names:
        axes[1].plot(
            obs.times,
            metrics[name]["per_time_rmse"],
            color=colors[name],
            ls="--" if name.startswith("sis") else "-",
            label=titles[name],
        )
    axes[1].set(xlabel="Months", ylabel="RMS (profiles / mm²)", ylim=(0, None))
    axes[1].legend(frameon=False, fontsize=8)
    m = metrics["sir_mixed"]
    axes[2].plot(
        obs.times, np.array(m["total_infected"]) / 1000, color=colors["sir_mixed"], label="I"
    )
    axes[2].plot(obs.times, np.array(m["total_removed"]) / 1000, color="#8966aa", label="R")
    axes[2].set(xlabel="Months", ylabel="Cells (thousands)", title="SIR, mixed", ylim=(0, None))
    axes[2].legend(frameon=False)
    save(fig, "sir_comparison")
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.7), layout="constrained")
    for ax, lab in zip(axes.flat, labels):
        for t, month in enumerate(obs.times):
            vals = y[(obs.labels == lab) & (obs.time_index == t)]
            if len(vals):
                ax.scatter(
                    month + np.linspace(-0.06, 0.06, len(vals)),
                    vals,
                    s=9,
                    color=".55",
                    alpha=0.55,
                    linewidth=0,
                )
                ax.plot(month, vals.mean(), "o", color="black", ms=4)
        for name in ["sis_mixed", "sir_mixed"]:
            row = metrics[name]["regional"][lab]
            mean = np.array(row["model"], dtype=float)
            se = np.array(row["mc_se"], dtype=float)
            ax.plot(
                obs.times,
                mean,
                color=colors[name],
                ls="--" if name.startswith("sis") else "-",
                lw=1.7,
                label=titles[name],
            )
            ax.fill_between(
                obs.times,
                np.maximum(0, mean - 1.96 * se),
                mean + 1.96 * se,
                color=colors[name],
                alpha=0.15,
                lw=0,
            )
        ax.set(title=lab[1:] + " (right)", xlabel="Months", ylim=(0, None))
    for ax in axes[:, 0]:
        ax.set_ylabel("Profiles / mm²")
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="outside lower center", ncols=2, frameon=False)
    save(fig, "sir_regional_trajectories")
    plot_slices(
        cases["sir_mixed"],
        args.data,
        args.out / "sir_mixed_slices",
        scale_um=6,
        times=(0.1, 1, 3, 6, 9),
        planes=(55, 72, 83),
        vmax=250,
        regional=True,
        title="SIR, mixed connectivity",
        font_scale=1.6,
    )
    rows = []
    for name in names:
        m = metrics[name]
        b, l, g = m["rates"]
        rows.append(
            f"{titles[name]} & {b:.4f} & {l:.4f} & {g:.4f} & {m['relative_rms']:.4f} & {m['weighted_r2']:.3f} \\\\"
        )
    table = (
        "\\begin{tabular}{lrrrrr}\\toprule\nModel & $\\beta_T$ & $\\beta_L$ & $\\gamma$ & Rel. RMS & $R_w^2$\\\\\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\\end{tabular}\n"
    )
    (args.out / "parameter_rows.tex").write_text(table)
    print(
        json.dumps(
            {
                n: {
                    k: v
                    for k, v in m.items()
                    if k in ("rates", "relative_rms", "weighted_r2", "total_removed")
                }
                for n, m in metrics.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
