"""Compare fixed-readout full-reference SIR ensembles with the same mouse data."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pixel_sis.observations import RegionalObservations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--revised", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline-label", default="Original start")
    p.add_argument("--revised-label", default="Revised start")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    obs = RegionalObservations.load(a.data)
    cases = []
    for path, name, color, style in [
        (a.baseline, a.baseline_label, ".5", "--"),
        (a.revised, a.revised_label, "#16877e", "-"),
    ]:
        with np.load(path) as f:
            samples = 6e6 * f["density_samples"][:, obs.time_index, obs.group]
        cases.append((name, color, style, samples))
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.5), layout="constrained")
    for ax, label in zip(axes.flat, ["iCP", "iMOs", "iPL", "iBLA", "iSNc", "iCOAp"]):
        for t, month in enumerate(obs.times):
            values = obs.values[(obs.labels == label) & (obs.time_index == t)]
            ax.scatter(
                month + np.linspace(-0.06, 0.06, len(values)),
                values,
                s=9,
                color=".65",
                alpha=0.6,
                linewidth=0,
            )
            ax.plot(month, values.mean(), "o", color="black", ms=3)
        for name, color, style, samples in cases:
            regional = np.stack(
                [
                    samples[:, (obs.labels == label) & (obs.time_index == t)].mean(axis=1)
                    for t in range(len(obs.times))
                ],
                axis=1,
            )
            mean = regional.mean(axis=0)
            se = regional.std(axis=0, ddof=1) / np.sqrt(len(regional))
            ax.plot(obs.times, mean, color=color, ls=style, lw=1.7, label=name)
            ax.fill_between(
                obs.times,
                np.maximum(0, mean - 1.96 * se),
                mean + 1.96 * se,
                color=color,
                alpha=0.15,
                lw=0,
            )
        ax.set(title=label[1:] + " (right)", xlabel="Months", ylim=(0, None))
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Profiles / mm²")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    for extension in ["png", "pdf"]:
        fig.savefig(a.out / ("sir_initial_comparison." + extension), dpi=200)


if __name__ == "__main__":
    main()
