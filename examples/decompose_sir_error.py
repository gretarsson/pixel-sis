"""Separate shared-trajectory error from scatter among observed mice.

This is an exact weighted least-squares identity on the observed dataset, not an
estimate of a population noise floor. It keeps the original observations/weights.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pixel_sis.observations import RegionalObservations


def decompose(values, weights, cell, prediction):
    """Return record-level within-cell and shared-mean SSE contributions."""
    y, w, cell, pred = map(np.asarray, (values, weights, cell, prediction))
    _, cell = np.unique(cell, return_inverse=True)
    mass = np.bincount(cell, weights=w)
    mean = np.divide(
        np.bincount(cell, weights=w * y),
        mass,
        out=np.zeros_like(mass),
        where=mass > 0,
    )[cell]
    pred_mean = np.divide(
        np.bincount(cell, weights=w * pred),
        mass,
        out=np.zeros_like(mass),
        where=mass > 0,
    )[cell]
    if not np.allclose(pred[w > 0], pred_mean[w > 0], atol=1e-10, rtol=1e-10):
        raise ValueError("Decomposition assumes one prediction per observation window/time")
    scatter = w * (y - mean) ** 2
    mismatch = w * (pred - mean) ** 2
    np.testing.assert_allclose(np.sum(scatter + mismatch), np.sum(w * (pred - y) ** 2), rtol=1e-12)
    return scatter, mismatch, mean


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    obs = RegionalObservations.load(a.data)
    cases = json.loads((a.report / "metrics.json").read_text())["cases"]
    norm = float(obs.weights @ obs.values**2)
    cell = obs.time_index * obs.operator.shape[0] + obs.group
    rows = []
    for case in cases:
        with np.load(a.outputs / case["source"]) as f:
            prediction = 6e6 * f["density_samples"].mean(0)[obs.time_index, obs.group]
        scatter, mismatch, mean = decompose(obs.values, obs.weights, cell, prediction)
        rows.append(
            dict(
                name=case["name"],
                source=case["source"],
                within_window_time_scatter=float(scatter.sum() / norm),
                mean_mismatch=float(mismatch.sum() / norm),
                relative_mean_rms=float(np.sqrt(mismatch.sum() / (obs.weights @ mean**2))),
                within_fraction_of_sse=float(scatter.sum() / (scatter.sum() + mismatch.sum())),
            )
        )
    # Report cases are ordered chronologically; inspect the latest completed case.
    label_rows = []
    for label in np.unique(obs.labels):
        k = obs.labels == label
        label_rows.append(
            dict(
                label=str(label),
                scatter=float(scatter[k].sum() / norm),
                mean_mismatch=float(mismatch[k].sum() / norm),
                records=int(k.sum()),
            )
        )
    label_rows.sort(key=lambda r: r["mean_mismatch"], reverse=True)
    result = dict(
        cases=rows,
        latest_regional_errors=label_rows,
        observed_window_time_cells=len(np.unique(cell)),
        interpretation="Exact within/between weighted SSE identity, with original area weights. Scatter combines animal variability, sampling and measurement differences. It is not a calibrated biological noise model or held-out error.",
        formula="sum_r w_r(y_r-p_g)^2 = sum_r w_r(y_r-ybar_g)^2 + sum_g W_g(ybar_g-p_g)^2; g is window/time, ybar_g is weighted",
    )
    (a.out / "summary.json").write_text(json.dumps(result, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    x = np.arange(len(rows))
    ax[0].bar(
        x,
        [r["within_window_time_scatter"] for r in rows],
        color=".75",
        label="Between-mouse scatter",
    )
    ax[0].bar(
        x,
        [r["mean_mismatch"] for r in rows],
        bottom=[r["within_window_time_scatter"] for r in rows],
        color="#16877e",
        label="Mean mismatch",
    )
    ax[0].set(
        xticks=x,
        xticklabels=[
            "Uniform",
            "Start",
            "Comp.\n(area)",
            "Comp.\n(regions)",
            "Regional\nrates",
            "Initial\nresist.",
            "Source\nrefit",
        ][: len(rows)],
        ylabel="Normalized squared error",
    )
    ax[0].tick_params(axis="x", labelsize=8)
    ax[0].legend(frameon=False, fontsize=8)
    top = label_rows[:8][::-1]
    ax[1].barh([r["label"] for r in top], [r["mean_mismatch"] for r in top], color="#16877e")
    ax[1].set_xlabel("Remaining mean mismatch")
    for axis in ax:
        axis.spines[["top", "right"]].set_visible(False)
    for ext in ["png", "pdf"]:
        fig.savefig(a.out / ("error_decomposition." + ext), dpi=200)
    print(json.dumps(dict(cases=rows, largest_mean_errors=label_rows[:8]), indent=2))


if __name__ == "__main__":
    main()
