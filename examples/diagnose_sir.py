"""Locate errors in the unchanged SIR baseline and regional observation rule."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from pixel_sis.observations import RegionalObservations


def diagnose(data, ensemble, out):
    out.mkdir(parents=True, exist_ok=True)
    obs = RegionalObservations.load(data)
    with np.load(ensemble) as f:
        count = f["count_mean"]
        N = f["population"]
        rates = f["rates"]
    prediction = 6 * obs.predictions((obs.operator @ count.T).T)
    y, w = obs.values, obs.weights
    norm = float(w @ (y * y))
    errors = w * (prediction - y) ** 2 / norm
    capacity = 6e6 * np.asarray(obs.operator @ N).ravel()[obs.group]
    label_rows = []
    for label in np.unique(obs.labels):
        m = obs.labels == label
        label_rows.append(
            {
                "label": str(label),
                "loss_contribution": float(errors[m].sum()),
                "signal_contribution": float((w[m] * y[m] ** 2).sum() / norm),
            }
        )
    label_rows.sort(key=lambda r: -r["loss_contribution"])
    time_rows = []
    for t, month in enumerate(obs.times):
        m = obs.time_index == t
        time_rows.append(
            {
                "month": float(month),
                "loss_contribution": float(errors[m].sum()),
                "signal_contribution": float((w[m] * y[m] ** 2).sum() / norm),
                "underprediction_contribution": float(errors[m & (prediction < y)].sum()),
                "overprediction_contribution": float(errors[m & (prediction >= y)].sum()),
            }
        )
    # Descriptive diagnostics only: separately fitted group amplitudes/time means
    # are not offered as a biological model or as held-out predictive performance.
    fitted_amp = np.zeros_like(y)
    fitted_mean = np.zeros_like(y)
    for g in np.unique(obs.group):
        m = obs.group == g
        denominator = np.sum(w[m] * prediction[m] ** 2)
        a = np.sum(w[m] * prediction[m] * y[m]) / denominator if denominator else 0
        fitted_amp[m] = a * prediction[m]
        for t in np.unique(obs.time_index[m]):
            k = m & (obs.time_index == t)
            fitted_mean[k] = np.average(y[k], weights=w[k])
    # Initial expected tracer arrivals in each observation window, before removal.
    U = sparse.load_npz(data / "raw_source_factors.npz")
    V = sparse.load_npz(data / "raw_target_factors.npz")
    initial = np.load(data / "fitting_initial_fine.npy")
    c = float(N @ (U @ (V @ np.ones(len(N)))) / N.sum())
    arrivals = (1 - initial / N) * N * (U @ (V @ (initial / N))) / c
    del U, V
    labels = ["iCP", "iBLA", "iSNc", "iVTA", "iMOs", "iSNr"]
    regions = {}
    groups = json.loads((data / "observation_groups.json").read_text())
    for label in labels:
        ids = np.flatnonzero(obs.labels == label)
        if not len(ids):
            continue
        # Prefer the most commonly reported observation support for a label.
        unique, numbers = np.unique(obs.group[ids], return_counts=True)
        g = int(unique[np.argmax(numbers)])
        voxels = obs.operator.indices[obs.operator.indptr[g] : obs.operator.indptr[g + 1]]
        regions[label] = {
            "group": g,
            "voxels": len(voxels),
            "population": int(N[voxels].sum()),
            "volume_mm3": groups[g]["model_volume_um3"] / 1e9,
            "maximum_profiles_per_mm2_at_6um": float(6e6 * (obs.operator @ N)[g]),
            "initial_tracer_arrivals_per_month_at_unit_beta": float(arrivals[voxels].sum()),
            "initial_tracer_arrivals_per_month_at_fitted_beta": float(
                rates[0] * arrivals[voxels].sum()
            ),
            "observed_means": [],
            "predicted_means": [],
        }
        for t in range(len(obs.times)):
            m = (obs.labels == label) & (obs.time_index == t)
            regions[label]["observed_means"].append(float(np.mean(y[m])) if m.any() else None)
            regions[label]["predicted_means"].append(
                float(np.mean(prediction[m])) if m.any() else None
            )
    result = {
        "baseline_rms": float(np.sqrt(errors.sum())),
        "capacity_exceeding_records": int(np.count_nonzero(y > capacity)),
        "capacity_exceeding_signal_fraction": float(
            np.sum(w[y > capacity] * y[y > capacity] ** 2) / norm
        ),
        "capacity_only_lower_bound_rms": float(
            np.sqrt(np.sum(w * np.maximum(y - capacity, 0) ** 2) / norm)
        ),
        "diagnostic_separate_window_amplitudes_rms": float(
            np.sqrt(np.sum(w * (fitted_amp - y) ** 2) / norm)
        ),
        "descriptive_window_time_means_rms": float(
            np.sqrt(np.sum(w * (fitted_mean - y) ** 2) / norm)
        ),
        "largest_label_errors": label_rows[:30],
        "time_errors": time_rows,
        "regions": regions,
        "caution": "Diagnostic in-sample fits do not establish a noise floor. Regional volumes are atlas supports, not sampled tissue volumes.",
    }
    (out / "diagnosis.json").write_text(json.dumps(result, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    r = label_rows[:10]
    ax[0].barh(
        [x["label"] for x in r][::-1], [x["loss_contribution"] for x in r][::-1], color="#588b8b"
    )
    ax[0].set_xlabel("Contribution to squared error")
    x = np.arange(len(time_rows))
    ax[1].bar(
        x, [r["underprediction_contribution"] for r in time_rows], label="Too low", color="#588b8b"
    )
    ax[1].bar(
        x,
        [r["overprediction_contribution"] for r in time_rows],
        bottom=[r["underprediction_contribution"] for r in time_rows],
        label="Too high",
        color="#d2a16e",
    )
    ax[1].set_xticks(x, [str(t) for t in obs.times])
    ax[1].set_xlabel("Months")
    ax[1].set_ylabel("Contribution to squared error")
    ax[1].legend(frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    for ext in ["png", "pdf"]:
        fig.savefig(out / ("error_diagnosis." + ext), dpi=180)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["regions", "largest_label_errors"]},
            indent=2,
        )
    )
    print(
        json.dumps(
            {k: {a: b for a, b in v.items() if "means" not in a} for k, v in regions.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ensemble", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    diagnose(a.data, a.ensemble, a.out)
