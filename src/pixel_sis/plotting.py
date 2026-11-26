"""Scientific figures with explicit observation units and fixed color scales."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm, ListedColormap
from matplotlib.patches import Patch

from .observations import RegionalObservations


def plot_slices(
    result,
    data_dir,
    out,
    *,
    scale_um,
    times=(0.1, 1.0, 3.0, 6.0, 9.0),
    planes=(55, 72, 83),
    vmax=1300.0,
    regional=False,
):
    """Model above data at every plane; gray means absent regional measurement."""
    if not np.isfinite(scale_um) or scale_um < 0 or not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Invalid display scale")
    folder = Path(data_dir)
    observations = RegionalObservations.load(folder)
    with np.load(folder / "geometry.npz", allow_pickle=False) as f:
        index, annotation = f["index"], f["annotation"]
    if any(not isinstance(p, (int, np.integer)) or p < 0 or p >= index.shape[0] for p in planes):
        raise ValueError("Coronal plane is outside the atlas")
    ti = np.searchsorted(result["times"], times)
    di = np.searchsorted(observations.times, times)
    if np.any(ti >= len(result["times"])) or np.any(di >= len(observations.times)):
        raise ValueError("Requested times are unavailable")
    if not np.array_equal(result["times"][ti], times) or not np.array_equal(
        observations.times[di], times
    ):
        raise ValueError("Slices require exact simulated and measured times")
    measured = observations.atlas_means()[di]
    mean = result["count_mean"][ti]
    # Reference voxel volume is (100 um)^3; the area-density conversion is a*count.
    modeled = scale_um * mean
    if regional:
        O = observations.operator
        densities = scale_um * 1e6 * (O @ mean.T).T
        sums = np.zeros_like(measured)
        numbers = np.zeros_like(sums, dtype=np.int16)
        for k, t in enumerate(di):
            for r in np.flatnonzero(observations.time_index == t):
                g = observations.group[r]
                ids = O.indices[O.indptr[g] : O.indptr[g + 1]]
                sums[k, ids] += densities[k, g]
                numbers[k, ids] += 1
        modeled = np.divide(sums, numbers, out=np.full_like(sums, np.nan), where=numbers > 0)
    if max(np.nanmax(modeled), np.nanmax(measured)) > vmax:
        raise ValueError("Display exceeds vmax; choose an explicit larger shared scale")
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(
        2 * len(planes),
        len(times),
        figsize=(12, 3.5 * len(planes)),
        layout="constrained",
        squeeze=False,
    )
    norm = AsinhNorm(linear_width=0.1, vmin=0, vmax=vmax)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad((0, 0, 0, 0))
    for j, plane in enumerate(planes):
        ids = index[plane]
        valid = ids >= 0
        yy, xx = np.where(annotation[plane] > 0)
        for k, t in enumerate(times):
            for r, values in [(2 * j, modeled[k]), (2 * j + 1, measured[k])]:
                ax = axes[r, k]
                canvas = np.full(ids.shape, np.nan)
                canvas[valid] = values[ids[valid]]
                ax.imshow(
                    np.where(annotation[plane] > 0, 1.0, np.nan),
                    cmap=ListedColormap(["#eeeeee"]),
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
                ax.imshow(
                    np.where(valid & ~np.isfinite(canvas), 1.0, np.nan),
                    cmap=ListedColormap(["#b8b8b8"]),
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
                image = ax.imshow(canvas, cmap=cmap, norm=norm, interpolation="nearest")
                ax.contour(annotation[plane] > 0, levels=[0.5], colors=["#777777"], linewidths=0.35)
                ax.set(xlim=(xx.min() - 2.5, xx.max() + 2.5), ylim=(yy.max() + 2.5, yy.min() - 2.5))
                ax.set_axis_off()
                if k == 0:
                    ax.text(
                        -0.04,
                        0.5,
                        "Model" if r % 2 == 0 else "Data",
                        rotation=90,
                        va="center",
                        ha="right",
                        transform=ax.transAxes,
                    )
            if j == 0:
                axes[0, k].set_title(f"{t:g} mo", fontsize=10)
        axes[2 * j, 0].text(
            0,
            1.02,
            {55: "CP plane", 72: "BLA plane", 83: "SNc plane"}.get(plane, f"AP index {plane}"),
            transform=axes[2 * j, 0].transAxes,
            color=".35",
            fontsize=8,
        )
    beta_t, beta_l, gamma = result["rates"]
    suffix = " · regional means" if regional else ""
    fig.suptitle(
        rf"SIS{suffix}  ($\beta_T={beta_t:g},\ \beta_L={beta_l:g},\ \gamma={gamma:g}$ mo$^{{-1}}$)",
        fontsize=11,
    )
    ticks = [x for x in [0, 0.1, 1, 10, 100, 1000, 10000] if x <= vmax]
    cb = fig.colorbar(image, ax=axes, shrink=0.7, fraction=0.018, pad=0.012, ticks=ticks)
    cb.set_label("Profiles / mm² · fixed model scale")
    fig.legend(
        handles=[Patch(facecolor="#b8b8b8", label="No regional measurement")],
        loc="outside lower center",
        frameon=False,
        fontsize=8,
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_totals(result, out):
    samples = result["total_samples"]
    mean = samples.mean(axis=0)
    se = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    fig, ax = plt.subplots(figsize=(6, 3.4), layout="constrained")
    ax.plot(result["times"], mean, color="#316fa0")
    ax.fill_between(
        result["times"],
        np.maximum(0, mean - 1.96 * se),
        mean + 1.96 * se,
        color="#316fa0",
        alpha=0.2,
        linewidth=0,
    )
    ax.set(xlabel="Time", ylabel="Mean affected cells", ylim=(0, None))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out, dpi=180)
    plt.close(fig)
