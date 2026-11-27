"""Compare paired neuritic-area and cell-body-count observations without fitting.

The quantities have different units. This audit compares reported positivity and
checks the area arithmetic; it does not equate neuritic area with infected somata.
"""

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path

import numpy as np

from pixel_sis.observations import RegionalObservations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    raw = {}
    precision = {}
    for path in sorted((a.data / "raw_pathology").glob("*.csv")):
        with path.open() as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            for row in reader:
                parent = row["daughter"] if row["daughter"] in ("SNc", "SNr") else row["parent"]
                key = ("".join(row["mouse"].split()), row["side"], parent)
                if key in raw:
                    raise ValueError("Duplicate raw measurement")
                raw[key] = [float(row[name]) for name in columns[-4:]]
                precision[key] = sum(
                    0.5 * 10 ** Decimal(row[name]).as_tuple().exponent
                    for name in columns[-4:-1]
                    if float(row[name]) != 0
                )
    with np.load(a.data / "cell_body_observations.npz") as f:
        selected = f["base_record_index"]
    records = json.loads((a.data / "observation_records.json").read_text())
    obs = RegionalObservations.load(a.data)
    paired = []
    tolerance = []
    for k in selected:
        record = records[k]
        side = "right" if record["label"].startswith("i") else "left"
        key = (record["mouse"], side, record["label"][1:])
        paired.append(raw[key])
        tolerance.append(precision[key])
    total, inclusion_area, neurite_area, density = np.asarray(paired).T
    np.testing.assert_allclose(density, obs.values, rtol=0, atol=0)
    finite_area = np.isfinite(total + inclusion_area + neurite_area)
    arithmetic_error = np.abs(total - inclusion_area - neurite_area)
    rows = []
    for t, month in enumerate(obs.times):
        mask = (obs.time_index == t) & finite_area
        body, neurite = density[mask] > 0, neurite_area[mask] > 0
        rows.append(
            dict(
                month=float(month),
                records=int(mask.sum()),
                body_positive=int(body.sum()),
                neurite_positive=int(neurite.sum()),
                neurite_positive_body_zero=int(np.sum(neurite & ~body)),
                body_positive_neurite_zero=int(np.sum(body & ~neurite)),
                both_positive=int(np.sum(body & neurite)),
            )
        )
    bad = np.flatnonzero(finite_area & (arithmetic_error > 1e-7))
    largest = bad[np.argsort(arithmetic_error[bad])[-10:]][::-1]
    result = dict(
        measurements_matched_exactly=len(density),
        missing_area_records=int(np.sum(~finite_area)),
        maximum_area_sum_discrepancy=float(arithmetic_error[finite_area].max()),
        area_sum_discrepancies_over_1e_minus_7=int(np.sum(arithmetic_error[finite_area] > 1e-7)),
        area_sum_excess_over_total=int(np.sum((inclusion_area + neurite_area - total)[bad] > 0)),
        discrepancies_beyond_written_decimal_precision=int(
            np.sum(arithmetic_error > np.array(tolerance) + 1e-12)
        ),
        large_discrepancies_beyond_written_precision=int(
            np.sum((arithmetic_error > 1e-7) & (arithmetic_error > np.array(tolerance) + 1e-12))
        ),
        positive_count_zero_inclusion_area=int(np.sum((density > 0) & (inclusion_area == 0))),
        positive_inclusion_area_zero_count=int(np.sum((inclusion_area > 0) & (density == 0))),
        maximum_count_density_with_zero_inclusion_area=float(density[inclusion_area == 0].max()),
        maximum_inclusion_area_with_zero_count=float(inclusion_area[density == 0].max()),
        largest_area_discrepancies=[
            dict(
                label=str(obs.labels[k]),
                month=float(obs.times[obs.time_index[k]]),
                total_area=float(total[k]),
                inclusion_area=float(inclusion_area[k]),
                neurite_area=float(neurite_area[k]),
                signed_difference=float(total[k] - inclusion_area[k] - neurite_area[k]),
            )
            for k in largest
        ],
        by_time=rows,
        interpretation="Descriptive paired observations with reported value >0; no new threshold or readout fitted. Neuritic area and cell-body profile density are different quantities, and neurites can originate from somata outside the observed region. Positivity is not a calibrated biological detection probability.",
        state_assumption="Current immediate SIR readout treats all I cells as cell-body positive. This audit does not establish a unique alternative observation model.",
    )
    (a.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
