# Data inputs and reproduction boundary

All inputs are supplied by an explicit directory argument. Nothing is downloaded
on import, and loading the reference does not modify its data directory.

## Required reference files

| File | Content |
|---|---|
| `geometry.npz` | `coords`: N×3 integer atlas indices; `index`: 3D grid with -1 outside the model and 0…N-1 inside; `annotation`: atlas IDs for plots |
| `merfish_population.npy` | N positive integer cell populations |
| `fitting_initial_fine.npy` | N integer affected counts, each between 0 and population |
| `raw_source_factors.npz` | SciPy sparse U, shape N×rank |
| `raw_target_factors.npz` | SciPy sparse V, shape rank×N |

Matrices use source rows and target columns. The current reference voxel side is
100 µm. The loader computes the reversed tracer sampling factors and the local
sampling envelopes from these files.

## Regional measurements

| File | Content |
|---|---|
| `observation_density_fine.npz` | Sparse operator O, shape groups×N; maps affected counts to density per µm³ |
| `observations.npz` | Original regional records, including `times`, `weights`, `time_index`, `group`, `mouse_index`, `labels`, `area` |
| `cell_body_observations.npz` | The cell-profile target: `values`, `base_record_index`, `times`, `time_index`, `group`, `mouse_index`, `labels`, `area` |

The cell-profile file is aligned against the original records before use. Times
are distinct observed months; missing measurements are omitted, not entered as
zero. Zero-valued observations are retained. The existing target contains 18,292
records from 55 mice at 0.1, 0.2, 0.3, 0.5, 1, 3, 6 and 9 months.

For a regional window of V µm³, O has value 1/V at its voxel columns. The current
prediction for record r is `a * 1e6 * [O E(I(t_r))]_group(r)`, in profiles/mm².
The weights use the supplied area proxy within mouse and equal mice/time points.
Those area summaries are not reconstructed pixel-level sampled-tissue masks.

One source table has a `log10` label inconsistent with its numeric scale. The
research preprocessing preserves the published numeric values without
exponentiation. The generating count/area pipeline still needs end-to-end checking.

## What is and is not distributed

The repository contains the original input checksums in
[`reference_expectations.json`](../tests/reference_expectations.json), but does not
redistribute prepared arrays, pathology archives, tissue images, or mouse metadata.
The self-contained demo and ordinary tests run without these data.

For an existing pixel-SIS workspace, pass its prepared `tracer_sis/data` directory.
For a new independent installation, obtaining or reconstructing the prepared bundle
is a separate requirement. This release does not yet automate the whole preparation
pipeline from the raw source studies. In particular, raw images for every mouse and
complete measured-tissue masks are not established as available.

Published upstream sources:

- Allen/Knox `nodes.csv.gz`, `weights.csv.gz`, and source/target mask parameters:
  https://download.alleninstitute.org/publications/A_high_resolution_data-driven_model_of_the_mouse_connectome/
- Allen CCFv3 atlas and structure hierarchy.
- [Registered MERFISH reference animal](https://alleninstitute.github.io/abc_atlas_access/descriptions/MERFISH-C57BL6J-638850.html).
- [Regional pathology and area tables, pinned analysis revision](https://github.com/jkbrynildsen/aSyn_spread/tree/895dc66cf16a3419c8018a3c7f40b08a035ca7fe).

The private/shared Nutil coordinate archive is an additional experimental input,
not a software dependency. Its point semantics, possible subsampling, and tissue
coverage must be established before it becomes a quantitative spatial target.
