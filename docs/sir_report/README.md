# SIR fitting and diagnostic extensions

This is an exploratory, in-sample comparison. The revised models improve the numerical fit but still miss important regional peaks. `R` contributes no pathology signal. The reference remains the 100-µm CTMC; coarse ODE models are search aids only.

Use the prepared reference directory described in `docs/data.md`. Data and historical output folders are not included in the public repository. Python dependencies are pinned in `pyproject.toml`. Run one numerical thread:

```sh
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1
```

The completed run folders are recorded in `metrics.json`. Reproduction scripts:

- `examples/run_sir_fits.py`: original three-start uniform SIR fit. The mixed family completed; the remaining tracer-only search was stopped when the research question changed. Do not report an unfinished tracer-only fit.
- `examples/diagnose_sir.py`: residuals, readout capacity, descriptive amplitude diagnostics and initial tracer arrivals.
- `examples/screen_sir_rates.py`: broad uniform-rate ODE screen.
- `examples/screen_sir_turnover.py`: separate low/high-turnover searches.
- `examples/screen_sir_initial_state.py`: fit initial burden and a tracer-defined projected fraction jointly with the uniform rates.
- `examples/check_sir_initial_state.py`: fresh full-reference CTMC check, with explicitly independent binomial initial counts.
- `examples/screen_sir_composition.py`: anatomical ODE screen for composition-dependent voxel rates. `--weighting regions` changes the loss; it does not change the mouse data. `--starts-from` records continued initialization explicitly.
- `examples/check_sir_composition.py`: fresh CTMC check using exact event thinning. `--start 2` selects the third finite-budget candidate; omitting it selects the screen's best candidate.
- `examples/build_sir_report.py`: assemble figures and scores from completed ensembles only.

For the runs documented here, use these output names and options:

```sh
python examples/screen_sir_rates.py --data tracer_sis/data \
  --out outputs/sir_broad_rates_positive_20261127 --iterations 50
python examples/screen_sir_turnover.py --data tracer_sis/data \
  --out outputs/sir_turnover_screen_20261127
python examples/screen_sir_initial_state.py --data tracer_sis/data \
  --out outputs/sir_initial_screen_20261127 --iterations 40
python examples/check_sir_initial_state.py --data tracer_sis/data \
  --screen outputs/sir_initial_screen_20261127 \
  --out outputs/sir_initial_reference_20261127 --runs 64
python examples/screen_sir_composition.py --data tracer_sis/data \
  --initial-source outputs/sir_initial_screen_20261127/surrogate.npz \
  --out outputs/sir_composition_screen_20261127 --max-evaluations 220
python examples/check_sir_composition.py --data tracer_sis/data \
  --screen outputs/sir_composition_screen_20261127 --start 2 \
  --initial-source outputs/sir_initial_screen_20261127/surrogate.npz \
  --out outputs/sir_composition_reference_start2_20261127 --runs 64
python examples/screen_sir_composition.py --data tracer_sis/data \
  --initial-source outputs/sir_initial_screen_20261127/surrogate.npz \
  --basis-source outputs/sir_composition_screen_20261127 \
  --starts-from outputs/sir_composition_screen_20261127/starts.json \
  --weighting regions --out outputs/sir_composition_regions_20261127 \
  --max-evaluations 500
python examples/check_sir_composition.py --data tracer_sis/data \
  --screen outputs/sir_composition_regions_20261127 \
  --initial-source outputs/sir_initial_screen_20261127/surrogate.npz \
  --out outputs/sir_composition_regions_reference_20261127 --runs 128
python examples/build_sir_report.py --data tracer_sis/data \
  --outputs outputs --report docs/sir_report
tectonic -o docs/sir_report docs/sir_report/sir_fit_report.tex
```

The coarse screens read precomputed population-weighted operators from the existing research dataset. They are not standalone Allen-data downloaders. The composition basis is rebuilt from the frozen voxel tracer factors, geometry, independent composition prior and local normalizers when absent. Do not substitute different inputs under the same cache/output path.

The initial-condition fit and composition fits are provisional, finite searches; most ended at their evaluation budgets. No global optimum, uniquely identified biological rates, held-out biological validation, or successful reconstruction of the actual section-sampling operator is claimed.
