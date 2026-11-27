# SIR fitting and diagnostic extensions

This is an exploratory, in-sample comparison. Regional-rate extensions now reproduce a rise and fall in amygdala and substantia nigra pathology, with remaining timing and amplitude errors. `R` contributes no pathology signal. The reference remains the 100-µm CTMC; coarse ODE models are search aids only.

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
- `examples/diagnose_sir_regional_rates.py`: fit bilateral BLA/SNc rate overrides to the two right-sided trajectories only; report whole-brain error separately.
- `examples/fit_sir_regional_extension.py`: jointly fit the seven composition parameters and four regional overrides against the whole cohort. `--at-risk` adds two initially eligible fractions, while preserving the earlier infected marginal.
- `examples/build_sir_report.py`: assemble figures and scores from completed ensembles only.
- `examples/decompose_sir_error.py`: exact weighted SSE decomposition into scatter among observations in the same window/time and mismatch with their weighted mean. This changes no data or objective.
- `examples/screen_sir_seed_position.py`: cover the interior of right CP with 24 geometry-selected candidate locations plus the original source, at fixed rates and initial burden. Infeasible initial probabilities are rejected.

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
python examples/diagnose_sir_regional_rates.py --data tracer_sis/data \
  --basis outputs/sir_composition_screen_20261127 \
  --baseline outputs/sir_composition_regions_20261127 \
  --out outputs/sir_regional_diagnostic_20261127 --iterations 35
python examples/fit_sir_regional_extension.py --data tracer_sis/data \
  --basis outputs/sir_composition_screen_20261127 \
  --baseline outputs/sir_composition_regions_20261127 \
  --diagnostic outputs/sir_regional_diagnostic_20261127 \
  --out outputs/sir_regional_joint_20261127 --max-evaluations 700
python examples/check_sir_composition.py --data tracer_sis/data \
  --screen outputs/sir_regional_joint_20261127 \
  --initial-source outputs/sir_initial_screen_20261127/surrogate.npz \
  --regional-source outputs/sir_regional_joint_20261127 \
  --prepare-susceptibility \
  --out outputs/sir_regional_joint_reference_20261127 --runs 64
python examples/fit_sir_regional_extension.py --data tracer_sis/data \
  --basis outputs/sir_composition_screen_20261127 \
  --baseline outputs/sir_composition_regions_20261127 \
  --diagnostic outputs/sir_regional_diagnostic_20261127 \
  --continue-from outputs/sir_regional_joint_20261127 --at-risk \
  --out outputs/sir_regional_atrisk_20261127 --max-evaluations 900
python examples/check_sir_composition.py --data tracer_sis/data \
  --screen outputs/sir_regional_atrisk_20261127 \
  --initial-source outputs/sir_initial_screen_20261127/surrogate.npz \
  --regional-source outputs/sir_regional_atrisk_20261127 \
  --prepare-susceptibility \
  --out outputs/sir_regional_atrisk_reference_20261127 --runs 64
python examples/screen_sir_composition.py --data tracer_sis/data \
  --initial-source outputs/sir_initial_screen_20261127/surrogate.npz \
  --out outputs/sir_seed_basis_200um_20261127 --cp-width-um 200 --prepare-only
python examples/screen_sir_seed_position.py --data tracer_sis/data \
  --basis outputs/sir_seed_basis_200um_20261127 \
  --rates-source outputs/sir_regional_joint_20261127 \
  --initial-source outputs/sir_initial_screen_20261127 \
  --out outputs/sir_seed_positions_20261127 --candidates 24
python examples/fit_sir_regional_extension.py --data tracer_sis/data \
  --basis outputs/sir_seed_basis_200um_20261127 \
  --baseline outputs/sir_composition_regions_20261127 \
  --diagnostic outputs/sir_regional_diagnostic_20261127 \
  --continue-from outputs/sir_regional_joint_20261127 \
  --fit-initial outputs/sir_seed_positions_20261127 \
  --out outputs/sir_seed_joint_initial_20261127 --max-evaluations 400
python examples/fit_sir_regional_extension.py --data tracer_sis/data \
  --basis outputs/sir_seed_basis_200um_20261127 \
  --baseline outputs/sir_composition_regions_20261127 \
  --diagnostic outputs/sir_regional_diagnostic_20261127 \
  --continue-from outputs/sir_regional_joint_20261127 \
  --fit-initial outputs/sir_initial_screen_20261127 \
  --out outputs/sir_original_seed_joint_initial_20261127 --max-evaluations 400
python examples/check_sir_composition.py --data tracer_sis/data \
  --screen outputs/sir_seed_joint_initial_20261127 \
  --initial-source outputs/sir_seed_joint_initial_20261127/surrogate.npz \
  --regional-source outputs/sir_seed_joint_initial_20261127 \
  --prepare-susceptibility \
  --out outputs/sir_seed_joint_initial_reference_20261127 --runs 64
python examples/build_sir_report.py --data tracer_sis/data \
  --outputs outputs --report docs/sir_report
tectonic -o docs/sir_report docs/sir_report/sir_fit_report.tex
```

The coarse screens read precomputed population-weighted operators from the existing research dataset. They are not standalone Allen-data downloaders. The composition basis is rebuilt from the frozen voxel tracer factors, geometry, independent composition prior and local normalizers when absent. Do not substitute different inputs under the same cache/output path.

The initial-condition fit and composition fits are provisional, finite searches; most ended at their evaluation budgets. No global optimum, uniquely identified biological rates, held-out biological validation, or successful reconstruction of the actual section-sampling operator is claimed.

The regional extensions were chosen after inspecting residuals. The joint rate
fit used three stated starts with 700 evaluations each; initial resistance used
three stated starts with 900 evaluations each. All six stopped at their budgets.
The initial-resistance fit keeps the infected marginal fixed and draws a joint
multinomial S/I/R state. Its initial R means pre-existing resistance; reported
new removals exclude this initial count. Model N, contact denominators and the
6-µm immediate readout stay fixed. The diagnostic that fits only BLA/SNc is not
included as a successful whole-brain fit. The eight-path initial-resistance
pilot is superseded by the 64-path check (same first eight seeds).

Prepared susceptibility sampling retains the exact generator while avoiding a
global maximum-rate envelope. Nonuniform removal now uses a weighted event tree.
For heterogeneous rates this changes seeded event paths relative to the earlier
sampler, while preserving their law. The default uniform SIS reference remains
bitwise unchanged. Validation includes 62 tests with the local reference data,
finite-state master equations, an independent ODE solver, initial-state moments
and repeat loading of the compiled composition kernel cache.

The source-location experiment keeps the seed Gaussian width at 150 µm and its
cutoff at 300 µm. Positions are chosen by deterministic farthest-point coverage
within right CP, at least 300 µm inside the boundary, plus the old provisional
position. Its ODE grid treats every CP location equally at 200 µm; other regions
remain anatomical compartments (3,117 total). The source choice is fitted to the
same cohort, not identified from registered injection data. Both subsequent
joint refinements use one local start and 400 evaluations, with no convergence
claim. They optimize the existing eleven rate parameters and the two expected
initial component counts. Their sum is at most 100,000 and each voxel's infection
probability must not exceed one. The prior initial-resistance extension is not
used here. The final comparison uses 64 full-reference paths.

`regional_provenance.json` records the new search configurations, candidate
locations, parameters and simulation summaries. The decomposition figure and
its exact numerical contributions are in `figures/decomposition/`.
