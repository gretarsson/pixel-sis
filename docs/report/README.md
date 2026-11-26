# Model and fit report

[Read the report](pixel_sis_report.pdf). It explains the reference model, its
construction data, pathology measurements, observation approximation, fitting
method, results and limitations.

The PDF can be rebuilt from the repository root with

```sh
tectonic --outdir docs/report docs/report/pixel_sis_report.tex
```

The figures and numerical summaries are already included. Regenerating the fits
requires the prepared mouse bundle described in [the data guide](../data.md).
`examples/run_section_fits.py` runs the fixed-depth fits;
`examples/analyze_section_fits.py` generates the figures and numerical summaries
from the saved ensembles. The latter also takes the historical freely scaled fit
as a comparison. See the report for commands and exact settings.

`reproducibility.json` records input hashes, random seeds, settings and the
simulation-code commit. `figures/metrics.json` contains the reported metrics;
`figures/numerical_checks.json` records the checks on the completed ensembles.
Neither the experimental inputs nor the large simulation arrays are distributed
in this repository.
