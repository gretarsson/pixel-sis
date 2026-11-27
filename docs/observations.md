# Observation model and next research step

The implemented baseline uses a single immediate observation map:

$$\widehat y_r=a\,10^6[O\,\mathbb E I(t_r)]_{g(r)}.$$

`I` contains affected-cell counts. `O` averages over the corresponding anatomical
window in µm³. `y` is positive cell-body profiles/mm². The global factor `a` has
units of µm and is an empirical conversion, not measured section thickness.
It is shared across regions and times. There is no hidden/visible subdivision,
biological delay, or temporal filter.

The SIR option uses the identical observation equation: removed cells `R` have
zero contribution. This is an explicit assumption, including when `R` is
interpreted as resistant rather than dead. Pathology measurements alone do not
distinguish those interpretations. The denominator remains the fixed regional
tissue volume, not the number of surviving cells.

The current fitting objective is weighted regional squared error:

$$J(\theta,a)=\frac{\sum_r w_r(\widehat y_r-y_r)^2}{\sum_r w_r y_r^2},
\qquad\theta=(\beta_T,\beta_L,\gamma).$$

For a candidate trajectory, the best nonnegative `a` is calculated analytically.
During manual comparisons the earlier fitted value is fixed across settings.

The fitter also accepts `observation_scale_um` to hold this conversion fixed.
The report's fixed-depth experiment uses `a = 6 µm`: an idealization in which
each affected cell is detected if its center lies in a physical tissue slab.
It assumes that the sampled tissue represents the modeled regional average.
Actual profile counting depends on object extent and detection as well, so this
choice is an explicit approximation, not a measured effective capture depth.
Only spreading and clearance rates are fitted in that experiment. The actual
registered-section operator below remains unimplemented for the full dataset.

## Planned explicit measurement model

The next task is to trace a measured mouse from tissue images or segmentation
exports through to reported regional values:

1. Establish whether each exported coordinate is a positive pixel, segmented
   object, cell-body profile, or another quantity, including any subsampling.
2. Recover actual section geometry and measured tissue boundaries, so unmeasured
   tissue is distinguished from measured tissue with zero pathology.
3. Reproduce the corresponding regional aggregation as a check.
4. Project model predictions through the same sampling/counting procedure and
   compare spatial patches at a resolution supported by registration accuracy.

Cell-body profiles, positive area, and raw fluorescence are different observations.
Section thickness, object geometry, and detection can affect the relationship to
modeled cells. These are measurement questions; this roadmap does not silently
introduce new spreading mechanisms. None of the four steps is claimed complete by
the public package.
