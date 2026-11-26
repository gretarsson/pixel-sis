# Reference model

The reference has 448,962 modeled voxels and 39,476,440 estimated neurons.
`I_i` is a count from 0 to `N_i`, rather than a binary state for the entire voxel.
Uniform contact rules within each reference voxel allow the underlying assumed
cell process to be simulated exactly as a count CTMC.

## Tracer contacts

Let `W = U V` be the prepared, source-row voxel projection matrix, `N` the fixed
reference populations and `D = diag(N)`. The forward normalization is

$$c=\frac{\sum_j N_j\sum_i W_{ji}}{\sum_j N_j}
=24.381317861482305.$$

The reference uses

$$Q=D^{-1}(W/c)^T D.$$

This reverses the assumed cell-level contacts. For unequal populations, simply
transposing the voxel matrix would describe a different process. Source strengths
are retained; `Q` is not normalized to sum to one in every row. The implementation
samples through sparse factors and never constructs the full voxel-by-voxel matrix.

The supplied factors come from an inferred Allen tracer-based connectivity model.
Left-hemisphere sources use the historical mirrored-source construction. Neither
source interpolation nor this symmetry assumption establishes individual synapses.

## Local contacts

For center distance `d_ji`,

$$L_{ji}=\frac{N_i\exp[-d_{ji}^2/(2\sigma^2)]\,1(d_{ji}\leq600\,\mu m)}
{\sum_kN_k\exp[-d_{jk}^2/(2\sigma^2)]\,1(d_{jk}\leq600\,\mu m)},
\quad\sigma=200\,\mu m.$$

The zero-distance term is included. Contacts are conditioned on the modeled brain
domain. Source normalization is a retained assumption, not a measurement of local
connection strength. At high occupancy the simulator switches to an equivalent
susceptible-target proposal to avoid wasted attempts; the generator is unchanged.

## Population and initial condition

The population prior uses registered neuronal MERFISH detections from a reference
animal, a factor-20 section-sampling correction, and source-normalized Gaussian
smoothing with sigma 200 µm. Integer populations conserve the estimated total and
have a floor of one on the model domain. They are not a census of the pathology mice.

The fixed initial condition contains 1,000 affected cells in a Gaussian focus in
right caudoputamen (sigma 150 µm, cutoff 300 µm). The exact focus is provisional;
registered injection coordinates are not known. The loader reads the original
initial vector rather than reconstructing or moving it.

The historical center-versus-edge coordinate convention in population preparation
still needs a dedicated registration audit. This release preserves the frozen
prepared arrays; no coordinate shift is silently introduced during cleanup.

## Simulation and limitations

Independent realizations use seed `seed_base + 100003*r`. Time is in months in the
mouse experiments. A simulation reaching its attempt limit raises without returning
an incomplete trajectory. Reported Monte Carlo standard errors quantify numerical
uncertainty in ensemble means, not biological uncertainty between animals.

Exact simulation of the specified count model does not imply that the inferred
contacts, homogeneous rates, SIS mechanism, or observation mapping are biologically
exact. Current fits capture only some broad propagation patterns.
