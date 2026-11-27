"""Validated model construction; data locations are explicit caller inputs."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve

from ._kernels import full_cdf, normalize_values


@dataclass(frozen=True)
class SISModel:
    """Prepared sampler for a specified finite-population contact model."""

    arguments: tuple
    metadata: dict = field(default_factory=dict)
    local_source_strength: object = field(default=None, repr=False)
    local_target_weight: object = field(default=None, repr=False)

    @property
    def population(self):
        return self.arguments[1]

    @property
    def initial(self):
        return self.arguments[0]


def with_susceptibility(model, susceptibility):
    """Prepare the identical target-weighted generator without a global bound.

    Q_ji and L_ji are multiplied by chi_i. Their resulting source strengths
    are retained, not normalized away. The capacity N_i remains unchanged.
    """
    from ._kernels import local_weight_sums, weighted_conditional_cdf

    if model.local_source_strength is not None:
        raise ValueError("Susceptibility has already been prepared for this model")
    chi = np.asarray(susceptibility, dtype=float)
    if chi.shape != model.population.shape or not np.isfinite(chi).all() or np.any(chi < 0):
        raise ValueError("Susceptibility must be a finite nonnegative population-shaped field")
    args = list(model.arguments)
    target_cdf, component_mass = weighted_conditional_cdf(args[6], args[7], args[8], chi)
    source_cdf, source_mass = weighted_conditional_cdf(args[3], args[4], args[5], component_mass)
    args[2] = args[2] * source_mass
    args[5] = source_cdf
    args[8] = target_cdf
    if args[23]:
        args[11], local_strength = weighted_conditional_cdf(args[9], args[10], args[11], chi)
        local_target = None
    else:
        local_target = args[1] * chi
        local_strength, forward_max = local_weight_sums(
            args[1], local_target, args[19], args[20], args[21], args[22]
        )
        args[17] = forward_max
    if not np.isfinite(args[2]).all() or not np.isfinite(local_strength).all():
        raise ValueError("Nonfinite susceptibility-weighted source intensity")
    metadata = dict(
        model.metadata,
        target_susceptibility_prepared=True,
        weighted_local_sampler="forward; source strengths retained",
    )
    return SISModel(tuple(args), metadata, local_strength, local_target)


def _counts(population, initial):
    arrays = []
    for value in (population, initial):
        a = np.asarray(value)
        if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
            raise ValueError("Counts must be nonempty finite vectors")
        if np.any(a != np.floor(a)) or np.any(a > np.iinfo(np.int32).max):
            raise ValueError("Counts must be integers representable by int32")
        if np.any(a < 0):
            raise ValueError("Counts must be nonnegative")
        arrays.append(a.astype(np.int64))
    N, I = arrays
    if N.shape != I.shape or np.any(N <= 0) or np.any(I > N):
        raise ValueError("Require equal shapes and 0 <= initial <= positive population")
    return N, I


def conditional(matrix):
    """Normalize nonempty rows without modifying the caller's sparse matrix."""
    out = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    if np.any(out.data < 0) or not np.isfinite(out.data).all():
        raise ValueError("Contact weights must be finite and nonnegative")
    mass = normalize_values(out.data, out.indptr)
    return out, mass


def from_contacts(population, initial, tracer, local=None):
    """Construct a small/sparse graph model, with source-row contact matrices.

    Tracer row sums retain their strengths. Local rows must sum to one.
    If omitted, local contacts stay within each compartment (the identity).
    """
    N, I = _counts(population, initial)
    n = len(N)
    Q = sparse.csr_matrix(tracer, dtype=np.float64)
    L = sparse.eye(n, format="csr") if local is None else sparse.csr_matrix(local, dtype=np.float64)
    if Q.shape != (n, n) or L.shape != (n, n):
        raise ValueError("Contact matrices must have one source and target per compartment")
    S, strength = conditional(Q)
    _, mass = conditional(L)
    if not np.allclose(mass, 1, rtol=1e-12, atol=1e-12):
        raise ValueError("Local contact rows must sum to one")
    incoming = np.asarray(L.T @ N).ravel()
    potential = incoming / N
    R = (
        L.T.multiply(N[None, :])
        .multiply((1 / np.where(incoming > 0, incoming, 1))[:, None])
        .tocsr()
    )
    args = (
        I,
        N,
        strength,
        *full_cdf(S),
        *full_cdf(sparse.eye(n)),
        *full_cdf(L),
        *full_cdf(R),
        potential,
        np.ones(1),
        np.ones(1),
        np.ones(1),
        np.zeros((1, 3), dtype=np.int16),
        np.zeros((1, 1, 1), dtype=np.int32),
        np.zeros((1, 3), dtype=np.int16),
        np.ones(1),
        True,
    )
    return SISModel(args, {"kind": "explicit-contact-matrices", "compartments": n})


def local_offsets(sigma_um=200.0, voxel_um=100.0, cutoff_sigmas=3.0):
    """Discrete Gaussian contact offsets, including zero displacement."""
    if not all(np.isfinite(x) and x > 0 for x in (sigma_um, voxel_um, cutoff_sigmas)):
        raise ValueError("Kernel dimensions must be positive and finite")
    r = int(np.ceil(cutoff_sigmas * sigma_um / voxel_um))
    if r > 100:
        raise ValueError("Local kernel radius exceeds the supported memory budget")
    offsets = np.stack(np.meshgrid(*([np.arange(-r, r + 1)] * 3), indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    distance2 = (voxel_um * offsets) ** 2
    keep = distance2.sum(axis=1) <= (cutoff_sigmas * sigma_um) ** 2
    offsets = offsets[keep].astype(np.int16)
    p = np.exp(-distance2[keep].sum(axis=1) / (2 * sigma_um * sigma_um))
    cdf = np.cumsum(p) / p.sum()
    cdf[-1] = 1.0
    return offsets, cdf


def local_tables(N, coords, index, offsets, cdf):
    """Source and target sampling envelopes; no dense voxel contact matrix."""
    p = np.diff(np.r_[0.0, cdf])
    radius = int(abs(offsets).max())
    kernel = np.zeros((2 * radius + 1,) * 3)
    kernel[tuple((offsets + radius).T)] = p
    grid = np.zeros(index.shape)
    grid[tuple(coords.T)] = N
    z = fftconvolve(grid, kernel, mode="same")[tuple(coords.T)]
    if np.any(z <= 0):
        raise ValueError("Empty local neighborhood")
    back_weight = N / z
    backgrid = np.zeros(index.shape)
    backgrid[tuple(coords.T)] = back_weight
    potential = fftconvolve(backgrid, kernel, mode="same")[tuple(coords.T)]
    forward_max = maximum_filter(grid, size=2 * radius + 1, mode="constant")[tuple(coords.T)]
    back_max = maximum_filter(backgrid, size=2 * radius + 1, mode="constant")[tuple(coords.T)]
    if not all(
        np.isfinite(a).all() and np.all(a > 0)
        for a in (potential, back_weight, forward_max, back_max)
    ):
        raise ValueError("Invalid local sampling envelope")
    return potential, back_weight, forward_max, back_max


def load_reference(data_dir):
    """Load the frozen 100-um Allen/MERFISH reference from prepared files.

    This reads existing data and computes sampling tables in memory. It never
    modifies the input directory, including any historical cache files there.
    """
    folder = Path(data_dir).expanduser().resolve()
    required = [
        "merfish_population.npy",
        "fitting_initial_fine.npy",
        "geometry.npz",
        "raw_source_factors.npz",
        "raw_target_factors.npz",
    ]
    missing = [p for p in required if not (folder / p).is_file()]
    if missing:
        raise FileNotFoundError("Missing reference inputs: " + ", ".join(missing))
    N, initial = _counts(
        np.load(folder / required[0], allow_pickle=False),
        np.load(folder / required[1], allow_pickle=False),
    )
    with np.load(folder / "geometry.npz", allow_pickle=False) as g:
        coords, index = g["coords"], g["index"]
    if coords.shape != (len(N), 3) or index.ndim != 3:
        raise ValueError("Geometry does not match population")
    if not np.issubdtype(coords.dtype, np.integer) or not np.issubdtype(index.dtype, np.integer):
        raise ValueError("Geometry must contain integer indices")
    if np.any(coords < 0) or np.any(coords >= np.array(index.shape)):
        raise ValueError("Coordinates are outside the grid")
    if np.count_nonzero(index >= 0) != len(N) or not np.array_equal(
        index[tuple(coords.T)], np.arange(len(N))
    ):
        raise ValueError("Grid indices are not a one-to-one population mapping")
    U = sparse.load_npz(folder / "raw_source_factors.npz")
    V = sparse.load_npz(folder / "raw_target_factors.npz")
    if U.shape[0] != len(N) or V.shape[1] != len(N) or U.shape[1] != V.shape[0]:
        raise ValueError("Tracer factors do not match the reference domain")
    for factor in (U, V):
        if np.any(factor.data < 0) or not np.isfinite(factor.data).all():
            raise ValueError("Invalid raw tracer factor")
    outgoing = np.asarray(U @ (V @ np.ones(len(N)))).ravel()
    normalizer = float(np.dot(N, outgoing) / N.sum())
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("Invalid tracer normalization")
    H = V.T.astype(np.float64).multiply((1 / (N * normalizer))[:, None]).tocsr()
    Traw = U.T.astype(np.float64).multiply(N[None, :]).tocsr()
    T, mass = conditional(Traw)
    if np.any(mass <= 0):
        raise ValueError("Empty tracer component")
    H = H.multiply(mass[None, :]).tocsr()
    strength = np.asarray(H.sum(axis=1)).ravel()
    S, check = conditional(H)
    np.testing.assert_allclose(check, strength, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(np.dot(N, strength) / N.sum(), 1, rtol=2e-12)
    offsets, cdf = local_offsets()
    tables = local_tables(N, coords, index, offsets, cdf)
    empty = sparse.csr_matrix((1, 1))
    args = (
        initial,
        N,
        strength,
        *full_cdf(S),
        *full_cdf(T),
        *full_cdf(empty),
        *full_cdf(empty),
        *tables,
        coords,
        index,
        offsets,
        cdf,
        False,
    )
    return SISModel(
        args,
        {
            "kind": "allen-merfish-reference",
            "voxel_um": 100,
            "compartments": len(N),
            "cells": int(N.sum()),
            "normalizer": normalizer,
            "tracer": "D^-1 (U V)^T D / c",
            "local_sigma_um": 200.0,
            "local_cutoff_um": 600.0,
            "local_self_contacts": True,
        },
    )
