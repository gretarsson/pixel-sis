"""SIR validation against an independently assembled finite-state generator."""

import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from pixel_sis import Rates, from_contacts, simulate_ensemble
from pixel_sis._kernels import run_strength
from pixel_sis.model import local_offsets, local_tables


def sir_generator(population, contact, gamma):
    per_node = [[(i, r) for i in range(n + 1) for r in range(n + 1 - i)] for n in population]
    states = np.array(list(itertools.product(*per_node)))
    index = {tuple(x.ravel()): k for k, x in enumerate(states)}
    G = np.zeros((len(states), len(states)))
    for k, x in enumerate(states):
        infected, removed = x[:, 0], x[:, 1]
        susceptible = population - infected - removed
        infection = susceptible / population * (contact.T @ infected)
        removal = gamma * infected
        for j in range(len(population)):
            for di, dr, rate in [(1, 0, infection[j]), (-1, 1, removal[j])]:
                if rate > 0:
                    destination = x.copy()
                    destination[j] += [di, dr]
                    G[k, index[tuple(destination.ravel())]] += rate
                    G[k, k] -= rate
    return states, index, G


@pytest.mark.parametrize("initial", [np.array([1, 0, 1]), np.array([2, 3, 2])])
@pytest.mark.parametrize("geometric_local", [False, True])
def test_sir_matches_master_equation(initial, geometric_local):
    N = np.array([2, 3, 2])
    Q = np.array([[0.05, 0.1, 0.05], [0, 0, 0], [0.9, 0.3, 1.8]])
    L = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.3, 0.5]])
    if geometric_local:
        positions = np.arange(3)
        L = np.exp(-((positions[:, None] - positions[None, :]) ** 2) / 2) * N[None, :]
        L /= L.sum(axis=1)[:, None]
    model = from_contacts(N, initial, Q, L)
    args = model.arguments
    if geometric_local:
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int16)
        grid = np.arange(3, dtype=np.int32).reshape(3, 1, 1)
        offsets, cdf = local_offsets(sigma_um=100, voxel_um=100)
        args = (
            *args[:15],
            *local_tables(N, coords, grid, offsets, cdf),
            coords,
            grid,
            offsets,
            cdf,
            False,
        )
    rates = Rates(1.1, 2.3, 0.7)
    states, lookup, G = sir_generator(N, rates.beta_tracer * Q + rates.beta_local * L, rates.gamma)
    start = np.column_stack([initial, np.zeros(3, dtype=int)])
    probability = expm(1.5 * G)[lookup[tuple(start.ravel())]]
    expected = probability @ states.reshape(len(states), -1)
    removed = np.zeros((1, 3), dtype=np.int32)
    samples = []
    for r in range(6000):
        infected = run_strength(
            *args, np.array([1.5]), *rates.values(), 20261127 + 7919 * r, 100000, True, removed
        )[0][-1]
        samples.append(np.column_stack([infected, removed[-1]]).ravel())
    samples = np.asarray(samples)
    se = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    assert np.all(np.abs(samples.mean(axis=0) - expected) < 5 * se + 1e-12)


def test_sir_without_removal_is_identical_to_sis():
    model = from_contacts([4, 7], [2, 3], [[0.4, 0.8], [1.1, 0.2]])
    kwargs = dict(runs=24, seed=818, observation=np.eye(2))
    sis = simulate_ensemble(model, [0, 1, 3], Rates(0.9, 1.2, 0), **kwargs)
    sir = simulate_ensemble(model, [0, 1, 3], Rates(0.9, 1.2, 0), dynamics="sir", **kwargs)
    for key in ("count_mean", "total_samples", "density_samples", "events"):
        np.testing.assert_array_equal(sis[key], sir[key])
    assert not sir["removed_total_samples"].any()


def test_removed_cells_neither_reinfect_nor_contribute_to_readout():
    N = np.array([5, 7])
    model = from_contacts(N, N, [[1.0, 1.0], [1.0, 1.0]])
    e = simulate_ensemble(
        model, [0, 1, 3, 10], Rates(2, 2, 1), runs=512, dynamics="sir", observation=np.eye(2)
    )
    assert not e["susceptible_total_samples"].any()
    assert np.all(np.diff(e["removed_total_samples"], axis=1) >= 0)
    np.testing.assert_array_equal(e["total_samples"] + e["removed_total_samples"], N.sum())
    np.testing.assert_allclose(e["density_samples"].mean(0), e["count_mean"], atol=1e-12)
    np.testing.assert_allclose(e["count_mean"] + e["removed_mean"], np.tile(N, (4, 1)), atol=1e-12)
    expected = N[None, :] * np.exp(-e["times"][:, None])
    assert np.all(np.abs(e["count_mean"] - expected) < 5 * e["count_mc_se"] + 0.01)


def test_unknown_dynamics_rejected():
    model = from_contacts([3], [1], [[1.0]])
    with pytest.raises(ValueError, match="Dynamics"):
        simulate_ensemble(model, [0, 1], Rates(1, 0, 0.2), dynamics="typo")
