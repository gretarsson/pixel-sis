"""Independent finite-state master equations check the CTMC implementation."""

import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from pixel_sis import Rates, from_contacts, simulate_ensemble
from pixel_sis._kernels import run_strength


def generator(population, contact, gamma):
    states = np.array(list(itertools.product(*(range(int(n) + 1) for n in population))))
    lookup = {tuple(x): i for i, x in enumerate(states)}
    G = np.zeros((len(states), len(states)))
    for k, x in enumerate(states):
        birth = (1 - x / population) * (contact.T @ x)
        death = gamma * x
        for j in range(len(population)):
            for step, rate in [(1, birth[j]), (-1, death[j])]:
                if rate > 0:
                    destination = x.copy()
                    destination[j] += step
                    G[k, lookup[tuple(destination)]] += rate
                    G[k, k] -= rate
    return states, lookup, G


@pytest.mark.parametrize("initial", [np.array([1, 0, 1]), np.array([2, 3, 1])])
def test_ensemble_matches_master_equation(initial):
    N = np.array([2, 3, 2])
    Q = np.array([[0.05, 0.1, 0.05], [0, 0, 0], [0.9, 0.3, 1.8]])
    L = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.3, 0.5]])
    rates = Rates(1.1, 2.3, 0.7)
    model = from_contacts(N, initial, Q, L)
    states, lookup, G = generator(N, rates.beta_tracer * Q + rates.beta_local * L, rates.gamma)
    expected = expm(G * 1.5)[lookup[tuple(initial)]] @ states
    samples = np.array(
        [
            run_strength(
                *model.arguments, np.array([1.5]), *rates.values(), 20261210 + 7919 * r, 100000
            )[0][-1]
            for r in range(6000)
        ]
    )
    se = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    assert np.all(np.abs(samples.mean(axis=0) - expected) < 5 * se)


def test_microscopic_lumping_identity():
    labels = np.array([0, 0, 1, 1])
    N = np.array([2, 2])
    Q = np.array([[0.1, 0.3], [1.0, 2.0]])
    A = Q[labels[:, None], labels[None, :]] / N[labels][None, :]
    for state in itertools.product([0, 1], repeat=4):
        x = np.array(state)
        counts = np.bincount(labels, weights=x)
        microscopic = np.bincount(labels, weights=(1 - x) * (A.T @ x))
        np.testing.assert_allclose(microscopic, (1 - counts / N) * (Q.T @ counts), atol=1e-14)


def test_no_spontaneous_infection_and_zero_rates():
    for initial, rates in [([0, 0], Rates(1, 1, 0.1)), ([1, 2], Rates(0, 0, 0))]:
        model = from_contacts([2, 3], initial, np.eye(2))
        result = simulate_ensemble(model, [0, 1, 10], rates, runs=3, seed=23)
        np.testing.assert_array_equal(result["count_mean"], np.tile(initial, (3, 1)))
        assert not result["events"].any()


def test_event_limit_does_not_return_partial_output():
    model = from_contacts([10], [5], [[1.0]])
    with pytest.raises(RuntimeError, match="event limit"):
        simulate_ensemble(model, [100.0], Rates(10.0, 0, 0), runs=2, seed=5, max_events=1)


@pytest.mark.parametrize("times", [[0, float("nan")], [1, 0], [-1], [], [float("inf")]])
def test_invalid_times_rejected(times):
    model = from_contacts([3], [1], [[1.0]])
    with pytest.raises(ValueError, match="Times"):
        simulate_ensemble(model, times, Rates(0.2, 0, 0.1), runs=2)


@pytest.mark.parametrize("population,initial", [([0], [0]), ([2], [3]), ([2.5], [1]), ([2], [-1])])
def test_invalid_populations_rejected(population, initial):
    with pytest.raises(ValueError):
        from_contacts(population, initial, [[1.0]])


def test_empty_incoming_local_columns_supported():
    model = from_contacts([2, 2], [1, 1], np.zeros((2, 2)), [[1.0, 0.0], [1.0, 0.0]])
    result = simulate_ensemble(model, [0, 2], Rates(0, 1, 0), runs=32)
    np.testing.assert_array_equal(result["count_mean"][:, 1], [1.0, 1.0])


def test_global_rng_and_results_are_reproducible():
    model = from_contacts([5, 7], [2, 1], [[0.2, 0.8], [0.7, 0.3]])
    kwargs = dict(runs=8, seed=123, observation=np.eye(2))
    first = simulate_ensemble(model, [0, 1, 3], Rates(0.4, 0.1, 0.2), **kwargs)
    second = simulate_ensemble(model, [0, 1, 3], Rates(0.4, 0.1, 0.2), **kwargs)
    for key in ("count_mean", "count_mc_se", "total_samples", "density_samples", "events"):
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_allclose(first["count_mean"], first["density_samples"].mean(0), atol=1e-14)
    np.testing.assert_allclose(
        first["count_mc_se"], first["density_samples"].std(0, ddof=1) / np.sqrt(8)
    )
