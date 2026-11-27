"""Exact SIS/SIR count CTMC kernel, extended from the research implementation.

Source rows, destination columns. Null-event thinning preserves the specified
finite-population process; the kernel does not use a mean-field closure.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def cdf_index(cdf, u, lo, hi):
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] <= u:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def tree_add(tree, i, change):
    j = i + 1
    while j < len(tree):
        tree[j] += change
        j += j & -j


@njit(cache=True)
def tree_find(tree, rank):
    i = 0
    bit = 1
    while bit * 2 < len(tree):
        bit *= 2
    while bit:
        j = i + bit
        if j < len(tree) and tree[j] <= rank:
            rank -= tree[j]
            i = j
        bit //= 2
    return i


@njit(cache=True)
def row_cumsum(data, indptr):
    result = np.empty(len(data), dtype=np.float64)
    for row in range(len(indptr) - 1):
        mass = 0.0
        for k in range(indptr[row], indptr[row + 1]):
            mass += data[k]
            result[k] = mass
    return result


@njit(cache=True)
def draw_row(indptr, indices, cdf, row):
    lo, hi = indptr[row], indptr[row + 1]
    u = np.random.random()
    if lo == hi or u >= cdf[hi - 1]:
        return -1
    return indices[cdf_index(cdf, u, lo, hi)]


@njit(cache=True)
def normalize_values(values, indptr):
    masses = np.zeros(len(indptr) - 1)
    for j in range(len(masses)):
        mass = 0.0
        for k in range(indptr[j], indptr[j + 1]):
            mass += values[k]
        masses[j] = mass
        if mass > 0:
            for k in range(indptr[j], indptr[j + 1]):
                values[k] /= mass
    return masses


@njit(cache=True)
def sample_neighbor(a, coords, index, offsets, cdf, weights, bounds):
    while True:
        d = cdf_index(cdf, np.random.random(), 0, len(cdf))
        x, y, z = coords[a] + offsets[d]
        if (
            x < 0
            or y < 0
            or z < 0
            or x >= index.shape[0]
            or y >= index.shape[1]
            or z >= index.shape[2]
        ):
            continue
        b = index[x, y, z]
        if b >= 0 and np.random.random() * bounds[a] < weights[b]:
            return b


@njit(cache=True)
def finish_conditional_cdf(cdf, indptr):
    for j in range(len(indptr) - 1):
        lo, hi = indptr[j], indptr[j + 1]
        if hi > lo:
            mass = cdf[hi - 1]
            if mass <= 0:
                raise ValueError("Nonpositive conditional row")
            for k in range(lo, hi):
                cdf[k] /= mass


def full_cdf(matrix):
    matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    cdf = row_cumsum(np.asarray(matrix.data, dtype=np.float64), matrix.indptr)
    finish_conditional_cdf(cdf, matrix.indptr)
    return matrix.indptr, matrix.indices, cdf


@njit(cache=True)
def fenwick_build(weights):
    n = len(weights)
    tree = np.zeros(n + 1, dtype=weights.dtype)
    tree[1:] = weights
    for j in range(1, n + 1):
        parent = j + (j & -j)
        if parent <= n:
            tree[parent] += tree[j]
    return tree


@njit(cache=True)
def run_strength(
    initial,
    N,
    strength,
    sp,
    si,
    sc,
    tp,
    ti,
    tc,
    lp,
    li,
    lc,
    rp,
    ri,
    rc,
    potential,
    back_weight,
    forward_max,
    back_max,
    coords,
    index,
    offsets,
    cdf,
    sparse_local,
    times,
    betaT,
    betaL,
    gamma,
    seed,
    max_events=100000000,
    sir=False,
    removed_out=None,
):
    if min(betaT, betaL, gamma) < 0 or not np.isfinite(betaT + betaL + gamma):
        raise ValueError("Invalid rates")
    if np.any(N <= 0) or np.any(initial < 0) or np.any(initial > N):
        raise ValueError("Invalid populations")
    if np.any(strength < 0) or not np.all(np.isfinite(strength)):
        raise ValueError("Invalid tracer intensities")
    if np.any(times < 0) or np.any(np.diff(times) < 0):
        raise ValueError("Invalid times")
    np.random.seed(seed)
    I = initial.copy()
    S = N - I
    tree = fenwick_build(I)
    tracer_weights = strength * I
    tracer_tree = fenwick_build(tracer_weights)
    susceptible_weights = S * potential
    susceptible_tree = fenwick_build(susceptible_weights)
    total = I.sum()
    tracer_total = tracer_weights.sum()
    bound = susceptible_weights.sum()
    out = np.zeros((len(times), len(N)), dtype=np.int32)
    t = 0.0
    obs = 0
    events = 0
    changes = 0
    while obs < len(times):
        reverse = bound < total
        localrate = betaL * max(0.0, min(float(total), bound))
        tracer_rate = betaT * max(0.0, tracer_total)
        rate = gamma * total + tracer_rate + localrate
        if rate <= 0 or total == 0:
            for k in range(obs, len(times)):
                out[k] = I
                if removed_out is not None:
                    removed_out[k] = N - I - S
            break
        t += np.random.exponential(1 / rate)
        while obs < len(times) and times[obs] < t:
            out[obs] = I
            if removed_out is not None:
                removed_out[obs] = N - I - S
            obs += 1
        if obs == len(times):
            break
        if events >= max_events:
            raise RuntimeError("Strength CTMC event limit reached")
        events += 1
        u = np.random.random() * rate
        b = -1
        change = 0
        if u < gamma * total:
            b = tree_find(tree, np.random.random() * total)
            change = -1
        elif u < gamma * total + tracer_rate:
            a = tree_find(tracer_tree, np.random.random() * tracer_total)
            if a >= len(N) or I[a] == 0 or strength[a] == 0:
                continue
            e = draw_row(sp, si, sc, a)
            if e < 0:
                raise ValueError("Positive-strength source has no tracer route")
            b = draw_row(tp, ti, tc, e)
            if b >= 0 and np.random.random() * N[b] >= N[b] - S[b]:
                change = 1
        elif reverse:
            b = tree_find(susceptible_tree, np.random.random() * bound)
            if b >= len(N) or S[b] <= 0:
                continue
            if sparse_local:
                a = draw_row(rp, ri, rc, b)
            else:
                a = sample_neighbor(b, coords, index, offsets, cdf, back_weight, back_max)
            if a >= 0 and np.random.random() * N[a] < I[a]:
                change = 1
        else:
            a = tree_find(tree, np.random.random() * total)
            if sparse_local:
                b = draw_row(lp, li, lc, a)
            else:
                b = sample_neighbor(a, coords, index, offsets, cdf, N, forward_max)
            if b >= 0 and np.random.random() * N[b] >= N[b] - S[b]:
                change = 1
        if change:
            I[b] += change
            tree_add(tree, b, change)
            total += change
            changes += 1
            delta = change * strength[b]
            tree_add(tracer_tree, b, delta)
            tracer_total += delta
            # Infection consumes S. SIS clearance replenishes S; SIR removal does not.
            susceptible_change = -change if change > 0 or not sir else 0
            S[b] += susceptible_change
            delta = susceptible_change * potential[b]
            tree_add(susceptible_tree, b, delta)
            bound += delta
    return out, events, changes
