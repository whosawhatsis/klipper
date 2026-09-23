"""Seeded, anchored two-line intersection.

Seed  : deployed V-min half-way crossing (robust LOCATION, biased low on the flank).
Per axis: segmented least squares - independent air line (z > k) and flank line
          (z_turn <= z <= k), k chosen by minimum total SSE; report the lines'
          intersection with a delta-method se.
Anchor: axis with the smallest se, searched over k in (seed, seed + reach_up].
Guided: every axis refit with k restricted to anchor +- tol; accept if the two
        slopes differ by >= min_t standard errors and the intersection lands
        within tol of the anchor.  Inverse-variance combine.
"""
import numpy as np
from anchor import ols, intersect, prep
import knee


def turning_point(z, y, z_seed, reach_down=0.12):
    """Deepest point of the flank the seed sits on: first extremum below it."""
    s = knee.movmed(y)
    top = z >= z_seed + 0.08
    air = np.median(y[top]) if top.sum() >= 3 else np.median(y[-5:])
    below = np.where((z <= z_seed) & (z >= z_seed - reach_down))[0][::-1]
    if len(below) < 3:
        return None
    sgn = 1 if s[below[0]] > air else -1
    i_e = below[0]
    for j in below[1:]:
        if sgn * s[j] >= sgn * s[i_e]:
            i_e = j
        else:
            break
    return z[i_e]


def anchored_turn(z, y, z0, look=0.025, reach_down=0.12):
    """First extremum below z0 on the side the trace leaves air toward."""
    am = z > z0 + 0.005
    if am.sum() < 5:
        return None
    m, b, _, sd = ols(z[am], y[am])
    s = knee.movmed(y)
    dev = s - (m * z + b)
    near = (z <= z0) & (z >= z0 - look)
    if near.sum() < 2:
        return None
    sgn = 1 if np.mean(dev[near]) > 0 else -1
    below = np.where((z <= z0) & (z >= z0 - reach_down))[0][::-1]
    i_e = below[0]
    for j in below[1:]:
        if sgn * dev[j] >= sgn * dev[i_e]:
            i_e = j
        elif sgn * (dev[i_e] - dev[j]) > sd:
            break
    return z[i_e]


def seg_fit(z, y, z_lo, k_lo, k_hi, min_air=5, min_fl=3):
    """-> (zc, se, t, k) for the best split k in [k_lo, k_hi], or None."""
    best = None
    for k in z[(z >= k_lo) & (z <= k_hi)]:
        am = z > k
        fm = (z >= z_lo) & (z <= k)
        if am.sum() < min_air or fm.sum() < min_fl:
            continue
        A, F = ols(z[am], y[am]), ols(z[fm], y[fm])
        sse = (A[3] ** 2) * (am.sum() - 2) + (F[3] ** 2) * max(fm.sum() - 2, 0)
        if best is None or sse < best[0]:
            best = (sse, A, F, k)
    if best is None:
        return None
    r = intersect(best[1], best[2])
    if r is None:
        return None
    return r[0], r[1], r[2], best[3]


def seeded_edge(ramp_axes, z_seed, reach_up=0.08, tol=0.015, min_t=4., detail=False):
    data = [prep(zs, a) for zs, a in ramp_axes]
    fits = []
    for z, y in data:
        zt = turning_point(z, y, z_seed)
        f = None if zt is None else seg_fit(z, y, zt, z_seed, z_seed + reach_up)
        fits.append((zt, f))
    cand = [(f[1], a, f[0]) for a, (zt, f) in enumerate(fits)
            if f and f[2] >= min_t and z_seed - 0.01 <= f[0] <= z_seed + reach_up]
    if not cand:
        return None
    se0, anc, z0 = min(cand)
    per = []
    for a, (z, y) in enumerate(data):
        # the flank to fit is the one that leaves air at the ANCHOR, which on a
        # noisy axis need not be the one under the seed
        zt = anchored_turn(z, y, z0)
        if zt is None:
            per.append(None); continue
        f = seg_fit(z, y, zt, z0 - tol, z0 + tol)
        ok = f is not None and f[2] >= min_t and abs(f[0] - z0) <= tol
        per.append(f if ok else None)
    acc = [p for p in per if p]
    if not acc:
        return None
    w = np.array([1. / p[1] ** 2 for p in acc])
    zc = float(np.sum(w * np.array([p[0] for p in acc])) / np.sum(w))
    return (zc, per, anc) if detail else zc
