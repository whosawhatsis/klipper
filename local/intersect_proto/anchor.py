"""Anchor-guided two-line intersection (user's proposal, 2026-09-23).

1. Fit every axis independently (knee.knee_edge); the axis whose intersection
   has the smallest standard error is the ANCHOR.
2. Given the anchor height z0, fit each axis's air line above z0 and its flank
   line below z0 (down to that axis's first turning point).  Accept the axis
   only if the two lines differ significantly and meet near z0.
3. Inverse-variance combine the accepted intersections; iterate once with the
   combined height as the new z0.
"""
import numpy as np
import knee


def ols(x, y):
    """-> (slope, intercept, cov 2x2, resid sd) or None."""
    n = len(x)
    if n < 3:
        return None
    X = np.c_[x, np.ones(n)]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    s2 = float(r @ r) / max(n - 2, 1)
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta[0], beta[1], cov, np.sqrt(s2)


def intersect(A, F):
    """Intersection of air line A and flank line F with delta-method se."""
    (ma, ba, ca, _), (mf, bf, cf, _) = A, F
    d = mf - ma
    if abs(d) < 1e-9:
        return None
    zc = (ba - bf) / d
    # zc = (ba-bf)/(mf-ma): gradient wrt (ma,ba) and (mf,bf)
    ga = np.array([zc / d, 1. / d])
    gf = np.array([-zc / d, -1. / d])
    var = ga @ ca @ ga + gf @ cf @ gf
    se_slope = np.sqrt(ca[0, 0] + cf[0, 0])
    return zc, float(np.sqrt(max(var, 0.))), abs(d) / se_slope


def prep(zs, amps):
    z = np.asarray(zs, float); o = np.argsort(z)
    return z[o], np.log(np.maximum(np.asarray(amps, float)[o], 1.))


def guided(z, y, z0, gap=0.005, reach=0.15, min_t=4.):
    """Two lines meeting near z0 on one axis -> (zc, se, t) or None."""
    A = ols(z[z > z0 + gap], y[z > z0 + gap])
    if A is None:
        return None
    ma, ba, _, sd = A
    s = knee.movmed(y)
    dev = s - (ma * z + ba)
    below = np.where((z <= z0) & (z >= z0 - reach))[0]
    if len(below) < 4:
        return None
    # direction: whichever side the trace sits on just below z0
    near = below[z[below] >= z0 - 0.03]
    sgn = 1 if np.mean(dev[near]) > 0 else -1
    # first turning point going deeper from z0
    idx = below[::-1]                       # z0 downward
    i_e = idx[0]
    for j in idx[1:]:
        if sgn * dev[j] >= sgn * dev[i_e]:
            i_e = j
        elif sgn * dev[i_e] - sgn * dev[j] > 2 * sd / np.sqrt(5):
            break                           # clearly turned
    fl = (z >= z[i_e]) & (z <= z0)
    F = ols(z[fl], y[fl])
    if F is None or fl.sum() < 3:
        return None
    r = intersect(A, F)
    if r is None or r[2] < min_t:
        return None
    return r


def combine(results, tol):
    ok = [(zc, se) for zc, se, _ in results if se > 0]
    if not ok:
        return None
    zc = np.array([o[0] for o in ok]); w = 1. / np.array([o[1] for o in ok]) ** 2
    return float(np.sum(w * zc) / np.sum(w))


def anchored_edge(ramp_axes, tol=0.015, iters=2, detail=False):
    """ramp_axes: [(zs, amps) per axis] for ONE ramp -> (edge, per-axis) or None."""
    data = [prep(zs, a) for zs, a in ramp_axes]
    best = None
    for a, (z, y) in enumerate(data):
        d = knee.knee_edge(z, np.exp(y), detail=True)
        if d is None:
            continue
        A = ols(z[d['i_d'] + 1:], y[d['i_d'] + 1:])
        F = ols(z[d['fl']], y[d['fl']])
        if A is None or F is None:
            continue
        r = intersect(A, F)
        if r and (best is None or r[1] < best[1]):
            best = (r[0], r[1], a)
    if best is None:
        return None
    z0 = best[0]
    per = [None] * len(data)
    for _ in range(iters):
        per = []
        for z, y in data:
            r = guided(z, y, z0)
            per.append(r if (r is not None and abs(r[0] - z0) <= tol) else None)
        c = combine([p for p in per if p], tol)
        if c is None:
            break
        z0 = c
    return (z0, per, best[2]) if detail else z0
