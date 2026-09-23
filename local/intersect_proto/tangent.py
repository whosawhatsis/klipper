"""Tangent-at-half-way knee: pivot a line on the half-way crossing, slope from a
local derivative there, and intersect it with the fitted air line."""
import numpy as np
from anchor import ols


def tangent_knee(zs, amps, min_step=0.10, slope='band', band=(0.2, 0.8), nloc=2, iters=2, mf_override=None):
    z = np.asarray(zs, float); o = np.argsort(z); z = z[o]
    y = np.log(np.maximum(np.asarray(amps, float)[o], 1.))
    n = len(z)
    if n < 10:
        return None
    m = max(3, n // 4); h = 2
    s = np.array([np.median(y[max(0, i - h):i + h + 1]) for i in range(n)])
    air0 = float(np.median(y[-m:]))
    i0 = int(np.argmin(s)); step = air0 - s[i0]
    if step < min_step:
        return None
    half = air0 - .5 * step
    j = next((i for i in range(i0, n - 1) if y[i] - half < 0 <= y[i + 1] - half), None)
    if j is None:
        return None
    zh = z[j] + (half - y[j]) * (z[j + 1] - z[j]) / (y[j + 1] - y[j])
    # --- slope at the crossing (raw samples)
    if slope == 'pair':                          # the straddling pair only
        mf = (y[j + 1] - y[j]) / (z[j + 1] - z[j])
    elif slope == 'local':                       # nloc samples each side of the crossing
        idx = np.arange(max(i0, j - nloc + 1), min(n, j + nloc + 1))
        mf = ols(z[idx], y[idx])[0]
    else:                                        # the 20-80% band of the rising flank
        seg = np.arange(i0, n)
        frac = (y[seg] - s[i0]) / step
        # contiguous run of the flank around the crossing
        lo_i = j
        while lo_i > i0 and frac[lo_i - 1 - i0] >= band[0]:
            lo_i -= 1
        hi_i = j + 1
        while hi_i < n - 1 and frac[hi_i + 1 - i0] <= band[1]:
            hi_i += 1
        idx = np.arange(lo_i, hi_i + 1)
        if len(idx) < 2:
            return None
        mf = ols(z[idx], y[idx])[0] if len(idx) >= 3 else (y[idx[-1]] - y[idx[0]]) / (z[idx[-1]] - z[idx[0]])
    if mf_override is not None:
        mf_own, mf = mf, mf_override
    if mf <= 1e-6:
        return None
    # --- air line, fitted above the current knee estimate
    knee = zh + (air0 - half) / mf
    for _ in range(iters):
        am = z > knee + 0.005
        if am.sum() < 5:
            break
        ma, ba = ols(z[am], y[am])[:2]
        # line through (zh, half) with slope mf meets y = ma z + ba
        if abs(mf - ma) < 1e-9:
            return None
        knee = (ba - half + mf * zh) / (mf - ma)
    return knee, step, zh, mf
