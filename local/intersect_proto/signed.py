"""Direction-agnostic anchor: first departure from air (either sign) -> pivoted tangent knee.

Per verify call: every axis of every down ramp gets an independent signed fit and a predicted
sigma; the anchor is the axis with the lowest median sigma across the call's ramps; every axis is
then refit guided at the anchor height; one inverse-variance average over the call.
"""
import numpy as np
import knee as KN
from anchor import prep
from tangent_axes import signed_tangent
import quality as Q


def signed_fit(zs, amps, min_exc=0.10, min_snr=5.):
    """Independent fit on one axis of one ramp -> (knee, sigma) or None."""
    d = KN.knee_edge(zs, amps, min_exc=min_exc, detail=True)
    if d is None:
        return None
    z, y = prep(zs, amps)
    z_dep = d['z'][d['i_d']]                   # where the trace last sat on the air line
    f = signed_tangent(z, y, z_dep, detail=True)
    if f is None or abs(f['exc']) < min_exc or f['snr'] < min_snr:
        return None
    s = Q.knee_sigma(z, y, f)
    return (f['knee'], s) if np.isfinite(s) else None


def guided_fits(zs, amps_axes, z0, tol=0.015):
    out = []
    for a in amps_axes:
        z, y = prep(zs, a)
        f = signed_tangent(z, y, z0, detail=True)
        if f is None or abs(f['knee'] - z0) > tol:
            out.append(None); continue
        s = Q.knee_sigma(z, y, f)
        out.append((f['knee'], s) if np.isfinite(s) else None)
    return out


def call_estimate(probe):
    """probe: [(zs, [amps_x, amps_y, amps_z])] down ramps of ONE verify call -> (z, anchor) ."""
    ind = [[signed_fit(zs, a) for a in amps] for zs, amps in probe]
    med = [np.median([r[a][1] for r in ind if r[a]] or [np.inf]) for a in range(3)]
    anc = int(np.argmin(med))
    if not np.isfinite(med[anc]):
        return np.nan, None
    fits = []
    for (zs, amps), r in zip(probe, ind):
        if not r[anc]:
            continue
        z0, s0 = r[anc]
        fits.append((s0, z0))
        for b, p in enumerate(guided_fits(zs, amps, z0)):
            if p and b != anc:
                fits.append((p[1], p[0]))
    if not fits:
        return np.nan, anc
    w = np.array([1. / s ** 2 for s, _ in fits])
    return float(np.sum(w * np.array([k for _, k in fits])) / w.sum()), anc
