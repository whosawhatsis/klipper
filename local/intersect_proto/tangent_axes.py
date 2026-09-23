"""Tangent-knee across all axes, anchored + guided.

Anchor: tangent.tangent_knee (V-min based) on the axis with the best step/noise.
Guided: for every axis, a SIGNED tangent knee located around the anchor z0 -
  air line above z0, flank = z0 down to its first turning point (either sign),
  half level between the air line and that extremum, crossing of it, slope over
  the 10-90% band, tangent pivoted on the crossing x air line.
"""
import numpy as np
from anchor import ols, prep
from seg import anchored_turn
import tangent


def signed_tangent(z, y, z0, gap=0.005, band=(0.1, 0.9), iters=2, detail=False):
    zt = anchored_turn(z, y, z0)
    if zt is None:
        return None
    am = z > z0 + gap
    if am.sum() < 5:
        return None
    ma, ba, _, sd = ols(z[am], y[am])
    fl = np.where((z >= zt) & (z <= z0 + gap))[0]
    if len(fl) < 3:
        return None
    dev = y[fl] - (ma * z[fl] + ba)
    i_ext = fl[0]                          # deepest flank sample = the turning point
    exc = y[i_ext] - (ma * z[i_ext] + ba)
    if abs(exc) < 3 * sd:
        return None
    frac = dev / exc                        # 0 at air, 1 at the turning point
    # half-way crossing on the flank, walking down from air
    zh = None
    for k in range(len(fl) - 1, 0, -1):
        a, b = frac[k], frac[k - 1]
        if a < .5 <= b:
            zh = z[fl[k]] + (.5 - a) * (z[fl[k - 1]] - z[fl[k]]) / (b - a)
            break
    if zh is None:
        return None
    sel = fl[(frac >= band[0]) & (frac <= band[1])]
    if len(sel) < 2:
        return None
    mf = ols(z[sel], y[sel])[0] if len(sel) >= 3 else (y[sel[-1]] - y[sel[0]]) / (z[sel[-1]] - z[sel[0]])
    yh = ma * zh + ba + .5 * exc            # half level at zh
    if abs(mf - ma) < 1e-9 or np.sign(mf - ma) != -np.sign(exc):
        return None                          # slope must point from air toward the extremum
    knee = (ba - yh + mf * zh) / (mf - ma)
    for _ in range(iters - 1):               # refit air above the knee itself
        am = z > knee + gap
        if am.sum() < 5:
            break
        ma, ba, _, sd = ols(z[am], y[am])
        if abs(mf - ma) < 1e-9:
            return None
        knee = (ba - yh + mf * zh) / (mf - ma)
    snr = abs(exc) / sd
    if detail:
        return dict(knee=knee, zh=zh, yh=yh, mf=mf, air=(ma, ba), zt=zt, exc=exc, sd=sd, snr=snr, sel=sel)
    return knee, snr, exc, sd


def anchored_knee(ramp_axes, vmin_snr, tol=0.015, min_snr=15., weight='fade', detail=False):
    """ramp_axes: [(zs, amps)] per axis for one ramp; vmin_snr: per-axis step/noise from the
    probe's _vmin_edge (worst rep), used only to choose the anchor axis."""
    # only an axis that passes the deployed step/noise gate may anchor - an
    # air-only ramp under a false halt must not find a 'knee' in noise
    order = [a for a in sorted(range(len(ramp_axes)), key=lambda a: -(vmin_snr[a] or 0))
             if (vmin_snr[a] or 0) >= min_snr]
    t = None
    for anc in order:
        t = tangent.tangent_knee(*ramp_axes[anc], band=(.1, .9))
        if t is not None:
            break
    if t is None:
        return None
    z0 = t[0]
    per = []
    for zs, a in ramp_axes:
        z, y = prep(zs, a)
        r = signed_tangent(z, y, z0)
        per.append(r if (r and abs(r[0] - z0) <= tol) else None)
    acc = [(a, p) for a, p in enumerate(per) if p]
    if not acc:
        return (z0, per, anc) if detail else z0
    if weight in ('keep', 'keep_eq'):
        # the anchor keeps its OWN tangent value at full weight; only the other
        # axes come in through the guided fits
        hs = [(a, p) for a, p in acc if a != anc]
        w = [min(max((p[1] - min_snr) / min_snr, 0.), 1.) if weight == 'keep' else 1. for _, p in hs]
        v = (z0 + sum(wi * p[0] for wi, (_, p) in zip(w, hs))) / (1. + sum(w))
        return (v, per, anc) if detail else v
    if weight == 'anchor':
        v = next((p[0] for a, p in acc if a == anc), z0)
    else:
        if weight == 'equal':
            w = [1.] * len(acc)
        else:                                # fade on the guided fit's own step/noise
            w = [min(max((p[1] - min_snr) / min_snr, 0.), 1.) for _, p in acc]
        if sum(w) == 0:
            v = z0
        else:
            v = sum(wi * p[0] for wi, (_, p) in zip(w, acc)) / sum(w)
    return (v, per, anc) if detail else v
