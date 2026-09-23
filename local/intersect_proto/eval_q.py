import numpy as np, sys, os, collections
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
from extras.resonance_probe import HaltingContactProbe as H
import eval_knee as E, tangent, tangent_axes as TA, quality as Q
from anchor import prep
TOL = 0.015
FIRST = '--first' in sys.argv

def axis_fits(zs, amps, rr):
    """Independent tangent fit + sigma per axis (gated on the deployed step/noise)."""
    out = []
    for a in range(3):
        if not rr[a] or rr[a][1] / rr[a][2] < 15:
            out.append(None); continue
        f = tangent.tangent_knee(zs, amps[a], band=(.1, .9), detail=True)
        if f is None:
            out.append(None); continue
        z, y = prep(zs, amps[a])
        out.append((f['knee'], Q.knee_sigma(z, y, f)))
    return out

def guided(zs, amps, z0):
    """Every axis fitted at the anchor height -> [(knee, sigma) or None]."""
    out = []
    for a in range(3):
        z, y = prep(zs, amps[a])
        f = TA.signed_tangent(z, y, z0, detail=True)
        if f is None or abs(f['knee'] - z0) > TOL:
            out.append(None); continue
        out.append((f['knee'], Q.knee_sigma(z, y, f)))
    return out

def ivw(pairs):
    pairs = [p for p in pairs if p and np.isfinite(p[1]) and p[1] > 0]
    if not pairs: return None
    w = np.array([1 / p[1] ** 2 for p in pairs]); return float(np.sum(w * [p[0] for p in pairs]) / w.sum())

if __name__ == '__main__':
    V = ['x alone', 'best by sigma, per ramp', 'inverse-variance, independent',
         'best + 2nd (guided), equal', 'best + 2nd (guided), inv-var', 'best + all guided, inv-var']
    out = {k: {} for k in V}; picks = collections.Counter(); seconds = collections.Counter(); sig = collections.defaultdict(list)
    for key, probes in E.CORPUS.items():
        acc = {k: [] for k in V}
        for probe in probes:
            vals = {k: [] for k in V}
            for zs, amps in probe:
                rr = [H._vmin_edge(zs, a, 0.10) for a in amps]
                t = tangent.tangent_knee(zs, amps[0], band=(.1, .9)); vals['x alone'].append(t[0] if t else np.nan)
                ind = axis_fits(zs, amps, rr)
                cand = [(p[1], a, p[0]) for a, p in enumerate(ind) if p and np.isfinite(p[1])]
            # contact begins at the FIRST departure from air: a well-determined knee far
            # below another is fitting a later flank (z's fall under its bump at (60,45))
            if FIRST:
                good = [c for c in cand if c[0] <= 0.005]
                if good:
                    top = max(c[2] for c in good)
                    cand = [c for c in cand if c[2] >= top - TOL]
                    ind = [p if (p and p[0] >= top - TOL) else None for p in ind]
                if not cand:
                    for k in V[1:]: vals[k].append(np.nan)
                    continue
                s0, best, z0 = min(cand); picks['xyz'[best]] += 1
                for a, p in enumerate(ind):
                    if p and np.isfinite(p[1]): sig['xyz'[a]].append(p[1] * 1e3)
                vals['best by sigma, per ramp'].append(z0)
                vals['inverse-variance, independent'].append(ivw(ind))
                g = guided(zs, amps, z0)
                helpers = sorted([(p[1], a, p[0]) for a, p in enumerate(g) if p and a != best and np.isfinite(p[1])])
                if helpers:
                    s2, a2, k2 = helpers[0]; seconds['xyz'[a2]] += 1
                    vals['best + 2nd (guided), equal'].append((z0 + k2) / 2)
                    vals['best + 2nd (guided), inv-var'].append(ivw([(z0, s0), (k2, s2)]))
                else:
                    seconds['none'] += 1
                    vals['best + 2nd (guided), equal'].append(z0); vals['best + 2nd (guided), inv-var'].append(z0)
                vals['best + all guided, inv-var'].append(ivw([(z0, s0)] + [(k, s) for s, a, k in helpers]))
            for k in V: acc[k].append(np.mean(vals[k]) if not any(v is None for v in vals[k]) else np.nan)
        for k in V:
            v = [x for x in acc[k] if x is not None and np.isfinite(x)]
            if len(v) >= 5: out[k][key] = (np.std(v, ddof=1) * 1e3, len(acc[k]) - len(v))
    ks = sorted(out['x alone'])
    print('%-32s' % '' + ''.join('%8s' % (k[0][-1] + k[1][0] + k[2] + k[3]) for k in ks) + '  median  mean*  lost')
    for n in V:
        r = out[n]; good = [r[k][0] for k in r if k[2:] != ('30', '70')]
        print('%-32s' % n + ''.join('%8.2f' % r[k][0] if k in r else '%8s' % '-' for k in ks) + '  %6.2f  %5.2f  %d' % (np.median([v[0] for v in r.values()]), np.mean(good), sum(v[1] for v in r.values())))
    print('* mean excluding (30,70)')
    print('best axis by sigma:', dict(picks), '  second axis:', dict(seconds))
    for a, v in sorted(sig.items()): print('  predicted sigma %s: median %.1fum  (n=%d)' % (a, np.median(v), len(v)))
