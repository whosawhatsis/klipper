"""Per-call anchor chosen by how well the OTHER axes confirm it (no state across probes)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
from extras.resonance_probe import HaltingContactProbe as H
import eval_knee as E, tangent
from eval_q import axis_fits, guided
def ivw(ps):
    ps = [(s, k) for s, k in ps if np.isfinite(s) and s > 0]
    if not ps: return np.nan
    w = np.array([1 / s ** 2 for s, _ in ps]); return float(np.sum(w * [k for _, k in ps]) / w.sum())
def call_estimate(probe, pf, mode):
    """-> (value, anchor axis) for one verify call."""
    cands = [a for a in range(3) if any(f[a] and np.isfinite(f[a][1]) for f in pf)]
    best = None
    for a in cands:
        fits, n_conf, chi = [], 0, 0.
        for (zs, amps), f in zip(probe, pf):
            if not f[a] or not np.isfinite(f[a][1]): continue
            z0, s0 = f[a]; fits.append((s0, z0))
            for b, p in enumerate(guided(zs, amps, z0)):
                if p and b != a and np.isfinite(p[1]):
                    fits.append((p[1], p[0])); n_conf += 1
                    chi += ((p[0] - z0) / np.hypot(p[1], s0)) ** 2
        if not fits: continue
        v = ivw(fits)
        # rank: confirmations first, then agreement, then the anchor's own precision
        med_s = np.median([s for s, _ in fits[:1]])
        key = (-n_conf, chi / max(n_conf, 1)) if mode == 'confirm' else (np.median([f[a][1] for f in pf if f[a] and np.isfinite(f[a][1])]),)
        if best is None or key < best[0]: best = (key, v, a)
    return (best[1], best[2]) if best else (np.nan, None)
V = ['x alone', 'B: lowest sigma anchor', 'B: best-confirmed anchor']
out = {k: {} for k in V}; picks = {}
for key, probes in E.CORPUS.items():
    fits = [[axis_fits(zs, amps, [H._vmin_edge(zs, a, 0.10) for a in amps]) for zs, amps in probe] for probe in probes]
    acc = {k: [] for k in V}; pk = []
    for probe, pf in zip(probes, fits):
        xs = [tangent.tangent_knee(zs, amps[0], band=(.1, .9)) for zs, amps in probe]
        acc['x alone'].append(np.mean([t[0] if t else np.nan for t in xs]))
        acc['B: lowest sigma anchor'].append(call_estimate(probe, pf, 'sigma')[0])
        v, a = call_estimate(probe, pf, 'confirm'); acc['B: best-confirmed anchor'].append(v); pk.append('xyz'[a] if a is not None else '-')
    picks[key] = ''.join(pk)
    for k in V:
        v = [x for x in acc[k] if np.isfinite(x)]
        if len(v) >= 5: out[k][key] = (np.std(v, ddof=1) * 1e3, len(acc[k]) - len(v))
ks = sorted(out['x alone'])
print('%-28s' % '' + ''.join('%8s' % (k[0][-1] + k[1][0] + k[2] + k[3]) for k in ks) + '  median  mean*  lost')
for n in V:
    r = out[n]; good = [r[k][0] for k in r if k[2:] != ('30', '70')]
    print('%-28s' % n + ''.join('%8.2f' % r[k][0] if k in r else '%8s' % '-' for k in ks) + '  %6.2f  %5.2f  %d' % (np.median([v[0] for v in r.values()]), np.mean(good), sum(v[1] for v in r.values())))
print('* mean excluding (30,70)')
for k in ks: print('  confirmed anchor %-24s %s' % (k, picks[k]))
