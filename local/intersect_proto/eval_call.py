"""Anchor chosen per verify CALL (no state across probes)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
from extras.resonance_probe import HaltingContactProbe as H
import eval_knee as E, tangent
from eval_q import axis_fits, guided
V = ['x alone', 'fixed per point (reference)', 'A: per call, IVW per ramp', 'B: per call, one IVW over call',
     'B, anchor only (no helpers)']
out = {k: {} for k in V}; picks = {}
def ivw(ps):
    ps = [(s, k) for s, k in ps if np.isfinite(s) and s > 0]
    if not ps: return np.nan
    w = np.array([1 / s ** 2 for s, _ in ps]); return float(np.sum(w * [k for _, k in ps]) / w.sum())
for key, probes in E.CORPUS.items():
    fits = [[axis_fits(zs, amps, [H._vmin_edge(zs, a, 0.10) for a in amps]) for zs, amps in probe] for probe in probes]
    def first_departure(f, tol=0.015, sure=0.005):
        """Contact starts at the FIRST departure from air: drop fits far below a well-determined one."""
        good = [p[0] for p in f if p and np.isfinite(p[1]) and p[1] <= sure]
        if not good: return f
        top = max(good); return [p if (p and p[0] >= top - tol) else None for p in f]
    fits = [[first_departure(f) for f in pf] for pf in fits]
    medp = [np.median([f[a][1] for pf in fits for f in pf if f[a] and np.isfinite(f[a][1])] or [np.inf]) for a in range(3)]
    anc_pt = int(np.argmin(medp))
    acc = {k: [] for k in V}; pk = []
    for probe, pf in zip(probes, fits):
        med = [np.median([f[a][1] for f in pf if f[a] and np.isfinite(f[a][1])] or [np.inf]) for a in range(3)]
        anc = int(np.argmin(med)); pk.append('xyz'[anc] if np.isfinite(med[anc]) else '-')
        xs, ref, per_ramp, allfits, aonly = [], [], [], [], []
        for (zs, amps), f in zip(probe, pf):
            t = tangent.tangent_knee(zs, amps[0], band=(.1, .9)); xs.append(t[0] if t else np.nan)
            for a_, lst in ((anc_pt, ref), (anc, None)):
                if not f[a_] or not np.isfinite(f[a_][1]):
                    if lst is not None: lst.append(np.nan)
                    else: per_ramp.append(np.nan)
                    continue
                z0, s0 = f[a_]
                g = guided(zs, amps, z0)
                fs = [(s0, z0)] + [(p[1], p[0]) for a, p in enumerate(g) if p and a != a_]
                if lst is not None: lst.append(ivw(fs))
                else:
                    per_ramp.append(ivw(fs)); allfits += fs; aonly.append((s0, z0))
        acc['x alone'].append(np.mean(xs)); acc['fixed per point (reference)'].append(np.mean(ref))
        acc['A: per call, IVW per ramp'].append(np.mean(per_ramp))
        acc['B: per call, one IVW over call'].append(ivw(allfits) if allfits else np.nan)
        acc['B, anchor only (no helpers)'].append(ivw(aonly) if aonly else np.nan)
    picks[key] = ''.join(pk)
    for k in V:
        v = [x for x in acc[k] if np.isfinite(x)]
        if len(v) >= 5: out[k][key] = (np.std(v, ddof=1) * 1e3, len(acc[k]) - len(v))
ks = sorted(out['x alone'])
print('%-32s' % '' + ''.join('%8s' % (k[0][-1] + k[1][0] + k[2] + k[3]) for k in ks) + '  median  mean*  lost')
for n in V:
    r = out[n]; good = [r[k][0] for k in r if k[2:] != ('30', '70')]
    print('%-32s' % n + ''.join('%8.2f' % r[k][0] if k in r else '%8s' % '-' for k in ks) + '  %6.2f  %5.2f  %d' % (np.median([v[0] for v in r.values()]), np.mean(good), sum(v[1] for v in r.values())))
print('* mean excluding (30,70)')
for k in ks: print('  per-call anchor %-24s %s' % (k, picks[k]))
