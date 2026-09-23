import numpy as np, sys, os, collections
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
from extras.resonance_probe import HaltingContactProbe as H
import eval_knee as E, tangent, tangent_axes as TA
from eval_q import axis_fits, guided
from anchor import prep
import quality as Q
V = ['x alone', 'fixed anchor', 'fixed anchor + 2nd guided (inv-var)', 'fixed anchor + all guided (inv-var)', 'fixed anchor + fade helpers']
out = {k: {} for k in V}; chosen = {}
for key, probes in E.CORPUS.items():
    fits = [[axis_fits(zs, amps, [H._vmin_edge(zs, a, 0.10) for a in amps]) for zs, amps in probe] for probe in probes]
    med = [np.median([f[a][1] for pf in fits for f in pf if f[a] and np.isfinite(f[a][1])] or [np.inf]) for a in range(3)]
    anc = int(np.argmin(med)); chosen[key] = 'xyz'[anc]
    acc = {k: [] for k in V}
    for probe, pf in zip(probes, fits):
        raw = [[H._vmin_edge(zs, a, 0.10) for a in amps] for zs, amps in probe]
        snr = [min((r[a][1] / r[a][2]) if r[a] else 0 for r in raw) for a in range(3)]
        vals = {k: [] for k in V}
        for (zs, amps), f in zip(probe, pf):
            t = tangent.tangent_knee(zs, amps[0], band=(.1, .9)); vals['x alone'].append(t[0] if t else np.nan)
            if not f[anc]:
                for k in V[1:]: vals[k].append(np.nan)
                continue
            z0, s0 = f[anc]
            vals['fixed anchor'].append(z0)
            g = guided(zs, amps, z0)
            hs = sorted([(p[1], p[0]) for a, p in enumerate(g) if p and a != anc and np.isfinite(p[1])])
            def ivw(ps):
                w = np.array([1 / s ** 2 for s, _ in ps]); return float(np.sum(w * [k for _, k in ps]) / w.sum())
            vals['fixed anchor + 2nd guided (inv-var)'].append(ivw([(s0, z0)] + hs[:1]))
            vals['fixed anchor + all guided (inv-var)'].append(ivw([(s0, z0)] + hs))
            sn = list(snr); order_snr = [anc] + [a for a in range(3) if a != anc]
            r = TA.anchored_knee([(zs, a) for a in amps], [99 if a == anc else (snr[a] if a != anc else 0) for a in range(3)], weight='keep', detail=True)
            vals['fixed anchor + fade helpers'].append(r[0] if r else np.nan)
        for k in V: acc[k].append(np.mean(vals[k]))
    for k in V:
        v = [x for x in acc[k] if np.isfinite(x)]
        if len(v) >= 5: out[k][key] = (np.std(v, ddof=1) * 1e3, len(acc[k]) - len(v))
ks = sorted(out['x alone'])
print('%-38s' % '' + ''.join('%8s' % (k[0][-1] + k[1][0] + k[2] + k[3]) for k in ks) + '  median  mean*  lost')
print('%-38s' % 'anchor chosen' + ''.join('%8s' % chosen[k] for k in ks))
for n in V:
    r = out[n]; good = [r[k][0] for k in r if k[2:] != ('30', '70')]
    print('%-38s' % n + ''.join('%8.2f' % r[k][0] if k in r else '%8s' % '-' for k in ks) + '  %6.2f  %5.2f  %d' % (np.median([v[0] for v in r.values()]), np.mean(good), sum(v[1] for v in r.values())))
print('* mean excluding (30,70)')
