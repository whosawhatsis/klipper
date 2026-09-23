import numpy as np, sys, os
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
from extras.resonance_probe import HaltingContactProbe as H
import eval_knee as E, seg, tangent
from anchor import prep
V = ['half x', 'free 2-line x', 'tangent x pair', 'tangent x local2', 'tangent x local3', 'tangent x band20-80', 'tangent x band10-90']
out = {k: {} for k in V}; gap = []
for key, probes in E.CORPUS.items():
    acc = {k: [] for k in V}
    for probe in probes:
        raw = [[H._vmin_edge(zs, a, 0.10) for a in amps] for zs, amps in probe]
        c = H._combine_axes(raw, 15.)
        if not c or not all(r[0] for r in raw): continue
        vals = {k: [] for k in V}
        for (zs, amps), seed, rr in zip(probe, c, raw):
            vals['half x'].append(rr[0][0])
            z, y = prep(zs, amps[0]); zt = seg.turning_point(z, y, seed)
            f = seg.seg_fit(z, y, zt, seed, seed + 0.08) if zt is not None else None
            vals['free 2-line x'].append(f[0] if f else np.nan)
            for k, kw in (('tangent x pair', dict(slope='pair')), ('tangent x local2', dict(slope='local', nloc=2)),
                          ('tangent x local3', dict(slope='local', nloc=3)), ('tangent x band20-80', {}),
                          ('tangent x band10-90', dict(band=(.1, .9)))):
                t = tangent.tangent_knee(zs, amps[0], **kw)
                vals[k].append(t[0] if t else np.nan)
        for k in V: acc[k].append(np.mean(vals[k]))
        if np.isfinite(acc['tangent x band20-80'][-1]): gap.append((acc['tangent x band20-80'][-1] - acc['half x'][-1]) * 1e3)
    for k in V:
        v = [x for x in acc[k] if np.isfinite(x)]
        if len(v) >= 5: out[k][key] = (np.std(v, ddof=1) * 1e3, len(acc[k]) - len(v))
ks = sorted(out['half x'])
print('%-22s' % '' + ''.join('%8s' % (k[0][-1] + k[1][0] + k[2] + k[3]) for k in ks) + '  median  lost')
for n in V:
    r = out[n]; print('%-22s' % n + ''.join('%8.2f' % r[k][0] if k in r else '%8s' % '-' for k in ks) + '  %6.2f  %d' % (np.median([v[0] for v in r.values()]), sum(v[1] for v in r.values())))
print('knee - half (band20-80): median %.1fum, range %.1f..%.1f' % (np.median(gap), min(gap), max(gap)))
