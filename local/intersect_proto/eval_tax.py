import numpy as np, sys, os, collections
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
from extras.resonance_probe import HaltingContactProbe as H
import eval_knee as E, tangent, tangent_axes as TA
V = ['tangent x only', 'anchored: fade', 'anchor kept + fade helpers', 'anchor kept + equal helpers']
out = {k: {} for k in V}; used = collections.Counter(); ancs = collections.Counter(); off = collections.defaultdict(list)
for key, probes in E.CORPUS.items():
    acc = {k: [] for k in V}
    for probe in probes:
        raw = [[H._vmin_edge(zs, a, 0.10) for a in amps] for zs, amps in probe]
        snr = [min((r[a][1] / r[a][2]) if r[a] else 0 for r in raw) for a in range(3)]
        vals = {k: [] for k in V}
        for zs, amps in probe:
            t = tangent.tangent_knee(zs, amps[0], band=(.1, .9)); vals['tangent x only'].append(t[0] if t else np.nan)
            axes = [(zs, a) for a in amps]
            for k, w in (('anchored: fade', 'fade'), ('anchor kept + fade helpers', 'keep'), ('anchor kept + equal helpers', 'keep_eq')):
                r = TA.anchored_knee(axes, snr, weight=w, detail=True)
                vals[k].append(r[0] if r else np.nan)
                if r and w == 'keep':
                    ancs['xyz'[r[2]]] += 1; used[''.join('xyz'[i] for i, p in enumerate(r[1]) if p) or '-'] += 1
                    for i, p in enumerate(r[1]):
                        if p: off['xyz'[i]].append((p[0] - r[0]) * 1e3)
        for k in V: acc[k].append(np.mean(vals[k]))
    for k in V:
        v = [x for x in acc[k] if np.isfinite(x)]
        if len(v) >= 5: out[k][key] = (np.std(v, ddof=1) * 1e3, len(acc[k]) - len(v))
ks = sorted(out['tangent x only'])
print('%-28s' % '' + ''.join('%8s' % (k[0][-1] + k[1][0] + k[2] + k[3]) for k in ks) + '  median  mean*  lost')
for n in V:
    r = out[n]; good = [r[k][0] for k in r if k[2:] != ('30', '70')]
    print('%-28s' % n + ''.join('%8.2f' % r[k][0] if k in r else '%8s' % '-' for k in ks) + '  %6.2f  %5.2f  %d' % (np.median([v[0] for v in r.values()]), np.mean(good), sum(v[1] for v in r.values())))
print('* mean excluding the (30,70) plate defect')
print('anchor:', dict(ancs), ' accepted sets:', dict(used.most_common(7)))
for a, v in sorted(off.items()): print('  %s: n=%d  offset from combined %+.1fum  IQR %.1fum' % (a, len(v), np.median(v), np.subtract(*np.percentile(v, [75, 25]))))
