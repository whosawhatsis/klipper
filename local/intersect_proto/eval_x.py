import numpy as np, sys, os
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
from extras.resonance_probe import HaltingContactProbe as H
import eval_knee as E, seg
from anchor import prep
V = {'half x (V-min)': [], 'intersect x (seeded seg)': [], 'intersect, equal-weight helpers': [], 'intersect x, 5 air pts nearest': []}
out = {k: {} for k in V}
for key, probes in E.CORPUS.items():
    acc = {k: [] for k in V}
    for probe in probes:
        raw = [[H._vmin_edge(zs, a, 0.10) for a in amps] for zs, amps in probe]
        c = H._combine_axes(raw, 15.)
        if not c or not all(r[0] for r in raw): continue
        h, ix, eq, near = [], [], [], []
        for (zs, amps), seed, rr in zip(probe, c, raw):
            h.append(rr[0][0])
            z, y = prep(zs, amps[0]); zt = seg.turning_point(z, y, seed)
            f = seg.seg_fit(z, y, zt, seed, seed + 0.08) if zt is not None else None
            ix.append(f[0] if f else np.nan)
            r = seg.seeded_edge([(zs, a) for a in amps], seed, detail=True)
            eq.append(np.mean([p[0] for p in r[1] if p]) if r else np.nan)
        for k, v in (('half x (V-min)', h), ('intersect x (seeded seg)', ix), ('intersect, equal-weight helpers', eq)):
            acc[k].append(np.mean(v))
    for k in acc:
        v = [x for x in acc[k] if np.isfinite(x)]
        if len(v) >= 5: out[k][key] = np.std(v, ddof=1) * 1e3
ks = sorted(out['half x (V-min)'])
print('%-32s' % '' + ''.join('%8s' % (k[0][-1] + k[1][0] + k[2] + k[3]) for k in ks) + '  median')
for n, r in out.items():
    if r: print('%-32s' % n + ''.join('%8.2f' % r[k] if k in r else '%8s' % '-' for k in ks) + '  %6.2f' % np.median(list(r.values())))
