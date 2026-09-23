import numpy as np, sys, os
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
import eval_knee as E, tangent
CFG = {'band10-90': dict(band=(.1, .9)), 'band20-80': {}, 'local3': dict(slope='local', nloc=3)}
out = {}
for name, kw in CFG.items():
    for pool in (False, True):
        r = {}
        for key, probes in E.CORPUS.items():
            vals = []
            for probe in probes:
                t = [tangent.tangent_knee(zs, amps[0], **kw) for zs, amps in probe]
                if any(x is None for x in t): continue
                if pool:
                    mf = np.mean([x[3] for x in t])
                    t = [tangent.tangent_knee(zs, amps[0], mf_override=mf, **kw) for zs, amps in probe]
                    if any(x is None for x in t): continue
                vals.append(np.mean([x[0] for x in t]))
            if len(vals) >= 5: r[key] = np.std(vals, ddof=1) * 1e3
        out['%s%s' % (name, ' pooled slope' if pool else '')] = r
ks = sorted(next(iter(out.values())))
print('%-26s' % '' + ''.join('%8s' % (k[0][-1] + k[1][0] + k[2] + k[3]) for k in ks) + '  median')
for n, r in out.items(): print('%-26s' % n + ''.join('%8.2f' % r.get(k, np.nan) for k in ks) + '  %6.2f' % np.median(list(r.values())))
