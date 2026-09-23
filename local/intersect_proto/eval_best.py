import numpy as np, sys, os, collections
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
from extras.resonance_probe import HaltingContactProbe as H
import eval_knee as E, tangent, tangent_axes as TA
V = ['x alone', 'best axis per probe', 'best axis per ramp', 'anchor + fade helpers']
out = {k: {} for k in V}; pick = collections.Counter()
for key, probes in E.CORPUS.items():
    acc = {k: [] for k in V}
    for probe in probes:
        raw = [[H._vmin_edge(zs, a, 0.10) for a in amps] for zs, amps in probe]
        snr_probe = [min((r[a][1] / r[a][2]) if r[a] else 0 for r in raw) for a in range(3)]
        vals = {k: [] for k in V}
        for (zs, amps), rr in zip(probe, raw):
            t = tangent.tangent_knee(zs, amps[0], band=(.1, .9)); vals['x alone'].append(t[0] if t else np.nan)
            # per probe: the anchor chosen on the probe's worst-rep step/noise
            r = TA.anchored_knee([(zs, a) for a in amps], snr_probe, weight='keep', detail=True)
            vals['anchor + fade helpers'].append(r[0] if r else np.nan)
            z0 = None
            for a in sorted(range(3), key=lambda a: -snr_probe[a]):
                if snr_probe[a] < 15: break
                t = tangent.tangent_knee(zs, amps[a], band=(.1, .9))
                if t: z0 = t[0]; break
            vals['best axis per probe'].append(z0 if z0 is not None else np.nan)
            # per ramp: this ramp's own step/noise
            snr_ramp = [(rr[a][1] / rr[a][2]) if rr[a] else 0 for a in range(3)]
            z0 = None
            for a in sorted(range(3), key=lambda a: -snr_ramp[a]):
                if snr_ramp[a] < 15: break
                t = tangent.tangent_knee(zs, amps[a], band=(.1, .9))
                if t: z0 = t[0]; pick['xyz'[a]] += 1; break
            vals['best axis per ramp'].append(z0 if z0 is not None else np.nan)
        for k in V: acc[k].append(np.mean(vals[k]))
    for k in V:
        v = [x for x in acc[k] if np.isfinite(x)]
        if len(v) >= 5: out[k][key] = (np.std(v, ddof=1) * 1e3, len(acc[k]) - len(v))
ks = sorted(out['x alone'])
print('%-24s' % '' + ''.join('%8s' % (k[0][-1] + k[1][0] + k[2] + k[3]) for k in ks) + '  median  mean*  lost')
for n in V:
    r = out[n]; good = [r[k][0] for k in r if k[2:] != ('30', '70')]
    print('%-24s' % n + ''.join('%8.2f' % r[k][0] if k in r else '%8s' % '-' for k in ks) + '  %6.2f  %5.2f  %d' % (np.median([v[0] for v in r.values()]), np.mean(good), sum(v[1] for v in r.values())))
print('* mean excluding (30,70);  per-ramp picks:', dict(pick))
