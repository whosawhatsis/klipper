import sys, os, json, numpy as np
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
from extras.resonance_probe import HaltingContactProbe as H
import eval_knee as E, tangent, tangent_axes as TA
from anchor import prep
import knee as KN
d = json.load(open('fitviz/data.json'))
keymap = {list(d)[0]: '-22', list(d)[1]: '22b'}
for run, pts in d.items():
    for pt, probes in pts.items():
        x, y = pt.split(',')
        corpus = E.CORPUS[(keymap[run], 'smoo', '%02d' % int(x), '%02d' % int(y))]
        assert len(corpus) == len(probes)
        for p, probe in zip(probes, corpus):
            raw = [[H._vmin_edge(zs, a, 0.10) for a in amps] for zs, amps in probe]
            snr = [min((r[a][1] / r[a][2]) if r[a] else 0 for r in raw) for a in range(3)]
            kneevals, xonly = [], []
            for ri, (zs, amps) in enumerate(probe):
                r = TA.anchored_knee([(zs, a) for a in amps], snr, weight='keep', detail=True)
                kneevals.append(r[0] if r else None)
                t = tangent.tangent_knee(zs, amps[0], band=(.1, .9)); xonly.append(t[0] if t else None)
                for ax in range(3):
                    fit = None
                    if r:
                        if ax == r[2]:
                            dd = tangent.tangent_knee(zs, amps[ax], band=(.1, .9), detail=True)
                            ok = True
                        else:
                            z, yy = prep(zs, amps[ax])
                            dd = TA.signed_tangent(z, yy, r[0], detail=True)
                            ok = r[1][ax] is not None
                        if dd:
                            ma, ba = dd['air']
                            fit = dict(knee=round(dd['knee'] * 1e3, 2), zh=round(dd['zh'] * 1e3, 2), yh=round(float(dd['yh']), 4),
                                       mf=round(float(dd['mf']) / 1e3, 6), ma=round(float(ma) / 1e3, 6), ba=round(float(ba), 4),
                                       zt=round(float(dd['zt']) * 1e3, 2), ok=bool(ok), anchor=bool(ax == r[2]))
                    p['reps'][ri][ax]['fit'] = fit
            p['knee'] = round(float(np.mean(kneevals)) * 1e3, 2) if all(v is not None for v in kneevals) else None
            p['knee_x'] = round(float(np.mean(xonly)) * 1e3, 2) if all(v is not None for v in xonly) else None
            p['half'] = p['result']
json.dump(d, open('fitviz/data.json', 'w'), separators=(',', ':'))
for run, pts in d.items():
    print(run, {k: (np.std([p['knee'] for p in v if p['knee']], ddof=1).round(2), np.std([p['half'] for p in v], ddof=1).round(2)) for k, v in pts.items()})
