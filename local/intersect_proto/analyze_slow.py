"""Slow vs control verify ramp: per point x arm repeatability for each estimator."""
import sys, os, json, collections
import numpy as np
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
from extras.resonance_probe import HaltingContactProbe as H
import tangent
from eval_q import axis_fits, guided
from anchor import prep

def ivw(ps):
    ps = [(s, k) for s, k in ps if np.isfinite(s) and s > 0]
    if not ps: return np.nan
    w = np.array([1 / s ** 2 for s, _ in ps]); return float(np.sum(w * [k for _, k in ps]) / w.sum())


D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/code/klipper-tooling/probe_traces_2026-09-23')
man = [m for m in json.load(open(os.path.join(D, sys.argv[2] if len(sys.argv) > 2 else 'slow_run_manifest.json'))) if m['arm'] != 'warmup']


def load(f):
    meta, cols, rows = {}, None, []
    for ln in open(os.path.join(D, f)):
        if ln.startswith('#'):
            for t in ln[1:].split():
                if '=' in t: k, v = t.split('=', 1); meta[k] = v
            continue
        p = ln.strip().split(',')
        if cols is None: cols = p
        elif len(p) == len(cols) and p[2] == 'down': rows.append(p)
    ai = [cols.index(c) for c in ('amp_x', 'amp_y', 'amp_z')]
    probe = []
    for rep in sorted(set(r[3] for r in rows)):
        r = [x for x in rows if x[3] == rep]
        probe.append(([float(x[0]) for x in r], [[float(x[i]) for x in r] for i in ai]))
    return meta, probe


def call_B(probe):
    pf = [axis_fits(zs, amps, [H._vmin_edge(zs, a, 0.10) for a in amps]) for zs, amps in probe]
    med = [np.median([f[a][1] for f in pf if f[a] and np.isfinite(f[a][1])] or [np.inf]) for a in range(3)]
    anc = int(np.argmin(med))
    if not np.isfinite(med[anc]): return np.nan, '-', med
    fits = []
    for (zs, amps), f in zip(probe, pf):
        if not f[anc] or not np.isfinite(f[anc][1]): continue
        z0, s0 = f[anc]; fits.append((s0, z0))
        fits += [(p[1], p[0]) for b, p in enumerate(guided(zs, amps, z0)) if p and b != anc]
    return (ivw(fits) if fits else np.nan), 'xyz'[anc], med


res = collections.defaultdict(lambda: collections.defaultdict(list))
for m in man:
    key = (m['x'], m['y'], m['arm'])
    for f in m['files']:
        meta, probe = load(f)
        raw = [[H._vmin_edge(zs, a, 0.10) for a in amps] for zs, amps in probe]
        c = H._combine_axes(raw, 15.)
        res[key]['half-way (deployed)'].append(np.mean(c) if c else np.nan)
        xs = [tangent.tangent_knee(zs, amps[0], band=(.1, .9), detail=True) for zs, amps in probe]
        res[key]['contact, x alone'].append(np.mean([t['knee'] for t in xs]) if all(xs) else np.nan)
        v, anc, med = call_B(probe)
        res[key]['contact, per-call B'].append(v); res[key]['_anchor'].append(anc)
        res[key]['_sigma_x'].append(med[0] * 1e3)
        # samples on the x slope (10-90% band between the knee and the half-way point's flank)
        n_fl = []
        for (zs, amps), t in zip(probe, xs):
            if not t: continue
            z, y = prep(zs, amps[0])
            n_fl.append(int(np.sum((z >= t['zt']) & (z <= t['knee']))))
        res[key]['_flank_samples'].append(np.mean(n_fl) if n_fl else np.nan)
        res[key]['_n_down'].append(len(probe[0][0]))
EST = ['half-way (deployed)', 'contact, x alone', 'contact, per-call B']
print('%-14s %3s %8s %8s  ' % ('point/arm', 'n', 'flank n', 'pred sx') + ''.join('%22s' % e for e in EST) + '   anchors')
for key in sorted(res, key=lambda k: (k[0], k[1], k[2])):
    r = res[key]
    sds = []
    for e in EST:
        v = np.array([x for x in r[e] if np.isfinite(x)])
        sds.append('%6.2f um (n=%d)' % (np.std(v, ddof=1) * 1e3, len(v)) if len(v) > 1 else '-')
    print('%-14s %3d %8.1f %8.2f  ' % ('(%d,%d) %s' % key, len(r[EST[0]]), np.nanmean(r['_flank_samples']), np.nanmedian(r['_sigma_x']))
          + ''.join('%22s' % s for s in sds) + '   ' + ''.join(r['_anchor']))
# means of the contact point per arm, to see whether the ramp speed shifts the answer
for pt in sorted(set(k[:2] for k in res)):
    a = {arm: np.nanmean(res[pt + (arm,)]['contact, x alone']) * 1e3 for arm in ('slow', 'ctrl') if pt + (arm,) in res}
    if len(a) == 2: print('(%d,%d) contact mean slow %.1f ctrl %.1f  diff %+.1f um' % (pt + (a['slow'], a['ctrl'], a['slow'] - a['ctrl'])))

print()
pts = sorted(set(k[:2] for k in res))
for e in EST:
    sd = {arm: [] for arm in ('ctrl', 'slow')}
    for pt in pts:
        for arm in sd:
            v = np.array([x for x in res[pt + (arm,)][e] if np.isfinite(x)]) if pt + (arm,) in res else []
            sd[arm].append(np.std(v, ddof=1) * 1e3 if len(v) > 1 else np.nan)
    c, s_ = np.array(sd['ctrl']), np.array(sd['slow'])
    ok = np.isfinite(c) & np.isfinite(s_)
    print('%-22s ctrl median %5.2f mean %5.2f | slow median %5.2f mean %5.2f | slow better at %d/%d points'
          % (e, np.nanmedian(c), np.nanmean(c), np.nanmedian(s_), np.nanmean(s_), int(np.sum(s_[ok] < c[ok])), int(ok.sum())))
