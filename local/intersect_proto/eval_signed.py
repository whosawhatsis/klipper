"""Score the signed per-call estimator against the V-min-based one, per group, all corpora."""
import sys, os, json, glob, collections, io, contextlib
import numpy as np
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
from extras.resonance_probe import HaltingContactProbe as H
import signed
from eval_q import axis_fits, guided
T = os.path.expanduser('~/code/klipper-tooling/')

def ivw(ps):
    ps = [(s, k) for s, k in ps if np.isfinite(s) and s > 0]
    if not ps: return np.nan
    w = np.array([1 / s ** 2 for s, _ in ps]); return float(np.sum(w * [k for _, k in ps]) / w.sum())

def old_B(probe):
    pf = [axis_fits(zs, amps, [H._vmin_edge(zs, a, 0.10) for a in amps]) for zs, amps in probe]
    med = [np.median([f[a][1] for f in pf if f[a] and np.isfinite(f[a][1])] or [np.inf]) for a in range(3)]
    anc = int(np.argmin(med))
    if not np.isfinite(med[anc]): return np.nan
    fits = []
    for (zs, amps), f in zip(probe, pf):
        if not f[anc] or not np.isfinite(f[anc][1]): continue
        z0, s0 = f[anc]; fits.append((s0, z0))
        fits += [(p[1], p[0]) for b, p in enumerate(guided(zs, amps, z0)) if p and b != anc]
    return ivw(fits)

def load(path):
    meta, cols, rows = {}, None, []
    for ln in open(path):
        if ln.startswith('#'):
            for t in ln[1:].split():
                if '=' in t: k, v = t.split('=', 1); meta[k] = v
            continue
        p = ln.strip().split(',')
        if cols is None: cols = p
        elif len(p) == len(cols) and p[2] == 'down': rows.append(p)
    if 'amp_x' not in (cols or []): return meta, None
    ai = [cols.index(c) for c in ('amp_x', 'amp_y', 'amp_z')]
    probe = []
    for rep in sorted(set(r[3] for r in rows)):
        r = [x for x in rows if x[3] == rep]
        probe.append(([float(x[0]) for x in r], [[float(x[i]) for x in r] for i in ai]))
    return meta, probe

groups = collections.defaultdict(list)
for d, first, manf in (('probe_traces_2026-09-21', 0, None), ('probe_traces_2026-09-22', 472, None), ('probe_traces_2026-09-22b', 493, None),
                       ('probe_traces_2026-09-23', 514, 'slow_run_manifest.json'), ('probe_traces_2026-09-23b', 556, 'fresh_run_manifest.json')):
    arm_of = {}
    if manf and os.path.exists(T + d + '/' + manf):
        for m in json.load(open(T + d + '/' + manf)):
            for f in m['files']: arm_of[f] = m['arm']
    for f in sorted(glob.glob(T + d + '/verify*.csv')):
        b = os.path.basename(f)
        if int(b[6:11]) < first or arm_of.get(b) == 'warmup': continue
        meta, probe = load(f)
        if probe is None or meta.get('freq') != '112.70': continue
        groups[(d[-3:], meta.get('note', '')[:4], int(float(meta['x'])), int(float(meta['y'])), arm_of.get(b, '-'))].append(probe)

rows = []
for key in sorted(groups):
    probes = groups[key]
    o = [old_B(p) for p in probes]
    s = [signed.call_estimate(p) for p in probes]
    sv = [v for v, _ in s]
    anc = ''.join('xyz'[a] if a is not None else '-' for _, a in s)
    sd = lambda v: np.std([x for x in v if np.isfinite(x)], ddof=1) * 1e3 if sum(np.isfinite(v)) > 1 else np.nan
    ok = lambda v: sum(np.isfinite(v))
    rows.append((key, sd(o), ok(o), sd(sv), ok(sv), len(probes), anc))
    print('%-34s old B %7.2f (%d)   signed %7.2f (%d) of %d   anchors %s' % (key, rows[-1][1], rows[-1][2], rows[-1][3], rows[-1][4], rows[-1][5], anc), flush=True)
o = np.array([r[1] for r in rows]); s = np.array([r[3] for r in rows])
print('\nold B:  median %.2f  scored %d/%d groups, %d/%d probes' % (np.nanmedian(o), np.sum(np.isfinite(o)), len(rows), sum(r[2] for r in rows), sum(r[5] for r in rows)))
print('signed: median %.2f  scored %d/%d groups, %d/%d probes' % (np.nanmedian(s), np.sum(np.isfinite(s)), len(rows), sum(r[4] for r in rows), sum(r[5] for r in rows)))
both = np.isfinite(o) & np.isfinite(s)
print('where both score: old median %.2f, signed median %.2f, signed better in %d/%d' % (np.median(o[both]), np.median(s[both]), np.sum(s[both] < o[both]), both.sum()))
