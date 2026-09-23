"""Score the all-axes hardware run: ledger z_reported, all-axes replay, x-only replay.
usage: score_run.py <trace_dir> <first_verify_number>"""
import sys, os, glob, collections
import numpy as np
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
from extras.resonance_probe import HaltingContactProbe as H

d, first = sys.argv[1], int(sys.argv[2])
led = {}
for ln in open(os.path.join(d, 'contacts.csv')):
    p = ln.strip().split(',')
    if len(p) == 11 and p[10].startswith('verify'):
        led[p[10]] = p
res = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(os.path.join(d, 'verify*.csv'))):
    name = os.path.basename(f)
    if int(name[6:11]) < first:
        continue
    meta, cols, rows = {}, None, []
    for ln in open(f):
        if ln.startswith('#'):
            for t in ln[1:].split():
                if '=' in t:
                    k, v = t.split('=', 1); meta[k] = v
            continue
        p = ln.strip().split(',')
        if cols is None:
            cols = p
        elif len(p) == len(cols) and p[2] == 'down':
            rows.append(p)
    key = (meta['x'], meta['y'])
    ai = [cols.index(c) for c in ('amp_x', 'amp_y', 'amp_z')]
    ramps = []
    for rep in sorted(set(r[3] for r in rows)):
        r = [x for x in rows if x[3] == rep]
        zs = [float(x[0]) for x in r]
        ramps.append([H._vmin_edge(zs, [float(x[i]) for x in r], 0.10) for i in ai])
    a = H._combine_axes(ramps, 15.)
    xo = [rp[0][0] for rp in ramps if rp[0] is not None]
    voters = [k for k in range(3) if all(rp[k] is not None and rp[k][1] / rp[k][2] > 15. for rp in ramps)]
    if name in led:
        res[key]['ledger'].append(float(led[name][2]))
        res[key]['ledger_down'].append(float(led[name][3]) if led[name][3] else np.nan)
    if a:
        res[key]['allaxes'].append(np.mean(a))
    if xo:
        res[key]['x_only'].append(np.mean(xo))
    res[key]['voters'].append(''.join('xyz'[k] for k in voters) or '-')
for key in sorted(res):
    r = res[key]
    sd = lambda v: np.nanstd(v, ddof=1) * 1e3 if len(v) > 1 else float('nan')
    print('%s,%s  n=%d  ledger sd=%.2fum (down col %.2f)  allaxes replay sd=%.2fum  x-only replay sd=%.2fum  mean=%.1fum  voters=%s'
          % (key[0][:2], key[1][:2], len(r['ledger']), sd(r['ledger']), sd(r['ledger_down']),
             sd(r['allaxes']), sd(r['x_only']), np.mean(r['allaxes']) * 1e3 if r['allaxes'] else 0,
             ' '.join(r['voters'])))
