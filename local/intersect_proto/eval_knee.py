import sys, os, glob, collections, numpy as np
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
from extras.resonance_probe import HaltingContactProbe as H
import knee
T = os.path.expanduser('~/code/klipper-tooling/')
def load_all():
    out = collections.defaultdict(list)
    for d, first in (('probe_traces_2026-09-21', 0), ('probe_traces_2026-09-22', 472), ('probe_traces_2026-09-22b', 493)):
        for f in sorted(glob.glob(T + d + '/verify*.csv')):
            if int(os.path.basename(f)[6:11]) < first: continue
            meta, cols, rows = {}, None, []
            for ln in open(f):
                if ln.startswith('#'):
                    for t in ln[1:].split():
                        if '=' in t: k, v = t.split('=', 1); meta[k] = v
                    continue
                p = ln.strip().split(',')
                if cols is None: cols = p
                elif len(p) == len(cols) and p[2] == 'down': rows.append(p)
            if meta.get('freq') != '112.70' or 'amp_x' not in (cols or []): continue
            ai = [cols.index(c) for c in ('amp_x', 'amp_y', 'amp_z')]
            probe = []
            for rep in sorted(set(r[3] for r in rows)):
                r = [x for x in rows if x[3] == rep]
                probe.append(([float(x[0]) for x in r], [[float(x[i]) for x in r] for i in ai]))
            out[(d[-3:], meta.get('note', '')[:4], meta['x'][:2], meta['y'][:2])].append(probe)
    return out
CORPUS = load_all()

def fade(ramps, lo):
    """ramps[rep][axis] = (edge, exc, sd, ...) -> weighted mean, same fade rule as in-tree."""
    wts = {}
    for a in range(3):
        g = [r[a] for r in ramps]
        if any(x is None for x in g): continue
        s = min(x[1] / x[2] for x in g)
        w = min((s - lo) / lo, 1.)
        if w > 0: wts[a] = w
    if not wts: return None
    return sum(w * np.mean([r[a][0] for r in ramps]) for a, w in wts.items()) / sum(wts.values())

def score(fn_axis, combine, groups=None):
    """-> {group: sd_um} for a per-axis estimator + combiner."""
    out = {}
    for key, probes in CORPUS.items():
        vals = []
        for probe in probes:
            ramps = [[fn_axis(zs, a) for a in amps] for zs, amps in probe]
            v = combine(ramps)
            if v is not None: vals.append(v)
        if len(vals) >= 5: out[key] = (np.std(vals, ddof=1) * 1e3, len(vals), len(probes))
    return out

def vmin(zs, a): return H._vmin_edge(zs, a, 0.10)
def table(name, res):
    ks = sorted(res)
    return name, [res[k][0] for k in ks], ks, sum(res[k][2] - res[k][1] for k in ks)

if __name__ == '__main__':
    rows = []
    rows.append(table('vmin fade15 (deployed)', score(vmin, lambda r: fade(r, 15.))))
    for ax in range(3):
        rows.append(table('vmin %s only' % 'xyz'[ax], score(vmin, lambda r, ax=ax: np.mean([x[ax][0] for x in r]) if all(x[ax] for x in r) else None)))
    rows.append(table('knee fade15', score(lambda z, a: knee.knee_edge(z, a), lambda r: fade(r, 15.))))
    for ax in range(3):
        rows.append(table('knee %s only' % 'xyz'[ax], score(lambda z, a: knee.knee_edge(z, a), lambda r, ax=ax: np.mean([x[ax][0] for x in r]) if all(x[ax] for x in r) else None)))
    ks = rows[0][2]
    print('%-24s' % '' + ''.join('%8s' % (k[0][-1] + k[1][0] + k[2] + k[3]) for k in ks) + '  median  lost')
    for name, sds, kk, lost in rows:
        m = dict(zip(kk, sds))
        print('%-24s' % name + ''.join('%8.2f' % m[k] if k in m else '%8s' % '-' for k in ks) + '  %6.2f  %d' % (np.median(sds), lost))
