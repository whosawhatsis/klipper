import numpy as np, collections, sys
import eval_knee as E, anchor
tol = float(sys.argv[1]) if len(sys.argv) > 1 else 0.015
res, used, anc, agree = {}, collections.Counter(), collections.Counter(), collections.defaultdict(list)
for key, probes in E.CORPUS.items():
    vals = []
    for probe in probes:
        per_rep = []
        for zs, amps in probe:
            r = anchor.anchored_edge([(zs, a) for a in amps], tol=tol, detail=True)
            if r is None: continue
            per_rep.append(r[0]); anc['xyz'[r[2]]] += 1
            used[''.join('xyz'[i] for i, p in enumerate(r[1]) if p) or '-'] += 1
            for i, p in enumerate(r[1]):
                if p: agree['xyz'[i]].append((p[0] - r[0]) * 1e3)
        if len(per_rep) == len(probe): vals.append(np.mean(per_rep))
    if len(vals) >= 5: res[key] = (np.std(vals, ddof=1) * 1e3, len(probes) - len(vals))
base = E.score(E.vmin, lambda r: E.fade(r, 15.))
ks = sorted(base)
print('%-18s' % '' + ''.join('%8s' % (k[0][-1] + k[1][0] + k[2] + k[3]) for k in ks) + '  median  lost')
print('%-18s' % 'vmin fade (deployed)' + ''.join('%8.2f' % base[k][0] for k in ks) + '  %6.2f' % np.median([base[k][0] for k in ks]))
print('%-18s' % ('anchored tol=%g' % tol) + ''.join('%8.2f' % res[k][0] if k in res else '%8s' % '-' for k in ks) + '  %6.2f  %d' % (np.median([res[k][0] for k in res]), sum(v[1] for v in res.values())))
print('anchor axis:', dict(anc), ' axes used:', dict(used.most_common(6)))
for a, v in sorted(agree.items()): print('  %s: n=%d  offset from combined median %+.1fum  IQR %.1fum' % (a, len(v), np.median(v), np.subtract(*np.percentile(v, [75, 25]))))
