import sys; sys.argv = ['x']
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import analyze_slow as A
import numpy as np
order = [(75, 85, 'slow'), (75, 85, 'ctrl'), (90, 70, 'ctrl'), (90, 70, 'slow'), (60, 45, 'slow'), (60, 45, 'ctrl')]
for k in order:
    r = A.res[k]
    v = np.array(r['contact, x alone']) * 1e3
    h = np.array(r['half-way (deployed)']) * 1e3
    print('(%d,%d) %-4s contact: %s   | slope n=%s' % (k[0], k[1], k[2], ' '.join('%6.1f' % x for x in v), ' '.join('%.0f' % x for x in r['_flank_samples'])))
    print('              half-way: %s   r(contact, order)=%+.2f' % (' '.join('%6.1f' % x for x in h), np.corrcoef(v, np.arange(len(v)))[0, 1]))
