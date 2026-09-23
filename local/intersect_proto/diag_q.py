import numpy as np, sys, os
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
from extras.resonance_probe import HaltingContactProbe as H
import eval_knee as E, tangent
from eval_q import axis_fits
for key in [('-21', 'text', '60', '60'), ('-22', 'smoo', '60', '45')]:
    print(key)
    for pi, probe in enumerate(E.CORPUS[key]):
        for ri, (zs, amps) in enumerate(probe):
            rr = [H._vmin_edge(zs, a, 0.10) for a in amps]
            ind = axis_fits(zs, amps, rr)
            print('  p%d r%d  ' % (pi + 1, ri + 1) + '   '.join('%s: %s' % ('xyz'[a], '-' if p is None else '%6.1fum s=%5.1f' % (p[0] * 1e3, p[1] * 1e3)) for a, p in enumerate(ind)))
