import numpy as np, matplotlib, sys, os; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser('~/code/klipper-tooling/klippy'))
from extras.resonance_probe import HaltingContactProbe as H
import eval_knee as E, seg
from anchor import prep, ols
key = ('-22', 'smoo', '60', '45')
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for row, pi in enumerate([2, 1]):
    probe = E.CORPUS[key][pi]; zs, amps = probe[0]
    raw = [[H._vmin_edge(z_, a, 0.10) for a in am] for z_, am in probe]
    seed = H._combine_axes(raw, 15.)[0]
    r = seg.seeded_edge([(zs, a) for a in amps], seed, detail=True)
    for a in range(3):
        A = ax[row][a]; z, y = prep(zs, amps[a])
        A.plot(z * 1e3, y, '.', ms=4, color='0.4')
        A.axvline(seed * 1e3, color='C0', ls=':', label='seed (half-way)')
        if r: A.axvline(r[0] * 1e3, color='k', lw=1, label='combined %.1f' % (r[0] * 1e3))
        p = r[1][a] if r else None
        zt = seg.turning_point(z, y, seed)
        if p:
            k = p[3]; am = z > k; fm = (z >= zt) & (z <= k)
            for m, c in ((am, 'C2'), (fm, 'C3')):
                s_, b_, *_ = ols(z[m], y[m]); xx = np.linspace(z[m].min(), max(z[m].max(), p[0]), 5)
                A.plot(xx * 1e3, s_ * xx + b_, color=c, lw=2)
                A.plot(z[m] * 1e3, y[m], 'o', mfc='none', color=c, ms=5)
            A.axvline(p[0] * 1e3, color='C3', ls='--', label='axis %.1f se %.1f t %.0f' % (p[0] * 1e3, p[1] * 1e3, p[2]))
        A.set_xlim(-60, 250); A.legend(fontsize=7); A.set_title('probe %d  accel %s%s' % (pi + 1, 'xyz'[a], '  ANCHOR' if r and r[2] == a else ''))
plt.tight_layout(); plt.savefig('peek3.png', dpi=70)
