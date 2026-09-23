import numpy as np, matplotlib, sys; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import knee, eval_knee as E
kw = eval(sys.argv[1]) if len(sys.argv) > 1 else {}
keys = [('-22', 'smoo', '75', '85'), ('-22', 'smoo', '90', '70'), ('-22', 'smoo', '60', '45')]
fig, ax = plt.subplots(3, 3, figsize=(15, 10))
for r, key in enumerate(keys):
    for a in range(3):
        A = ax[r][a]
        for pi, probe in enumerate(E.CORPUS[key][:3]):
            zs, amps = probe[0]; c = 'C%d' % pi
            z = np.array(zs) * 1e3; y = np.log(np.maximum(amps[a], 1))
            A.plot(z, y, '.', ms=3, color=c, alpha=.4)
            d = knee.knee_edge(zs, amps[a], detail=True, **kw)
            if d is None: A.text(.02, .9 - .08 * pi, 'probe %d: none' % pi, transform=A.transAxes, color=c); continue
            zz = d['z'] * 1e3
            A.plot(zz, d['s'], '-', color=c, lw=.8)
            m, b = d['air']; A.plot(zz[d['i_d']:], m * d['z'][d['i_d']:] + b, '--', color=c, lw=1.5)
            m, b = d['flank']; zf = np.linspace(d['z'][d['i_e']], d['edge'] + .01, 10); A.plot(zf * 1e3, m * zf + b, '-', color=c, lw=2)
            A.plot(zz[d['fl']], d['y'][d['fl']], 'o', mfc='none', color=c)
            A.axvline(d['edge'] * 1e3, color=c, ls=':')
        A.set_title('%s,%s accel %s' % (key[2], key[3], 'xyz'[a])); A.set_xlim(-80, 250)
plt.tight_layout(); plt.savefig('peek2.png', dpi=70)
