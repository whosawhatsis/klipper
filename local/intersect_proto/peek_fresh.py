import json, os, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import io, contextlib, sys
sys.argv = ['x', os.path.expanduser('~/code/klipper-tooling/probe_traces_2026-09-23b'), 'fresh_run_manifest.json']
with contextlib.redirect_stdout(io.StringIO()):
    import analyze_slow as A
man = A.man
fig, ax = plt.subplots(4, 3, figsize=(15, 12))
for row, (pt, arm) in enumerate([((85, 25), 'slow'), ((85, 25), 'ctrl'), ((55, 25), 'slow'), ((105, 45), 'slow')]):
    files = next(m['files'] for m in man if (m['x'], m['y']) == pt and m['arm'] == arm)
    for pi, f in enumerate(files[:3]):
        meta, probe = A.load(f)
        zs, amps = probe[0]
        for a in range(3):
            z = np.array(zs) * 1e3; y = np.log(np.maximum(amps[a], 1))
            ax[row][a].plot(z, y, '.-', ms=3, lw=.6, color='C%d' % pi, label=f[6:11] + ' zc=%.0f' % (float(meta['z_cand']) * 1e3))
    for a in range(3):
        ax[row][a].set_title('(%d,%d) %s accel %s' % (pt + (arm, 'xyz'[a]))); ax[row][a].legend(fontsize=7)
plt.tight_layout(); plt.savefig('peek_fresh.png', dpi=65)
