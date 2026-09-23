#!/usr/bin/env python3
"""Which single-point statistic predicts worst-LOCATION detectability?

Calibration measures one bed location, but the probe must work everywhere, and
which axis carries contact changes with position.  Detection succeeds if ANY
armed axis fires, so:

  ground truth for a mode = min over locations of ( max over axes of margin )

i.e. at its worst location, how much margin does its best axis still have.
That credits a mode with one strong axis and penalises one that is uniformly
marginal - which plain min-over-axes gets backwards.

The corpus samples different random points per trace, so the ground truth is
computable here.  The question is which statistic of a SINGLE trace best
reproduces that ranking, since that is all calibration will have.
"""
import glob, os, statistics as st, sys
from collections import defaultdict

sys.path.insert(0, os.path.expanduser('~/code/klipper/klippy'))
from extras.analog_contact import estimate_noise, detector_margin

FLOOR, NSIGMA = 0.10, 8.


def load(path):
    meta, rows = {}, []
    for line in open(path):
        if line.startswith('#'):
            for kv in line[1:].split():
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    meta[k] = v
        elif not line.startswith('mm_below'):
            rows.append([float(x) for x in line.split(',')])
    return meta, rows


def max_drawdown(v):
    pk, m = v[0], 0.
    for a in v:
        pk = max(pk, a)
        if pk > 0:
            m = max(m, (pk - a) / pk)
    return m


def trace_margins(rows):
    out = []
    for ai in (1, 2, 3):
        v = [r[ai] for r in rows]
        base = st.median(v[:40])
        sd = estimate_noise(v[:40]) / base if base > 0 else 0.
        out.append(detector_margin(max_drawdown(v), sd, FLOOR, NSIGMA))
    return out


# ---- candidate single-point statistics -------------------------------------
def s_min(m):        return min(m)
def s_max(m):        return max(m)
def s_second(m):     return sorted(m, reverse=True)[1]
def s_mean(m):       return sum(m) / len(m)
def s_redundant(m):
    """Best axis, discounted when it has no backup.

    Rationale: a strong axis is what actually triggers, but if it is the ONLY
    axis above threshold, a location that shifts the signal off it leaves
    nothing.  Each additional axis clearing 1x removes part of that discount.
    """
    m = sorted(m, reverse=True)
    backups = sum(1 for x in m[1:] if x >= 1.0)
    return m[0] * (0.5 + 0.25 * backups)
def s_softmin(m):
    """Smooth blend of best and worst - credits strength, still punishes a
    dead axis, without min()'s all-or-nothing behaviour."""
    m = sorted(m, reverse=True)
    return 0.6 * m[0] + 0.3 * m[1] + 0.1 * m[2]


CANDIDATES = [("min-over-axes (current)", s_min),
              ("max-over-axes", s_max),
              ("2nd-best axis", s_second),
              ("mean of axes", s_mean),
              ("best x redundancy", s_redundant),
              ("weighted 0.6/0.3/0.1", s_softmin)]


def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: -v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1)) if n > 2 else float('nan')


def main():
    per = defaultdict(list)
    here = os.path.dirname(os.path.abspath(__file__))
    for f in sorted(glob.glob(os.path.join(here, 'corpus', '*.csv'))):
        meta, rows = load(f)
        if 'freq' not in meta or len(rows) < 60:
            continue
        if not meta['speed'].startswith('0.2'):
            continue
        per[(float(meta['freq']), float(meta['accel_per_hz']))].append(
            trace_margins(rows))

    modes = sorted(per)
    truth = {}
    for k in modes:
        # worst location, best axis there
        truth[k] = min(max(m) for m in per[k])

    print("GROUND TRUTH - worst location, best axis (min over loc of max over axes)")
    for k in sorted(modes, key=lambda k: -truth[k]):
        best_per_loc = sorted(max(m) for m in per[k])
        print("  %6.1f Hz aph %-4.0f  worst-loc %5.1fx   per-location best: %s"
              % (k[0], k[1], truth[k],
                 " ".join("%.1f" % b for b in best_per_loc)))

    print()
    print("HOW WELL EACH SINGLE-POINT STATISTIC PREDICTS THAT (n=%d modes)"
          % len(modes))
    tv = [truth[k] for k in modes]
    for name, fn in CANDIDATES:
        pv = [st.median([fn(m) for m in per[k]]) for k in modes]
        rho = spearman(tv, pv)
        order = sorted(modes, key=lambda k: -st.median(
            [fn(m) for m in per[k]]))
        print("  %-24s rho=%+.2f  picks %.0f/%.0f   order %s"
              % (name, rho, order[0][0], order[0][1],
                 ">".join("%.0f" % k[0] for k in order)))


if __name__ == '__main__':
    main()
