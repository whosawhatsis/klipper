#!/usr/bin/env python3
"""Offline selector evaluation against the descent-trace corpus.

The premise: a mode-selection metric is only useful if it predicts what the
LIVE detector will do.  So rather than compare proxy metrics to each other,
simulate the detector on every trace and score modes by its actual margin,
then ask which cheap metric would have picked the same winner.
"""
import glob, os, statistics as st, sys
from collections import defaultdict

sys.path.insert(0, os.path.expanduser('~/code/klipper/klippy'))
from extras.analog_contact import ContactDetector, estimate_noise, DRAWDOWN

AXES = ('x', 'y', 'z')


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


def contact_depth(depth, vals):
    """Ground-truth contact: the deepest point where the value is still within
    noise of the air baseline, i.e. the last window before sustained collapse.

    Deliberately independent of any detector, so it can referee them."""
    n = len(vals)
    air = st.median(vals[:max(5, n // 4)])
    if air <= 0:
        return None
    # Walk backwards to the last window above 90% of air that is followed by a
    # sustained (3-window) stay below 80%.
    for i in range(n - 4, 2, -1):
        if vals[i] >= 0.90 * air and all(v < 0.80 * air for v in vals[i+1:i+4]):
            return depth[i]
    return None


def simulate(depth, vals, nsigma=8., floor=0.10, lookback=0.12):
    """Run the real detector; return (fire_depth, threshold, air_sd)."""
    hop = (depth[-1] - depth[0]) / max(len(depth) - 1, 1)
    warm = min(40, len(vals) // 3)
    air = vals[:warm]
    base = st.median(air) if air else 0.
    sd = (estimate_noise(air) / base) if base > 0 else 0.
    thr = max(floor, nsigma * sd)
    det = ContactDetector(DRAWDOWN, thr, 0.2, 0.2 / max(hop, 1e-9),
                          relative=True, persist_mm=2 * hop,
                          lookback_mm=lookback)
    for i in range(warm, len(vals)):
        if det.update(vals[i], position=depth[i]):
            return depth[i], thr, sd
    return None, thr, sd


def max_drawdown(vals, lookback_n=None):
    pk, m = vals[0], 0.
    win = []
    for v in vals:
        if lookback_n:
            win.append(v)
            if len(win) > lookback_n:
                win.pop(0)
            pk = max(win)
        else:
            pk = max(pk, v)
        if pk > 0:
            m = max(m, (pk - v) / pk)
    return m


def main():
    per_mode = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__),
                                           'corpus', '*.csv'))):
        meta, rows = load(f)
        if 'freq' not in meta or len(rows) < 60:
            continue
        # Only 0.2mm/s: higher speeds are hop-confounded and ring-down limited,
        # so mixing them would compare modes under unequal conditions.
        if not meta['speed'].startswith('0.2'):
            continue
        freq, aph = float(meta['freq']), float(meta['accel_per_hz'])
        depth = [r[0] for r in rows]
        rec = {'file': os.path.basename(f), 'freq': freq, 'aph': aph}
        for ai, ax in enumerate(AXES, start=1):
            vals = [r[ai] for r in rows]
            cz = contact_depth(depth, vals)
            fire, thr, sd = simulate(depth, vals)
            rec[ax] = {
                'sd': sd, 'thr': thr, 'maxdd': max_drawdown(vals),
                'contact': cz, 'fire': fire,
                'margin': (max_drawdown(vals) / thr) if thr else 0.,
                'overtravel': (fire - cz) if (fire and cz) else None,
            }
        per_mode[(freq, aph)].append(rec)

    print("=" * 78)
    print("SIMULATED DETECTOR OUTCOME per mode (0.2 mm/s traces only)")
    print("=" * 78)
    print("%7s %5s %3s | %-26s | %-22s" % ("freq", "aph", "n",
                                           "fire rate x/y/z",
                                           "median margin x/y/z"))
    ranked = []
    for k in sorted(per_mode):
        recs = per_mode[k]
        rates, margins, overs = [], [], []
        for ax in AXES:
            fired = [r[ax]['fire'] is not None for r in recs]
            rates.append(sum(fired) / len(fired))
            margins.append(st.median([r[ax]['margin'] for r in recs]))
            ov = [r[ax]['overtravel'] for r in recs
                  if r[ax]['overtravel'] is not None]
            overs.append(st.median(ov) if ov else float('nan'))
        # A mode is only as good as its WEAKEST axis, because which axis carries
        # the signal varies with bed location - that was measured, not assumed.
        worst_margin = min(margins)
        ranked.append((worst_margin, k, rates, margins, overs))
        print("%7.1f %5.0f %3d | %8s %8s %8s | %6.1fx %6.1fx %6.1fx"
              % (k[0], k[1], len(recs),
                 "%.0f%%" % (rates[0] * 100), "%.0f%%" % (rates[1] * 100),
                 "%.0f%%" % (rates[2] * 100),
                 margins[0], margins[1], margins[2]))

    print()
    print("=" * 78)
    print("RANKING by WEAKEST-AXIS margin (robust to location-dependent axis)")
    print("=" * 78)
    for worst, k, rates, margins, overs in sorted(ranked, reverse=True):
        print("  %7.1f Hz aph %-4.0f  weakest-axis margin %5.1fx   "
              "all-axis fire %s"
              % (k[0], k[1], worst,
                 "yes" if min(rates) == 1.0 else
                 "NO (%.0f%%)" % (min(rates) * 100)))


if __name__ == '__main__':
    main()
