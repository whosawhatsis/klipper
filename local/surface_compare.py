#!/usr/bin/env python3
# Score DETECTABILITY on the smooth vs the textured plate side, over the two
# labelled-overshoot sets in probe_traces/LABELS.md.  Both sets were recorded at
# 172.9Hz / accel_per_hz 120 / SPEED=0.2 / win_n=148 at the SAME four XY points,
# so the comparison is PAIRED BY POINT: air noise turns out to vary more with bed
# position than with surface, and unpaired medians hide that completely.
#
# Reuses compare_detectors.py's loader and its two live detectors rather than
# re-implementing them, so "when would the probe have fired" means the same
# thing here as it does there.
#
# !! THE AIR-NOISE AND SNR COLUMNS BELOW ARE NOT A VALID SURFACE COMPARISON for
# !! the two sets currently listed.  The textured set was recorded through
# !! RESONANCE_PROBE_CONTACT with its old hardcoded WARMUP default of 0.5s while
# !! the machine is configured for 0.8s, so ~0.3s of excitation ring-up landed
# !! inside every textured trace and inside its air baseline.  The surface cannot
# !! cause this - ring-up happens in air ~2mm above contact - it is purely a
# !! capture difference.  Fire depth IS comparable (see trap 1).  Re-record the
# !! textured set at the configured warmup to answer detectability.
#
# Two measurement traps this deliberately avoids:
#
# 1. The first few windows of a descent can be the excitation still RINGING UP
#    (seen as 13, 21, 1838, then ~4100).  Including them in the air baseline
#    reports ~20% "air noise" that is not noise at all, and a linear detrend
#    does not remove it because it is a step, not a trend.  Fire DEPTH survives
#    it: 'mm_below_arm' is anchored at the arming time and the warm-up runway
#    scales with warmup, so depth 0 is the same physical Z either way - dropping
#    the ring-up changed the replayed fire depth on 15 of 16 traces by 0um.  The
#    16th was an under-warmed descent that "fired" 1478um ABOVE contact; with a
#    clean baseline it fires at +117um like everything else, which is how the
#    warmup defect was found.
# 2. Ground truth ('truth' = 2.0 - contact_z) is the ARMED probe's own contact at
#    that point.  That reference is itself surface-dependent, so fire depth
#    relative to truth cannot separate "damps earlier" from "armed probe read
#    lower".  Amplitude measured ABOVE truth needs no detector or threshold.

import importlib.util
import os
import sys

import numpy as np
import os as _os
_TRACES = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                       "..", "probe_traces")

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "compare_detectors", os.path.join(HERE, "compare_detectors.py"))
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)

# point -> {surface: [(trace, contact_z), ...]}, from probe_traces/LABELS.md.
# smooth = 2026-07-27 (no note=); textured = 2026-07-29 (note=textured).
POINTS = [
    ("64.9,73.5", {'smooth': [(140, -0.001267), (141, -0.001267)],
                   'textured': [(300, 0.008109), (302, 0.008109)]}),
    ("73.2,60.9", {'smooth': [(143, -0.010620), (144, -0.010620)],
                   'textured': [(312, -0.004786), (314, -0.004786)]}),
    ("80.2,33.7", {'smooth': [(146, 0.011829), (147, 0.011829)],
                   'textured': [(320, -0.017552), (322, -0.017552)]}),
    ("31.4,69.6", {'smooth': [(149, -0.000344), (150, -0.000344)],
                   'textured': [(328, -0.021813), (330, -0.021813)]}),
]
AIR_N = 30      # air windows used for the baseline, after ring-up
Z_AXIS = 2      # accel_axis = z is the live configuration
AIR_TOL = 0.10  # a window is "settled" within this fraction of the settled level
AIR_RUN = 5     # ...and only once this many consecutive windows are settled
DRIVE_MIN = 0.6  # settled level below this fraction of the set median = anomaly


def air_window(a):
    """(n_skipped, air samples, settled level), excitation ring-up removed.

    A 50%-of-settled cutoff is NOT enough: a trace can climb 5, 15, 1168, 2233
    and sit at 2233 for several windows while still ringing up, which reports as
    ~9% air noise.  Require a run of consecutive windows inside a tight band
    instead, and take the settled level from a region that is certainly past
    ring-up rather than from the head of the trace.
    """
    settled = float(np.median(a[20:60]))
    for i in range(len(a) - AIR_RUN):
        w = a[i:i + AIR_RUN]
        if np.all(np.abs(w - settled) / settled < AIR_TOL):
            return i, a[i:i + AIR_N], settled
    return 0, a[:AIR_N], settled


def measure(trace, contact_z):
    hdr, zs, amps = cd.load(_os.path.join(_TRACES, "descent%05d.csv" % trace))
    z = np.array(zs)
    a = np.array(amps)
    col = a[:, Z_AXIS]
    truth = 2.0 - contact_z
    skipped, air, settled = air_window(col)
    base = air.mean()
    noise = air.std() / base

    def frac(lo, hi):
        sel = np.where((z >= lo) & (z < hi))[0]
        return col[sel].mean() / base if len(sel) else np.nan

    fire = cd.fire_drawdown(z, a)
    return {
        'trace': trace, 'skipped': skipped, 'noise': noise,
        'settled': settled,
        'above': 1. - frac(truth - 0.12, truth - 0.08),
        'drop100': 1. - frac(truth + 0.08, truth + 0.12),
        'deep': 1. - col[-5:].mean() / base,
        'fire_um': None if fire is None else (fire - truth) * 1000.,
        'note': hdr.get('note', '-'),
    }


def main():
    print("Paired smooth-vs-textured detectability, 172.9Hz aph120 SPEED=0.2")
    print("air noise and drop are fractions of the air baseline (z-axis);")
    print("drop@+100um is the fall 100um BELOW the armed probe's contact;")
    print("SNR = drop@+100um / air noise;  fire = drawdown vs truth.\n")

    rows = {'smooth': [], 'textured': []}
    for name, surfaces in POINTS:
        for surf in ('smooth', 'textured'):
            for (tr, cz) in surfaces[surf]:
                r = measure(tr, cz)
                r['point'] = name
                r['snr'] = r['drop100'] / r['noise'] if r['noise'] else np.nan
                rows[surf].append(r)

    # Flag traces whose excitation never reached the drive level the rest of the
    # set ran at - their drop fractions are not comparable.  Reported, never
    # silently dropped.
    med_settled = float(np.median([r['settled']
                                   for v in rows.values() for r in v]))
    for v in rows.values():
        for r in v:
            r['bad_drive'] = r['settled'] < DRIVE_MIN * med_settled

    print("%-11s %-9s %5s %4s %8s %7s %8s %7s %7s %8s" % (
        "point", "surface", "trace", "skip", "settled", "noise", "drop@100",
        "SNR", "deep", "fire_um"))
    for name, _ in POINTS:
        for surf in ('smooth', 'textured'):
            for r in rows[surf]:
                if r['point'] != name:
                    continue
                print("%-11s %-9s %5d %4d %8.0f %7.4f %8.3f %7.2f %7.3f %8s%s"
                      % (name, surf, r['trace'], r['skipped'], r['settled'],
                         r['noise'], r['drop100'], r['snr'], r['deep'],
                         "n/a" if r['fire_um'] is None
                         else "%+.0f" % r['fire_um'],
                         "  <-- LOW DRIVE" if r['bad_drive'] else ""))
        print()

    excluded = [r['trace'] for v in rows.values() for r in v if r['bad_drive']]
    if excluded:
        print("EXCLUDED from the pooled stats (drive below %.0f%% of the set "
              "median %.0f): %s\n"
              % (100 * DRIVE_MIN, med_settled,
                 ", ".join(str(t) for t in excluded)))

    def keep(surf):
        return [r for r in rows[surf] if not r['bad_drive']]

    print("=== per-point SNR (the paired comparison) ===")
    print("%-11s %9s %9s %9s" % ("point", "smooth", "textured", "ratio"))
    for name, _ in POINTS:
        vals = {}
        for surf in ('smooth', 'textured'):
            v = [r['snr'] for r in keep(surf) if r['point'] == name]
            vals[surf] = float(np.median(v)) if v else np.nan
        print("%-11s %9.2f %9.2f %8.2fx"
              % (name, vals['smooth'], vals['textured'],
                 vals['textured'] / vals['smooth']))

    print("\n=== pooled medians (low-drive traces excluded) ===")
    for key, lab in (('noise', 'air noise'), ('drop100', 'drop@+100um'),
                     ('snr', 'SNR@+100um'), ('deep', 'deep drop'),
                     ('settled', 'air level')):
        sm = float(np.nanmedian([r[key] for r in keep('smooth')]))
        tx = float(np.nanmedian([r[key] for r in keep('textured')]))
        print("  %-14s smooth %9.4f   textured %9.4f   delta %+9.4f  (%.2fx)"
              % (lab, sm, tx, tx - sm, tx / sm if sm else np.nan))

    # The headline: does the surface move WHERE it fires?
    print("\n=== fire depth vs truth (the press-depth question) ===")
    for surf in ('smooth', 'textured'):
        f = [r['fire_um'] for r in keep(surf) if r['fire_um'] is not None]
        good = [v for v in f if v > 0]
        false_halts = [r['trace'] for r in keep(surf)
                       if r['fire_um'] is not None and r['fire_um'] < 0]
        print("  %-9s median %+.0f um, range %+.0f..%+.0f (n=%d)%s"
              % (surf, np.median(good), min(good), max(good), len(good),
                 "  false halts above contact: %s"
                 % (false_halts,) if false_halts else ""))

    # The confound: both sets were recorded point-by-point in a fixed order, so
    # anything that drifts during a run masquerades as a surface effect.
    print("\n=== run-order confound ===")
    for surf in ('smooth', 'textured'):
        ns = [r['noise'] for r in rows[surf]]
        r = float(np.corrcoef(np.arange(len(ns)), ns)[0, 1])
        print("  %-9s corr(position in run, air noise) = %+.2f   [%s]"
              % (surf, r, " ".join("%.3f" % n for n in ns)))
    print("  Order was NOT reversed between the two sets, so a drift within a"
          " run cannot be separated from the surface here.")

    # A set recorded at a shorter warmup carries ring-up in its air baseline, so
    # its noise column means something different.  Detect it from the data (the
    # old traces predate the 'warmup=' header field) and say so loudly rather
    # than letting the SNR column be read as a surface result.
    ring = {s: sum(1 for r in rows[s] if r['skipped'] > 0)
            for s in ('smooth', 'textured')}
    print("\n=== capture-parity check ===")
    print("  traces carrying excitation ring-up: smooth %d/%d, textured %d/%d"
          % (ring['smooth'], len(rows['smooth']),
             ring['textured'], len(rows['textured'])))
    if ring['smooth'] != ring['textured']:
        print("  *** AIR NOISE AND SNR ABOVE ARE NOT COMPARABLE between these")
        print("  *** sets: they were captured at different warmups, which the")
        print("  *** surface cannot influence (ring-up is in air, ~2mm up).")
        print("  *** Fire depth is unaffected and remains comparable.")


if __name__ == "__main__":
    sys.exit(main())
