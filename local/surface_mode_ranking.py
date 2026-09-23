#!/usr/bin/env python3
# Does the plate surface REORDER the mode ranking, or just shift it?
#
# Compares the 2026-07-28 smooth-side frequency sweep (traces 169-185) against
# the 2026-07-29 textured-side sweep recorded by scripts/textured_freq_sweep.py.
# Identical protocol: same two XY points, same four candidate modes, aph 120,
# SPEED 0.2, detectors disarmed, ~220um of overshoot, frequency order reversed at
# the second point.
#
# The measurement is the one the existing smooth-side mode table uses: MAX
# DRAWDOWN from the running peak, per axis, plus the air noise it has to beat.
# Ranking modes needs a per-mode scalar, and the project's own finding is that
# robustness across axes matters more than the best single axis (a location can
# move the signal onto another axis), so the headline score is the SECOND-BEST
# axis - matching the live selector's rule.
#
# Hypothesis under test: texture should not reorder modes, but may add a gradient
# proportional (or inversely proportional) to frequency on top of the smooth
# ranking - e.g. via a change in ring-down.  A gradient shows up as a straight
# line in (textured - smooth) versus frequency; a reordering shows up as a rank
# change in the second-best-axis score.
#
# AIR NOISE IS NOT A SURFACE PROPERTY AND IS NOT PART OF THE VERDICT.  The air
# measurement is taken with the nozzle ~2mm clear of the bed: nothing about the
# surface it is about to touch can reach it.  The two sweeps ran on different
# days (smooth 07-28, textured 07-29), so a difference in the air floor is an
# EXTERNAL/session factor - ambient, machine state, belt temperature - and
# attributing it to the plate is a category error.  It is printed for the record
# and as a session-drift figure, never divided into the drawdown.  Only the
# contact drawdown is surface-attributable, so the verdict rests on that alone.

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

# 2026-07-28 smooth sweep, from probe_traces/LABELS.md.  Verified ring-up free.
SMOOTH = {
    65.5:  [(169, 0.002519), (170, 0.002519), (184, -0.017753), (185, -0.017753)],
    148.0: [(171, 0.002519), (172, 0.002519), (182, -0.017753), (183, -0.017753)],
    172.9: [(173, 0.002519), (174, 0.002519), (180, -0.017753), (181, -0.017753)],
    212.2: [(175, 0.002519), (176, 0.002519), (178, -0.017753), (179, -0.017753)],
}
# Filled in from the sweep's own output (scripts/textured_freq_sweep.py prints
# ready-made rows).  Kept as data, not derived, so the pairing is auditable.
TEXTURED = {
    65.5:  [(338, -0.010186), (340, -0.010186), (374, -0.078417),
            (376, -0.078417)],
    148.0: [(342, -0.010186), (344, -0.010186), (370, -0.078417),
            (372, -0.078417)],
    172.9: [(346, -0.010186), (348, -0.010186), (366, -0.078417),
            (368, -0.078417)],
    212.2: [(350, -0.010186), (352, -0.010186), (362, -0.078417),
            (364, -0.078417)],
}

AIR_N = 40


def air_start(c, tol=0.10, run=5):
    settled = float(np.median(c[20:60]))
    for i in range(len(c) - run):
        if np.all(np.abs(c[i:i + run] - settled) / settled < tol):
            return i
    return 0


def measure(trace, contact_z):
    """Per-axis max drawdown from the running peak, and per-axis air noise."""
    hdr, zs, amps = cd.load(_os.path.join(_TRACES, "descent%05d.csv" % trace))
    a = np.array(amps)
    z = np.array(zs)
    dd, sd = [], []
    for ax in range(3):
        c = a[:, ax]
        k = air_start(c)
        air = c[k:k + AIR_N]
        sd.append(100. * air.std() / air.mean())
        peak = np.maximum.accumulate(c)
        dd.append(100. * float(np.max(1. - c / peak)))
    # Ring-up means the drive had not come up yet: windows far BELOW the settled
    # level.  Do not reuse air_start's tolerance band for this - at 65.5Hz the
    # air is genuinely 4-17% noisy, so a "no settled run yet" index reads as
    # contamination on a trace that starts at full amplitude.
    settled = float(np.median(a[20:60, 2]))
    ring = 0
    while ring < len(a) and a[ring, 2] < 0.5 * settled:
        ring += 1
    return {'dd': dd, 'sd': sd, 'end': z[-1], 'truth': 2.0 - contact_z,
            'warmup': hdr.get('warmup', 'unrecorded'), 'ring': ring}


def summarize(sets, label):
    """freq -> per-axis medians plus the second-best-axis score."""
    out = {}
    print("\n=== %s ===" % label)
    print("%7s %-24s %-22s %9s %8s" % (
        "freq", "maxDD %% x / y / z", "air sd %% x / y / z", "2nd-best",
        "overshoot"))
    for f in sorted(sets):
        D, S, over, rings = [], [], [], []
        for (tr, cz) in sets[f]:
            m = measure(tr, cz)
            D.append(m['dd'])
            S.append(m['sd'])
            over.append(m['end'] - m['truth'])
            rings.append(m['ring'])
        D = np.median(np.array(D), axis=0)
        S = np.median(np.array(S), axis=0)
        second = float(sorted(D)[1])          # second-best axis, the live rule
        out[f] = {'dd': D, 'sd': S, 'second': second,
                  'over': float(np.median(over))}
        note = "  RING-UP" if any(r > 0 for r in rings) else ""
        print("%7.1f %-24s %-22s %9.1f %8.3f%s" % (
            f, "%.0f / %.0f / %.0f" % tuple(D),
            "%.2f / %.2f / %.2f" % tuple(S), second,
            out[f]['over'], note))
    return out


def main():
    if not TEXTURED:
        sys.exit("TEXTURED is empty - paste the trace numbers from "
                 "scripts/textured_freq_sweep.py output into TEXTURED first.")
    sm = summarize(SMOOTH, "SMOOTH side (2026-07-28, traces 169-185)")
    tx = summarize(TEXTURED, "TEXTURED side (2026-07-29)")

    freqs = sorted(set(sm) & set(tx))
    print("\n=== ranking by second-best axis (higher = more detectable) ===")
    rank_s = sorted(freqs, key=lambda f: -sm[f]['second'])
    rank_t = sorted(freqs, key=lambda f: -tx[f]['second'])
    print("  smooth  : %s" % " > ".join("%.1f" % f for f in rank_s))
    print("  textured: %s" % " > ".join("%.1f" % f for f in rank_t))
    print("  REORDERED: %s" % ("NO - same order" if rank_s == rank_t else
                               "YES"))

    print("\n=== is the difference a gradient in frequency? ===")
    print("%7s %9s %9s %9s" % ("freq", "smooth", "textured", "delta"))
    d = []
    for f in freqs:
        delta = tx[f]['second'] - sm[f]['second']
        d.append(delta)
        print("%7.1f %9.1f %9.1f %+9.1f" % (f, sm[f]['second'],
                                            tx[f]['second'], delta))
    d = np.array(d)
    fa = np.array(freqs, dtype=float)
    if len(freqs) >= 3:
        slope, intercept = np.polyfit(fa, d, 1)
        pred = slope * fa + intercept
        ss_res = float(((d - pred) ** 2).sum())
        ss_tot = float(((d - d.mean()) ** 2).sum())
        r2 = 1. - ss_res / ss_tot if ss_tot else float('nan')
        print("\n  linear fit of delta vs frequency:")
        print("    slope %+.3f pp/Hz, intercept %+.1f pp, R^2 %.2f"
              % (slope, intercept, r2))
        # An inverse-frequency form is the other shape worth checking.
        islope, iint = np.polyfit(1. / fa, d, 1)
        ipred = islope / fa + iint
        ir2 = 1. - float(((d - ipred) ** 2).sum()) / ss_tot if ss_tot else \
            float('nan')
        print("    inverse fit: %+.1f/f %+.1f, R^2 %.2f" % (islope, iint, ir2))
        best = "proportional to f" if r2 >= ir2 else "inversely prop. to f"
        print("    better-fitting shape: %s" % best)
        print("    mean offset %+.1f pp, scatter about the fit %.1f pp"
              % (d.mean(), float(np.std(d - (pred if r2 >= ir2 else ipred)))))

    print("\n=== verdict, on drawdown alone (the surface-attributable part) ===")
    at = [(f, tx[f]['second'] - sm[f]['second']) for f in freqs]
    better = [f for f, d in at if d > 0]
    worse = [f for f, d in at if d < 0]
    print("  textured MORE detectable at: %s"
          % (", ".join("%.1f" % f for f in better) or "none"))
    print("  textured LESS detectable at: %s"
          % (", ".join("%.1f" % f for f in worse) or "none"))
    print("  scatter about the gradient is the yardstick: a |delta| under it is"
          " not a difference.")

    print("\n=== air noise: SESSION DRIFT, NOT A SURFACE EFFECT ===")
    print("  Measured with the nozzle ~2mm clear of the bed, so the surface")
    print("  cannot influence it.  Smooth ran 07-28, textured 07-29; this is")
    print("  day-to-day drift of the same machine and is NOT divided into the")
    print("  drawdown above.  Listed because a drifting air floor matters on")
    print("  its own: floors characterised one day are stale the next.")
    print("%7s %11s %11s %8s" % ("freq", "07-28", "07-29", "drift"))
    drifts = []
    for f in freqs:
        s2 = float(sorted(sm[f]['sd'])[1])
        t2 = float(sorted(tx[f]['sd'])[1])
        drifts.append(t2 / s2 if s2 else float('nan'))
        print("%7.1f %10.2f %% %10.2f %% %7.2fx" % (f, s2, t2, drifts[-1]))
    print("  session drift %.2fx..%.2fx (median %.2fx) across modes"
          % (min(drifts), max(drifts), float(np.median(drifts))))


if __name__ == "__main__":
    sys.exit(main())
