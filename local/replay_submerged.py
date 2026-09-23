#!/usr/bin/env python3
# Replay _submerged_fraction against the LABELLED OVERSHOOT descents.
#
# Fix 3 asks: when verify's ramp starts BELOW the surface, is its 'air' end
# (z_cand + VERIFY_UP) measurably damped against the air this same descent saw
# at its start?  The labelled overshoot traces answer it directly - detectors
# disarmed, driven past a contact whose Z was measured immediately before, with
# per-axis amplitude recorded the whole way down.
#
# For a hypothetical candidate at depth d below contact, verify's ramp top sits
# at (d - VERIFY_UP).  Reading the descent's amplitude there and dividing by its
# start-of-descent amplitude reproduces exactly the statistic the detector
# computes.  No numpy: the amplitudes are already in the CSVs.
#
# Caveat this CANNOT test: the ramp is a different motion from the descent
# (0.5mm/s vs 0.2mm/s, and it arrives from below on the up-ramp), so this
# measures the amplitude-vs-height curve, not the ramp's own dynamics.
import glob
import os
import sys

GRID = (-0.05, 0.00, 0.05, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40)

VERIFY_UP = 0.15          # gcmd VERIFY_UP default
ARM_Z = 2.0               # traces are mm_below_arm with the arm at Z=2.0
N_START = 8               # _dbg_amp0 keeps the first 8 windows

# From LABELS.md "Labelled overshoot descents".  truth = ARM_Z - contact_z.
LABELLED = [
    ("00140", 64.9, 73.5, -0.001267), ("00141", 64.9, 73.5, -0.001267),
    ("00143", 73.2, 60.9, -0.010620), ("00144", 73.2, 60.9, -0.010620),
    ("00146", 80.2, 33.7, +0.011829), ("00147", 80.2, 33.7, +0.011829),
    ("00149", 31.4, 69.6, -0.000344), ("00150", 31.4, 69.6, -0.000344),
    ("00173", 41.2, 94.6, +0.002519), ("00174", 41.2, 94.6, +0.002519),
    ("00180", 39.6, 28.3, -0.017753), ("00181", 39.6, 28.3, -0.017753),
]


def load(path):
    rows, hdr = [], {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                for tok in line[1:].split():
                    if '=' in tok:
                        k, v = tok.split('=', 1)
                        hdr[k] = v
                continue
            if not line or line.startswith('mm_below_arm'):
                continue
            p = line.split(',')
            if len(p) < 4:
                continue
            try:
                rows.append(tuple(float(x) for x in p[:4]))
            except ValueError:
                pass
    return hdr, rows


def amp_at(rows, depth):
    """Per-axis amplitude at a depth, nearest recorded window."""
    best = min(rows, key=lambda r: abs(r[0] - depth))
    if abs(best[0] - depth) > 0.05:
        return None
    return best[1:4]


def start_air(rows):
    n = min(N_START, len(rows))
    if not n:
        return None
    return [sum(r[1 + a] for r in rows[:n]) / n for a in range(3)]


def frac(rows, depth):
    """max over axes of amp(depth)/amp(start) - the detector's statistic."""
    a = amp_at(rows, depth)
    ref = start_air(rows)
    if a is None or ref is None:
        return None
    pairs = [(a[i] / ref[i], i) for i in range(3) if ref[i] > 1e-9]
    return max(pairs) if pairs else None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "probe_traces"
    print("candidate depths are BELOW the true contact; verify's ramp top sits")
    print("VERIFY_UP=%.2fmm above the candidate.  A ramp top below contact is a" % VERIFY_UP)
    print("TRUE overshoot and the detector must fire.\n")
    print("%-9s %8s | %s" % ("trace", "truth", "max-axis frac at ramp top, by candidate overshoot"))
    print("%-9s %8s | %s" % ("", "(mm)", "  ".join("%+.2f" % d for d in
                                                   GRID)))
    print("-" * 88)
    submerged_vals, air_vals = [], []
    for tag, x, y, cz in LABELLED:
        path = os.path.join(root, "descent%s.csv" % tag)
        if not os.path.exists(path):
            print("%-9s MISSING" % tag)
            continue
        hdr, rows = load(path)
        if not rows:
            print("%-9s empty" % tag)
            continue
        truth = ARM_Z - cz
        deepest = max(r[0] for r in rows)
        cells = []
        for d in GRID:
            top_depth = truth + d - VERIFY_UP
            if top_depth > deepest:
                cells.append("  n/a")
                continue
            f = frac(rows, top_depth)
            if f is None:
                cells.append("  n/a")
                continue
            cells.append("%5.2f" % f[0])
            (submerged_vals if top_depth > truth else air_vals).append(
                (f[0], tag, d, 'xyz'[f[1]]))
        print("%-9s %8.4f | %s" % (tag, truth, "  ".join(cells)))

    print("\n--- separation ---")
    for name, vals in (("ramp top BELOW contact (true overshoot)", submerged_vals),
                       ("ramp top ABOVE contact (legitimate)", air_vals)):
        if not vals:
            print("%-42s no samples" % name)
            continue
        fs = sorted(v[0] for v in vals)
        print("%-42s n=%2d  min %.2f  median %.2f  max %.2f"
              % (name, len(fs), fs[0], fs[len(fs) // 2], fs[-1]))
    if submerged_vals and air_vals:
        worst_sub = max(v[0] for v in submerged_vals)
        best_air = min(v[0] for v in air_vals)
        print("\nA threshold must sit ABOVE %.2f to catch every overshoot here,"
              % worst_sub)
        print("and BELOW %.2f to never fire on a legitimate contact." % best_air)
        if worst_sub < best_air:
            print("SEPARABLE - any threshold in (%.2f, %.2f) works; midpoint %.2f."
                  % (worst_sub, best_air, 0.5 * (worst_sub + best_air)))
        else:
            print("NOT SEPARABLE on this statistic - the classes overlap.")


if __name__ == '__main__':
    main()
