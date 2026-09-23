#!/usr/bin/env python3
# What happens if a descent IGNORES an amplitude rise and keeps going?
#
# Verify currently aborts a point when contact RAISES the measured amplitude
# ("inverted coupling"), because descending past it once over-pressed 0.282mm.
# The labelled overshoot descents can price the alternative: detectors were
# disarmed, the nozzle was driven 0.17-0.30mm past a contact whose Z was measured
# immediately before, and every axis was recorded the whole way.
#
# For each trace, amplitude is normalised to that trace's own start-of-descent
# air level and tabulated against depth RELATIVE TO TRUE CONTACT.  Three
# questions:
#   1. does a rise actually occur at contact, and on which axes?
#   2. if it does, how far BELOW contact must the descent continue before a
#      confirmable drop appears?
#   3. is any axis still dropping cleanly when the others rise - i.e. would
#      all-axis verification have found contact without descending at all?
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_submerged import LABELLED, load, ARM_Z, N_START  # noqa: E402

DEPTHS = (-0.10, -0.05, -0.02, 0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
CONFIRM_DROP = 0.15          # VERIFY_DROP: a 15% fall is a confirmable contact
RISE_ABORT = 0.10            # verify_rise_abort: a 10% climb aborts the point


def amp_at(rows, depth, tol=0.02):
    best = min(rows, key=lambda r: abs(r[0] - depth))
    return best[1:4] if abs(best[0] - depth) <= tol else None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "probe_traces"
    print("Amplitude relative to each trace's own start-of-descent air,")
    print("tabulated by depth BELOW true contact.  >1.00 = rise (inverted).\n")
    hdr = "  ".join("%+5.2f" % d for d in DEPTHS)
    print("%-9s %-4s %s" % ("trace", "axis", hdr))
    print("-" * (16 + len(hdr)))

    per_axis = dict((a, dict((d, []) for d in DEPTHS)) for a in range(3))
    rise_then_drop = []
    for tag, x, y, cz in LABELLED:
        path = os.path.join(root, "descent%s.csv" % tag)
        if not os.path.exists(path):
            continue
        hdrs, rows = load(path)
        if not rows:
            continue
        truth = ARM_Z - cz
        ref = [sum(r[1 + a] for r in rows[:N_START]) / float(N_START)
               for a in range(3)]
        for a in range(3):
            cells = []
            for d in DEPTHS:
                v = amp_at(rows, truth + d)
                if v is None or ref[a] <= 1e-9:
                    cells.append("  n/a")
                    continue
                f = v[a] / ref[a]
                cells.append("%5.2f" % f)
                per_axis[a][d].append(f)
            print("%-9s %-4s %s" % (tag if a == 0 else "", 'xyz'[a],
                                    "  ".join(cells)))
        # Where would a rise-then-continue strategy have ended up?
        for a in range(3):
            if ref[a] <= 1e-9:
                continue
            rose_at = None
            for d in DEPTHS:
                v = amp_at(rows, truth + d)
                if v is None:
                    continue
                f = v[a] / ref[a]
                if rose_at is None and d >= 0. and f >= 1. + RISE_ABORT:
                    rose_at = d
                if rose_at is not None and f <= 1. - CONFIRM_DROP:
                    rise_then_drop.append((tag, 'xyz'[a], rose_at, d, d - rose_at))
                    break

    print("\n--- median across traces ---")
    print("%-9s %-4s %s" % ("", "axis", hdr))
    for a in range(3):
        cells = []
        for d in DEPTHS:
            v = sorted(per_axis[a][d])
            cells.append("%5.2f" % v[len(v) // 2] if v else "  n/a")
        print("%-9s %-4s %s" % ("", 'xyz'[a], "  ".join(cells)))

    print("\n--- if a rise is IGNORED and the descent continues ---")
    if not rise_then_drop:
        print("  No axis that rose >=%.0f%% at or below contact ever reached a"
              % (RISE_ABORT * 100.))
        print("  %.0f%% drop within the recorded travel." % (CONFIRM_DROP * 100.))
        print("  So continuing past a rise does NOT find a confirmable contact")
        print("  in this corpus - it just presses deeper.")
    else:
        for tag, ax, rose, drop, extra in rise_then_drop:
            print("  %s axis %s: rose at %+.2f, confirmable drop at %+.2f"
                  " -> %.0f um of extra press" % (tag, ax, rose, drop,
                                                  extra * 1000.))
        ex = sorted(r[4] for r in rise_then_drop)
        print("  median extra press: %.0f um   worst: %.0f um"
              % (ex[len(ex) // 2] * 1000., ex[-1] * 1000.))

    print("\n--- was any axis dropping while others rose, AT contact? ---")
    for d in (0.00, 0.02, 0.05):
        vals = [(sorted(per_axis[a][d])[len(per_axis[a][d]) // 2], 'xyz'[a])
                for a in range(3) if per_axis[a][d]]
        if not vals:
            continue
        lo = min(vals)
        print("  at %+.2f: strongest DROP is %s at %.2f; per-axis %s"
              % (d, lo[1], lo[0], ", ".join("%s=%.2f" % (b, f) for f, b in vals)))


if __name__ == '__main__':
    main()
