#!/usr/bin/env python3
# Does (refined_edge - z_cand) detect a SHALLOW overshoot, where the air-plateau
# test cannot?
#
# The air test fails below ~0.15mm because the contact signal is flat-then-cliff:
# the ramp's top plateau is still air-like when the candidate is only slightly
# too deep.  But the EDGE does not care about the plateau level - if the
# candidate is overshot by d, the cliff sits d higher relative to it, so the
# offset between the candidate and the edge should track d one-for-one, for as
# long as the cliff stays inside the ramp.
#
# Simulated against the same 12 labelled overshoot descents: take the descent's
# own amplitude-vs-depth curve over the span a verify ramp would cover, and run
# the same step-based edge finder _ramp_edge uses on its DOWN ramp.
#
# What this cannot test: the up ramp (different dynamics, and the corpus has no
# rising traces at these points), and the ramp's 0.5mm/s vs the descent's
# 0.2mm/s.  Treat the numbers as the shape of the effect, not its calibration.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_submerged import LABELLED, load, ARM_Z, VERIFY_UP  # noqa: E402

VERIFY_DOWN = 0.25
GRID = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def edge_in_span(rows, top_depth, bot_depth, axis):
    """Step-based edge, the same rule as _ramp_edge's descending branch: the
    most NEGATIVE fractional step between adjacent windows, taken at the
    midpoint of the straddling pair.  Returns a depth, or None."""
    span = [r for r in rows if top_depth <= r[0] <= bot_depth]
    if len(span) < 4:
        return None
    span.sort(key=lambda r: r[0])          # descending in Z == increasing depth
    amps = [r[1 + axis] for r in span]
    steps = [(amps[i + 1] - amps[i]) / max(amps[i], 1e-9)
             for i in range(len(amps) - 1)]
    if not steps:
        return None
    j = min(range(len(steps)), key=lambda i: steps[i])
    if steps[j] >= 0.:
        return None                        # no drop anywhere in the span
    return 0.5 * (span[j][0] + span[j + 1][0])


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "probe_traces"
    print("Edge offset = (candidate depth - edge depth), in mm.")
    print("POSITIVE means the edge sits ABOVE the candidate, i.e. the candidate")
    print("was already past the surface.  If this tracks the overshoot d, it is")
    print("a detector for exactly the range the air-plateau test is blind to.\n")
    print("%-9s | %s" % ("trace", "  ".join("d=%.2f" % d for d in GRID)))
    print("-" * 76)
    per_d = dict((d, []) for d in GRID)
    for tag, x, y, cz in LABELLED:
        path = os.path.join(root, "descent%s.csv" % tag)
        if not os.path.exists(path):
            continue
        hdr, rows = load(path)
        if not rows:
            continue
        truth = ARM_Z - cz
        deepest = max(r[0] for r in rows)
        # Judge on the axis with the biggest overall drop, as the fixed code now
        # picks the strongest responder.
        ref = [sum(r[1 + a] for r in rows[:8]) / 8. for a in range(3)]
        deep = [min(r[1 + a] for r in rows) for a in range(3)]
        axis = max(range(3), key=lambda a: (ref[a] - deep[a]) / max(ref[a], 1e-9))
        cells = []
        for d in GRID:
            cand = truth + d
            top, bot = cand - VERIFY_UP, min(cand + VERIFY_DOWN, deepest)
            e = edge_in_span(rows, top, bot, axis)
            if e is None:
                cells.append("   n/a")
                continue
            off = cand - e
            cells.append("%+6.3f" % off)
            per_d[d].append(off)
        print("%-9s | %s" % (tag, "  ".join(cells)))

    print("\n--- median offset by true overshoot ---")
    base = None
    for d in GRID:
        v = sorted(per_d[d])
        if not v:
            print("  d=%.2f  no samples" % d)
            continue
        med = v[len(v) // 2]
        if base is None:
            base = med
        print("  d=%.2f  n=%2d  median %+.3f   (rise over d=0: %+.3f)"
              % (d, len(v), med, med - base))
    print("\nIf 'rise over d=0' tracks d one-for-one, the offset measures the")
    print("overshoot directly and a threshold on it catches the shallow case.")


if __name__ == '__main__':
    main()
