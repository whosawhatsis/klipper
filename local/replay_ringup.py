#!/usr/bin/env python3
# Is the excitation still ringing UP when detection starts?
#
# The warmup gate drops windows before _armed_time, so a trace begins at the
# moment detection begins.  If the amplitude is still climbing across the early
# part of a trace, the gate let detection start too soon - and a rising, low
# amplitude is where the gradient test is least reliable, which is the shape of
# the recurring false halt near the start height.
#
# Statistic per trace: mean of the first 8 windows (what _dbg_amp0 reports, and
# what the air guard uses as a reference) divided by that trace's own plateau,
# taken as the median of windows 40-120.  Below 1.0 means still ringing up.
#
# Only traces long enough to HAVE a plateau can be scored, which biases the
# sample toward descents that did not halt early - i.e. this UNDERSTATES the
# problem, because the earliest false halts are unscoreable.
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_submerged import load  # noqa: E402

N_START = 8
PLATEAU_FROM, PLATEAU_TO = 40, 120


def score(rows):
    if len(rows) < PLATEAU_FROM + 20:
        return None
    out = []
    for a in range(3):
        early = sum(r[1 + a] for r in rows[:N_START]) / float(N_START)
        mid = sorted(r[1 + a] for r in rows[PLATEAU_FROM:PLATEAU_TO])
        if not mid:
            return None
        plateau = mid[len(mid) // 2]
        if plateau <= 1e-9:
            return None
        out.append(early / plateau)
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "probe_traces"
    files = sorted(glob.glob(os.path.join(root, "descent*.csv")))
    scored, short = [], 0
    for path in files:
        hdr, rows = load(path)
        s = score(rows)
        if s is None:
            short += 1
            continue
        scored.append((s, os.path.basename(path), hdr.get('freq', '?')))
    print("descent traces: %d total, %d scoreable, %d too short to have a plateau"
          % (len(files), len(scored), short))
    if not scored:
        return
    # Report on the axis with the strongest signal, matching how the air guard
    # and the fixed verify both judge on the best channel.
    best = sorted(max(s) for s, _, _ in scored)
    n = len(best)

    def pct(p):
        return best[min(n - 1, int(p * n))]
    print("\nfirst-8-window amplitude as a fraction of the trace's own plateau")
    print("  (1.00 = fully rung up when detection began; <1 = still rising)")
    print("  p05 %.2f   p25 %.2f   median %.2f   p75 %.2f   p95 %.2f"
          % (pct(.05), pct(.25), pct(.50), pct(.75), pct(.95)))
    for thr in (0.95, 0.90, 0.80, 0.70, 0.50):
        k = sum(1 for b in best if b < thr)
        print("  below %.2f: %4d / %d  (%.1f%%)" % (thr, k, n, 100. * k / n))
    print("\nworst 12:")
    for s, name, freq in sorted(scored, key=lambda t: max(t[0]))[:12]:
        print("  %-22s freq=%-7s x=%.2f y=%.2f z=%.2f"
              % (name, freq, s[0], s[1], s[2]))


if __name__ == '__main__':
    main()
