#!/usr/bin/env python3
# Drawdown vs derivative on the same recorded windows.
#
# Live, the two race and the winner truncates the descent, so a halting trace
# can never show what the loser would have done.  The disarmed overshoot traces
# can: they run PAST contact with every detector off, so both detectors can be
# replayed over the identical window stream, and the labelled ones carry a real
# contact Z rather than a guess.
#
# The question is whether drawdown wins on merit or merely on sensitivity.  A
# detector that fires sooner AND false-alarms more is just tuned hotter; one
# that fires sooner at the SAME false-alarm rate is genuinely better.
import os, sys, glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'klippy', 'extras'))
import analog_contact
import os as _os
_TRACES = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                       "..", "probe_traces")

WARM = 40           # windows drawdown needs before it can arm at all
DETECT_ZONE = 0.15  # mm; a fire above (truth - this) is a false alarm

# Floors as recorded in the headers of armed descents at the current config
# (172.9Hz / aph120).  The disarmed traces carry halt_floor=1.0 by construction,
# so they cannot supply their own - these are what the live probe would use.
GRAD_FLOOR = (0.050, 0.074, 0.060)
DERIV_FLOOR = tuple(max(0.08, 1.5 * g) for g in GRAD_FLOOR)
DD_SENS, DD_NSIGMA = 0.10, 8.


def load(path):
    hdr, zs, amps = {}, [], []
    for ln in open(path):
        if ln.startswith('#'):
            for kv in ln[1:].split():
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    hdr[k] = v
        elif ',' in ln and not ln.startswith('mm_below'):
            f = ln.split(',')
            zs.append(float(f[0]))
            amps.append([float(x) for x in f[1:4]])
    return hdr, zs, amps


def fire_drawdown(zs, amps, lookback=0.06):
    """First trigger, mirroring the live path (armed after WARM windows)."""
    first = None
    for ax in range(3):
        air = [a[ax] for a in amps[:WARM]]
        base = sum(air) / len(air)
        sd = analog_contact.estimate_noise(air) / base if base > 1e-9 else 0.
        det = analog_contact.ContactDetector(
            analog_contact.DRAWDOWN, max(DD_SENS, DD_NSIGMA * sd), 0.2, 0.2 / 0.01,
            relative=True, persist_mm=0.02, lookback_mm=lookback)
        for z, a in zip(zs[WARM:], amps[WARM:]):
            if det.update(a[ax], z):
                first = z if first is None else min(first, z)
                break
    return first


def fire_derivative(zs, amps, start=1, persist=2):
    """Two consecutive per-window relative down-steps past the axis floor.

    Mirrors resonance_probe.py: step = (v - prev)/prev, no memory beyond one
    window.  `start` exists because the live derivative is NOT gated by
    drawdown's 40-window warmup - it is exposed to the whole descent.
    """
    first = None
    for ax in range(3):
        thr = DERIV_FLOOR[ax]
        run, prev = 0, None
        for i, (z, a) in enumerate(zip(zs, amps)):
            v = a[ax]
            if prev is not None and prev > 1e-9 and i >= start:
                if (v - prev) / prev <= -thr:
                    run += 1
                    if run >= persist:
                        first = z if first is None else min(first, z)
                        break
                else:
                    run = 0
            prev = v
    return first


def main(corpus):
    # --- labelled overshoot traces: contact Z known, data continues past it ---
    # trace -> (contact_z, freq, aph)
    LABELLED = {}
    for tr, cz in ((140, -0.001267), (141, -0.001267), (143, -0.010620),
                   (144, -0.010620), (146, 0.011829), (147, 0.011829),
                   (149, -0.000344), (150, -0.000344),
                   (173, 0.002519), (174, 0.002519),      # 172.9 in freq sweep
                   (180, -0.017753), (181, -0.017753),
                   (189, -0.006817), (190, -0.006817),    # aph120 in aph sweep
                   (196, -0.003029), (197, -0.003029)):
        LABELLED[tr] = cz

    print("=== labelled overshoot descents (172.9Hz, aph120, contact known) ===")
    print("%-6s %8s %10s %10s %9s" % ("trace", "truth", "drawdown", "derivative",
                                      "deriv-dd"))
    dd_d, dv_d, dd_early, dv_early, dd_miss, dv_miss = [], [], 0, 0, 0, 0
    for tr in sorted(LABELLED):
        p = os.path.join(corpus, "descent%05d.csv" % tr)
        if not os.path.exists(p):
            continue
        hdr, zs, amps = load(p)
        truth = 2.0 - LABELLED[tr]
        dd = fire_drawdown(zs, amps)
        dv = fire_derivative(zs, amps)
        def score(f, depths):
            if f is None:
                return "none", 1, 0
            if f < truth - DETECT_ZONE:
                return "EARLY %.3f" % f, 0, 1
            depths.append((f - truth) * 1000.)
            return "%+.0fum" % ((f - truth) * 1000.), 0, 0
        s1, m1, e1 = score(dd, dd_d)
        s2, m2, e2 = score(dv, dv_d)
        dd_miss += m1; dd_early += e1
        dv_miss += m2; dv_early += e2
        gap = ("%+.0fum" % ((dv - dd) * 1000.)) if (dd and dv) else "-"
        print("%-6d %8.3f %10s %10s %9s" % (tr, truth, s1, s2, gap))

    def summ(name, depths, early, miss, n):
        d = ("%.0f..%.0fum (med %.0f)"
             % (min(depths), max(depths),
                sorted(depths)[len(depths) // 2])) if depths else "-"
        print("  %-11s detected %2d/%2d  depth %-26s early %d  miss %d"
              % (name, len(depths), n, d, early, miss))
    n = len([t for t in LABELLED if os.path.exists(
        os.path.join(corpus, "descent%05d.csv" % t))])
    print()
    summ("drawdown", dd_d, dd_early, dd_miss, n)
    summ("derivative", dv_d, dv_early, dv_miss, n)

    # --- false alarms: descents that never touched the bed at all ---
    print("\n=== never-touched descents (any fire is a false alarm) ===")
    air_traces = []
    for p in sorted(glob.glob(os.path.join(corpus, 'descent*.csv'))):
        hdr, zs, amps = load(p)
        if len(zs) < WARM + 5 or not hdr.get('halt_floor', '').startswith('1.0000'):
            continue
        if os.path.basename(p) in ["descent%05d.csv" % t for t in LABELLED]:
            continue
        # a descent that never touched: no window ever falls far below its peak
        pk = [0.] * 3
        touched = False
        for a in amps:
            for ax in range(3):
                pk[ax] = max(pk[ax], a[ax])
                if pk[ax] > 0 and a[ax] <= 0.5 * pk[ax]:
                    touched = True
        if not touched:
            air_traces.append((hdr, zs, amps))
    dd_fa = sum(1 for h, z, a in air_traces if fire_drawdown(z, a) is not None)
    dv_fa = sum(1 for h, z, a in air_traces if fire_derivative(z, a) is not None)
    dv_fa_late = sum(1 for h, z, a in air_traces
                     if fire_derivative(z, a, start=WARM) is not None)
    print("  n=%d pure-air descents" % len(air_traces))
    print("  drawdown            false alarms: %2d  (%.0f%%)"
          % (dd_fa, 100. * dd_fa / max(len(air_traces), 1)))
    print("  derivative          false alarms: %2d  (%.0f%%)"
          % (dv_fa, 100. * dv_fa / max(len(air_traces), 1)))
    print("  derivative (warmed) false alarms: %2d  (%.0f%%)   <- if it were"
          " gated like drawdown" % (dv_fa_late,
                                    100. * dv_fa_late / max(len(air_traces), 1)))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else _TRACES)
