#!/usr/bin/env python3
# Sweep drawdown detector settings against the descent-trace corpus.
#
# The live detector fires a false halt in mid-air on a sizeable fraction of
# descents (verify catches them, so they cost time rather than accuracy).  This
# replays every recorded descent through analog_contact.ContactDetector so the
# threshold / persistence / lookback question is answered offline instead of by
# wearing the build plate.
#
# Ground truth comes from the trace itself.  A live descent halts AT its
# trigger, so the recorded span says which kind of halt it was:
#   span >= CONTACT_SPAN   halt was at the bed  -> all but the last stretch is air
#   span <  CONTACT_SPAN   halt was in mid-air  -> the WHOLE trace is air
# Traces with drawdown_thresh=unset are verify sweeps or disarmed runs, not
# live descents, and are excluded.
#
# The replay mirrors resonance_probe.py exactly: the detector is CREATED after
# WARM air windows (no earlier history), its threshold is max(sens, nsigma*sd)
# from those windows, and any of the three axes may trigger.
import os, sys, glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'klippy', 'extras'))
import analog_contact

CONTACT_SPAN = 2.05     # mm; a halt shallower than this was in mid-air
DETECT_ZONE = 0.15      # mm above the halt that counts as "found the bed"
WARM = 40               # air windows used for the noise estimate


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


def replay(zs, amps, sens, nsigma, persist_mm, lookback_mm, step_z, speed):
    """First axis trigger as mm_below_arm, or None.  Mirrors the live path."""
    rate = speed / step_z          # 'samples' here are detection windows
    first = None
    for ax in range(3):
        air = [a[ax] for a in amps[:WARM]]
        base = sum(air) / len(air)
        sd = analog_contact.estimate_noise(air) / base if base > 1e-9 else 0.
        det = analog_contact.ContactDetector(
            analog_contact.DRAWDOWN, max(sens, nsigma * sd), speed, rate,
            relative=True, persist_mm=persist_mm, lookback_mm=lookback_mm)
        for z, a in zip(zs[WARM:], amps[WARM:]):
            if det.update(a[ax], z):
                if first is None or z < first:
                    first = z
                break
    return first


def evaluate(traces, sens, nsigma, persist_mm, lookback_mm):
    early, air_fire, miss, press = 0, 0, 0, []
    for hdr, zs, amps, is_real in traces:
        hit = replay(zs, amps, sens, nsigma, persist_mm, lookback_mm,
                     float(hdr.get('step_z', 0.01)),
                     float(hdr.get('speed', 0.2)))
        if is_real:
            if hit is None:
                miss += 1
            elif hit < zs[-1] - DETECT_ZONE:
                early += 1
            else:
                press.append(zs[-1] - hit)
        elif hit is not None:
            air_fire += 1
    return early, air_fire, miss, press


def main(corpus):
    traces = []
    for p in sorted(glob.glob(os.path.join(corpus, 'descent*.csv'))):
        hdr, zs, amps = load(p)
        if len(zs) < WARM + 5:
            continue
        if hdr.get('drawdown_thresh', 'unset').startswith('unset'):
            continue           # verify sweep or disarmed run, not a descent
        traces.append((hdr, zs, amps, (zs[-1] - zs[0]) >= CONTACT_SPAN))
    n_real = sum(1 for t in traces if t[3])
    print("corpus: %d live descents (%d ended at contact, %d false halts)"
          % (len(traces), n_real, len(traces) - n_real))
    print("live baseline is sens=0.10 nsigma=8 persist=2win lookback=0.12\n")
    print("%-5s %-3s %-7s %-8s | %5s %5s %5s | %6s %s"
          % ("sens", "ns", "persist", "lookback", "early", "air", "miss",
             "press", "verdict"))
    rows = []
    for sens in (0.10, 0.15, 0.20, 0.25, 0.30):
        for nsigma in (8., 12., 16.):
            for persist_mm in (0.02, 0.03, 0.05, 0.08):
                for lookback_mm in (0.06, 0.12, 0.25):
                    e, a, m, press = evaluate(traces, sens, nsigma,
                                              persist_mm, lookback_mm)
                    rows.append((e + a + m, e, a, m, sens, nsigma, persist_mm,
                                 lookback_mm,
                                 sum(press) / len(press) if press else -1.))
    rows.sort()
    for tot, e, a, m, sens, ns, pm, lb, mp in rows[:18]:
        print("%-5.2f %-3d %-7.2f %-8.2f | %5d %5d %5d | %6.4f %s"
              % (sens, ns, pm, lb, e, a, m, mp,
                 "CLEAN" if tot == 0 else ("%d bad" % tot)))
    print("\n(early = fired >%.2fmm above the real contact; air = fired at all"
          " on a known false-halt trace; miss = never fired)" % DETECT_ZONE)


if __name__ == '__main__':
    main(sys.argv[1])
