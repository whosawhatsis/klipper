#!/usr/bin/env python3
# What do the toolhead fans do to the accelerometer the probe depends on?
#
# The probe does not use broadband noise - it measures the amplitude of a SINGLE
# DFT bin at the excitation frequency, and derives its halt threshold from the
# scatter of that value in air.  So the question is not "are the fans loud", it
# is "do the fans put energy at, or near, the frequency we excite".  A fan tone
# landing on the excitation frequency raises the in-band noise, which raises the
# self-derived threshold max(0.10, 8*sd), which shrinks detection margin.
#
# Usage: fan_noise_eval.py <dir-with-adxl345-*.csv>
import os, sys, glob, math, cmath

CANDIDATES = (65.5, 148.0, 172.9, 212.2)   # modes we have characterised
BAND = 6.0                                  # +/- Hz counted as "in band"


def load(path):
    """Klipper adxl345 CSV: time,accel_x,accel_y,accel_z."""
    t, ax, ay, az = [], [], [], []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln[0] not in '0123456789-.':
            continue
        f = ln.split(',')
        if len(f) < 4:
            continue
        t.append(float(f[0]))
        ax.append(float(f[1]))
        ay.append(float(f[2]))
        az.append(float(f[3]))
    return t, (ax, ay, az)


def goertzel(vals, freq, rate):
    """Single-bin DFT amplitude - the same quantity the probe thresholds on."""
    n = len(vals)
    if n < 8:
        return 0.
    w = 2. * math.pi * freq / rate
    acc = sum(v * cmath.exp(-1j * w * i) for i, v in enumerate(vals))
    return 2. * abs(acc) / n


def spectrum(vals, rate, fmax=400., step=2.):
    out = []
    f = step
    while f <= fmax:
        out.append((f, goertzel(vals, f, rate)))
        f += step
    return out


def rms(vals):
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def main(d):
    paths = sorted(glob.glob(os.path.join(d, 'adxl345-*.csv')))
    if not paths:
        print("no adxl345-*.csv in %s" % d)
        return
    results = {}
    for p in paths:
        name = os.path.basename(p).replace('adxl345-', '').replace('.csv', '')
        t, axes = load(p)
        if len(t) < 100:
            print("  %s: too few samples (%d)" % (name, len(t)))
            continue
        rate = (len(t) - 1) / max(t[-1] - t[0], 1e-9)
        results[name] = (rate, axes)
        print("%-12s %6d samples @ %.0f Hz   rms x=%.1f y=%.1f z=%.1f"
              % (name, len(t), rate, rms(axes[0]), rms(axes[1]), rms(axes[2])))

    if not results:
        return
    base = 'nofan' if 'nofan' in results else sorted(results)[0]
    print("\nin-band single-bin amplitude at each candidate mode"
          " (ratio vs '%s' in brackets):" % base)
    print("%-12s %s" % ("state", "  ".join("%9.1fHz" % f for f in CANDIDATES)))
    ref = {}
    for name in [base] + [n for n in sorted(results) if n != base]:
        rate, axes = results[name]
        cells = []
        for f in CANDIDATES:
            # worst axis: the probe halts on whichever axis fires first
            amp = max(goertzel(a, f, rate) for a in axes)
            if name == base:
                ref[f] = amp
                cells.append("%11.1f" % amp)
            else:
                r = amp / ref[f] if ref.get(f) else float('nan')
                cells.append("%6.1f(%.1fx)" % (amp, r))
        print("%-12s %s" % (name, "  ".join(cells)))

    print("\nstrongest tones below 400Hz (worst axis), to spot fan lines:")
    for name in [base] + [n for n in sorted(results) if n != base]:
        rate, axes = results[name]
        sp = {}
        for a in axes:
            for f, v in spectrum(a, rate):
                sp[f] = max(sp.get(f, 0.), v)
        top = sorted(sp.items(), key=lambda kv: -kv[1])[:6]
        # Only flag a tone that is actually a TONE.  Ranking by amplitude always
        # yields a "top 6" even in pure noise, and flagging those on proximity
        # alone reports a fan line where there is none (seen on a noise-only
        # synthetic capture).  Require it to stand above the typical bin.
        level = sorted(sp.values())[len(sp) // 2] or 1e-9
        near = [f for f, v in top
                if v >= 4. * level and any(abs(f - c) <= BAND for c in CANDIDATES)]
        print("  %-12s %s%s" % (name,
              "  ".join("%.0fHz:%.0f" % (f, v) for f, v in top),
              ("   <-- WITHIN %.0fHz OF A MODE: %s"
               % (BAND, near)) if near else ""))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
