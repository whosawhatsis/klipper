#!/usr/bin/env python3
"""Contact detection as CHANGE-POINT detection, not threshold crossing.

In air the toolhead is a linear driven resonator: after ring-up its per-axis
amplitude follows a smooth, predictable trajectory (slow drift as Z changes).
Contact makes all three axes depart from that trajectory.  WHAT they do varies
with press depth, platform stiffness and texture - z may fall while y rises -
so no fixed sign or magnitude can be assumed.  What is reliable is that the
departures are COINCIDENT.

So: fit a model to recent air behaviour, FREEZE it, and score how badly it
predicts the next few windows.  Freezing matters - a trailing fit that keeps
adapting absorbs the contact and stops flagging it (measured: peak z of 66 that
never survives three consecutive windows).

The statistic is joint across axes rather than three detectors plus a
coincidence check: on the 2026-09-20 capture the joint form fired on 3/3
bursts where per-axis fired on 1/3, and never later.

Threshold is relative to each burst's own baseline, so nothing is calibrated
per bed position - which is the point, since absolute contact damping varies
2x across this bed (83% at centre, 37% at (60,25)).

Input: raw ACCELEROMETER_MEASURE csv (#time,accel_x,accel_y,accel_z).
  ACCELEROMETER_MEASURE CHIP=adxl345 NAME=x   <- before
  ...probe...
  ACCELEROMETER_MEASURE CHIP=adxl345 NAME=x   <- after
Trace CSVs are NOT usable: they store post-DFT amp_x/y/z, raw samples gone.
"""
import sys
import numpy as np

FREQ = 73.0
CYCLES = 10        # analysis window, in excitation cycles
FIT = 15           # windows of air used for the frozen model
FWD = 6            # windows judged against it


def windows(t, a, freq=FREQ, cycles=CYCLES):
    fs = 1.0 / np.median(np.diff(t))
    w = int(round(cycles / freq * fs))
    hop = max(1, w // 4)
    ref = np.exp(-2j * np.pi * freq * np.arange(w) / fs)
    out = []
    for s in range(0, len(t) - w, hop):
        seg = a[s:s + w] - a[s:s + w].mean(0)
        out.append([t[s + w // 2]] + [2.0 / w * abs(seg[:, ax] @ ref)
                                      for ax in range(3)])
    return np.array(out), w / fs, hop / fs


def bursts(amp, frac=0.3):
    live = amp > frac * amp.max()
    out, start = [], None
    for i, v in enumerate(live):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(live) - 1))
    return out


def joint(y_axes):
    """Mean over axes of frozen-model prediction error, in sigma."""
    n = len(y_axes[0])
    j = np.full(n, np.nan)
    for i in range(FIT + 2, n - FWD):
        per = []
        for y in y_axes:
            seg = y[i - FIT:i]
            x = np.arange(FIT)
            c = np.polyfit(x, seg, 1)
            sd = max((seg - np.polyval(c, x)).std(), 1e-6)
            per.append(np.mean([abs(y[i + k] - np.polyval(c, FIT + k)) / sd
                                for k in range(FWD)]))
        j[i] = np.mean(per)
    return j


def main(path):
    d = np.loadtxt(path, delimiter=',', skiprows=1)
    W, wlen, hop = windows(d[:, 0], d[:, 1:4])
    t = W[:, 0] - W[0, 0]
    print("windows=%d  len=%.0fms  hop=%.0fms" % (len(W), wlen * 1e3, hop * 1e3))
    for k, (a, b) in enumerate(bursts(W[:, 1])):
        if t[b] - t[a] < 0.3:
            continue
        rel = t[a:b] - t[a]
        ys = [np.log(np.maximum(W[a:b, 1 + ax], 1.0)) for ax in range(3)]
        j = joint(ys)
        ok = ~np.isnan(j)
        base = np.median(j[ok][:max(len(j[ok]) // 3, 1)])
        thr = max(4 * base, 6.0)
        hit = [i for i in range(len(rel)) if ok[i] and j[i] > thr]
        print("burst %d (%.2f-%.2fs): base=%.2f peak=%.1f thr=%.1f -> %s"
              % (k + 1, t[a], t[b], base, np.nanmax(j), thr,
                 "onset %.3fs" % rel[hit[0]] if hit else "none"))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'adxl345-rawprobe.csv')
