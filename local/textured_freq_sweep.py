#!/usr/bin/env python3
# Mirror the 2026-07-28 smooth-side frequency sweep on the TEXTURED plate side,
# to test whether the surface reorders the mode ranking or merely applies a
# frequency-dependent gradient on top of it.
#
# Protocol copied from the smooth sweep (traces 169-185 in probe_traces/LABELS.md)
# so the two are directly comparable: same two XY points, same four candidate
# modes, same accel_per_hz, same descent speed, detectors disarmed, driven ~220um
# past a contact measured immediately before.  Frequency order is REVERSED at the
# second point so sequence wear cannot masquerade as a frequency effect - and,
# after the 2026-07-29 finding that textured air noise correlated with position
# in run at -0.61, so it cannot masquerade as a surface effect either.
#
# WARMUP is deliberately NOT passed.  It now defaults to the live configured
# value (0.8 here) instead of the old hardcoded 0.5; leaving it unset means these
# traces also verify that fix.  Every trace should report warmup=0.800 in its
# header and carry no excitation ring-up.
#
# Run on the printer host (zer0).  Homes itself: a deploy leaves it unhomed.

import json
import os
import sys
import time
import urllib.error
import urllib.request

MOONRAKER = "http://localhost:7125"
TRACE_DIR = "/home/who/probe_traces"

# Same points as the smooth sweep.  Both are >=7.5mm clear of every spot already
# touched on this side of the plate.
FREQS = [65.5, 148.0, 172.9, 212.2]
POINTS = [("41.2,94.6", 41.2, 94.6, FREQS),
          ("39.6,28.3", 39.6, 28.3, list(reversed(FREQS)))]

ARM_Z = 2.0
OVERSHOOT = 0.22
REPS = 2
Z_SANITY = 0.15

CONTACT_TMPL = (
    "RESONANCE_PROBE_CONTACT POINT={x},{y},{armz} FREQ={freq} ACCEL_AXIS=z"
    " ACCEL_PER_HZ=120 CAL_MATCH_LIVE=1 SPEED=0.2 HALT_SENS_XYZ=1,1,1"
    " ZMIN={zmin:.4f} SAMPLES=1"
)


def gcode(script, timeout=900):
    req = urllib.request.Request(
        MOONRAKER + "/printer/gcode/script",
        data=json.dumps({"script": script}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, r.read().decode()
    except urllib.error.HTTPError as e:
        return False, e.read().decode()


def responses(since):
    with urllib.request.urlopen(
            MOONRAKER + "/server/gcode_store?count=200", timeout=30) as r:
        store = json.load(r)["result"]["gcode_store"]
    return [e["message"] for e in store if e["time"] > since]


def csv_count():
    return len([f for f in os.listdir(TRACE_DIR) if f.endswith(".csv")])


def trace_facts(n):
    """(final depth, warmup header, ring-up windows) for descent<n>.csv."""
    path = os.path.join(TRACE_DIR, "descent%05d.csv" % (n,))
    warmup, amps, last = None, [], None
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("#"):
                if "warmup=" in ln:
                    warmup = ln.split("warmup=")[1].split()[0]
                continue
            if ln.startswith("mm_below") or not ln.strip():
                continue
            last = ln
            amps.append(float(ln.split(",")[3]))   # amp_z
    depth = float(last.split(",")[0]) if last else 0.0
    # Ring-up: leading windows below 50% of the settled level.  With the warmup
    # fix in place this must be 0; if it is not, the fix did not take.
    ring = 0
    if len(amps) > 60:
        settled = sorted(amps[20:60])[20]
        while ring < len(amps) and amps[ring] < 0.5 * settled:
            ring += 1
    return depth, warmup, ring


def probe_contact_z(x, y):
    t0 = time.time() - 1
    ok, body = gcode("G1 X%.1f Y%.1f Z%.1f F3000\nPROBE" % (x, y, ARM_Z))
    if not ok:
        raise RuntimeError("PROBE failed at %.1f,%.1f: %s" % (x, y, body))
    zs, result = [], None
    for msg in responses(t0):
        if "contact at z=" not in msg:
            continue
        try:
            z = float(msg.rsplit("contact at z=", 1)[1].split()[0])
        except (ValueError, IndexError):
            continue
        if msg.lstrip("/ ").startswith("Result:"):
            result = z
        else:
            zs.append(z)
    if result is not None:
        return result
    if not zs:
        raise RuntimeError("no contact Z from PROBE at %.1f,%.1f" % (x, y))
    return zs[-1]


def main():
    ok, body = gcode("SET_IDLE_TIMEOUT TIMEOUT=3600")
    if not ok:
        sys.exit("idle timeout: %s" % body)
    print("homing", flush=True)
    ok, body = gcode("G28")
    if not ok:
        sys.exit("G28 failed: %s" % body)

    log = []
    for (name, x, y, freqs) in POINTS:
        z = probe_contact_z(x, y)
        print("\npoint %s contact_z=%+.6f  (freq order: %s)"
              % (name, z, ", ".join("%.1f" % f for f in freqs)), flush=True)
        if abs(z) > Z_SANITY:
            sys.exit("ABORT: contact_z=%+.4f at %s is more than %.2fmm from "
                     "z=0; re-set z=0 before driving %.2fmm past contact."
                     % (z, name, Z_SANITY, OVERSHOOT))
        truth = ARM_Z - z
        for freq in freqs:
            for rep in range(REPS):
                n = csv_count()
                ok, body = gcode(CONTACT_TMPL.format(
                    x=x, y=y, armz=ARM_Z, freq=freq, zmin=z - OVERSHOOT))
                if not ok and "no contact detected" not in body:
                    print("  %6.1fHz rep%d: unexpected error: %s"
                          % (freq, rep + 1, body.strip()[:160]), flush=True)
                if csv_count() <= n:
                    sys.exit("ABORT: %.1fHz rep%d at %s wrote no trace - not "
                             "producing corpus data. Last response: %s"
                             % (freq, rep + 1, name, body.strip()[:250]))
                depth, warmup, ring = trace_facts(n)
                short = truth + OVERSHOOT - depth
                # 65.5Hz is KNOWN to stop short of its commanded floor on the
                # smooth side, so a short descent is data, not a failure - warn
                # and keep going rather than abort the sweep at its first mode.
                flag = ""
                if ring:
                    flag += "  RING-UP=%d (warmup fix did not take!)" % ring
                if short > 0.05:
                    flag += "  SHORT by %.0fum" % (short * 1000.)
                print("  %6.1fHz rep%d: descent%05d.csv warmup=%s reached "
                      "%.3f (overshoot %+.3f)%s"
                      % (freq, rep + 1, n, warmup, depth, depth - truth, flag),
                      flush=True)
                log.append((name, freq, n, z, depth - truth, warmup, ring))
        gcode("G1 Z5.0 F600")

    print("\n=== LABELS.md rows ===", flush=True)
    for (name, x, y, freqs) in POINTS:
        cz = [r[3] for r in log if r[0] == name]
        print("point %s contact_z=%+.6f" % (name, cz[0] if cz else float('nan')),
              flush=True)
        for freq in freqs:
            tr = [str(r[2]) for r in log if r[0] == name and r[1] == freq]
            print("| %s | %.1f | %s |" % (", ".join(tr), freq, name), flush=True)
    bad = [r[2] for r in log if r[6]]
    if bad:
        print("\nWARNING traces with ring-up: %s" % (bad,), flush=True)


if __name__ == "__main__":
    sys.exit(main())
