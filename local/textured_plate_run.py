#!/usr/bin/env python3
# Record a labelled-overshoot trace set on the TEXTURED plate side, mirroring
# the 2026-07-27 recipe in probe_traces/LABELS.md so the two surfaces are
# comparable: same XY points, same live detection window, same drive level.
#
# Per point: one ARMED probe for ground-truth contact_z, then two DISARMED
# descents driven ~0.2mm past it.  Disarmed means HALT_SENS_XYZ=1,1,1 - the
# ratio halt can never fire and the drawdown/derivative detectors are unset on
# the CONTACT path - so the descent runs to ZMIN and the trace continues past
# contact.  RESONANCE_PROBE_CONTACT then reports "no contact detected", which
# is the EXPECTED outcome here, not a failure: the trace is autosaved before
# the error is raised.
#
# Run on the printer host (zer0).  A deploy service-restarts klippy and leaves
# the printer unhomed, so this homes itself.

import json
import sys
import time
import urllib.error
import urllib.request

MOONRAKER = "http://localhost:7125"

# Same four points as the 2026-07-27 smooth-side set, so surface is the only
# variable that changed.
POINTS = [(64.9, 73.5), (73.2, 60.9), (80.2, 33.7), (31.4, 69.6)]

ARM_Z = 2.0        # trace coordinates are mm below this; LABELS.md assumes 2.0
OVERSHOOT = 0.20   # mm driven past contact on the disarmed descents
REPS = 2           # disarmed descents per point
# A flipped plate can sit at a different height than the side z=0 was set on.
# Refuse to drive blind past a contact that is nowhere near where the machine
# thinks the bed is - that is the case where OVERSHOOT would bury the nozzle.
Z_SANITY = 0.15

# POINT takes THREE coords (x,y,z) and moves there itself - it is the arm-height
# move, so no separate G1 is needed.
CONTACT_TMPL = (
    "RESONANCE_PROBE_CONTACT POINT={x},{y},{armz} FREQ=172.9 ACCEL_AXIS=z"
    " ACCEL_PER_HZ=120 CAL_MATCH_LIVE=1 SPEED=0.2 HALT_SENS_XYZ=1,1,1"
    " ZMIN={zmin:.4f} SAMPLES=1"
)


def gcode(script, timeout=900):
    # Returns (ok, text).  A Klipper command error comes back as HTTP 400 with
    # the message in the body; the disarmed descents rely on reading it.
    req = urllib.request.Request(
        MOONRAKER + "/printer/gcode/script",
        data=json.dumps({"script": script}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, r.read().decode()
    except urllib.error.HTTPError as e:
        return False, e.read().decode()


def responses(since):
    url = MOONRAKER + "/server/gcode_store?count=200"
    with urllib.request.urlopen(url, timeout=30) as r:
        store = json.load(r)["result"]["gcode_store"]
    return [e["message"] for e in store if e["time"] > since]


def csv_count():
    # Mirrors the module's own numbering (it counts every .csv in trace_dir),
    # so the new files can be named exactly rather than guessed.
    import os
    d = "/home/who/probe_traces"
    return len([f for f in os.listdir(d) if f.endswith(".csv")])


def final_depth(n):
    # Last mm_below_arm in descent<n>.csv - how far the descent actually got.
    path = "/home/who/probe_traces/descent%05d.csv" % (n,)
    last = None
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("#") or ln.startswith("mm_below"):
                continue
            if ln.strip():
                last = ln
    return float(last.split(",")[0]) if last else 0.0


def probe_contact_z(x, y):
    t0 = time.time() - 1
    ok, body = gcode("G1 X%.1f Y%.1f Z%.1f F3000\nPROBE" % (x, y, ARM_Z))
    if not ok:
        raise RuntimeError("PROBE failed at %.1f,%.1f: %s" % (x, y, body))
    # The probe prints one "probe: at X,Y bed will contact at z=NNN" per sample
    # and then "Result: at X,Y estimate contact at z=NNN".  Prefer the Result
    # line; fall back to the last per-sample line.  Both end in "contact at
    # z=", so match that and let the Result line win.
    zs, result_z = [], None
    for msg in responses(t0):
        if "contact at z=" not in msg:
            continue
        try:
            z = float(msg.rsplit("contact at z=", 1)[1].split()[0])
        except (ValueError, IndexError):
            continue
        if msg.lstrip("/ ").startswith("Result:"):
            result_z = z
        else:
            zs.append(z)
    if result_z is not None:
        return result_z
    if not zs:
        raise RuntimeError("could not read a contact Z from PROBE at %.1f,%.1f"
                           % (x, y))
    return zs[-1]


def main():
    log = []
    ok, body = gcode("SET_IDLE_TIMEOUT TIMEOUT=3600")
    if not ok:
        sys.exit("idle timeout: %s" % body)
    print("homing", flush=True)
    ok, body = gcode("G28")
    if not ok:
        sys.exit("G28 failed: %s" % body)

    for (x, y) in POINTS:
        z = probe_contact_z(x, y)
        print("point %.1f,%.1f contact_z=%+.6f" % (x, y, z), flush=True)
        if abs(z) > Z_SANITY:
            sys.exit("ABORT: contact_z=%+.4f at %.1f,%.1f is more than %.2fmm "
                     "from z=0 - the flipped plate has shifted the bed height. "
                     "Re-set z=0 before driving %.2fmm past contact."
                     % (z, x, y, Z_SANITY, OVERSHOOT))
        traces = []
        for rep in range(REPS):
            n = csv_count()   # the descent about to be written takes this index
            ok, body = gcode(CONTACT_TMPL.format(x=x, y=y, armz=ARM_Z,
                                                 zmin=z - OVERSHOOT))
            # A disarmed descent normally runs all the way to ZMIN and returns
            # ok; it can also end with "no contact detected".  Both are fine -
            # what matters is whether the trace actually continued PAST contact,
            # which is checked from the file below.
            if not ok and "no contact detected" not in body:
                print("  rep %d: unexpected error: %s"
                      % (rep + 1, body.strip()[:200]), flush=True)
            after = csv_count()
            if after > n:
                traces.append(n)
                # The whole point of this class of trace is that it continues
                # past contact: an unlabelled trace that stopped AT contact
                # cannot measure trigger depth.  Verify it rather than assume.
                depth = final_depth(n)
                truth = ARM_Z - z
                print("  rep %d: descent%05d.csv reached %.3f mm below arm "
                      "(contact %.3f, overshoot %+.3f)"
                      % (rep + 1, n, depth, truth, depth - truth), flush=True)
                if depth < truth + 0.5 * OVERSHOOT:
                    sys.exit("ABORT: that descent stopped %.3fmm short of a "
                             "usable overshoot - something halted it, so the "
                             "rest of the set would be unusable too."
                             % (truth + OVERSHOOT - depth,))
            else:
                # Every contact wears the plate.  A descent that writes no
                # trace has spent one for nothing, so stop rather than spend
                # the rest the same way.
                sys.exit("ABORT: rep %d at %.1f,%.1f wrote no trace - the "
                         "descent is not producing corpus data.  Last "
                         "response: %s" % (rep + 1, x, y, body.strip()[:300]))
        log.append((x, y, z, traces))
        gcode("G1 Z5.0 F600")

    print("\n=== LABELS.md rows ===", flush=True)
    print("| traces | point | contact_z |", flush=True)
    print("| --- | --- | --- |", flush=True)
    for (x, y, z, traces) in log:
        print("| %s | %.1f, %.1f | %+.6f |"
              % (", ".join(str(t) for t in traces), x, y, z), flush=True)


if __name__ == "__main__":
    main()
