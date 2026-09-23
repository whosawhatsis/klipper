#!/usr/bin/env python3
# Cold platform + HOT nozzle: the combination the earlier runs never tested, and
# the one the results point at as the best way to mesh.
#
# Why this combination:
#  - A COLD bed avoids the transient bow.  While heating, the platform holds an
#    internal temperature differential and deviates from flat, then relaxes as it
#    soaks.  A mesh cannot even measure that (a 3x3 takes 372s while the
#    transient peaks around 6 min), and worse, bed_mesh probes in serpentine
#    ORDER, so a moving surface appears as a plausible-looking bed SHAPE.  A cold
#    bed is static while you sweep it, which is the only condition under which a
#    serial mesh means what it claims.
#  - A HOT nozzle should stop plastic on the tip from holding the probe off.
#    Residue reads HIGH and it has now shown up three times: +69um on the first
#    contact after a print, and again on the first contact after the hot phase.
#    Molten plastic should deform instead of supporting the nozzle.
#
# What this measures: the nozzle-only thermal term, with the bed held constant,
# which the previous run never got cleanly (its hot-nozzle phase had the bed hot
# too AND was wrecked by ooze, sd 34um).
#
# Phases: conditioning (discarded) -> cold baseline -> nozzle hot -> nozzle off
# and returned to the SAME temperature (the reversibility control, which is what
# separates a real thermal shift from the site or the nozzle changing).

import json
import os
import sys
import time
import urllib.error
import urllib.request

MOONRAKER = "http://localhost:7125"
OUT = "/home/who/cold_bed_hot_nozzle.csv"

# 85.0,55.0 is carried over: it PASSED its reversibility control in run 2b
# (returned to -2.9um) and repeats to sub-micron, so it is the reference site.
# 30.0,45.0 is virgin (12.0mm clear of everything).  70.0,45.0 is dropped - it
# came back +35.8um high at matched temperature, so it fails its own control.
POINTS = [(85.0, 55.0), (30.0, 45.0)]

NOZZLE_TARGET = 210.0
CONDITION_N = int(os.environ.get('CONDITION_N', '2'))
# Deeper than the 10mm used before: run 2b's hot-nozzle phase scattered to
# sd 34um at a point that otherwise repeats to 0.2um, and ooze is the prime
# suspect.  Pulling the melt further up the heatbreak is the only mitigation
# available without unloading the filament.
RETRACT_MM = 20.0


def get_json(url, timeout=30, tries=4):
    """GET with retry.  Read-only, so retrying is always safe.

    Moonraker drops a connection occasionally (RemoteDisconnected); a long
    unattended run must not die from one dropped request.
    """
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:          # transport-level, not an HTTP status
            last = e
            time.sleep(2 * (i + 1))
    raise last


def post(script, timeout=900):
    """POST a gcode script.  Returns (ok, body).

    Deliberately does NOT retry.  A gcode POST is not idempotent - the command
    may well have executed before the connection dropped, and a blind retry
    would probe the plate twice.  A transport failure is reported like a command
    failure so the caller can record the sample as lost and carry on, which
    costs one datapoint instead of the whole run.
    """
    req = urllib.request.Request(
        MOONRAKER + "/printer/gcode/script",
        data=json.dumps({"script": script}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, r.read().decode()
    except urllib.error.HTTPError as e:
        return False, e.read().decode()
    except Exception as e:
        return False, "transport error (command may or may not have run): %s" % e


def query(objs):
    return get_json(MOONRAKER + "/printer/objects/query?"
                    + "&".join(objs))["result"]["status"]


def temps():
    s = query(["heater_bed", "extruder"])
    return s["heater_bed"]["temperature"], s["extruder"]["temperature"]


def responses(since):
    store = get_json(MOONRAKER
                     + "/server/gcode_store?count=200")["result"]["gcode_store"]
    return [e["message"] for e in store if e["time"] > since]


def probe_at(x, y):
    t0 = time.time() - 1
    ok, body = post("G1 X%.1f Y%.1f Z3.0 F3000\nPROBE SAMPLES=1" % (x, y))
    if not ok:
        # Recorded as a blank z and skipped, not raised: one lost contact is a
        # missing datapoint, an exception here is the whole run.
        print("    PROBE lost at %.1f,%.1f: %s" % (x, y, body.strip()[:160]),
              flush=True)
        return None
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
    return result if result is not None else (zs[-1] if zs else None)


def wait_settled(what, tol_per_min=0.5, window=60., timeout_s=2400, poll=20):
    """Wait for thermal equilibrium rather than a fixed temperature."""
    t0 = time.time()
    prev, prev_t = temps()[0], time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(poll)
        b = temps()[0]
        now = time.time()
        if now - prev_t >= window:
            rate = abs(b - prev) / ((now - prev_t) / 60.)
            if rate <= tol_per_min:
                print("  %s settled at %.1fC (drift %.2f C/min) after %.0fs"
                      % (what, b, rate, now - t0), flush=True)
                return b
            prev, prev_t = b, now
    b = temps()[0]
    print("  %s TIMEOUT at %.1fC - proceeding" % (what, b), flush=True)
    return b


def wait_nozzle(target, what, timeout_s=1800, poll=10):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        b, n = temps()
        if (target > 50. and n >= target - 1.) or (target <= 50. and n <= target):
            print("  %s (bed %.1f nozzle %.1f) after %.0fs"
                  % (what, b, n, time.time() - t0), flush=True)
            return True
        time.sleep(poll)
    print("  TIMEOUT waiting for %s" % what, flush=True)
    return False


def round_of_probes(fh, phase, t_start):
    for (x, y) in POINTS:
        b, n = temps()
        z = probe_at(x, y)
        dt = time.time() - t_start
        fh.write("%s,%.0f,%.1f,%.1f,%.1f,%.1f,%s\n"
                 % (phase, dt, x, y, b, n, "" if z is None else "%.6f" % z))
        fh.flush()
        print("  %-11s t=%5.0fs  %5.1f,%-5.1f  bed %5.1f noz %6.1f  z=%s"
              % (phase, dt, x, y, b, n,
                 "FAILED" if z is None else "%+.4f" % z), flush=True)


def main():
    ok, body = post("SET_IDLE_TIMEOUT TIMEOUT=7200")
    if not ok:
        sys.exit("idle timeout: %s" % body)
    # Bed stays OFF for the whole run - that is the point of this test.
    post("M140 S0")
    print("homing", flush=True)
    ok, body = post("G28")
    if not ok:
        sys.exit("G28 failed: %s" % body)

    fh = open(OUT, "w")
    fh.write("phase,t_s,x,y,bed_c,nozzle_c,z\n")
    t_start = time.time()

    print("\n== conditioning: %d discarded contacts per point ==" % CONDITION_N,
          flush=True)
    for _ in range(CONDITION_N):
        round_of_probes(fh, "conditioning", t_start)

    print("\n== phase 0: cold baseline, nozzle cold ==", flush=True)
    post("M104 S0")
    baseline_bed = wait_settled("bed")
    for _ in range(3):
        round_of_probes(fh, "cold", t_start)
        time.sleep(60)

    print("\n== phase 1: nozzle -> %.0fC, BED STAYS OFF ==" % NOZZLE_TARGET,
          flush=True)
    post("M104 S%.0f" % NOZZLE_TARGET)
    wait_nozzle(NOZZLE_TARGET, "nozzle at target")
    post("M83\nG1 E-%.1f F300" % RETRACT_MM)
    t_hot = time.time()
    # The hotend reaches temperature in ~60s but the block and mount keep
    # warming for minutes, so sample the settling rather than one point.
    for wait in (0, 240, 480, 720):
        while time.time() - t_hot < wait:
            time.sleep(10)
        round_of_probes(fh, "noz_hot", t_start)

    print("\n== phase 2: nozzle off, return to baseline (the control) ==",
          flush=True)
    post("M104 S0")
    wait_nozzle(50., "nozzle back below 50C", timeout_s=3600)
    wait_settled("bed on return")
    for _ in range(3):
        round_of_probes(fh, "cooled", t_start)
        time.sleep(60)

    fh.close()
    print("\n=== done -> %s ===" % OUT, flush=True)
    print("bed held OFF throughout; baseline bed was %.1fC" % baseline_bed,
          flush=True)


if __name__ == "__main__":
    main()
