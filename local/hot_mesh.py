#!/usr/bin/env python3
# BED_MESH_CALIBRATE with a HOT nozzle over a COLD bed.
#
# The combination the measurements point at:
#  - COLD bed: no transient bow.  A mesh is a serial sweep, so a moving surface
#    maps onto probe ORDER and appears as a plausible bed SHAPE.  A cold bed is
#    static while you sweep it.
#  - HOT nozzle: melts deposit off the tip instead of letting it hold the probe
#    off.  Deposit reads HIGH - up to +64um on a first hot contact - and clears
#    with contact TIME, which is what verify_contact_dwell (0.5s) and
#    verify_reps (3) now buy at every point.
#
# Runs detached and writes everything here, so nothing needs to hold a
# connection open: tailnet traffic to this host causes "Timer too close" MCU
# shutdowns, and a vibrating descent is the most fragile moment for that.

import json
import sys
import time
import urllib.error
import urllib.request

MOONRAKER = "http://localhost:7125"
CAMERA = "http://localhost:8080/?action=snapshot"
NOZZLE_TARGET = 210.0
RETRACT_MM = 20.0
PASSES = int(sys.argv[1]) if len(sys.argv) > 1 else 1
# Seconds to hold at temperature WITHOUT probing before the first sweep.  Runs
# 13/14 confounded two causes of the front-row error: the transient mesh was
# also the unconditioned one, the soaked mesh also the already-conditioned one.
# A soak with no contacts in it is soaked-but-unconditioned and separates them.
# Homing moves after the soak so the sweep starts hot, as run 14's did.
SOAK_S = float(sys.argv[2]) if len(sys.argv) > 2 else 0.
# Extra params appended to BED_MESH_CALIBRATE, e.g. "VERIFY_REPS=3 VERBOSE=1".
EXTRA = (" " + sys.argv[3]) if len(sys.argv) > 3 else ""


def post(script, timeout=1800):
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
        return False, "transport error: %s" % e


def get_json(url, timeout=30, tries=4):
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise last


def temps():
    s = get_json(MOONRAKER
                 + "/printer/objects/query?heater_bed&extruder")["result"]["status"]
    return s["heater_bed"]["temperature"], s["extruder"]["temperature"]


def snapshot(path):
    # Post-home frame, grabbed BEFORE anything else moves: a sensorless false
    # home has no software symptom, and a frame taken after the toolhead has
    # driven off has nothing to compare against.
    try:
        with urllib.request.urlopen(CAMERA, timeout=15) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
        print("  post-home frame: %s (%d bytes)" % (path, len(data)), flush=True)
    except Exception as e:
        print("  SNAPSHOT FAILED (%s) - home is unverified" % e, flush=True)


def report_mesh():
    try:
        bm = get_json(MOONRAKER
                      + "/printer/objects/query?bed_mesh")["result"]["status"]["bed_mesh"]
        m = bm["probed_matrix"]
        print("\nprofile %r  min %s  max %s" % (bm["profile_name"], bm["mesh_min"],
                                                bm["mesh_max"]), flush=True)
        print("probed matrix (row 0 = FRONT):", flush=True)
        for i, row in enumerate(m):
            print("  y%d: %s" % (i, " ".join("%+7.4f" % v for v in row)), flush=True)
        vals = [v for r in m for v in r]
        if vals:
            print("  range %.4f .. %.4f   spread %.1f um"
                  % (min(vals), max(vals), (max(vals) - min(vals)) * 1000.),
                  flush=True)
    except Exception as e:
        print("could not read the mesh back: %s" % e, flush=True)


def heat_and_retract():
    print("heating nozzle to %.0fC (bed off)" % NOZZLE_TARGET, flush=True)
    post("M104 S%.0f" % NOZZLE_TARGET)
    t0 = time.time()
    while time.time() - t0 < 1800:
        b, n = temps()
        if n >= NOZZLE_TARGET - 1.:
            print("  nozzle %.1fC, bed %.1fC after %.0fs"
                  % (n, b, time.time() - t0), flush=True)
            break
        time.sleep(10)
    # Retract once hot enough to move filament, to limit ooze.  This matters
    # more with a soak: minutes at temperature with filament in the melt zone
    # would drool, and a blob on the tip is itself a conditioning confound -
    # exactly the variable this run is trying to hold fixed.
    post("M83\nG1 E-%.1f F300" % RETRACT_MM)
    time.sleep(2)


def main():
    post("SET_IDLE_TIMEOUT TIMEOUT=7200")
    post("M140 S0")                      # bed stays OFF - that is the point

    if SOAK_S:
        heat_and_retract()
        print("soaking %.0fs at temperature WITHOUT probing" % SOAK_S, flush=True)
        t0 = time.time()
        while time.time() - t0 < SOAK_S:
            time.sleep(30)
            b, n = temps()
            print("  +%4.0fs  nozzle %.1fC  bed %.1fC"
                  % (time.time() - t0, n, b), flush=True)

    for p in range(1, PASSES + 1):
        print("\n===== pass %d/%d =====" % (p, PASSES), flush=True)
        print("homing", flush=True)
        ok, body = post("G28")
        if not ok:
            post("M104 S0")
            sys.exit("G28 failed: %s" % body)
        time.sleep(3)
        snapshot("/tmp/posthome_pass%d.jpg" % p)

        if p == 1 and not SOAK_S:
            heat_and_retract()
        # Later passes keep the nozzle hot: re-heating from cold would put a
        # thermal change alongside the re-home, and the re-home is the variable.

        b, n = temps()
        print("\nBED_MESH_CALIBRATE%s  (bed %.1fC, nozzle %.1fC)" % (EXTRA, b, n), flush=True)
        t1 = time.time()
        ok, body = post("BED_MESH_CALIBRATE" + EXTRA)
        print("  returned ok=%s after %.0fs" % (ok, time.time() - t1), flush=True)
        if not ok:
            print("  %s" % body.strip()[:600], flush=True)
        report_mesh()

    # Nozzle off before anything else - do not leave it hot unattended.
    post("M104 S0")
    print("\n=== done, nozzle commanded off ===", flush=True)


if __name__ == "__main__":
    main()
