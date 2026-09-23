#!/usr/bin/env python3
# Two transects to test the cantilever hypothesis for the re-excitation artifact.
#
# The platform sits on a three-point mount on a cantilevered Z stage attached at
# the back, behind (60,100).  If pressing rotates that shelf and that is what
# re-excites the structure, then:
#
#   X-transect at y=100  - coupling should PEAK at x=60, where the X excitation
#                          is tangential to rotation about the attachment, and
#                          fall off toward x=20 and x=100 where the force gains
#                          a radial component that does no work on the mode.
#
#   Y-transect at x=60   - tests the radius term.  Simple rotation about a point
#                          behind y=100 predicts coupling GROWS with distance
#                          from the attachment (toward y=25).  The observations
#                          so far show the opposite, so this is the measurement
#                          that either rescues or kills the simple picture.
#
# Readout is dwell/ramp-min per capture, computed offline from the verify traces
# - which now record XY.  A point that FAILS still writes its captures, so a
# failure is data here rather than a lost sample.
import json
import sys
import time
import urllib.error
import urllib.request

MOONRAKER = "http://localhost:7125"
X_TRANSECT = [(x, 100.) for x in (20., 45., 60., 75., 100.)]
Y_TRANSECT = [(60., y) for y in (25., 50., 75.)]      # (60,100) shared above


def post(script, timeout=600):
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


def state():
    try:
        with urllib.request.urlopen(MOONRAKER + "/printer/info", timeout=15) as r:
            return json.load(r)["result"]["state"]
    except Exception:
        return "unknown"


def main():
    post("SET_IDLE_TIMEOUT TIMEOUT=7200")
    post("M104 S0")
    post("M140 S0")
    ok, body = post("G28")
    if not ok:
        sys.exit("G28 failed: %s" % body[:200])

    for label, pts in (("X-transect y=100", X_TRANSECT),
                       ("Y-transect x=60", Y_TRANSECT)):
        print("\n===== %s =====" % label, flush=True)
        for x, y in pts:
            if state() != "ready":
                print("  printer not ready (%s) - stopping" % state(), flush=True)
                return
            post("G90\nG1 X%.2f Y%.2f Z2 F1500" % (x, y))
            t0 = time.time()
            ok, body = post("PROBE_ACCURACY SAMPLES=1 VERIFY_REPS=3 VERBOSE=1")
            if ok:
                print("  (%5.1f,%5.1f)  ok        %4.0fs" % (x, y, time.time() - t0),
                      flush=True)
            else:
                # A failed point still wrote its verify captures, which is the
                # measurement this run is actually after.
                msg = body.strip().replace("\n", " ")[:90]
                print("  (%5.1f,%5.1f)  FAILED    %4.0fs  %s"
                      % (x, y, time.time() - t0, msg), flush=True)
            post("G91\nG1 Z3 F600\nG90")

    print("\n=== done ===", flush=True)


if __name__ == '__main__':
    main()
