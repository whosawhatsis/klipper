#!/usr/bin/env python3
# Measure how the probed contact height moves as the machine heats and soaks.
#
# The question: a mesh probed cold then printed hot came out UNDER-compressed.
# Thermal expansion is the obvious confound, but its SIGN is not obvious - the
# frame is several materials with different expansion rates, so something that
# expands more can push the nozzle away from the bed rather than toward it.
# This measures the sign and magnitude instead of assuming either.
#
# Design:
#  - TWO points, probed in rotation.  A replicate, and it halves the wear per
#    spot.  Each point is differenced against ITS OWN cold baseline, so the two
#    never need to agree in absolute terms.
#  - SAMPLES=1 to keep total contacts down (~24).  Probing wears the plate, and
#    a polished spot changes its own reading - which is exactly the confound
#    this test could otherwise manufacture for itself.
#  - A COOL-BACK-DOWN phase at the end.  This is the control that makes the
#    result interpretable: thermal drift is REVERSIBLE, plate wear is NOT.  If
#    the height returns to its cold baseline, the excursion was thermal.  If it
#    does not, some of it was the spot changing under repeated probing.
#
# Every probe records bed and nozzle temperature alongside the height, so the
# result is a curve against temperature rather than a before/after pair.
#
# Writes CSV to ~/heat_soak_drift.csv.  Run on the printer host.

import json
import sys
import time
import urllib.error
import urllib.request

MOONRAKER = "http://localhost:7125"
OUT = "/home/who/heat_soak_drift.csv"

# 70.0,45.0 is virgin (15.2mm clear of everything touched so far).  85.0,55.0 is
# carried over from the 2026-08-06 run deliberately: it behaved throughout
# (7.7um across three hot readings) so it gives continuity with that baseline.
# 55.0,80.0 was DROPPED - it scattered 44-90um from the start and then failed
# outright, driving the nozzle ~400um past the surface; it is a compromised site,
# not a fair test location.
POINTS = [(70.0, 45.0), (85.0, 55.0)]

BED_TARGET = 60.0
NOZZLE_TARGET = 210.0
COLD_BED = 32.0        # unused: the baseline now waits for EQUILIBRIUM,
                       # not a fixed temperature (the plate approaches room
                       # temperature asymptotically, so a threshold either
                       # never trips or trips arbitrarily on the curve)
RETRACT_MM = 10.0      # before the nozzle goes hot, to limit ooze
CONDITION_N = 2        # discarded contacts per point before the baseline


def post(script, timeout=900):
    req = urllib.request.Request(
        MOONRAKER + "/printer/gcode/script",
        data=json.dumps({"script": script}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, r.read().decode()
    except urllib.error.HTTPError as e:
        return False, e.read().decode()


def query(objs):
    q = "&".join(objs)
    with urllib.request.urlopen(
            MOONRAKER + "/printer/objects/query?" + q, timeout=30) as r:
        return json.load(r)["result"]["status"]


def temps():
    s = query(["heater_bed", "extruder"])
    return s["heater_bed"]["temperature"], s["extruder"]["temperature"]


def responses(since):
    with urllib.request.urlopen(
            MOONRAKER + "/server/gcode_store?count=200", timeout=30) as r:
        store = json.load(r)["result"]["gcode_store"]
    return [e["message"] for e in store if e["time"] > since]


def probe_at(x, y):
    """One verified contact at (x,y); returns the reported Z or None."""
    t0 = time.time() - 1
    ok, body = post("G1 X%.1f Y%.1f Z3.0 F3000\nPROBE SAMPLES=1" % (x, y))
    if not ok:
        print("    PROBE failed at %.1f,%.1f: %s" % (x, y, body.strip()[:160]),
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


def wait_for(cond, what, timeout_s, poll=10):
    """Poll until cond(bed, noz) or timeout.  Prints progress as it goes."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        b, n = temps()
        if cond(b, n):
            print("  reached %s (bed %.1f nozzle %.1f) after %.0fs"
                  % (what, b, n, time.time() - t0), flush=True)
            return True
        time.sleep(poll)
    b, n = temps()
    print("  TIMEOUT waiting for %s (bed %.1f nozzle %.1f)" % (what, b, n),
          flush=True)
    return False


def wait_settled(what, tol_per_min=0.5, window=60., timeout_s=2400, poll=20):
    """Wait for the bed to reach thermal EQUILIBRIUM, not a fixed temperature.

    The plate approaches room temperature asymptotically, so an absolute
    threshold either waits forever or trips at an arbitrary point on the curve
    depending on how warm the room is that day.  What a baseline actually needs
    is a bed that has stopped moving; the temperature it settles at is recorded
    with every probe anyway, so the number itself does not need to be chosen in
    advance.  Returns the settled temperature.
    """
    t0 = time.time()
    prev, prev_t = temps()[0], time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(poll)
        b = temps()[0]
        now = time.time()
        if now - prev_t >= window:
            rate = abs(b - prev) / ((now - prev_t) / 60.)
            if rate <= tol_per_min:
                print("  %s settled at %.1fC (drifting %.2f C/min) after %.0fs"
                      % (what, b, rate, now - t0), flush=True)
                return b
            prev, prev_t = b, now
    b = temps()[0]
    print("  %s TIMEOUT at %.1fC after %.0fs - proceeding" % (what, b, timeout_s),
          flush=True)
    return b


def round_of_probes(fh, phase, t_start):
    for (x, y) in POINTS:
        b, n = temps()
        z = probe_at(x, y)
        dt = time.time() - t_start
        fh.write("%s,%.0f,%.1f,%.1f,%.1f,%.1f,%s\n"
                 % (phase, dt, x, y, b, n, "" if z is None else "%.6f" % z))
        fh.flush()
        print("  %-10s t=%5.0fs  %5.1f,%-5.1f  bed %5.1f noz %5.1f  z=%s"
              % (phase, dt, x, y, b, n,
                 "FAILED" if z is None else "%+.4f" % z), flush=True)


def main():
    ok, body = post("SET_IDLE_TIMEOUT TIMEOUT=7200")
    if not ok:
        sys.exit("idle timeout: %s" % body)
    print("homing", flush=True)
    ok, body = post("G28")
    if not ok:
        sys.exit("G28 failed: %s" % body)

    fh = open(OUT, "w")
    fh.write("phase,t_s,x,y,bed_c,nozzle_c,z\n")
    t_start = time.time()

    # --- CONDITIONING: throw the first contacts away, by construction --------
    # The first probe after a print reads HIGH - measured 2026-08-06 at +69um on
    # the very first contact and +22um on the second, decreasing as it went.
    # Residual plastic on the nozzle triggers early, and the first touches
    # flatten it off.  Left in, that artifact swamped the thermal signal and
    # even reversed its sign (-34um vs +0.6um at one point).
    #
    # These land in the CSV tagged 'conditioning' so they are visible, and are
    # excluded from every phase mean.  Discarding by design beats discarding
    # afterwards: a post-hoc exclusion is a judgement call about your own data,
    # which is exactly where a wrong result gets rationalised into a right one.
    print("\n== conditioning: %d discarded contacts per point ==" % CONDITION_N,
          flush=True)
    for _ in range(CONDITION_N):
        round_of_probes(fh, "conditioning", t_start)

    # --- PHASE 0: cold baseline -------------------------------------------
    post("M140 S0\nM104 S0")
    print("\n== phase 0: cold baseline (waiting for the bed to SETTLE) ==",
          flush=True)
    baseline_bed = wait_settled("bed")
    for _ in range(2):
        round_of_probes(fh, "cold", t_start)
        time.sleep(120)

    # --- PHASE 1: bed hot, then soak --------------------------------------
    print("\n== phase 1: bed -> %.0fC ==" % BED_TARGET, flush=True)
    post("M140 S%.0f" % BED_TARGET)
    wait_for(lambda b, n: b >= BED_TARGET - 1., "bed target", 1800)
    t_hot = time.time()
    for wait in (0, 180, 360, 600, 900):
        while time.time() - t_hot < wait:
            time.sleep(10)
        round_of_probes(fh, "bed_hot", t_start)

    # --- PHASE 2: nozzle hot too ------------------------------------------
    print("\n== phase 2: nozzle -> %.0fC (retracting first) ==" % NOZZLE_TARGET,
          flush=True)
    post("M104 S%.0f" % NOZZLE_TARGET)
    wait_for(lambda b, n: n >= NOZZLE_TARGET - 1., "nozzle target", 1800)
    # Retract only once the nozzle is actually hot enough to move filament.
    post("M83\nG1 E-%.1f F300" % RETRACT_MM)
    t_noz = time.time()
    for wait in (0, 300, 720):
        while time.time() - t_noz < wait:
            time.sleep(10)
        round_of_probes(fh, "noz_hot", t_start)

    # --- PHASE 3: cool back down (the reversibility control) --------------
    print("\n== phase 3: heaters off, cooling (the control) ==", flush=True)
    post("M104 S0\nM140 S0")
    # The control only works if the return is measured at the SAME temperature
    # the baseline was: comparing a 33C baseline against a 38C "cooled" reading
    # would fold a real thermal difference into the reversibility check.
    print("  waiting to return to the baseline temperature (%.1fC)"
          % baseline_bed, flush=True)
    wait_for(lambda b, n: b <= baseline_bed + 1.5 and n <= 50.,
             "baseline temperature", 5400)
    wait_settled("bed on return")
    for _ in range(2):
        round_of_probes(fh, "cooled", t_start)
        time.sleep(180)

    fh.close()
    print("\n=== done -> %s ===" % OUT, flush=True)


if __name__ == "__main__":
    main()
