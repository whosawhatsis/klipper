#!/usr/bin/env python3
# Slower-verify-ramp test at 10 FRESH points, 2026-09-23, smooth PEI, 112.7Hz. Run on zer0 (python 3.7).
# A discarded warm-up block runs first so no measured block follows homing directly.
# SLOW: VERIFY_RAMP_SPEED 0.15 over UP 0.14 + DOWN 0.10 (2 reps = 1984 segments < 2000 budget).
# CTRL: 0.30 over the SAME shortened ramp, so ramp length is not a confound.
# Arm order alternates by point against drift. Manifest maps verify files to arms.
import json, os, glob, time, urllib.request
M = "http://localhost:7125"
TD = "/home/who/probe_traces"
POINTS = [(25, 25), (55, 25), (85, 25), (105, 45), (40, 45), (70, 60), (45, 80), (25, 100), (60, 100), (95, 95)]
WARMUP = (60, 45)
N = 6
ARMS = {"slow": 0.15, "ctrl": 0.30}
def gcode(s):
    req = urllib.request.Request(M + "/printer/gcode/script", data=json.dumps({"script": s}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=2400) as r:
            return "ok"
    except Exception as e:
        return "ERR %s" % (getattr(e, 'read', lambda: str(e).encode())().decode()[:300])
def verifies():
    return set(os.path.basename(f) for f in glob.glob(TD + "/verify*.csv"))
manifest = []
t0 = time.time()
print("== warm-up %d,%d" % WARMUP, flush=True)
before = verifies()
r = gcode("RESONANCE_PROBE_CONTACT POINT=%d,%d,2 FREQ=112.7 ACCEL_AXIS=x SPEED=0.2 SAMPLES=5 VERIFY_REPS=2"
          " VERIFY_UP=0.14 VERIFY_DOWN=0.10 VERIFY_RAMP_SPEED=0.30" % WARMUP)
manifest.append({"x": WARMUP[0], "y": WARMUP[1], "arm": "warmup", "speed": 0.30, "files": sorted(verifies() - before), "result": r})
print("  warmup %s" % r, flush=True)
for i, (x, y) in enumerate(POINTS):
    order = ["slow", "ctrl"] if i % 2 == 0 else ["ctrl", "slow"]
    print("== %d,%d  order %s" % (x, y, order), flush=True)
    print(gcode("RESONANCE_PROBE_CHARACTERIZE_NOISE POINT=%d,%d,2 FREQ=112.7 ACCEL_AXIS=x REPS=4 SPEED=0.2 AIR_ZMIN=0.20" % (x, y)), flush=True)
    for arm in order:
        before = verifies()
        r = gcode("RESONANCE_PROBE_CONTACT POINT=%d,%d,2 FREQ=112.7 ACCEL_AXIS=x SPEED=0.2 SAMPLES=%d VERIFY_REPS=2"
                  " VERIFY_UP=0.14 VERIFY_DOWN=0.10 VERIFY_RAMP_SPEED=%.2f VERBOSE=1" % (x, y, N, ARMS[arm]))
        new = sorted(verifies() - before)
        manifest.append({"x": x, "y": y, "arm": arm, "speed": ARMS[arm], "files": new, "result": r})
        print("  %s %s -> %d files %s" % (arm, r, len(new), (new[0] + ".." + new[-1]) if new else ""), flush=True)
        with open("/home/who/fresh_run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=1)
print(gcode("G1 Z10 F600"))
store = json.load(urllib.request.urlopen(M + "/server/gcode_store?count=1000"))["result"]["gcode_store"]
with open("/home/who/fresh_run_responses.json", "w") as f:
    json.dump([e for e in store if e["time"] >= t0 - 1], f)
print("DONE", flush=True)
