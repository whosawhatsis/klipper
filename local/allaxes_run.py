#!/usr/bin/env python3
# All-axes verify hardware check, 2026-09-22: repeat the evening's smooth-PEI
# recipe at 112.7Hz on the new code. Run on zer0 (python 3.7, no f-strings).
import json, time, urllib.request
M = "http://localhost:7125"
POINTS = [(75, 85), (90, 70), (60, 45)]
def gcode(s):
    req = urllib.request.Request(M + "/printer/gcode/script",
        data=json.dumps({"script": s}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return "ok"
    except Exception as e:
        return "ERR %s" % (getattr(e, 'read', lambda: str(e).encode())().decode()[:300])
t0 = time.time()
for x, y in POINTS:
    print("== %d,%d" % (x, y), flush=True)
    print(gcode("RESONANCE_PROBE_CHARACTERIZE_NOISE POINT=%d,%d,2 FREQ=112.7 ACCEL_AXIS=x REPS=4 SPEED=0.2 AIR_ZMIN=0.20" % (x, y)), flush=True)
    print(gcode("RESONANCE_PROBE_CONTACT POINT=%d,%d,2 FREQ=112.7 ACCEL_AXIS=x SPEED=0.2 SAMPLES=7 VERIFY_REPS=2 VERIFY_UP=0.30 VERIFY_DOWN=0.10 VERIFY_RAMP_SPEED=0.30 VERBOSE=1" % (x, y)), flush=True)
print(gcode("G1 Z10 F600"))
store = json.load(urllib.request.urlopen(M + "/server/gcode_store?count=1000"))["result"]["gcode_store"]
with open("/home/who/allaxes_run_responses.json", "w") as f:
    json.dump([e for e in store if e["time"] >= t0 - 1], f)
print("DONE", flush=True)
