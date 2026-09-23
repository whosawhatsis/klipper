#!/usr/bin/env python3
# Two bed mesh profiles from ONE probing pass: 'upmesh' and 'downmesh'.
#
# Every verify computes a down-ramp and an up-ramp contact height and reports
# one of them.  This runs a single BED_MESH_CALIBRATE, takes the reported mesh
# as one profile, and rebuilds the other by substituting the other estimate at
# each point from the contact ledger.
#
# ONE pass on purpose.  Re-probing to get the second mesh would differ by the
# estimator AND by run-to-run scatter, and single-point scatter has been
# measured at 206um - the same size as the effect being tested.  Same contacts
# means the estimator is the only difference.  It also halves plate wear, which
# measurably degrades contact-finding (see LABELS.md, worn vs virgin).
#
# The nozzle is soaked before the sweep: a mesh started right after reaching
# temperature carries a ~30um error concentrated in the first row probed
# (MESH_RUNS.md run 15).  Both profiles would inherit it equally, but it would
# sit on top of whatever the print test is trying to see.
import json
import os
import sys
import time
import urllib.error
import urllib.request

MOONRAKER = "http://localhost:7125"
CFG = os.path.expanduser("~/klipper_config/printer.cfg")
LEDGER = os.path.expanduser("~/probe_traces/contacts.csv")
NOZZLE_TARGET = 210.0
SOAK_S = 480.
RETRACT_MM = 20.0
MARKER = "#*# <---------------------- SAVE_CONFIG ---------------------->"


def post(script, timeout=2400):
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


def get(path, timeout=30):
    with urllib.request.urlopen(MOONRAKER + path, timeout=timeout) as r:
        return json.load(r)


def temps():
    s = get("/printer/objects/query?heater_bed&extruder")["result"]["status"]
    return s["heater_bed"]["temperature"], s["extruder"]["temperature"]


def ledger_offset():
    """Byte offset to read the ledger from, so only THIS run's contacts count."""
    return os.path.getsize(LEDGER) if os.path.exists(LEDGER) else 0


def read_ledger(offset):
    """Accepted contacts appended after 'offset'. Returns {(x,y): (down, up)}.

    Bounded by offset rather than filtered afterwards: the ledger accumulates
    across sessions, and the proximity match below would happily pair a mesh
    point with a contact from a different run at nearly the same XY.
    """
    if not os.path.exists(LEDGER):
        return {}
    out = {}
    with open(LEDGER) as f:
        f.seek(offset)
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6 or p[0] == 'x':
                continue
            try:
                x, y = float(p[0]), float(p[1])
                down = float(p[3]) if p[3] else None
                up = float(p[4]) if p[4] else None
            except ValueError:
                continue
            if down is None or up is None:
                continue
            out[(round(x, 1), round(y, 1))] = (down, up)
    return out


def nearest(ledger, x, y, tol=1.5):
    """Mesh points get nudged, so match by proximity rather than equality."""
    best, bd = None, tol
    for (lx, ly), v in ledger.items():
        d = ((lx - x) ** 2 + (ly - y) ** 2) ** 0.5
        if d <= bd:
            best, bd = v, d
    return best


def wait_ready(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if get("/printer/info")["result"]["state"] == "ready":
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def clone_profile(src, dst, matrix):
    """Copy the [bed_mesh src] block Klipper wrote, substituting new points.

    Klipper writes the authoritative block - algo, tension, pps and the mesh
    bounds all come from its own state.  Hand-writing those means guessing, and
    a wrong 'algo' produces a mesh that looks right and interpolates wrong.
    """
    with open(CFG) as f:
        lines = f.read().split("\n")
    try:
        i = lines.index("#*# [bed_mesh %s]" % src)
    except ValueError:
        sys.exit("Klipper did not write [bed_mesh %s] - nothing to clone" % src)
    j = i + 1
    while j < len(lines) and not lines[j].startswith("#*# ["):
        j += 1
    block = lines[i:j]

    out, in_points = ["#*# [bed_mesh %s]" % dst], False
    for line in block[1:]:
        if line.startswith("#*# points"):
            out.append(line)
            in_points = True
            for row in matrix:
                out.append("#*# \t" + ", ".join("%.6f" % v for v in row))
            continue
        if in_points:
            # skip the source's own point rows, keep everything after them
            if line.startswith("#*# \t") or line.startswith("#*# \t"):
                continue
            in_points = False
        out.append(line)

    # Replace any previous copy of dst, then append.
    text = "\n".join(lines)
    marker = "#*# [bed_mesh %s]" % dst
    if marker in lines:
        k = lines.index(marker)
        m = k + 1
        while m < len(lines) and not lines[m].startswith("#*# ["):
            m += 1
        lines = lines[:k] + lines[m:]
    backup = CFG + ".dualmesh.bak"
    with open(backup, "w") as f:
        f.write(text)
    with open(CFG, "w") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n" + "\n".join(out) + "\n")
    print("  wrote [bed_mesh %s] cloned from [bed_mesh %s] (backup %s)"
          % (dst, src, backup))


def main():
    post("SET_IDLE_TIMEOUT TIMEOUT=7200")
    post("M140 S0")                       # bed OFF, as every mesh here has been

    print("heating to %.0fC" % NOZZLE_TARGET, flush=True)
    post("M104 S%.0f" % NOZZLE_TARGET)
    t0 = time.time()
    while time.time() - t0 < 1800:
        b, n = temps()
        if n >= NOZZLE_TARGET - 1.:
            print("  reached %.1fC after %.0fs" % (n, time.time() - t0), flush=True)
            break
        time.sleep(10)
    post("M83\nG1 E-%.1f F300" % RETRACT_MM)
    print("soaking %.0fs WITHOUT probing" % SOAK_S, flush=True)
    time.sleep(SOAK_S)

    ok, body = post("G28")
    if not ok:
        post("M104 S0")
        sys.exit("G28 failed: %s" % body)

    mark = ledger_offset()
    print("\nBED_MESH_CALIBRATE (reporting the UP estimate)", flush=True)
    ok, body = post("BED_MESH_CALIBRATE PROFILE=upmesh VERIFY_COMBINE=2 VERBOSE=1")
    post("M104 S0")
    if not ok:
        sys.exit("mesh failed: %s" % body.strip()[:500])

    bm = get("/printer/objects/query?bed_mesh")["result"]["status"]["bed_mesh"]
    up_matrix = [list(r) for r in bm["probed_matrix"]]
    ledger = read_ledger(mark)
    print("  ledger has %d accepted contacts" % len(ledger), flush=True)

    xs = [bm["mesh_min"][0] + i * (bm["mesh_max"][0] - bm["mesh_min"][0])
          / max(1, len(up_matrix[0]) - 1) for i in range(len(up_matrix[0]))]
    ys = [bm["mesh_min"][1] + j * (bm["mesh_max"][1] - bm["mesh_min"][1])
          / max(1, len(up_matrix) - 1) for j in range(len(up_matrix))]

    down_matrix, missing, biases = [], 0, []
    for j, row in enumerate(up_matrix):
        out = []
        for i, z_up in enumerate(row):
            v = nearest(ledger, xs[i], ys[j])
            if v is None:
                missing += 1
                out.append(z_up)          # no ledger entry: leave it alone
                continue
            down, up = v
            out.append(z_up + (down - up))
            biases.append((abs(down - up) * 1000., xs[i], ys[j], down, up))
        down_matrix.append(out)

    print("\n  per-point |down-up| bias:")
    for b, x, y, d, u in sorted(biases, reverse=True):
        flag = "  <-- estimators disagree" if b > 20. else ""
        print("    (%5.1f,%5.1f)  %7.1f um   down %+.4f  up %+.4f%s"
              % (x, y, b, d, u, flag))
    if missing:
        print("  %d point(s) had no ledger entry and are IDENTICAL in both"
              " profiles" % missing)
    big = [b for b, _, _, _, _ in biases if b > 20.]
    print("\n  %d of %d points differ by more than 20um."
          % (len(big), len(biases)))
    if not big:
        print("  The two profiles are effectively the SAME MESH - a print test"
              "\n  cannot distinguish them.  That is a real result: it means"
              "\n  every contact in this mesh was clean.")

    print("\nsaving the reported (up) mesh as a profile", flush=True)
    post("BED_MESH_PROFILE SAVE=upmesh")
    post("SAVE_CONFIG")                   # this RESTARTS klippy
    if not wait_ready():
        sys.exit("klippy did not come back after SAVE_CONFIG")
    clone_profile("upmesh", "downmesh", down_matrix)
    post("FIRMWARE_RESTART")
    wait_ready()

    print("\nFIRMWARE_RESTART, then:")
    print("  BED_MESH_PROFILE LOAD=upmesh     # or LOAD=downmesh")
    print("=== done, nozzle commanded off ===")


if __name__ == '__main__':
    main()
