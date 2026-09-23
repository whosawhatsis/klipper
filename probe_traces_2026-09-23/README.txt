2026-09-23 slower verify ramp test, smooth PEI, 112.7Hz, deployed code e5bfdf8db (unchanged).
verify00514-00555: 7 probes per arm at (75,85), (90,70), (60,45); arms in slow_run_manifest.json.
  slow: VERIFY_RAMP_SPEED 0.15, VERIFY_UP 0.14, VERIFY_DOWN 0.10, 2 reps (1984 segments)
  ctrl: VERIFY_RAMP_SPEED 0.30, same shortened ramp
Arm order alternated per point (slow first at 75,85 and 60,45). Recipe: local/slow_run.py.
Analysis: local/intersect_proto/analyze_slow.py, seq_slow.py.
slow_store.jsonl = Moonraker gcode_store snapshots every 30s (VERBOSE verify lines).
