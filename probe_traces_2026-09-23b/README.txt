2026-09-23 slow-vs-control verify ramp at 10 FRESH smooth-PEI points, 112.7Hz, deployed e5bfdf8db.
Warm-up block (5 probes at 60,45, discarded) then per point 6 probes per arm, order alternated:
  slow 0.15mm/s, ctrl 0.30mm/s, both VERIFY_UP 0.14 + VERIFY_DOWN 0.10, 2 reps.
Points: (25,25) (55,25) (85,25) (105,45) (40,45) (70,60) (45,80) (25,100) (60,100) (95,95).
Files -> arms: fresh_run_manifest.json. Recipe: local/fresh_run.py.
MISSED HALTS (over-press, salvage recovered): (25,25) 0.211mm and 0.794mm [ctrl block aborted],
(25,100) 0.049mm. Both points may now carry press marks.
fresh_store.jsonl = gcode_store snapshots (VERBOSE verify lines).
