# Tooling + evidence branch

This branch exists because `local/`, `klippy/extras/test_*.py` and the trace corpus are in
`.git/info/exclude` on the feature branch — deliberately, to keep the upstream PR diff clean — but
they are **not reproducible** and must reach other machines.

It is `resonance-nozzle-probe` plus:

- **`local/`** — all project tooling, unversioned until now. Includes `refine_descent.py` (the
  offline estimator that reaches the 2 µm target), `deploy_resonance.sh`, `check_home.sh`, and the
  `replay_*` analysis tools.
- **`klippy/extras/test_*.py`** — six self-check suites, 174 checks. Run as modules from `klippy/`:
  `cd klippy && python -m extras.test_verify_ramp`. **Append new tests ABOVE the
  `if __name__ == '__main__':` block** or they are never collected.
- **`probe_traces_2026-09-21/`** — the measurement corpus. These are irreplaceable: re-creating any
  of them means re-probing, which wears the plate, and several points on the textured face are
  already damaged.

**Do not open a PR from this branch.** Rebase tooling changes onto it; keep code changes on
`resonance-nozzle-probe`.

Project state and all engineering knowledge: `~/AI Brain/wiki/project-klipper-resonance-probe-state.md`
(vault is pushed to github.com/whosawhatsis/ai-brain).
