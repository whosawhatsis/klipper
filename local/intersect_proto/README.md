# Two-line intersection prototypes (2026-09-23, offline only, NOT in tree)

User proposal: report where the flank line meets the air line, not the half-way
crossing; then anchor on the strongest axis and fit the noisier axes guided by it.

- `knee.py`    - independent per-axis air-line x first-departure flank line.
- `anchor.py`  - anchor = knee axis with smallest delta-method se; guided refits.
- `seg.py`     - seeded by the deployed half-way estimate; segmented least squares
                 (split k by min SSE); anchor-guided refit of all axes.
- `eval_*.py`  - scorers over probe_traces_2026-09-21/-22/-22b (run from this dir,
                 PYTHONPATH=../../klippy).  `peek*.py` draw the fits.

Result (16 groups, median per-point sd): half-way x 2.03um, deployed all-axes 2.27um,
intersection x 3.89um, seeded+guided 3.46-4.48um.  Intersections AGREE across axes
(guided y/z within 0-2um of x vs 20-25um for half-way) and sit ~30um above the half-way
point, but are ~2x less repeatable at ~6.7um sample spacing (4-7 samples on the flank).

## Tangent at the half-way crossing (user, 2026-09-23)

`tangent.py`: pivot a line on the half-way crossing, slope = local derivative there (fit over
the 10-90% band of the flank), intersect with the air line fitted above the knee. x only:
median 2.67um (free 2-line 3.89, half-way 2.03). Knee sits a median 22.6um above half-way.
Half-way position and (knee - half) are NEGATIVELY correlated in 12/16 groups (-0.3..-0.97):
the knee cancels flank-shape variation that moves the half-way point. Pooling slope over the two
reps of a probe does not help (2.67 -> 2.68), so the slope varies PROBE-TO-PROBE (6-26%), not
from per-ramp sampling noise.

## Anchored all-axes tangent knee (2026-09-23)

`tangent_axes.py`: anchor = tangent knee on the axis with the best step/noise (must pass the
deployed 15 gate, else air-only false-halt ramps "find" knees - (60,60) went to 508um without it);
every axis refitted with `signed_tangent` (either slope sign) at the anchor height; accepted if within
15um. Best combine = anchor keeps its own value + helpers fade-weighted on their own step/noise:
median 2.88um, mean(excl 30,70) 2.66um vs x alone 2.67 / 2.90. Helps (75,85) 5.70->1.99um, hurts
(90,70) run 1 1.20->4.52um. Helper knees sit +1.5..+1.9um from the anchor (IQR 7-9um).
Slope drift: |r(slope, probe order)| > 0.5 in 12/16 groups (chance ~4), sign split 6/6; slope vs
x air amplitude has no consistent sign (median r +0.22; amplitude varied only 1-5%).
