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

## Choosing the anchor axis (2026-09-23)

`quality.py`: predicted knee sigma = sqrt((sd/s)^2 + (d*se_slope/s)^2 + (se_air/s)^2), s = |slope - air slope|.
CALIBRATED (predicted vs actual rep-to-rep: x 1.8/2.1um, z 3.1/3.6um). But picking the best axis
PER RAMP from independent fits is worse than x alone (3.1-4.1um median): independent z knees sit a
point-dependent -25..+43um from x's (z follows its bump/fall), so switching axes re-creates the
offset jumps. `eval_fixed.py`: choose the anchor ONCE per point (lowest median predicted sigma) ->
2.35um median / 2.43um mean (x alone 2.67 / 2.90); + all guided helpers inverse-variance 2.13 / 2.46;
+ only the second-best guided 2.35 / 2.51. Per-point choice uses the scored data (mild selection
bias); in-tree it would be made at CALIBRATE time from separate probes.

## Stateless: anchor chosen per verify CALL (user constraint, 2026-09-23)

No state across probes; choosing once per verify/refine call, and one inverse-variance average over
every axis x down-ramp fit in the call, are allowed. `eval_call.py`: B (one IVW over the call,
anchor = lowest median predicted sigma over the call's ramps) 2.44um median / 3.15um mean (x alone
2.67 / 2.90). The mean is one call anchoring on z's later fall at (60,45) (11.4um); the
first-departure rule fixes that (1.73um) but breaks (90,70) run 1 (2.08 -> 5.14um, z's own knee sits
above x there) -> 2.44 / 2.68. `eval_consist.py`: choosing the anchor by how well the other axes
confirm it is WORSE (4.53um mean) - guided fits search +-15um around any anchor, so confirmation is
circular. Ceiling with a stable anchor (fixed per point): 2.11 / 2.47.
