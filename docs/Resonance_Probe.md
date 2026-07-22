# Resonance nozzle probe

The `[resonance_probe]` module turns an accelerometer (for example an
[adxl345](Measuring_Resonances.md)) already mounted on the toolhead into a
nozzle-contact probe.  No dedicated probe hardware (no inductive sensor, no
load cell, no strain gauge) and no micro-controller firmware changes are
required - detection is performed entirely on the host from the standard
accelerometer data stream.

This document describes how the probe works, how to configure it, and how to
calibrate it.  The probe presents the standard Klipper [probe](Probe_Calibrate.md)
interface, so once configured it is usable by `PROBE`, `PROBE_ACCURACY`,
`PROBE_CALIBRATE`, `[bed_mesh]`, `[safe_z_home]`, etc.

This feature is experimental.

## How it works

The toolhead is vibrated back and forth along a printer axis (typically X) at a
fixed frequency chosen to match a mechanical resonance of the toolhead/gantry.
Because the motion is at resonance, a small commanded excitation produces a
relatively large, easily measured oscillation at the accelerometer.

When the nozzle touches the build platform the contact damps that resonance and
the measured oscillation amplitude drops.  The host watches the response
amplitude (a single-frequency DFT of the accelerometer signal, with the DC
component recomputed continuously so a contact-induced offset cannot mask the
drop) and reports contact when the amplitude falls below a threshold.

Because the detection is purely a *relative* amplitude drop, the exact
resonance frequency is not critical - the probe works across a range of nearby
frequencies as long as the response is strong enough to measure.

Contact damping is not always strongest on the accelerometer axis being
excited: at some modes (typically the lower ones) the toolhead's response to
contact couples mostly into a *different* accelerometer axis than the one
driven. The `hostdriven` halt and the calibration/ranking commands all watch
all three accelerometer axes at once and trigger on whichever axis shows the
drop, so these cross-coupled modes are detected reliably instead of being
missed by watching only the driven axis.

## Requirements and limitations

* An accelerometer mounted on the toolhead, configured as a bulk-sensor chip
  (e.g. `[adxl345]`) with a working `start_internal_client` (the same chip used
  for [input shaper](Measuring_Resonances.md) calibration is ideal).
* A toolhead/gantry with a usable lateral resonance (most printers have one in
  the 40-90 Hz range on X).
* The excitation amplitude must exceed roughly one motor microstep of travel,
  or quantization noise dominates the measurement.  Calibration handles this
  automatically (see below).
* The resonance frequency and the response amplitude vary somewhat with
  toolhead position (height and bed location).  This is normal and the probe
  tolerates it; calibrate at a representative point near the bed.
* Probing necessarily vibrates the nozzle against the platform.  On a
  spring-mounted bed the small overshoot is harmless.  On a rigidly mounted bed
  use the `hostdriven` mode (below), which halts the descent on contact.
* The descent speed is limited by the excitation frequency.  The amplitude
  measurement needs a window spanning several excitation cycles, and that window
  maps to a span of Z that grows with descent speed; once it exceeds the (~0.1
  mm) contact transition the detection smears.  In practice a ~47 Hz resonance
  is reliable up to roughly 1 mm/s.  A higher-frequency mode does *not* lift
  this ceiling: the excitation acceleration scales with the square of the
  frequency, so the achievable displacement (and signal) falls off, and the
  faster descent that a shorter cycle would allow is given back to a weaker,
  noisier measurement.  Keep `speed` at or below ~1 mm/s.

## Probe modes

The `probe_mode` config option selects how the descent and detection are
performed:

* `stepwise` - descend a small fixed increment, stop, measure the amplitude
  while stationary, repeat.  Safe and immune to motion-queue desync, but the
  resolution is one `probe_step` and it is slow.
* `hostdriven` (recommended) - vibrate while descending, and halt the descent
  the moment a host task detects contact (using a loose `halt_sensitivity`
  threshold).  A fine post-halt analysis then locates the precise contact Z.
  This bounds the over-drive (good for rigidly mounted beds) while keeping the
  reported height independent of the halt latency.

## Configuration

```
[resonance_probe]
accel_chip: adxl345
#   The accelerometer to use.  Required.
vibrate_axis: x
#   Printer axis to vibrate (x, y, z, or a "dx,dy,dz" vector).  The default
#   is x.  Vibrating z is refused by default (it drives the nozzle toward the
#   bed); use a lateral axis.
accel_axis: x
#   Accelerometer output axis that responds most strongly to the vibration
#   (often, but not always, the same letter as vibrate_axis).  Determined by
#   calibration.
excitation_frequency: 53.3
#   Resonance frequency to excite, in Hz.  Determined by calibration.  Required.
accel_per_hz: 120
#   Excitation strength.  Peak lateral displacement is approximately
#   accel_per_hz / (4 * pi^2 * excitation_frequency) millimetres; it must
#   exceed about one microstep.  Determined by calibration.
probe_mode: hostdriven
#   One of stepwise, hostdriven (see above).  The default is stepwise.
sensitivity: 0.06
#   Fractional drop in the resonance amplitude that indicates contact, used by
#   the fine (precise) detection.  The default is 0.3.
halt_sensitivity: 0.15
#   Loose fractional drop used only to HALT the descent in hostdriven mode (to
#   bound over-drive); the precise height still comes from the fine
#   'sensitivity' analysis, so this can be generous.  The default is 0.15.
warmup: 0.8
#   Seconds the excitation runs before detection is enabled, so the oscillation
#   reaches steady state first.  The default is 0.3.
#descend_speed:
#   Descent speed while vibrating, in mm/s.  Unset (the default) uses the
#   standard probe 'speed' parameter, so there is a single speed knob like any
#   other probe; set this only to descend slower than other probe moves.  Keep
#   it at or below ~1 mm/s (see the limitations above).
#probe_distance: 2.5
#   How far below the start/ceiling height the vibrating descent may travel
#   before giving up (also bounds how far a failed detection can drive the
#   nozzle).  The default is 0.5.
#probe_start_z:
#   Optional fixed height to rapid-move to before each probe descent.  Caps the
#   queued descent length and gives the probe a known starting Z regardless of
#   where the toolhead was parked.  Unset = probe from the current height.
#probe_start_speed:
#   Speed of the (non-vibrating) rapid move to probe_start_z.  Defaults to the
#   probe lift_speed.
#retune_range: 0
#   Per-probe-session re-tune: before each session, sweep +/-retune_range Hz
#   around the configured excitation_frequency and adopt the measured peak.
#   0 (the default) disables it.  Useful when the printer is moved or
#   reconfigured often, since the resonance drifts with the environment;
#   unnecessary if nothing changes.  See "Re-tuning for drift" below.
#retune_step: 1.0
#   Frequency step (Hz) of the re-tune's driven-response scan; the peak is
#   parabolically interpolated between steps.
#retune_time: 0.4
#   Excitation dwell (s) per step of the re-tune scan.
#freq_mesh:
#   Optional per-point excitation frequency, for machines whose resonance shifts
#   enough across the bed that one frequency will not do.  A grid of frequencies
#   whose coordinate extents are taken from the [bed_mesh] section (so there is no
#   separate min/max here).  Rows are Y (front to back), columns are X, matching
#   bed_mesh's ordering; the value at a probe point is bilinearly interpolated and
#   clamped to the grid edges.  Rows may be RAGGED: a row with a single value is
#   constant across X at that Y, so only the rows that actually vary carry extra
#   X-points.  Examples: a single 1x1 value is constant everywhere (same as the
#   scalar excitation_frequency); an Nx1 column varies with Y only; a 1xM row
#   varies with X only.  Normally produced by RESONANCE_PROBE_CALIBRATE_MESH, not
#   entered by hand.  See "Per-point frequency" below.
#freq_mesh_interp: bilinear
#   How freq_mesh is interpolated between grid points: bilinear (default) or
#   nearest.
#probe_step: 0.05
#   stepwise mode: descent increment.
#probe_nudge_radius: 0
#   If a repeat touch at the same point exceeds the standard 'samples_tolerance'
#   (disagreement between repeated samples), retry at a point on a circle of
#   this radius (mm) around the requested XY instead of the exact same spot -
#   a bad local spot (a plastic blob, a dead/low-friction patch) usually causes
#   one wild sample, and re-probing it just reproduces the disagreement.
#   Retries are spread evenly around the circle (same idea as the
#   CALIBRATE_MESH nudge - see "Building a freq_mesh automatically" below). A
#   good radius is about the nozzle tip's outer diameter.  0 (the default)
#   reproduces the standard [probe] retry behavior.
#z_offset: 0
#speed:
#   Standard [probe] parameters.  This is a CONTACT probe, so z_offset is 0
#   (the nozzle itself is the trigger); 'speed' is the descent speed.
```

`[resonance_probe]` also accepts the standard `[probe]` parameters
(`z_offset`, `speed`, `samples`, `sample_retract_dist`, `x_offset`, `y_offset`,
etc.); see [Config_Reference](Config_Reference.md#probe).

The probe value sections above (frequency, axis, amplitude, sensitivities) are
normally produced by the automatic calibration rather than entered by hand.

## Calibration

### Calibration helper module

Add the calibration helper, which runs the test moves and can write the
results back to the `[resonance_probe]` section:

```
[resonance_probe_calibrate]
accel_chip: adxl345
#   Default accelerometer for the calibration commands.
#accel_per_hz: 10
#max_accel_per_hz: 30
#   Excitation strength bounds used while sweeping.
```

### Automatic calibration

`RESONANCE_PROBE_CALIBRATE` performs the full setup automatically:

1. sweeps for candidate resonance modes (the same mode scan `RESONANCE_PROBE_
   RANK_FREQ` uses - see "Building a freq_mesh automatically" below for how
   candidates are found and why a quiet high mode can beat a loud low one),
   then picks whichever candidate damps most cleanly ON CONTACT, not just the
   loudest peak in air;
2. descends to the platform (at the full excitation amplitude) and measures
   the contact amplitude drop at the chosen frequency, deriving `sensitivity`
   and `halt_sensitivity`;
3. saves the resulting values into the `[resonance_probe]` section
   (`SAVE_CONFIG` is required to keep them).

Procedure:

1. Home the printer.
2. **Set Z=0 with the paper method first.**  With a sheet of paper as a feeler
   gauge between the nozzle and the bed, set Z=0 so the nozzle just pinches the
   paper - this leaves Z=0 about one paper thickness (~0.1 mm) above the bed, so
   true contact is at roughly Z=-0.1.  This is a **safety requirement**, not just
   a convenience: the calibration uses a hard descent floor of Z=-0.2 (see
   `CONTACT_ZMIN`), which is only safe if the bed sits a little *below* Z=0.  If
   the bed were *above* Z=0 the nozzle could be driven into it - exactly the same
   crash risk as normal printing with a mis-set Z=0, so set it carefully.  Also
   level the bed so the nozzle is a similar small distance above the platform
   everywhere.
3. Position the nozzle over a representative point, a small distance above the
   platform, e.g. `G1 X<center> Y<center> Z0.5`.
4. Run `RESONANCE_PROBE_CALIBRATE`.  It surveys candidate modes, finds contact
   with whichever mode confirms it, then runs a bounded, gentle amplitude
   search around that contact point at the chosen frequency.
5. `SAVE_CONFIG`.  (This is a *contact* probe, so `z_offset` is 0 - the nozzle
   itself is the trigger - and no separate `PROBE_CALIBRATE` is needed.)

The resonance frequency and amplitude vary with bed location, but the probe
detects the *relative* amplitude drop, so a single calibration at a central
point is generally sufficient for the whole bed.  Verify with `PROBE_ACCURACY`
(repeatability of a few microns is typical).  On the occasional machine whose
resonance shifts too much across the bed for one frequency, see "Per-point
frequency (freq_mesh)" below.

Useful parameters:

* `POINT=x,y,z` - move here before calibrating.
* `AXIS=x` - printer axis to vibrate.
* `FREQ=<hz>` - skip the mode search and calibrate at this exact frequency
  (also needs `ACCEL_AXIS=` if it differs from `AXIS`) - for a known-good
  resonance or a fast repeatable diagnostic run.
* `N_PEAKS=`, `PEAK_THRESHOLD=` - how many candidate modes the scan admits and
  how loud (relative to the single loudest peak) a peak must be to qualify
  outright; see "Building a freq_mesh automatically" below for the full
  candidate-selection logic (prominence, cross-axis candidates, etc).
* `MODE_ORDER=high` (default) tries candidates highest-first and makes the
  primary (first-tried) candidate the contact arbiter, same as
  `CALIBRATE_MESH` - a non-primary candidate reading clean while the primary
  stays silent is a shallow-false-halt signature and is not trusted, so a
  weak, position-sensitive mode cannot spuriously win the frequency choice.
  `MODE_ORDER=low` tries lowest-first instead.
* `CANDIDATES=f1,f2,...` - skip the mode scan and rank these explicit
  frequencies instead.
* `FREQ_START=`, `FREQ_END=` - frequency band searched for candidates (default
  5-135 Hz).
* `SAVE=0` - report the suggested values without modifying the config.
* `CONTACT_ZMIN=` - hard descent floor (default -0.2; see the paper-method note
  above).  `CONTACT_SPEED=` - first-contact descent speed.  `CONTACT_WARMUP=` -
  excitation warm-up before detection.
* `CONTACT_UP=`, `CONTACT_DOWN=`, `CONTACT_LEVELS=`, `CONTACT_CYCLES=`,
  `CONTACT_DWELL=`, `CONTACT_RAMP_SPEED=`, `CONTACT_MIN_DROP=`,
  `CONTACT_TARGET_NOISE=` - bounds and targets for the bounded amplitude search
  around the contact point (`CONTACT_DWELL` is the air/contact reference dwell
  time; `CONTACT_RAMP_SPEED` is the Z speed of the measured ramp through
  contact).

### Manual calibration helpers

Two lower-level commands are available if you prefer to choose values by hand:

* `RESONANCE_PROBE_FIND_FREQ [AXIS=x] [CHIP=]` - sweep frequencies and report
  the dominant resonance peak per accelerometer axis, with a suggested
  `accel_axis` and frequency.
* `RESONANCE_PROBE_MEASURE FREQ=<hz> [AXIS=x] [ACCEL_PER_HZ=] [DURATION=]` -
  vibrate at a fixed frequency and report the steady-state response amplitude
  per accelerometer axis.  Compare the amplitude with and without nozzle
  contact (made by hand) to choose `sensitivity`.

## Calibrating another probe's Z offset

If your machine already has a non-contact probe (inductive, eddy-current,
optical, etc.) you can use resonance probing as a *one-shot* nozzle-contact
reference to calibrate that probe's `z_offset` - without configuring
`[resonance_probe]` as the machine probe.  Only the `[resonance_probe_calibrate]`
helper is needed:

```
[resonance_probe_calibrate]
accel_chip: adxl345
```

Then:

1. Home the printer and **set Z=0 with the paper method** (as in the calibration
   procedure above) so true contact is around Z=-0.1 and the -0.2 floor is safe.
2. Position the nozzle over a point your real probe can also reach, roughly
   1 mm above the bed.
3. Run `RESONANCE_PROBE_CONTACT`.  It finds the lateral resonance, vibrates
   while descending, and **halts in real time** on contact (the same host-driven
   halt the `hostdriven` probe mode uses - safe on a rigid bed, no over-drive),
   reports the contact Z, and retracts to where it started.  Add `SAMPLES=3` to
   repeat and report the median and spread.
4. `PROBE` at the same X/Y to read the Z at which your real probe triggers, then
   set `z_offset = (probe trigger Z) - (reported contact Z)`.

The descent halts on contact, so it does not drive into the bed; the `ZMIN`
floor (default -0.2) is only a backstop for a detection miss.
`RESONANCE_PROBE_CONTACT` also accepts `FREQ=` and `ACCEL_AXIS=` (to skip the
resonance search if you already know them), `ACCEL_PER_HZ=`, `POINT=x,y,z`,
`SENSITIVITY=`, `HALT_SENSITIVITY=`, `SPEED=` (descent speed, default 1 mm/s),
`WARMUP=`, and `ZMIN=`/`DISTANCE=` (safety floor and max descent).

## Tuning

* `accel_per_hz` too low gives a noisy amplitude measurement (the swing is near
  or below one microstep); too high risks a harder nozzle contact.  Aim for a
  displacement of a few microsteps.
* `sensitivity` must be larger than the baseline measurement noise (to avoid
  false triggers) and smaller than the contact drop.  Raise it if probing
  triggers early/erratically; lower it (toward the noise floor) for more
  precision.
* `halt_sensitivity` (hostdriven) sets how far the descent over-drives before
  halting: a larger value halts later (more over-drive) but more reliably.  The
  reported height is unaffected by this value.
* `speed` (or `descend_speed`) trades probe time against over-drive; slower is
  gentler.  Stay at or below ~1 mm/s - faster smears the detection (see the
  limitations above).

### Re-tuning for drift

The lateral resonance frequency drifts with the machine's environment (mounting,
temperature, belt tension, even moving the printer to a new location).  If you
leave the printer set up and untouched, the calibrated `excitation_frequency` is
stable and no re-tuning is needed.

If instead the machine is moved or reconfigured frequently, set `retune_range`
to a few Hz.  At the start of every probe session the probe then scans the
driven response over `excitation_frequency` +/- `retune_range` (a fixed-frequency
scan stepped by `retune_step`, dwelling `retune_time` per step, with the peak
parabolically interpolated) and excites the measured peak instead.  A *driven*
scan is used rather than a swept spectrum because it measures how hard the
toolhead actually resonates when driven at each frequency, which is the property
the probe uses.  The toolhead position is restored before probing, so the cost is
only a second or two.  Leaving `retune_range` at 0 disables the step.

When a `freq_mesh` is configured, re-tuning is done *lazily per mesh cell*: the
first probe point that falls nearest to a given cell re-tunes that cell in place
(no detour, since the toolhead is already there) and the refined value is reused
for the rest of the session.  So the number of re-tunes equals the number of mesh
cells actually used - fewer than the number of probe points when the mesh is
coarse or collapsed to a single column/value.

## Per-point frequency (freq_mesh)

Most machines have one lateral resonance that works across the whole bed, so a
single `excitation_frequency` (optionally re-tuned for drift) is enough.  On some
machines the useful resonance shifts enough with bed position that no single
frequency detects well everywhere.  For those, `freq_mesh` supplies a per-point
excitation frequency.

The mesh borrows its coordinate extents and grid from the `[bed_mesh]` section
(so there is nothing to keep in sync by hand); rows are Y, columns are X.  At each
probe point the frequency is bilinearly interpolated from the mesh and clamped to
the grid edges.  Rows may be *ragged* - a row with a single value is constant
across X at that Y - so only the rows that genuinely vary carry extra X-points,
and only those cost extra re-tunes.  The common cases collapse naturally:

* a single `1x1` value - constant everywhere (identical to the scalar
  `excitation_frequency`);
* an `Nx1` column - varies with Y only, constant across X;
* a `1xM` row - varies with X only, constant across Y.

### Building a freq_mesh automatically

`RESONANCE_PROBE_CALIBRATE_MESH` surveys the `[bed_mesh]` grid and writes a
`freq_mesh` (run `SAVE_CONFIG` to keep it).  It selects the frequency at each
point by **contact damping**, not by in-air response.

Candidate identification runs once at the bed center: a fast continuous sweep
locates the modes, then a slower discrete driven scan refines them within a
narrow `CAND_WINDOW` of each swept peak.  The search spans the full input-shaper
range (`FREQ_START`-`FREQ_END`, default 5-135 Hz) - a real resonance can land
anywhere in it, so bad modes are rejected by *behaviour* (below), not by a
machine-specific band.

At each grid point the candidates are tried highest-first by default (see
`MODE_ORDER` below), but **the primary (first-tried) mode is the contact
arbiter**, whichever end of the range it comes from.  A clean detection is
trusted only when the surface is really reached: either the primary candidate
is itself the clean one, or it at least *confirms* contact (it damps a little).
A halt where only a *non-primary* mode reads clean while the primary is silent
is the signature of a shallow false halt - a weak, position-sensitive mode
blipping a "drop" above the true surface - and is **not** accepted; the point is
retried for a deeper contact.  This arbiter rule is also safe regardless of
order: contact is always *found* with a reliably damping mode, and the other
candidates are only ever measured by a *bounded* oscillation around the
already-known contact height, never by a blind descent that could press a
weakly-damping mode into the platform.  The highest mode is preferred by
default because it has proven the more reliable all-around detector - it damps
strongly on contact even on slick plates where the low modes barely couple.

If a point still yields no clean detection - typically a low-friction or dirty
micro-spot that couples poorly - it is retried at a small lateral **nudge** off
the point: points spread evenly around a circle of `NUDGE_RADIUS` (see below).
Because the offset is tiny, bed height and resonance are effectively unchanged,
so this is a far better proxy than an adjacent grid point; the arbiter applies
at the nudged spot too, so a bad micro-region cannot sneak in a spurious clean
reading from a non-primary mode.  A neighbour's frequency is never substituted -
the bed height there is genuinely different, which is what `[bed_mesh]` measures.

Points that agree collapse to a single value, so a uniform machine yields a
scalar and only a truly varying bed yields a full or ragged mesh.

Motion reuses the configured `[resonance_probe]` so calibration moves like real
probing: the Z lift/retract uses the probe's `lift_speed` and the ring-up
`warmup` is inherited from it.  The point-to-point travel runs at the machine's
max velocity (the speed real bed-mesh probing travels at), not the near-bed lift
speed.  The contact *descent* stays deliberately slow and the `CONTACT_ZMIN`
floor deliberately shallow - both calibration-specific, for accuracy and
hands-off safety.

The same paper-method Z=0 safety setup as `RESONANCE_PROBE_CALIBRATE` applies
(this command touches the bed at the survey points).  Useful parameters:

* `CONTACT_POINTS=all` (default) tests every bed_mesh point; `=corners` tests only
  the center and four corners (fast - if they agree, the bed is uniform).
* `FREQ_START=`, `FREQ_END=` - the frequency band searched for candidate modes
  (default 5-135 Hz, Klipper's own input-shaper sweep range).  A bad far-high
  mode is ruled out by the low-mode arbiter above, not by narrowing this band.
* `CAND_WINDOW=` - half-width (Hz, default 8) of the discrete driven-scan window
  around each swept peak, so candidate refinement does not re-scan the whole
  range point by point.
* `NUDGE_RADIUS=`, `NUDGE_TRIES=` - radius (mm, default 1.0) and max retries
  (default 4) of the circular nudge used to step off a point that gives no clean
  detection.  Retries are spread evenly around a circle of that radius (at a
  random starting angle), spreading wear and avoiding the same bad micro-spot.
  A good radius is about the nozzle tip's outer diameter.
* `MODE_ORDER=high` (default) tries candidate modes highest-first when picking
  the primary (arbiter) mode at the center point, since the highest mode is
  usually the better all-around detector; `=low` tries lowest-first (the
  original, more conservative order described above) instead.  Either way the
  chosen primary mode still has to confirm contact, or the next candidate in
  the order is tried.
* `TRACK_WINDOW=` - half-width (Hz, default 5) of the narrow re-scan used to
  re-locate the primary mode's peak at a new grid point if it appears to have
  drifted (0 disables this recovery re-scan).
* `CANDIDATES=f1,f2,...` - skip the sweep and rank these explicit frequencies
  as the candidate modes instead.
* `DRIP_TIME=` - MCU step-buffer look-ahead for the vibrating descent (s, default
  0.3).  The host-driven halt normally feeds the MCU just ~0.1 s ahead so it can
  stop promptly; a long, dense, or high-frequency vibrating descent can then
  outrun the host and starve the step pipeline (an MCU "Timer too close"
  shutdown).  Buffering further ahead prevents that at the cost of a little
  over-travel after the halt (`~DRIP_TIME x CONTACT_SPEED`, bounded by
  `CONTACT_ZMIN`; the precise contact Z comes from the anchored analysis, not the
  stop position).
* `VIB_SPAN=` - alternative overrun guard (mm, default 0 = off).  Instead of a
  deeper buffer, vibrate only the final `VIB_SPAN` mm just above the floor,
  reaching it by a fast non-vibrating approach.  This bounds the drip regardless
  of host speed, but the shorter vibrating run raises the measurement noise, so
  `DRIP_TIME` is preferred; if used, `VIB_SPAN` must exceed the bed's height
  variation.
* `MESH_TRAVEL=` - override the point-to-point travel speed (defaults to the
  toolhead's max velocity).
* `CONTACT_ZMIN=` - hard descent floor (default -0.2; see the paper-method note).
* `SAVE=0` - survey and report without writing the config.

### Picking the best frequency at one point

`RESONANCE_PROBE_RANK_FREQ` does the mode comparison at the current point without
writing anything: it finds the candidate modes, finds contact, and ranks each
mode by how strongly it damps on contact relative to its noise.  Use it to
understand a machine's modes, or to choose a single `excitation_frequency` by the
right criterion (contact damping) rather than by which mode is loudest in air.

## Development status and planned work

This feature is experimental and under active development.  What is implemented
and validated on hardware, and what is still in progress, as of this writing:

**Working and validated**

* The `hostdriven` probe path: vibrate-while-descend with a real-time host halt
  on contact, then an *anchored* fine analysis that pins the precise contact Z
  by refining around the halt position (rather than re-searching the whole
  stream).  Repeatability of ~5-15 um over 8 touches was measured.
* A shared `HaltingContactProbe` helper implements that halting descent + anchored
  analysis; both `[resonance_probe]` (hostdriven mode) and the
  `[resonance_probe_calibrate]` utility use it, so every user-facing descent
  halts in real time (no over-drive).
* The safety model for calibration: the paper-method Z=0 setup plus a tight
  Z=-0.2 hard floor bounds any detection miss to ~0.1 mm of over-travel.  Note
  that the floor must sit at least `detect_confirm_z` (~0.05 mm) below true
  contact for the halt to be able to confirm, which the paper-method geometry
  provides.
* Automatic amplitude selection during `RESONANCE_PROBE_CALIBRATE`: one firm
  first contact at a strong amplitude, then a bounded, gentle oscillation around
  that known contact that measures the air-vs-contact amplitude drop and noise to
  derive `sensitivity`/`halt_sensitivity`.
* **Frequency selection by contact damping.**  Candidate resonance modes are
  ranked by how strongly each damps on contact (relative to its noise), not by
  in-air loudness - so the mode a probe actually needs is chosen even on a
  multi-modal gantry.  Exposed as `RESONANCE_PROBE_RANK_FREQ` and used internally
  by `RESONANCE_PROBE_CALIBRATE_MESH`.
* **High mode preferred as the universal probe frequency.**  Candidates are
  tested highest-first by default (`MODE_ORDER=high`; see "Building a freq_mesh
  automatically" above) since the highest structural mode has proven the more
  reliable all-around contact detector, including on slick plates where the low
  modes barely couple.  The contact-*find* (not just the damping comparison)
  escalates through the same highest-first order if the current candidate never
  locates contact, so a poor detector at one point does not stall the whole
  survey - validated on hardware.
* **Per-point frequency (`freq_mesh`) with lazy per-cell re-tune**, populated by
  `RESONANCE_PROBE_CALIBRATE_MESH` (contact-damping selection, ragged/collapsing
  mesh; see "Per-point frequency" above).  On a machine whose resonance is uniform
  the whole survey collapses to a single value; only a genuinely varying bed
  produces a full or ragged mesh.
* **Cross-axis detection.**  Both the real-time `hostdriven` halt and the
  offline contact-damping measurement (`RANK_FREQ`, `CALIBRATE_MESH`, etc.)
  watch all three accelerometer axes simultaneously and trigger on whichever
  axis damps, instead of only the driven axis.  This matters most for
  low-frequency modes, whose contact response is often carried almost
  entirely by a cross-coupled axis rather than the excited one - validated on
  hardware across the full frequency range, including on a smooth bed surface
  where the on-axis drop alone is too weak to detect reliably.

**In progress / planned**

* **Live nudge-on-consistency-failure (`probe_nudge_radius`).**  A `PROBE`/bed
  mesh touch whose repeat samples disagree beyond `samples_tolerance` is
  retried at a point on an evenly-spread ring around the requested XY (see
  "Configuration" above) instead of the exact same spot, mirroring
  `CALIBRATE_MESH`'s nudge.  Implemented; not yet hardware-validated.  A generic
  version of the same mechanism for any touch probe (not just this one) is
  planned as a separate, later change.
* **`RESONANCE_PROBE_RETUNE` command.**  Break the pre-probe re-tune (currently
  the `retune_range` option inside every probe session) out into a standalone
  command that can be placed in the user's print-start macro - an abbreviated
  re-calibration of just the frequency.  Keeps probing and tuning as separate
  concerns; `retune_range` remains available for hands-off use.
* **Continuous-excitation contact reversal (motion enhancement).**  Today the
  descent *halts* (stops) on contact.  A future toolhead extension could let the
  contact detector *reverse* the move instead, so the excitation runs
  continuously through the turnaround - no warm-up transient, and never rung up
  while touching the bed.  Most valuable for the probe itself (a second
  release-edge per touch, no per-touch warm-up).  This touches core motion code
  and is deferred as its own careful effort.

**Remaining hardware validation before this is production-ready**

* Probing with a hot nozzle (contact damping may change with a softer tip).
* A different platform texture.
* Actually using the probe for a real print.

## Errors

* "no contact detected" - the descent reached its floor without a detectable
  amplitude drop; lower the start height, increase `accel_per_hz`, or lower
  `sensitivity`. If the excitation is too weak (near or below one microstep of
  travel) the amplitude measurement is dominated by quantization noise and the
  drop cannot be seen - raise `accel_per_hz`, or the chosen frequency may not be
  a real resonance.
* "Accelerometer measured no data while probing" - the accelerometer returned
  no samples for the vibrating move; check the `accel_chip` wiring and that it
  streams (the same chip used for input-shaper calibration should work).
* "start Z ... is at or below the descent floor" - there is no room to descend
  from the current height to the floor (`z_min`/`probe_distance`); raise the
  nozzle before probing or increase `probe_distance`.
