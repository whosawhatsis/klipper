# Host-side contact detection for probes that report an analog value
#
# Copyright (C) 2026  Klipper contributors
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#
# This is the host-side counterpart to trigger_analog.c.  Firmware can only
# evaluate a trigger on a value it actually receives, so a probe whose contact
# signal must be COMPUTED (a DFT bin amplitude, a filtered ratio, a value fused
# across several channels) cannot use the MCU triggers.  The detectors here
# deliberately mirror trigger_analog's semantics so that a probe behaves the
# same whichever side evaluates it.
#
# The value is assumed to be sampled while descending at a known speed, so
# thresholds are expressed per-mm of Z and converted to per-sample internally
# (as probe_eddy_current does for tap_threshold).  A per-sample threshold would
# silently change meaning with probe speed or sample rate.
import math


# Detector modes.  Named for what the SIGNAL does at contact, not for the
# sensor, since the same sensor can present differently at different settings.
#
#   DRAWDOWN  value falls by `threshold` below its running peak.  Mirrors
#             trigger_analog's diff_peak_gt.  This is the right choice whenever
#             the value RISES on approach and drops at contact: the running
#             peak absorbs the approach ramp, so no baseline model is needed.
#             Accumulating the fall (rather than differencing adjacent samples)
#             is what makes it insensitive to sample spacing - a fixed drop
#             split across more samples still reaches the same drawdown, where
#             a per-sample difference shrinks toward the noise.
#
#   RUNUP     the mirror image: value rises by `threshold` above its running
#             trough.  For sensors that read LOW in free air (a load cell, or
#             the release edge when lifting off a surface already touched).
#
#   FLATTEN   the rate of change falls below `threshold` and stays there.  For
#             sensors whose value tracks the GAP and therefore stops changing
#             once contact is made and further Z motion goes into deflecting
#             the bed or the mount rather than closing the gap - capacitive
#             nozzle sensing works this way.  Note this one needs an
#             `arm_slope`: it must first observe the value changing, or a
#             stationary sensor reads as "contact" from the first sample.
DRAWDOWN = 'drawdown'
RUNUP = 'runup'
FLATTEN = 'flatten'
MODES = (DRAWDOWN, RUNUP, FLATTEN)


class ContactDetector:
    """Detect contact in a stream of analog samples taken during a descent.

    thresh_per_mm - contact threshold, in value-units per mm of Z travel.  For
        DRAWDOWN/RUNUP this is the excursion from the running extreme; for
        FLATTEN it is the slope at or below which the value counts as no longer
        changing.  Per-mm so that it does not change meaning with probe speed.
    speed - descent speed (mm/s), used with the sample rate to convert the
        per-mm threshold into per-sample terms.
    persist - consecutive qualifying samples required.  A single sample is one
        noise excursion away from a false halt.
    persist_mm - persistence expressed as Z travel instead, converted to a
        sample count at runtime.  Prefer this: a sample count has the same
        speed-dependence the per-mm threshold exists to avoid, since the same
        physical event spans fewer samples when sampling is coarser.  A rule
        tuned at one probe speed then silently demands a longer event at
        another.
    relative - when True the threshold is a FRACTION of the running reference
        rather than an absolute value-unit amount.  Correct for sensors whose
        signal scales with drive level (our resonance amplitude does), so the
        same config works at any excitation strength.
    """

    def __init__(self, mode, thresh_per_mm, speed, sample_rate, persist=2,
                 relative=False, arm_slope_per_mm=None, persist_mm=None,
                 lookback_mm=None):
        if mode not in MODES:
            raise ValueError("unknown contact mode '%s'" % (mode,))
        if thresh_per_mm <= 0.:
            raise ValueError("thresh_per_mm must be positive")
        if speed <= 0. or sample_rate <= 0.:
            raise ValueError("speed and sample_rate must be positive")
        self.mode = mode
        self.relative = relative
        # mm of Z advanced per sample; the bridge between the per-mm config and
        # the per-sample stream.
        self.mm_per_sample = speed / sample_rate
        if persist_mm is not None:
            self.persist = max(1, int(round(persist_mm / self.mm_per_sample)))
        else:
            self.persist = max(1, int(persist))
        if mode == FLATTEN:
            # A slope threshold stays per-mm; it is compared against a measured
            # per-mm slope directly.
            self.threshold = thresh_per_mm
            self.arm_slope = (arm_slope_per_mm if arm_slope_per_mm is not None
                              else thresh_per_mm * 3.)
        else:
            # An excursion threshold is per-mm of travel, so the amount that
            # counts over one sample scales with how far a sample covers.  Note
            # this is NOT multiplied by mm_per_sample for the drawdown total -
            # the drawdown accumulates over however many samples the event
            # spans, and the config value is the total excursion that counts.
            self.threshold = thresh_per_mm
            self.arm_slope = None
        # Bounded lookback for the running extreme.  An UNbounded peak (what
        # trigger_analog's diff_peak_gt uses) is correct for a smooth,
        # monotonic sensor, but on a noisy signal that rises during approach
        # any slow dip keeps accumulating against a peak set long ago, and the
        # detector eventually fires in mid-air.  Measured on real descents: a
        # global peak fired ~0.5mm above contact on 4 of 24 axis-traces.
        # Limiting the peak to the last lookback_mm keeps the baseline immunity
        # (the peak still tracks the approach ramp) while capping how much slow
        # wander can accumulate.  None = unbounded, matching the MCU detector.
        self.lookback = None
        if lookback_mm:
            self.lookback = max(2, int(round(lookback_mm
                                             / max(self.mm_per_sample, 1e-12))))
        # Monotonic deque of (index, value) holding the candidates for the
        # window extreme, newest last - O(1) amortised per sample.
        self._win = []
        self._n = 0
        self._run = 0
        self._armed = False
        self._prev = None
        self._triggered_at = None
        self.last_excursion = 0.

    def reset(self):
        self._win = []
        self._n = 0
        self._run = 0
        self._armed = False
        self._prev = None
        self._triggered_at = None
        self.last_excursion = 0.

    def _excursion(self, value):
        """How far the value has moved against the running extreme."""
        better = ((lambda a, b: a >= b) if self.mode == DRAWDOWN
                  else (lambda a, b: a <= b))
        # Drop candidates that this sample dominates: they can never be the
        # window extreme again.
        while self._win and better(value, self._win[-1][1]):
            self._win.pop()
        self._win.append((self._n, value))
        # Expire the extreme once it leaves the lookback window.
        if self.lookback is not None:
            while self._win and self._win[0][0] <= self._n - self.lookback:
                self._win.pop(0)
        self._n += 1
        extreme = self._win[0][1]
        delta = (extreme - value) if self.mode == DRAWDOWN else (value - extreme)
        if delta <= 0.:
            return 0.
        if self.relative:
            ref = abs(extreme)
            if ref < 1e-12:
                return 0.
            return delta / ref
        return delta

    def update(self, value, position=None):
        """Feed one sample.  Returns True when contact is detected.

        `position` is the Z (or time) to report as the trigger point; the
        detector does not interpret it beyond storing it.
        """
        if value is None or (isinstance(value, float) and math.isnan(value)):
            # A NaN would poison the running extreme permanently - a peak of
            # NaN compares False against everything, so the detector would go
            # silently blind for the rest of the descent.
            return False
        if self.mode == FLATTEN:
            hit = self._update_flatten(value)
        else:
            exc = self._excursion(value)
            self.last_excursion = exc
            hit = exc >= self.threshold
        if hit:
            self._run += 1
            if self._run >= self.persist:
                if self._triggered_at is None:
                    self._triggered_at = position
                return True
        else:
            self._run = 0
        return False

    def _update_flatten(self, value):
        # Slope per mm between adjacent samples.
        if self._prev is None:
            self._prev = value
            return False
        slope = abs(value - self._prev) / max(self.mm_per_sample, 1e-12)
        if self.relative:
            ref = abs(self._prev)
            slope = slope / ref if ref > 1e-12 else 0.
        self._prev = value
        # Must see real movement before "stopped moving" means anything:
        # otherwise a sensor sitting still in free air triggers immediately.
        if not self._armed:
            if slope >= self.arm_slope:
                self._armed = True
            return False
        return slope <= self.threshold

    @property
    def triggered_at(self):
        return self._triggered_at


def estimate_noise(values, mm_per_sample=None):
    """Sample-to-sample noise of a value stream, as a robust scale estimate.

    Used to derive a threshold from the signal itself rather than from stored
    per-location config: every descent observes free air before it observes
    contact, so the noise can be measured fresh on each descent.  That is what
    lets one threshold follow a machine across bed positions, surface finishes
    and removable plates, none of which stay put well enough to calibrate per
    point.

    Returns the median absolute sample-to-sample step (robust to the odd
    outlier, unlike a standard deviation), in value units per sample - or per
    mm if mm_per_sample is given.
    """
    if len(values) < 3:
        return 0.
    steps = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    steps.sort()
    n = len(steps)
    med = (steps[n // 2] if n % 2 else 0.5 * (steps[n // 2 - 1]
                                              + steps[n // 2]))
    if mm_per_sample:
        return med / max(mm_per_sample, 1e-12)
    return med


def detector_margin(drop, noise, floor, nsigma):
    """How many times over its own threshold a candidate's drop actually is.

    This is deliberately the DETECTOR'S arithmetic, not a separate scoring
    formula.  A selector that models noise differently from the thing it is
    selecting for will mis-rank wherever the two disagree - measured on real
    descents: a `0.5*drop - (1.15*noise + 0.015)` headroom ranked a candidate
    SECOND that the detector could not use at all (0% trigger rate), because
    a 1.15x noise term is far gentler than the nsigma bar the threshold uses.
    """
    thr = max(floor, nsigma * noise)
    if thr <= 0.:
        return 0.
    return drop / thr


def robust_axis_margin(per_axis, floor, nsigma):
    """Score a candidate by its SECOND-best axis.

    per_axis: iterable of (drop, noise), or None for an axis with no reading.

    Detection succeeds if ANY armed axis fires, so the score must credit a
    strong axis.  But which axis carries contact changes with position on the
    bed (measured across random points: z can go from inert to the strongest
    detector over 10mm while x halves), so a candidate resting on ONE good axis
    can go blind where the signal moves off it.  The second-best axis captures
    both: it is high only when at least two axes are usable, and it rises with
    their strength.

    Chosen empirically, not by taste.  Against a ground truth computable from
    the trace corpus - worst location, best axis there, i.e.
    min over locations of (max over axes) - the candidates ranked:

        2nd-best axis      rho = +0.89
        mean of axes       rho = +0.71
        best x redundancy  rho = +0.71
        max over axes      rho = +0.66
        MIN over axes      rho = +0.37   <- the intuitive "weakest link" rule

    min() is the worst of them: it ranked a candidate LAST (0.6x) that the
    ground truth places SECOND (5.4x), because that candidate had two strong
    axes and one dead one - and a dead third axis costs nothing when the other
    two work everywhere.

    Returns 0. when fewer than two axes are readable: a single reading is not
    evidence of redundancy.
    """
    vals = []
    for entry in per_axis:
        if entry is None:
            continue
        drop, noise = entry
        vals.append(detector_margin(drop, noise, floor, nsigma))
    if len(vals) < 2:
        return 0.
    return sorted(vals, reverse=True)[1]


class FindThenRefine:
    """Two-phase probing: find contact fast, then measure it slowly.

    Conventional probing uses ONE descent for both jobs, which forces a
    tradeoff - slow enough to be accurate means slow everywhere, and the usual
    remedy (probe twice, require the two to agree) pays for the variance twice
    over rather than removing it.

    Splitting them removes the tradeoff, because the two phases are limited by
    different things:

      find   - only has to stop the toolhead near the surface.  Its error
               budget is the over-travel it commits to, so it can run as fast
               as the detector's margin and the motion buffer allow.
      refine - runs on a stationary or slowly-moving toolhead over a short,
               bounded span around the now-known contact height.  Its cost does
               not scale with how far the probe descended, so it can be as slow
               and as thorough as accuracy requires.

    Measured on the resonance probe: detection margin was flat from 0.2 to
    0.8 mm/s while the reported precision came almost entirely from the refine
    pass - i.e. descent speed traded against over-travel, not accuracy.

    This class owns only the SEQUENCING and the safety invariants.  The two
    callbacks do the probe-specific work.
    """

    def __init__(self, find_cb, refine_cb, max_overtravel=None):
        self.find_cb = find_cb
        self.refine_cb = refine_cb
        self.max_overtravel = max_overtravel

    def run(self, *args, **kwargs):
        """Returns (z, refined) - refined is False when only find succeeded."""
        found = self.find_cb(*args, **kwargs)
        if found is None:
            return None, False
        refined = self.refine_cb(found, *args, **kwargs)
        if refined is None:
            return found, False
        # A refine that lands far from where find halted is not a better
        # measurement of the same event - it is evidence the two disagree about
        # WHICH event they measured, and silently preferring the refined value
        # would launder a false halt into a confident answer.
        if (self.max_overtravel is not None
                and abs(refined - found) > self.max_overtravel):
            return found, False
        return refined, True
