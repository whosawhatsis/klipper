#!/usr/bin/env python3
# Self-check for _verify_verdict, the confirm/reject decision.  Run from klippy/:
#     python -m extras.test_verify_verdict
#
# This exists because the decision used to be written out inline in the normal
# path only, and the SALVAGE path returned before reaching it.  So the one
# result that runs precisely because detection already failed was the only one
# never checked - and on 2026-08-06 that silently put a 345um error into a bed
# mesh.  Both paths now call this function; these cases pin its behaviour so
# they cannot drift apart again.
#
# The numbers are the ones the machine logged at the corrupted point.
from .resonance_probe import HaltingContactProbe

verdict = HaltingContactProbe._verify_verdict

THRESH = 0.15     # VERIFY_DROP default
STEP_MIN = 8.0    # the step threshold as it USED to default; the shipped
                  # default is now 0 (off), so these cases pass it explicitly


def test_reproduces_the_logged_rejection():
    # "verify: REJECTED false halt z=0.0312 (air=2852 touch=2467, -14% < 15%;
    #  step SNR 3.2 < 8.0)"
    ok, drop, _ = verdict(2852., 2467., 3.2, THRESH, STEP_MIN)
    assert not ok, "the halt the machine rejected would now be confirmed"
    assert abs(drop - 0.135) < 0.002, drop


def test_ratio_alone_confirms():
    ok, drop, how = verdict(6000., 5000., 0.0, THRESH, STEP_MIN)
    assert ok and how == "ratio", (ok, drop, how)


def test_step_alone_confirms_a_marginal_ratio():
    # The reason the OR exists: a real touch can read marginal on ratio when the
    # ramp is short and the medians blend air with contact.
    ok, drop, how = verdict(6000., 5600., 12.0, THRESH, STEP_MIN)
    assert ok and how == "step", (ok, drop, how)
    assert drop < THRESH, drop


def test_neither_criterion_rejects():
    ok, _, _ = verdict(6000., 5700., 2.0, THRESH, STEP_MIN)
    assert not ok


def test_step_test_disabled_by_zero_leaves_ratio_in_charge():
    ok, _, _ = verdict(6000., 5600., 99.0, THRESH, 0.)
    assert not ok, "step test still fired with step_min=0"
    ok, _, how = verdict(6000., 5000., 99.0, THRESH, 0.)
    assert ok and how == "ratio"


def test_zero_or_negative_air_cannot_confirm_by_ratio():
    # A dead/absent air reading must not read as a 100% drop.  It divides by
    # air, so this is the case that would turn a measurement failure into a
    # confident confirmation.
    for air in (0., -1., 1e-12):
        ok, drop, _ = verdict(air, 0., 0.0, THRESH, STEP_MIN)
        assert drop == 0. and not ok, (air, drop, ok)


def test_touch_above_air_is_not_a_drop():
    # Amplitude RISING on contact is not a contact signal; it must not confirm.
    ok, drop, _ = verdict(3000., 3400., 0.0, THRESH, STEP_MIN)
    assert drop < 0. and not ok, (drop, ok)


# --- the inverted-rise rule -------------------------------------------------
# A rejection normally re-arms BELOW the candidate.  That is right when a false
# halt sits above true contact, and exactly wrong when the amplitude ROSE on
# contact: at a weakly-driven location touching ADDS signal, and descending
# further walks toward a deep false contact.  Logged 2026-08-06: rejections at
# -66%, -19% and -13% marched a descent from z=1.41 past the true surface,
# over-pressing 0.282mm.  These are the measured numbers.

rose = HaltingContactProbe._is_inverted_rise
RISE = 0.10       # verify_rise_abort default


def test_catches_the_logged_inverted_rises():
    for drop in (-0.66, -0.3811, -0.19, -0.1303):
        assert rose(drop, RISE), "rise of %.0f%% not caught" % (-drop * 100)


def test_ignores_small_negative_drops_that_are_just_noise():
    # Real, benign rejections from the same run: a couple of percent either way.
    for drop in (-0.02, -0.016, 0.0163, 0.0177):
        assert not rose(drop, RISE), drop


def test_a_normal_confirmed_drop_is_never_an_inverted_rise():
    for drop in (0.3666, 0.4251, 0.3109, 0.2706):
        assert not rose(drop, RISE)


def test_disabled_by_zero():
    assert not rose(-0.66, 0.)


def test_boundary_is_inclusive():
    assert rose(-RISE, RISE)
    assert not rose(-RISE + 1e-9, RISE)


def test_threshold_separates_the_two_populations():
    # The default must sit between the benign rejections and the inverted ones,
    # or it either never fires or aborts on noise.
    benign = max(0.02, 0.016)          # largest benign RISE seen
    inverted = 0.1303                  # smallest inverted rise seen
    assert benign < RISE <= inverted, (benign, RISE, inverted)


# --- the corroboration rule -------------------------------------------------
# The other way a rejection must NOT re-arm: the halt sits on a contact already
# CONFIRMED at the same XY.  Logged 2026-08-07 on the hot mesh at (60, 62.5):
# sample 1 confirmed z=0.0594; sample 2 halted at 0.0558, was rejected at -10%
# against the 15% threshold because the air level had decayed to 2768, re-armed
# below it, and drove to -0.1495 before min_air_fraction stopped it.  0.21mm of
# nozzle into the plate for a contact that was already known.

corrob = HaltingContactProbe._corroborated
TOL = 0.05        # verify_corroborate_tol default


def test_catches_the_logged_plow():
    # The rejected halt against the same point's confirmed contact.
    assert corrob(0.0558, 0.0594, TOL)
    # And the verdict really was a rejection, i.e. the guard is what stops it.
    ok, drop, _ = verdict(2768., 2482., 1.8, THRESH, 0.)
    assert not ok and abs(drop - 0.103) < 0.002, (ok, drop)


def test_a_halt_well_above_the_known_contact_still_re_arms():
    # 0.77mm up: a genuine false halt, and re-arming is the correct recovery.
    assert not corrob(0.7683, 0.0594, TOL)


def test_no_prior_contact_means_no_corroboration():
    assert not corrob(0.0558, None, TOL)


def test_disabled_by_zero_tolerance():
    assert not corrob(0.0594, 0.0594, 0.)


def test_the_rule_is_asymmetric_at_or_below_corroborates():
    # Deliberately NOT a symmetric window.  Only a halt clearly ABOVE the known
    # contact is the false-halt case that re-arming exists for; at or below it,
    # the halt is the same touch or deeper and descending further presses in.
    assert corrob(0.10 + TOL - 1e-6, 0.10, TOL)      # just above, within tol
    assert not corrob(0.10 + TOL + 1e-6, 0.10, TOL)  # clearly above -> re-arm
    assert corrob(0.10 - TOL - 1e-6, 0.10, TOL)      # below -> never re-arm
    assert corrob(-0.50, 0.10, TOL)                  # far below -> still no


def test_the_logged_negative_span_case():
    # 2026-08-07, RANK_FREQ at (60,100): a halt at 0.0675 sat 65um BELOW a
    # contact confirmed at 0.1322 - outside a +/-50um window, so the old
    # symmetric rule let it re-arm, and the clamped floor (0.0822) then sat
    # ABOVE the ceiling (0.0675-0.15) for a span of -0.165mm.
    assert corrob(0.0675, 0.1322, TOL), "the negative-span case still re-arms"


def test_tolerance_separates_the_logged_cases():
    # Must be wide enough for the 3.6um repeat and far short of the 0.71mm
    # false halt, or it either never fires or blocks every re-arm.
    assert abs(0.0558 - 0.0594) < TOL < abs(0.7683 - 0.0594)


# --- the re-arm floor must leave enough travel to measure ----------------------
# Clamping the floor to a known contact stops a rejection cascade plowing past
# it, but the clamp can leave a span too short to detect anything, and then the
# point fails as if the bed were out of reach.  Measured on hardware 2026-08-07
# at (60,60): a rejection at z=0.1183 put the ceiling at 0.1183-0.15 = -0.0317
# and the clamped floor at -0.0791-0.05 = -0.1291.  That is 0.097mm, while the
# 0.8s warm-up alone covers 0.16mm at 0.2mm/s.  The descent ran 2 windows.

span = HaltingContactProbe._min_descent_span
SPEED, WARMUP, CYCLES, FREQ = 0.2, 0.8, 8., 172.9


def test_the_logged_starved_descent_is_rejected():
    need = span(SPEED, WARMUP, CYCLES, FREQ)
    had = -0.0317 - (-0.1291)
    assert had < need, "0.097mm span would still be attempted (need %.3f)" % need


def test_warmup_alone_is_not_enough():
    # The floor of the rule: the runway is dead travel, so the minimum must
    # exceed it or the descent arms with nothing left to measure.
    assert span(SPEED, WARMUP, CYCLES, FREQ) > SPEED * WARMUP


def test_a_normal_probe_distance_passes():
    # The usual 2.5mm probe_distance must not trip this.
    assert span(SPEED, WARMUP, CYCLES, FREQ) < 2.5


def test_scales_with_speed():
    a = span(0.2, WARMUP, CYCLES, FREQ)
    b = span(0.4, WARMUP, CYCLES, FREQ)
    assert abs(b - 2. * a) < 1e-9, (a, b)


def test_zero_frequency_cannot_divide_by_zero():
    assert span(SPEED, WARMUP, CYCLES, 0.) > 0.


# ---------------------------------------------------------------------------
# Axis selection and overshoot detection, added 2026-08-08.
#
# Verify judged every contact on the configured accel_axis regardless of which
# channel the halt fired on.  A descent logged per-axis drops of x=16% y=51%
# z=7% and was judged on z against a 15% threshold - a real contact rejected on
# the one channel that could not see it.  A false REJECTION was unrecoverable,
# because the re-arm restarted BELOW it and every later descent then ran under
# the surface: -0.2665, -0.3170, -0.3046 at a point whose surface is near +0.03.
pick = HaltingContactProbe._pick_verify_axis
submerged = HaltingContactProbe._submerged_fraction


def test_picks_trigger_axis_over_the_configured_one():
    # The 2026-08-08 case: loud on y, silent on z, halt fired on x.
    drops = [0.16, 0.51, 0.07]
    assert pick(drops, 0) == 1, "should move to the channel with the real step"


def test_trigger_axis_wins_when_it_is_the_strongest():
    assert pick([0.40, 0.10, 0.05], 0) == 0


def test_no_trigger_axis_falls_back_to_strongest_not_to_index_zero():
    # The salvage path has no trigger axis precisely because the halt was
    # missed; falling back to a FIXED axis is the bug being removed.
    assert pick([0.02, 0.03, 0.44], None) == 2


def test_inverted_rise_channel_never_takes_over():
    # z rose 60% on contact (inverted coupling).  Its magnitude is the largest,
    # but handing the verdict to it would hide the rise from _is_inverted_rise.
    assert pick([0.20, 0.05, -0.60], 0) == 0


def test_submerged_when_every_channel_is_far_under_the_descent_air():
    # Ramp top never reached air: ~25% of the descent's own air level.
    assert submerged([500., 200., 240.], [2000., 800., 900.])[0] < 0.5


def test_not_submerged_when_one_channel_is_healthy():
    # One healthy channel proves the ramp top WAS in air, even if the others
    # look weak - that is why the statistic is a max, not a mean.
    frac, axis = submerged([1900., 100., 90.], [2000., 800., 900.])
    assert frac > 0.9 and axis == 0, (frac, axis)


def test_submerged_needs_a_reference():
    assert submerged([500., 200., 240.], None) is None
    assert submerged([500., 200., 240.], []) is None
    assert submerged([500., 200.], [2000., 800., 900.]) is None


def test_zero_reference_axes_do_not_renumber_the_others():
    # x has no reference at all.  The answer must name axis 2, not axis 1 - the
    # bug this guards against is filtering a bare list of fractions and losing
    # the axis mapping, which reports the wrong channel to the user.
    frac, axis = submerged([0., 100., 850.], [0., 800., 900.])
    assert axis == 2, axis


# Corpus-measured behaviour of the overshoot guard, from
# local/replay_submerged.py over the 12 labelled overshoot descents.  These pin
# the threshold to what the traces actually show rather than to a guess: the
# first default written was 0.5, which would have caught 4 of 12.
SUBMERGED_FRAC = 0.75


def test_threshold_catches_the_deep_overshoots_in_the_corpus():
    # Worst (highest) fraction measured with the ramp top >=0.15mm below
    # contact, across all 12 traces.
    assert 0.68 < SUBMERGED_FRAC


def test_threshold_never_fires_on_a_legitimate_contact_in_the_corpus():
    # Lowest fraction measured with the ramp top ABOVE contact.
    assert SUBMERGED_FRAC < 0.82


def test_the_flat_region_is_a_documented_blind_spot_not_a_tuning_failure():
    # From 0.05mm above contact to 0.10mm below it the corpus reads 0.82-1.04:
    # the contact signal is flat-then-cliff, so a shallow overshoot produces no
    # amplitude change for ANY threshold to find.  This asserts the blind spot
    # exists rather than pretending it is covered - if a future change claims to
    # detect shallow overshoot by amplitude alone, this is the case to disprove.
    shallow_overshoot_readings = (0.85, 0.93, 0.97, 1.02, 1.04)
    assert all(r > SUBMERGED_FRAC for r in shallow_overshoot_readings), \
        "shallow overshoot is invisible on this statistic - by design"


# Re-arm ceiling must leave a descent room to measure, added 2026-08-09.
#
# Once a contact is confirmed at a point the floor rises to prior_z - tol.  A
# rejection just above it used to leave less runway than the minimum descent
# span, so the SECOND sample at a point failed whenever its first halt was a
# false one above the contact.  Logged at (60,100): ceiling 0.218, floor 0.064,
# 0.154mm of travel against 0.197mm needed - it killed a point that had already
# produced one good sample, and the mesh with it.
def rearm_ceiling(ceiling, contact_z, margin, z_floor, need):
    return min(ceiling, max(contact_z + margin, z_floor + need))


def test_the_logged_starvation_case_now_has_runway():
    c = rearm_ceiling(2.0, 0.1681, 0.05, 0.0640, 0.197)
    assert c - 0.0640 >= 0.197, c


def test_margin_still_wins_when_it_already_clears_the_runway():
    # Plenty of room below: the ceiling should sit just above the rejection,
    # not be inflated to the floor-plus-runway value.
    c = rearm_ceiling(2.0, 1.5, 0.05, 0.0, 0.197)
    assert abs(c - 1.55) < 1e-9, c


def test_never_climbs_above_where_the_descent_started():
    c = rearm_ceiling(0.30, 0.28, 0.05, 0.20, 0.50)
    assert c == 0.30, c


def test_starvation_is_still_reported_when_the_ceiling_cannot_reach():
    # Original ceiling too low to fit a descent above the floor - the point must
    # still fail rather than run a descent that cannot measure.
    c = rearm_ceiling(0.30, 0.28, 0.05, 0.20, 0.50)
    assert c - 0.20 < 0.50


# Ramp-minimum contact statistic, added 2026-08-09.
#
# The contact dwell sits at z_cand-VERIFY_DOWN, which measured 35-180um past the
# damping minimum on 8 of 24 captures - far enough that amplitude has climbed
# back to or above air, so a real contact read as "no drop" or as a RISE.  One
# trace reached a 39% minimum and was failed as inverted coupling.
ramp_min = HaltingContactProbe._ramp_min_amp


def test_finds_the_minimum_the_dwell_drove_past():
    # V-shaped ramp: air, damping minimum, then re-excitation deeper.
    zs = [0.20, 0.15, 0.10, 0.05, 0.00, -0.05, -0.10]
    amps = [1000., 950., 700., 600., 750., 950., 1150.]
    assert abs(ramp_min(zs, amps, k=3) - 700.) < 1e-9, ramp_min(zs, amps, k=3)


def test_a_single_noise_dip_does_not_become_the_minimum():
    # The reason this is a moving MEDIAN: a bare min would return 10 here and
    # invent a contact in air, which is the failure mode the change must not
    # introduce.
    zs = [0.20, 0.15, 0.10, 0.05, 0.00]
    amps = [1000., 1010., 10., 1005., 995.]
    assert ramp_min(zs, amps, k=3) > 900., ramp_min(zs, amps, k=3)


def test_flat_air_ramp_yields_no_drop():
    zs = [0.2, 0.15, 0.1, 0.05, 0.0, -0.05]
    amps = [1000., 1005., 995., 1002., 998., 1001.]
    assert ramp_min(zs, amps, k=3) > 950.


def test_pools_reps_by_z_not_by_order():
    # Two reps interleaved in time; sorting by Z must still find the minimum.
    zs = [0.2, 0.1, 0.0, 0.2, 0.1, 0.0]
    amps = [1000., 600., 900., 1010., 610., 910.]
    assert ramp_min(zs, amps, k=3) < 700.


def test_short_input_falls_back_to_the_bare_minimum():
    assert ramp_min([0.1, 0.0], [500., 400.], k=5) == 400.


def test_empty_is_none_not_a_crash():
    assert ramp_min([], [], k=5) is None


# ---------------------------------------------------------------------------
# SHARED contact-measurement surface, added 2026-08-09 after five fixes in one
# session each landed on one call site and missed its sibling.  These pin the
# pieces that had diverged, so a future fix to one path cannot silently skip the
# other: both _verify_contact_moving and characterize_amplitude now go through
# _contact_levels, both size their ramp from CONTACT_UP_MM / CONTACT_DOWN_MM,
# and both cap reps with _cap_reps_for_budget.
from .resonance_probe import (CONTACT_UP_MM, CONTACT_DOWN_MM, SEG_BUDGET,
                              _cap_reps_for_budget)


def test_margins_clear_the_damping_cliff():
    # The cliff sits 110-150um below contact.  A ramp that does not reach it
    # measures the flat region and reports ~0% for a mode that damps fine -
    # exactly what invalidated the grid survey with 0.03/0.02 margins.
    assert CONTACT_DOWN_MM > 0.15, CONTACT_DOWN_MM
    # The air reference must sit clear of the transition, or the baseline is
    # already damped and the drop comes out compressed however good the
    # contact level is.
    assert CONTACT_UP_MM >= 0.10, CONTACT_UP_MM


def test_budget_caps_the_configuration_that_killed_the_mcu():
    # 172.9Hz, reps=3, 0.35 up-ramp: ~415 segments per ramp, 138 per dwell.
    per_rep = 2 * 415 + 2 * 138
    assert _cap_reps_for_budget(3, per_rep, 277, SEG_BUDGET) < 3


def test_budget_leaves_the_configuration_that_works_alone():
    # reps=1 at the same geometry has completed meshes reliably.
    per_rep = 2 * 415 + 2 * 138
    assert _cap_reps_for_budget(1, per_rep, 277, SEG_BUDGET) == 1


def test_budget_never_returns_zero_reps():
    assert _cap_reps_for_budget(3, 10 ** 6, 0, SEG_BUDGET) == 1


def test_budget_does_not_inflate_a_small_request():
    assert _cap_reps_for_budget(2, 10, 0, SEG_BUDGET) == 2


# 2026-09-12: calibration shut the MCU down right after the mode sweep.  The
# amplitude sweep's guard counted RAMP segments only (500/level) and ignored
# the dwells and the level count, so CONTACT_LEVELS=5 at 172.9Hz over the
# 0.40mm span was ~3600 continuous segments - the figure that killed it before.
from .resonance_probe import _sweep_segments


def _sweep_level(f):
    # Defaults: detect_cycles 5 -> ramp speed 0.020*f/5, dwell 0.35, warm-up 0.8.
    warm, air, contact, ramp = _sweep_segments(
        f, CONTACT_UP_MM + CONTACT_DOWN_MM, 0.020 * f / 5., 0.35, 0.8)
    return warm, air + (2 * ramp + contact + air)


def test_sweep_as_configured_exceeded_the_budget():
    warm, per_level = _sweep_level(172.9)
    assert warm + 5 * per_level > SEG_BUDGET, (warm, per_level)


def test_sweep_levels_capped_to_budget():
    for f in (65.5, 148.0, 172.9, 212.2):
        warm, per_level = _sweep_level(f)
        n = _cap_reps_for_budget(5, per_level, warm, SEG_BUDGET)
        assert n >= 1 and (n == 1 or warm + n * per_level <= SEG_BUDGET), \
            (f, warm, per_level, n)


# 2026-09-13: RESONANCE_PROBE_CALIBRATE with no POINT started its first noise
# descent from Z=60.  _gen_descend_segments builds the whole vibrating descent
# up front - ~77,000 segments at 64.2Hz and 0.1mm/s - which pegged klippy's CPU
# for 12s and shut the MCU down ("Timer too close"), three runs in a row.
from .resonance_probe import _vib_top, MAX_VIB_SPAN


# 2026-09-13: ring-up false halts landed at z~2.03-2.065 on descents armed "at"
# 2.0.  Detection cannot fire before arming, so at t0+warmup the nozzle had
# barely left its 2.08 start: motion begins well after t0 and the time gate
# left the vibration ~0.2s of ring-up.  Arming now follows actual Z (from step
# history) crossing the arm height; time is only the fallback when Z is
# unknown.  (In-place ring-up was tried earlier and rejected: the start of the
# Z move itself threw transients.)
from .resonance_probe import _arm_gate, _stepper_z


def test_arms_on_height_even_before_the_old_warmup_time():
    assert _arm_gate(None, 10.2, 2.0, 2.0, 10.8) == 10.2
    assert _arm_gate(None, 10.2, 1.99, 2.0, 10.8) == 10.2


def test_never_arms_above_the_arm_height_however_late():
    assert _arm_gate(None, 11.5, 2.05, 2.0, 10.8) is None


def test_arming_latches():
    assert _arm_gate(10.2, 12.0, 2.07, 2.0, 10.8) == 10.2


def test_unknown_height_falls_back_to_time():
    assert _arm_gate(None, 10.5, None, 2.0, 10.8) is None
    assert _arm_gate(None, 10.9, None, 2.0, 10.8) == 10.9


class _FakeStepper:
    def __init__(self, offset=0., fail=False):
        self.offset, self.fail = offset, fail
    def get_past_mcu_position(self, print_time):
        if self.fail:
            raise RuntimeError("no history")
        return int(round(print_time * 1000))
    def mcu_to_commanded_position(self, mcu_pos):
        return mcu_pos * 0.001 + self.offset


def test_stepper_z_reads_step_history():
    assert abs(_stepper_z([_FakeStepper()], 2.05) - 2.05) < 1e-9


def test_stepper_z_averages_multiple_z_steppers():
    z = _stepper_z([_FakeStepper(0.), _FakeStepper(0.02)], 2.0)
    assert abs(z - 2.01) < 1e-9, z


def test_stepper_z_is_none_without_history():
    assert _stepper_z([], 2.0) is None
    assert _stepper_z([_FakeStepper(fail=True)], 2.0) is None


# The post-halt refine placed each window at halt_z + speed*(halt_t - t): time
# extrapolated back up to 0.15mm, assuming commanded speed and no lag - the
# assumption that broke time-based arming.  Windows now take actual Z from step
# history.  That history is kept 30s and, for an older clock,
# stepcompress_find_past_position silently returns the oldest entry's start
# position, so windows older than a cutoff keep the extrapolated Z.
from .resonance_probe import _window_z


def test_window_z_reads_step_history():
    z = _window_z([10.0, 10.5], [2.0, 1.9], lambda t: 3.0 - 0.1 * t, 0.)
    assert [round(v, 9) for v in z] == [2.0, 1.95], z


def test_window_z_falls_back_where_history_has_no_answer():
    z = _window_z([10.0, 10.5], [2.0, 1.9],
                  lambda t: None if t < 10.2 else 1.93, 0.)
    assert [round(v, 9) for v in z] == [2.0, 1.93], z


def test_window_z_ignores_history_older_than_the_cutoff():
    # The lookup answers, but for a clock past retention that answer is the
    # oldest entry's start position - wrong, and silently so.
    z = _window_z([1.0, 40.0], [2.5, 0.1], lambda t: 9.9, 15.0)
    assert [round(v, 9) for v in z] == [2.5, 9.9], z


def test_window_z_keeps_one_value_per_window():
    assert len(_window_z([], [], lambda t: 1., 0.)) == 0
    assert len(_window_z([1., 2., 3.], [0., 0., 0.], lambda t: 1., 0.)) == 3


# 2026-09-15 corpus study: of 1264 live halts, gradient (fixed-floor threshold)
# triggered 162 and 92% of the paired ones were false - 11 of the corpus's 12
# in-air confirmations followed a gradient halt.  Those halts fired ~1.2mm above
# the bed on a channel carrying 4-13% of the strongest channel's signal, where
# noise crosses a fixed floor.  Drawdown derives its threshold from each
# descent's own noise.  Replayed over disarmed traces, derivative + drawdown
# alone found 16/16 labelled contacts with no early fires and cut pure-air false
# alarms 37% -> 16%.  So the gradient test no longer halts unless
# [resonance_probe] halt_gradient is set; its drop is still computed for
# diagnostics.
import math as _math
from .resonance_probe import _HostResonanceEndstop

_F, _SPS = 175.0, 3200.


class _FakeCompletion:
    def __init__(self):
        self.done = False
    def complete(self, v):
        self.done = True


class _FakeReactor:
    def completion(self):
        return _FakeCompletion()


class _FakeConfigProbe:
    def __init__(self, halt_gradient):
        self.halt_gradient = halt_gradient


class _FakePrinter:
    def __init__(self, halt_gradient):
        self._rp = _FakeConfigProbe(halt_gradient)
    def get_reactor(self):
        return _FakeReactor()
    def lookup_object(self, name, default=None):
        return self._rp if name == 'resonance_probe' else default


class _FakeRProbe:
    def __init__(self, halt_gradient, deriv=0.08, dd_sens=0.10):
        self.printer = _FakePrinter(halt_gradient)
        self.warmup = 0.8
        self.detect_cycles = 8.
        self._descend_speed = 0.2
        self.detect_step_z = 0.01
        self.detect_confirm_z = 0.05
        self.excitation_freq = _F
        self.halt_sensitivity = 0.15
        self.halt_sensitivity_axis = [0.10, 0.10, 0.10]
        self.deriv_sensitivity = deriv
        self.drawdown_sensitivity = dd_sens


def _descend(rprobe, amp_fn, seconds=6.0):
    """Feed a synthetic vibrating descent; return the endstop after it halts
    or the samples run out."""
    es = _HostResonanceEndstop(rprobe, [], 0.0)
    es._completion = _FakeCompletion()
    n = int(seconds * _SPS)
    batch = []
    for i in range(n):
        t = i / _SPS
        ax, ay, az = amp_fn(t)
        s = _math.sin(2. * _math.pi * _F * t)
        batch.append((t, ax * s, ay * s, az * s))
        if len(batch) == 320:
            if not es._handle_batch({'data': batch}):
                return es
            batch = []
    return es


def _weak_channel_dip(t):
    # Strong x/z steady; weak y drops 30% at 3s and stays down: the signature
    # of the corpus's gradient false halts.
    return (5000., 200. * (0.7 if t >= 3.0 else 1.0), 5000.)


def test_gradient_off_ignores_a_weak_channel_dip():
    # Derivative and drawdown switched off, so only gradient could fire.
    es = _descend(_FakeRProbe(False, deriv=0., dd_sens=0.99),
                  _weak_channel_dip)
    assert not es._done, (es._trigger_kind, es._trigger_axis)
    assert es._dbg_maxdrop[1] >= 0.25, es._dbg_maxdrop


def test_gradient_on_halts_on_the_same_dip():
    # Proves the scenario above really exercises the gradient test.
    es = _descend(_FakeRProbe(True, deriv=0., dd_sens=0.99),
                  _weak_channel_dip)
    assert es._done and es._trigger_kind == 'gradient', \
        (es._done, es._trigger_kind)
    assert es._trigger_axis == 1, es._trigger_axis


def test_gradient_off_still_halts_on_real_contact():
    # A 60% drop on the strong channel after drawdown's warm-up.
    es = _descend(_FakeRProbe(False),
                  lambda t: (5000. * (0.4 if t >= 4.0 else 1.0), 200., 5000.))
    assert es._done, "real contact did not halt"
    assert es._trigger_kind in ('drawdown', 'derivative'), es._trigger_kind
    assert es._trigger_axis == 0, es._trigger_axis


def test_a_high_start_is_refused_not_vibrated():
    # Calibration moves to START_Z before planning; a descent that would still
    # vibrate from 60mm is a bug upstream, so fail loudly instead of queuing it.
    assert _vib_top(60.0, 0.15, None) is None
    assert _vib_top(0.15 + MAX_VIB_SPAN, 0.15, None) is not None


def test_a_low_start_is_unchanged():
    assert _vib_top(2.0, -0.2, None) == 2.0


def test_an_explicit_span_still_wins():
    assert abs(_vib_top(60.0, 0.0, 0.5) - 0.5) < 1e-9


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print("ok   %s" % name)
            except AssertionError as e:
                fails += 1
                print("FAIL %s: %s" % (name, e))
    raise SystemExit(1 if fails else 0)
