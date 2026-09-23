#!/usr/bin/env python3
# Self-check for the weak-excitation guard (_air_reject_reason).  Run from
# klippy/:
#     python -m extras.test_air_guard
#
# This exists because of a real corrupted bed mesh (2026-08-06).  At X=60,Y=100
# the mode's air response was ~40% down on every axis while air NOISE stayed at
# 1-2% - a WEAK spot, not a noisy one.  The descent missed its halt, salvage
# over-pressed 0.282mm, and the point landed 345um below its neighbours.
#
# The part that makes this worth a guard: the machine's own consistency check
# passed.  Two touches, from two different degraded paths, agreed to 23um -
# inside samples_tolerance - because a location error is SYSTEMATIC, so repeat
# sampling reproduces it instead of exposing it.  The numbers below are the
# measured ones from that run.
from .resonance_probe import HaltingContactProbe

reject = HaltingContactProbe._air_reject_reason

# z-axis air baselines of the healthy mesh points from that run.
HEALTHY = [4514., 4755., 5293., 5258., 5540., 5165., 5199., 5637., 6245.]
BAD = 3162.          # the weak point's baseline
MIN_FRAC = 0.7       # shipped default


def test_catches_the_real_bad_point():
    r = reject(BAD, HEALTHY, MIN_FRAC)
    assert r is not None, "the 3162 baseline that corrupted the mesh was accepted"
    assert "too weak" in r, r


def test_accepts_every_healthy_point_from_that_run():
    # No false rejection, judged against the same session it came from.
    for b in HEALTHY:
        assert reject(b, HEALTHY, MIN_FRAC) is None, \
            "healthy baseline %.0f rejected" % b


def test_unmeasurable_baseline_is_refused():
    # "no plateau above halt" leaves last_baseline None and returns the raw
    # halt, which is documented as biased deep - that is the other path the bad
    # point took, so None must not sail through.
    r = reject(None, HEALTHY, MIN_FRAC)
    assert r is not None and "no air plateau" in r, r


def test_silent_until_a_reference_exists():
    # Fewer than 3 accepted points cannot establish a reference; the guard must
    # not fire on guesswork at the start of a session.
    assert reject(BAD, [], MIN_FRAC) is None
    assert reject(BAD, [5000., 5200.], MIN_FRAC) is None
    assert reject(BAD, [5000., 5200., 5300.], MIN_FRAC) is not None


def test_disabled_by_zero():
    assert reject(BAD, HEALTHY, 0.) is None
    assert reject(None, HEALTHY, 0.) is None


def test_median_reference_resists_one_bad_point():
    # If a weak reading did get accepted, the reference must not sag toward it
    # and start waving through the next one.  A mean would; a median does not.
    polluted = HEALTHY + [BAD]
    assert reject(BAD, polluted, MIN_FRAC) is not None


def test_threshold_separates_the_two_populations():
    # The default must sit between the worst healthy point and the bad one, or
    # it is either useless or trips constantly.
    ref = sorted(HEALTHY)[len(HEALTHY) // 2]
    worst_healthy = min(HEALTHY) / ref
    bad_frac = BAD / ref
    assert bad_frac < MIN_FRAC < worst_healthy, (
        "default %.2f does not separate bad %.2f from worst healthy %.2f"
        % (MIN_FRAC, bad_frac, worst_healthy))


# --- the guard reads EVERY channel, not the triggering one ---------------------
# The air level used to be read off whichever axis triggered the live halt, and
# the axes sit at very different absolute levels: on 2026-08-07 x-triggered
# baselines ran a median 8758 against 3986 for z, a factor of 2.2.  A session
# that mixes trigger axes then compares unlike things - that policy refused 12 of
# 37 descents on 2026-08-07, nearly all of them healthy.
#
# Keeping a separate history PER AXIS fixes those false refusals but breaks the
# case this guard exists for.  Replaying the 2026-08-06 mesh that corrupted a
# point: the weak region supplied 3 of the first 4 z-triggered baselines, so the
# z reference normalised the weakness away and the bad descent passed at 0.90.
#
# What holds on both: judge each channel against its OWN history and refuse only
# when EVERY channel is weak.  On the corrupted mesh that refuses the bad descent
# at 0.65 of its strongest channel; on 2026-08-07 it refuses 1 of 37.
# Numbers below are the measured plateaus from those runs.
reject_axes = HaltingContactProbe._air_reject_reason_axes

# Healthy points of the 2026-08-06 mesh, as [x, y, z] air levels.
MESH_HEALTHY = [[8646., 2900., 5539.], [8752., 2900., 5154.],
                [8898., 2900., 5178.], [9801., 2900., 5612.],
                [9012., 2900., 5236.]]
MESH_BAD = [5775., 2000., 3134.]      # descent00424 - the corrupted point
HIST = {ax: [p[ax] for p in MESH_HEALTHY] for ax in (0, 1, 2)}


def test_catches_the_corrupted_mesh_point():
    r = reject_axes(MESH_BAD, HIST, MIN_FRAC)
    assert r is not None, "the descent that corrupted the mesh was accepted"
    assert "too weak" in r, r


def test_accepts_the_healthy_points_of_that_mesh():
    for p in MESH_HEALTHY:
        assert reject_axes(p, HIST, MIN_FRAC) is None, p


def test_one_strong_channel_is_enough_to_pass():
    # The false-refusal case: a descent whose z is low but whose x is normal is
    # a trigger-axis artefact, not weak excitation.
    assert reject_axes([8700., 2900., 3200.], HIST, MIN_FRAC) is None


def test_weak_on_every_channel_is_refused():
    assert reject_axes([4000., 1200., 2400.], HIST, MIN_FRAC) is not None


def test_silent_until_every_axis_has_a_reference():
    thin = {0: [8646., 8752., 8898.], 1: [2900.], 2: [5539.]}
    assert reject_axes(MESH_BAD, thin, MIN_FRAC) is None


def test_missing_air_measurement_is_refused_not_assumed_good():
    assert reject_axes(None, HIST, MIN_FRAC) is not None


def test_disabled_by_zero_fraction():
    assert reject_axes(MESH_BAD, HIST, 0.) is None


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
