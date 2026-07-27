#!/usr/bin/env python3
# Self-check for analog_contact.  Run from klippy/:
#     python -m extras.test_analog_contact
from .analog_contact import (ContactDetector, FindThenRefine, estimate_noise,
                             DRAWDOWN, RUNUP, FLATTEN)

SPEED, SPS = 0.2, 20.          # 0.010mm per sample, matching the live probe


def det(mode, thresh, **kw):
    return ContactDetector(mode, thresh, SPEED, SPS, **kw)


def feed(d, values):
    """Return the index of the sample that triggered, or None."""
    for i, v in enumerate(values):
        if d.update(v, position=i):
            return i
    return None


def test_drawdown_fires_on_a_drop_from_the_running_peak():
    d = det(DRAWDOWN, 100.)
    # Rising approach, then a 300-unit collapse over three samples.
    assert feed(d, [1000.] * 10 + [1100., 1200.]) is None
    d.reset()
    assert feed(d, [1000, 1100, 1200, 1100, 1000, 900]) is not None


def test_rising_baseline_never_triggers():
    # THE failure mode of a plateau/threshold detector: the approach ramp is
    # monotonically rising, and must not look like an event no matter how far
    # it travels.
    d = det(DRAWDOWN, 100.)
    assert feed(d, [1000. + 40. * i for i in range(60)]) is None


def test_drawdown_is_insensitive_to_sample_SPACING():
    # The property a per-sample difference does NOT have, and the reason this
    # replaces it: the same total drop split across more samples must still
    # trigger.  Measured failure of the old rule: oversampling shrank each
    # step toward the noise and detection got WORSE.
    #
    # Persistence is given in mm so the two cases demand the same PHYSICAL
    # event length; a sample count would silently ask more of the coarse case.
    # Both series include the post-contact tail a real touch produces: the
    # value stays down once damped, it does not bounce back.
    coarse = [1000., 1000., 1000., 700., 400.] + [380.] * 3
    fine = ([1000.] * 3 + [1000. - 60. * i for i in range(1, 11)]
            + [380.] * 18)
    d = ContactDetector(DRAWDOWN, 500., SPEED, SPS, persist_mm=0.005)
    assert feed(d, coarse) is not None
    # Same event, 6x finer sampling: same speed, higher rate.
    d = ContactDetector(DRAWDOWN, 500., SPEED, SPS * 6., persist_mm=0.005)
    assert feed(d, fine) is not None


def test_persist_mm_scales_with_sample_rate():
    # 0.02mm of persistence is 2 samples at 20sps/0.2mm/s, 12 at 120sps.
    d = ContactDetector(DRAWDOWN, 1., SPEED, SPS, persist_mm=0.02)
    assert d.persist == 2, d.persist
    d = ContactDetector(DRAWDOWN, 1., SPEED, SPS * 6., persist_mm=0.02)
    assert d.persist == 12, d.persist


def test_lookback_stops_slow_air_wander_from_accumulating():
    # A noisy signal that RISES overall but dips slowly.  Against a global
    # peak the dip keeps accumulating and eventually fires in mid-air - this
    # happened on 4 of 24 real descent traces, once ~0.5mm above contact.
    # A bounded lookback caps the accumulation while still tracking the ramp.
    wander = ([1000. + 8. * i for i in range(40)]        # rise to 1312
              + [1312. - 6. * i for i in range(40)])     # slow dip to 1078
    # Global peak: the dip reaches 18% below peak -> false fire.
    d = ContactDetector(DRAWDOWN, 0.15, SPEED, SPS, relative=True)
    assert feed(d, wander) is not None
    # Bounded lookback: the reference follows the signal down, so the slow
    # dip never accumulates that far.
    d = ContactDetector(DRAWDOWN, 0.15, SPEED, SPS, relative=True,
                        lookback_mm=0.10)
    assert feed(d, wander) is None
    # ...and a real step still fires with the lookback in place.
    d = ContactDetector(DRAWDOWN, 0.15, SPEED, SPS, relative=True,
                        lookback_mm=0.10)
    assert feed(d, wander + [700.] * 5) is not None


def test_lookback_window_expires_by_distance_not_count():
    d = ContactDetector(DRAWDOWN, 1., SPEED, SPS, lookback_mm=0.05)
    assert d.lookback == 5, d.lookback          # 0.05mm / 0.010mm per sample
    d = ContactDetector(DRAWDOWN, 1., SPEED, SPS * 4., lookback_mm=0.05)
    assert d.lookback == 20, d.lookback


def test_relative_threshold_tracks_signal_scale():
    # Same 30% drop at two drive levels must behave identically.
    for scale in (1., 37.):
        d = det(DRAWDOWN, 0.25, relative=True)
        vals = [v * scale for v in (1000, 1000, 1000, 800, 650, 600)]
        assert feed(d, vals) is not None, scale
    # ...and a 10% drop must not trigger a 25% threshold at either scale.
    for scale in (1., 37.):
        d = det(DRAWDOWN, 0.25, relative=True)
        vals = [v * scale for v in (1000, 1000, 1000, 950, 900, 900)]
        assert feed(d, vals) is None, scale


def test_persistence_rejects_a_single_outlier():
    d = det(DRAWDOWN, 100., persist=2)
    assert feed(d, [1000, 1000, 500, 1000, 1000]) is None
    d = det(DRAWDOWN, 100., persist=2)
    assert feed(d, [1000, 1000, 500, 400, 1000]) is not None


def test_runup_is_the_mirror_image():
    d = det(RUNUP, 100.)
    assert feed(d, [1000, 900, 800, 900, 1000, 1100]) is not None
    d = det(RUNUP, 100.)
    assert feed(d, [1000. - 40. * i for i in range(60)]) is None


def test_flatten_needs_movement_before_stillness_counts():
    # Capacitive-style: value tracks the gap, stops changing at contact.
    # A sensor sitting still in free air must NOT read as contact.
    d = det(FLATTEN, 1.0, arm_slope_per_mm=50.)
    assert feed(d, [500.] * 30) is None
    # Now: closing the gap fast, then contact pins it.
    d = det(FLATTEN, 1.0, arm_slope_per_mm=50.)
    closing = [500. + 20. * i for i in range(20)]     # 2000/mm slope
    assert feed(d, closing + [900.] * 5) is not None


def test_nan_does_not_blind_the_detector():
    # A NaN peak compares False against everything, which would silently
    # disable detection for the rest of the descent.
    d = det(DRAWDOWN, 100.)
    assert feed(d, [1000, float('nan'), 1000, 1000, 700, 600]) is not None


def test_estimate_noise_is_robust_to_an_outlier():
    quiet = [1000. + (i % 2) for i in range(40)]
    assert estimate_noise(quiet) <= 1.5
    spiked = list(quiet)
    spiked[20] = 5000.
    assert estimate_noise(spiked) <= 1.5     # median, not std


def test_find_then_refine_rejects_a_distant_refine():
    ftr = FindThenRefine(lambda: -0.05, lambda found: -0.055,
                         max_overtravel=0.02)
    z, refined = ftr.run()
    assert refined and abs(z - (-0.055)) < 1e-9
    # A refine 0.5mm away is a disagreement, not a better measurement.
    ftr = FindThenRefine(lambda: -0.05, lambda found: -0.55,
                         max_overtravel=0.02)
    z, refined = ftr.run()
    assert not refined and abs(z - (-0.05)) < 1e-9


def test_find_then_refine_survives_a_failed_refine():
    ftr = FindThenRefine(lambda: -0.05, lambda found: None)
    z, refined = ftr.run()
    assert z == -0.05 and not refined
    ftr = FindThenRefine(lambda: None, lambda found: -0.05)
    assert ftr.run() == (None, False)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print("ok", name)
    print("all passed")
