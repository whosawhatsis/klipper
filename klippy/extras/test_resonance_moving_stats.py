#!/usr/bin/env python3
# Self-check for _moving_stats: the down-ramp must be split by HEIGHT, not
# summarised whole.  Run from klippy/:
#     python -m extras.test_resonance_moving_stats
import numpy as np
from .resonance_probe import _moving_stats, LIVE_WIN_Z

CONTACT_Z, UP = 0.0, 0.10
# A ramp with the real defaults: 0.10mm of air above contact, 0.04mm below.
ZS = np.round(np.arange(0.10, -0.041, -0.01), 4)


def ramp(air_amp, contact_amp):
    return np.where(ZS > CONTACT_Z, air_amp, contact_amp)


def test_split_recovers_the_true_drop():
    # 1000 in air, 300 in contact = a real 70% drop.
    drop, noise = _moving_stats(ramp(1000., 300.), ZS, CONTACT_Z, UP)
    assert abs(drop - 0.70) < 1e-6, drop
    assert noise < 1e-9, noise
    # The whole-ramp median this replaces lands in AIR and sees almost nothing,
    # which is the bug: it reported ~0-6% on a genuine 70% drop.
    naive = 1. - float(np.median(ramp(1000., 300.))) / 1000.
    assert naive < 0.05, naive


def test_noise_comes_from_the_moving_air_window():
    amps = ramp(1000., 300.).astype(float)
    amps[ZS > CONTACT_Z] += np.array([50., -50.] * 10)[:(ZS > CONTACT_Z).sum()]
    drop, noise = _moving_stats(amps, ZS, CONTACT_Z, UP)
    assert 0.04 < noise < 0.06, noise      # ~5% scatter on a 1000 baseline
    assert abs(drop - 0.70) < 0.02, drop   # drop survives the added noise


def test_no_contact_reads_as_no_drop():
    drop, _ = _moving_stats(ramp(1000., 1000.), ZS, CONTACT_Z, UP)
    assert drop == 0., drop


def test_too_few_windows_is_not_a_false_signal():
    # A ramp entirely in air must NOT report a drop just because the contact
    # population is empty - that would arm a floor against nothing.
    zs = np.array([0.10, 0.09, 0.08, 0.07])
    assert _moving_stats(np.array([1000.] * 4), zs, CONTACT_Z, UP) == (0., 1.)


def test_transition_band_is_excluded():
    # Windows between contact_z and contact_z+up/2 belong to neither
    # population; a mid-ramp value must not drag the air baseline down.
    amps = ramp(1000., 300.).astype(float)
    amps[(ZS > CONTACT_Z) & (ZS < CONTACT_Z + 0.5 * UP)] = 600.
    drop, _ = _moving_stats(amps, ZS, CONTACT_Z, UP)
    assert abs(drop - 0.70) < 1e-6, drop


def test_depth_dependent_damping_is_discounted():
    # The 148Hz/z case: damping that grows with press depth rather than
    # appearing at first contact.  The live detector only ever sees the first
    # LIVE_WIN_Z, so the metric must report the shallow part, not the deep part.
    depth = np.clip(CONTACT_Z - ZS, 0., None)
    amps = 1000. * (1. - 20. * depth)        # -80% by 40um of press
    drop, _ = _moving_stats(amps, ZS, CONTACT_Z, UP)
    whole_below = 1. - float(np.median(amps[ZS <= CONTACT_Z])) / 1000.
    # Both see damping, but the onset band must rate it substantially lower -
    # that gap is the entire point of the change.
    assert drop < whole_below - 0.15, (drop, whole_below)
    assert 0.1 < drop < 0.3, drop


def test_damping_at_the_surface_still_reads_high():
    # The counterpart: damping present from first contact must survive the
    # narrower band, or the change would just suppress everything.
    amps = np.where(ZS > CONTACT_Z, 1000., 300.)
    drop, _ = _moving_stats(amps, ZS, CONTACT_Z, UP)
    assert abs(drop - 0.70) < 1e-6, drop


def test_onset_band_is_respected_exactly():
    # A window sitting just outside the band must not count toward contact.
    zs = np.array([0.10, 0.09, 0.08, -0.001, -(LIVE_WIN_Z + 0.005)])
    amps = np.array([1000., 1000., 1000., 1000., 10.])
    drop, _ = _moving_stats(amps, zs, CONTACT_Z, UP)
    assert drop == 0., drop     # only the in-band window counts, and it is air


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print("ok", name)
    print("all passed")
