#!/usr/bin/env python3
# Self-check for _ramp_edge: the UP ramp must be read as a rising edge, and it
# must land on the same Z as the DOWN ramp when the signal is symmetric.  Run
# from klippy/:
#     python -m extras.test_verify_ramp
#
# This exists because verify used to analyse the down ramp only, so an up/down
# bias (backlash, ring-down) could not be seen at all - a wrong answer that
# looked like a confident one.
import numpy as np
from .resonance_probe import HaltingContactProbe

AIR, CONTACT = 1000., 300.
EDGE = 0.0                       # true contact Z
MID = 0.5 * (AIR + CONTACT)

# _ramp_edge never touches self, so it can be exercised off the class directly
# rather than standing up a printer object.
edge_of = HaltingContactProbe._ramp_edge


# A little noise on purpose: the step SNR is measured against the spread of the
# flat part of the ramp, so a perfectly noiseless synthetic gives sd=0 and no
# SNR at all.  Real windows always jitter.
RNG = np.random.default_rng(7)


def _noisy(vals):
    return vals * (1. + RNG.normal(0., 0.004, len(vals)))


def down_ramp(edge=EDGE, step=0.01):
    """High Z -> low Z, amplitude falls through the edge."""
    zs = np.round(np.arange(0.10, -0.101, -step), 4)
    return zs, _noisy(np.where(zs > edge, AIR, CONTACT))


def up_ramp(edge=EDGE, step=0.01):
    """Low Z -> high Z, amplitude rises through the edge."""
    zs = np.round(np.arange(-0.10, 0.101, step), 4)
    return zs, _noisy(np.where(zs > edge, AIR, CONTACT))


def test_down_ramp_finds_the_edge():
    zs, amps = down_ramp()
    edge, snr = edge_of(None, zs, amps, True, AIR, CONTACT)
    assert abs(edge - EDGE) <= 0.011, edge
    assert snr > 0., snr


def test_up_ramp_finds_the_same_edge():
    zs, amps = up_ramp()
    edge, snr = edge_of(None, zs, amps, False, AIR, CONTACT)
    assert abs(edge - EDGE) <= 0.011, edge
    assert snr > 0., snr


def test_no_bias_when_the_signal_is_symmetric():
    d, _ = edge_of(None, *down_ramp(), True, AIR, CONTACT)
    u, _ = edge_of(None, *up_ramp(), False, AIR, CONTACT)
    assert abs(u - d) <= 0.011, (d, u)


def test_an_injected_offset_shows_up_as_bias():
    # The up ramp releases 40um higher than the down ramp engages - exactly the
    # signature backlash or ring-down would leave.
    d, _ = edge_of(None, *down_ramp(edge=0.0), True, AIR, CONTACT)
    u, _ = edge_of(None, *up_ramp(edge=0.04), False, AIR, CONTACT)
    assert 0.025 <= (u - d) <= 0.055, (d, u, u - d)


def test_the_estimate_does_not_depend_on_sample_ORDER():
    # The estimator sorts by Z, so it reads the same edge whatever order the
    # windows arrive in.  That is what makes the up/down split meaningful: the
    # two directions differ because their DATA differs (where the amplitude
    # actually changed), not because of arrival order.
    zs, amps = down_ramp()
    ref, _ = edge_of(None, zs, amps, True, AIR, CONTACT)
    perm = RNG.permutation(len(zs))
    shuffled, _ = edge_of(None, zs[perm], amps[perm], True, AIR, CONTACT)
    assert abs(shuffled - ref) < 1e-9, (ref, shuffled)


def test_flat_ramp_reports_nothing():
    zs = np.round(np.arange(0.10, -0.101, -0.01), 4)
    edge, snr = edge_of(None, zs, np.full(len(zs), AIR), True, AIR, CONTACT)
    assert edge is None, edge


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
