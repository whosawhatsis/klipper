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
import extras.resonance_probe as rp
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


# Measured ramps, PROBE_ACCURACY at (80,85), 175.3 Hz, 2026-09-14 (verify
# captures 01392/01396/01400), median amplitude in 25um Z bins.  Past the
# damping minimum, pressing harder RE-EXCITES the structure back toward the air
# level, so a ramp that reaches deep crosses the air/contact midpoint more than
# once.  The up ramp starts at that deep end: taking the FIRST upward crossing
# read the re-excitation climb as the release, -58um and -110um where the
# release was at +103um / +111um, and dragged the reported contact ~80um low on
# five probes in a row.  The edge is the crossing nearest AIR, in both
# directions.
S1_AIR, S1_CON = 2150.459, 1332.060      # sample 1: first crossing was right
S1_UP = ([-0.125, -0.100, -0.075, -0.050, -0.025, 0.000, 0.025, 0.050, 0.075,
          0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275],
         [1874, 1884, 1931, 2009, 2148, 2022, 1824, 1516, 1330,
          1937, 2385, 2355, 2297, 2190, 2181, 2186, 2106])
S5_AIR, S5_CON = 2264.297, 1310.087      # sample 5: first crossing -0.058
S5_UP = ([-0.125, -0.100, -0.075, -0.050, -0.025, 0.000, 0.025, 0.050, 0.075,
          0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275],
         [1593, 1615, 1736, 1812, 1966, 2044, 1696, 1375, 1272,
          1709, 2307, 2378, 2313, 2270, 2235, 2326, 2246])
S5_DOWN = ([0.250, 0.225, 0.200, 0.175, 0.150, 0.125, 0.100, 0.075, 0.050,
            0.025, 0.000, -0.025, -0.050, -0.075, -0.100, -0.125],
           [2366, 2367, 2354, 2344, 2329, 2327, 2307, 2199, 1344,
            1407, 1750, 2032, 2092, 1923, 1772, 1722])
S9_AIR, S9_CON = 2583.892, 1653.291      # sample 9: first crossing -0.110
S9_UP = ([-0.125, -0.100, -0.075, -0.050, -0.025, 0.000, 0.025, 0.050, 0.075,
          0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275],
         [2039, 2150, 2282, 2428, 2603, 2607, 2313, 1932, 1567,
          1689, 2645, 2696, 2562, 2606, 2598, 2591, 2612])


def _real(ramp):
    return np.asarray(ramp[0], dtype=np.float64), \
        np.asarray(ramp[1], dtype=np.float64)


def test_up_edge_ignores_the_deep_re_excitation_climb():
    edge, _ = edge_of(None, *_real(S5_UP), False, S5_AIR, S5_CON)
    assert 0.090 <= edge <= 0.115, edge


def test_up_edge_ignores_it_on_the_worst_capture_too():
    edge, _ = edge_of(None, *_real(S9_UP), False, S9_AIR, S9_CON)
    assert 0.100 <= edge <= 0.120, edge


def test_up_edge_unchanged_where_first_crossing_was_already_right():
    edge, _ = edge_of(None, *_real(S1_UP), False, S1_AIR, S1_CON)
    assert 0.080 <= edge <= 0.100, edge


def test_down_edge_keeps_the_crossing_nearest_air():
    # The same deep re-excitation makes the DOWN ramp re-cross too (~-0.098);
    # the down edge must stay the first crossing coming from air (~+0.063).
    edge, _ = edge_of(None, *_real(S5_DOWN), True, S5_AIR, S5_CON)
    assert 0.055 <= edge <= 0.070, edge


def test_flat_ramp_reports_nothing():
    zs = np.round(np.arange(0.10, -0.101, -0.01), 4)
    edge, snr = edge_of(None, zs, np.full(len(zs), AIR), True, AIR, CONTACT)
    assert edge is None, edge



# --- verify CSV carries every axis -------------------------------------------
# The refinement pass (change-point on a disarmed press) needs the JOINT
# departure across axes - a single channel cannot distinguish contact from an
# air excursion, and at contact z can fall while y rises.  _window_amps_tagged
# already computes all three; the writer used to discard two of them.
# The per-axis columns are APPENDED, not substituted: local/replay_pressdepth,
# replay_rampmin and verify_strategy_eval all read 'amp' from these files.

def test_verify_rows_keep_legacy_columns_first():
    hdr, rows = rp._verify_rows([0.1], [[1.0], [2.0], [3.0]], ['down'], [0], 1)
    assert hdr.split(',')[:4] == ['z', 'amp', 'tag', 'rep'], hdr
    assert rows[0].split(',')[:4] == ['0.10000', '2.000', 'down', '0'], rows[0]


def test_verify_rows_append_all_three_axes():
    hdr, rows = rp._verify_rows([0.1], [[1.0], [2.0], [3.0]], ['down'], [0], 1)
    assert hdr.split(',')[4:] == ['amp_x', 'amp_y', 'amp_z'], hdr
    assert rows[0].split(',')[4:] == ['1.000', '2.000', '3.000'], rows[0]


def test_verify_rows_legacy_amp_follows_output_axis():
    # out_idx picks which channel the legacy 'amp' column mirrors.
    for out_idx, want in ((0, '1.000'), (1, '2.000'), (2, '3.000')):
        _, rows = rp._verify_rows([0.1], [[1.0], [2.0], [3.0]], ['down'], [0],
                                  out_idx)
        assert rows[0].split(',')[1] == want, (out_idx, rows[0])


# --- VERIFY_DWELL must outlast the analysis window ---------------------------
# The contact level is measured from windows tagged 'contact', and one window is
# detect_cycles/freq seconds long.  A dwell shorter than ~2 windows yields too
# few to form a level, so _contact_levels returns nothing, air_amp/contact_amp
# come back 0.0, and EVERY contact is rejected as a false halt - whereupon the
# reject path re-arms below the halt and walks the nozzle to the z_min floor.
# Cost of learning this on hardware: ~10min of a cold nozzle pressing to Z=-0.5
# at four bed locations, 2026-09-21, from VERIFY_DWELL=0.15 at 54.7Hz.
# The old bound was a flat above=0.1, which is fine at 150Hz and broken at 55.

def test_min_verify_dwell_is_two_analysis_windows():
    # 8 cycles at 54.7Hz = 146ms per window -> needs ~293ms
    assert abs(rp._min_verify_dwell(8., 54.7) - 0.2925) < 1e-3


def test_min_verify_dwell_scales_with_frequency():
    # the same cycle count is cheap at high frequency
    assert rp._min_verify_dwell(8., 150.) < rp._min_verify_dwell(8., 54.7)
    assert abs(rp._min_verify_dwell(8., 150.) - 0.1067) < 1e-3


def test_the_dwell_that_broke_the_sweep_is_rejected_at_54hz():
    assert 0.15 < rp._min_verify_dwell(8., 54.7), "0.15 must not be allowed"


def test_the_default_dwell_is_still_allowed_at_54hz():
    assert 0.4 >= rp._min_verify_dwell(8., 54.7), "0.4 default must survive"


# --- the contact ledger must name its own capture -----------------------------
# Pairing contacts.csv rows to verify*.csv by filename ORDER is unreliable:
# _autosave_verify numbers files by COUNTING existing ones, and a rejected
# verify writes a file too, so a probe can emit more than one.  Checked on the
# 2026-09-21 sweep: 12 of 25 rows paired to a capture whose z_cand could not
# contain the row's z_down.  Without a link, "which probes should we distrust"
# cannot be answered from the ledger at all.
# The column is APPENDED, so existing readers keep working.

def test_contact_row_appends_the_trace_name():
    d = {'down': 0.1, 'up': 0.2, 'bias': 0.1, 'down_spread': 0.0,
         'up_spread': 0.0, 'reps': 1}
    row = rp._contact_row(80., 30., 0.15, d, 54.7, '/traces/verify00207.csv')
    assert row.split(',')[:3] == ['80.000', '30.000', '0.150000'], row
    assert row.split(',')[-1] == 'verify00207.csv', row


def test_contact_row_survives_a_missing_trace():
    d = {'down': None, 'up': None, 'bias': None, 'reps': 0}
    row = rp._contact_row(1., 2., 0.3, d, 54.7, None)
    assert row.split(',')[-1] == '', row
    assert row.count(',') == 10, row


def test_contact_header_matches_the_row_width():
    d = {'down': 0.1, 'up': 0.2, 'bias': 0.1, 'reps': 1}
    row = rp._contact_row(1., 2., 0.3, d, 54.7, 'x/verify1.csv')
    assert rp._CONTACT_HEADER.count(',') == row.count(','), (
        rp._CONTACT_HEADER, row)

# --- _vmin_edge / _combine_axes: the measurement estimator -------------------
# V-minimum anchor + half-way crossing on the rising (air) side, in log
# amplitude, per axis per ramp.  The contact response is V-SHAPED on some axes -
# it falls to a minimum and climbs back as the nozzle presses deeper - so any
# level taken from the deepest samples is contaminated; the ramp's own minimum
# is not.  EVERY axis is evaluated; _combine_axes lets the data decide which
# ones are good witnesses rather than picking one in advance.
vmin_edge = HaltingContactProbe._vmin_edge
combine_axes = HaltingContactProbe._combine_axes


def _v_ramp(edge=0.05, zmin=0.02, step=0.005, noise=0.):
    """Down ramp: log-amplitude flat in air, linear fall to a minimum at
    `zmin`, then a climb back as the press deepens."""
    zs = np.round(np.arange(0.30, -0.10, -step), 4)
    y = np.where(zs >= edge + 0.02, 7.0,
        np.where(zs >= zmin, 7.0 - 0.4 * (edge + 0.02 - zs) / (edge + 0.02 - zmin),
                 6.6 + 0.3 * (zmin - zs) / (zmin + 0.10)))
    return zs, np.exp(y + RNG.normal(0., noise, len(zs)) if noise else y)


def test_vmin_edge_reads_the_half_way_crossing_above_the_minimum():
    zs, amps = _v_ramp()
    # air 7.0, min 6.6 -> half 6.8, reached half-way down the 0.02..0.07 flank
    edge, step, _ = vmin_edge(zs, amps, 0.10)
    assert abs(edge - 0.045) <= 0.003, edge
    assert abs(step - 0.4) <= 0.03, step   # moving median rounds the tip


def test_vmin_edge_ignores_the_deep_climb_back():
    # The climb back reaches ABOVE half (6.9 > 6.8) - a crossing search from
    # the deep end would find that first; the V anchor must not.
    zs, amps = _v_ramp()
    assert vmin_edge(zs, amps, 0.10)[0] > 0.02


def test_vmin_edge_rejects_an_air_only_ramp():
    # 2026-09-22 (60,60) false halts: ramps entirely in air stepped 0.01-0.06
    # log units against 0.28-0.36 for real contacts.
    zs = np.round(np.arange(0.30, -0.10, -0.005), 4)
    amps = np.exp(7.0 + 0.02 * np.sin(zs * 300.))
    assert vmin_edge(zs, amps, 0.10) is None


def test_vmin_edge_does_not_depend_on_sample_order():
    zs, amps = _v_ramp()
    perm = RNG.permutation(len(zs))
    assert abs(vmin_edge(zs[perm], amps[perm], 0.10)[0]
               - vmin_edge(zs, amps, 0.10)[0]) < 1e-12


def test_combine_drops_a_noisy_axis_and_averages_the_rest():
    # (edge, step, noise) per axis per rep.  y has a big step but air noise
    # like the measured accel y (step/noise ~4) - it must not vote.
    good_x = (0.050, 0.30, 0.005)
    good_z = (0.052, 0.32, 0.012)
    noisy_y = (0.300, 0.53, 0.093)
    ramps = [[good_x, noisy_y, good_z], [good_x, noisy_y, good_z]]
    assert np.allclose(combine_axes(ramps, 10.), [0.051, 0.051])


def test_combine_needs_an_axis_to_pass_on_every_rep():
    # An axis that drops in and out changes the MIX, and each axis crosses at
    # its own height - so a part-time witness moves the answer between probes.
    x = (0.050, 0.30, 0.005)
    z = (0.070, 0.32, 0.012)
    assert combine_axes([[x, None, z], [x, None, None]], 10.) == [0.050, 0.050]


def test_combine_reports_nothing_when_no_axis_is_good():
    assert combine_axes([[None, (0.3, 0.53, 0.093), None]], 10.) == []
    assert combine_axes([], 10.) == []


def _corpus_down_ramps():
    """112.7Hz textured-PEI verify captures, 2026-09-22 -> {(x,y): [probe]},
    each probe a list over reps of [(zs, amps) per axis].  {} if absent."""
    import os, glob, collections
    d = os.path.join(os.path.dirname(__file__), '..', '..',
                     'probe_traces_2026-09-21')
    out = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(d, 'verify*.csv'))):
        head = open(f).read(400)
        if 'freq=112.70' not in head or 'textured' not in head:
            continue
        xy = head.split('# x=')[1].split('\n')[0]
        cols, rows = None, []
        for ln in open(f):
            if ln.startswith('#'):
                continue
            p = ln.strip().split(',')
            if cols is None:
                cols = p
            elif len(p) == len(cols) and p[2] == 'down':
                rows.append(p)
        ai = [cols.index(c) for c in ('amp_x', 'amp_y', 'amp_z')]
        probe = []
        for rep in sorted(set(r[3] for r in rows)):
            r = [x for x in rows if x[3] == rep]
            zs = [float(x[0]) for x in r]
            probe.append([(zs, [float(x[i]) for x in r]) for i in ai])
        out[xy].append(probe)
    return out


def test_all_axes_reproduce_the_measured_repeatability():
    # Offline on this corpus the rule scored 1.40-3.25um at five of six
    # points; (30,70) is a known plate defect (11.7um on every estimator).
    # Median across points must hold at or under 3.5um.
    corpus = _corpus_down_ramps()
    if not corpus:
        return
    sds = []
    for xy, probes in corpus.items():
        vals = []
        for probe in probes:
            v = combine_axes([[vmin_edge(zs, a, 0.10) for zs, a in rep]
                              for rep in probe], 10.)
            if v:
                vals.append(np.mean(v))
        if len(vals) > 1:
            sds.append(float(np.std(vals, ddof=1)) * 1e3)
    assert len(sds) == 6, sds
    assert float(np.median(sds)) <= 3.5, sorted(sds)


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
