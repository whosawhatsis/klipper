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


def test_combine_drops_a_noisy_axis_and_weights_the_rest():
    # (edge, step, noise) per axis per rep.  y has a big step but air noise
    # like the measured accel y (step/noise ~6) - it must not vote.  x (snr 60)
    # is past full weight; z (snr 26.7) is partway up the 15..30 fade.
    good_x = (0.050, 0.30, 0.005)
    good_z = (0.052, 0.32, 0.012)
    noisy_y = (0.300, 0.53, 0.093)
    ramps = [[good_x, noisy_y, good_z], [good_x, noisy_y, good_z]]
    wz = (0.32 / 0.012 - 15.) / 15.
    want = (0.050 + wz * 0.052) / (1. + wz)
    assert np.allclose(combine_axes(ramps, 15.), [want, want])


def test_an_axis_FADES_in_rather_than_jumping_in():
    # Each axis crosses at its own height (z-x offsets -20..+25um across the
    # bed, measured 2026-09-22), so an axis that jumps from no say to an equal
    # say as it crosses the threshold moves the answer by half that offset.
    # Just above the threshold it must carry almost no weight.
    x = (0.050, 0.30, 0.005)                     # snr 60: full weight
    z_marginal = (0.070, 0.16, 0.010)            # snr 16: weight 1/15
    got = combine_axes([[x, None, z_marginal]], 15.)[0]
    assert abs(got - 0.050) < 0.0015, got


def test_combine_needs_an_axis_on_every_rep():
    # An axis missing from a rep has no crossing there, so it cannot vote.
    x = (0.050, 0.30, 0.005)
    z = (0.070, 0.32, 0.012)
    assert combine_axes([[x, None, z], [x, None, None]], 15.) == [0.050, 0.050]


def test_combine_weights_on_the_WORST_rep():
    # One noisy rep is enough to demote an axis for the whole probe - weights
    # are shared across reps so every rep reports the same mix.
    x = (0.050, 0.30, 0.005)
    z_good, z_bad = (0.070, 0.32, 0.005), (0.070, 0.32, 0.030)   # snr 64, 10.7
    assert combine_axes([[x, None, z_good], [x, None, z_bad]], 15.) \
        == [0.050, 0.050]


def test_combine_reports_nothing_when_no_axis_is_good():
    assert combine_axes([[None, (0.3, 0.53, 0.093), None]], 15.) == []
    assert combine_axes([], 15.) == []


def _corpus_down_ramps():
    """Every 112.7Hz verify capture in the two corpora -> {group: [probe]},
    grouped by (corpus, surface, x, y); each probe a list over reps of
    [(zs, amps) per axis].  {} if the corpora are absent."""
    import os, glob, collections
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    out = collections.defaultdict(list)
    for corpus in ('probe_traces_2026-09-21', 'probe_traces_2026-09-22'):
        for f in sorted(glob.glob(os.path.join(root, corpus, 'verify*.csv'))):
            meta, cols, rows = {}, None, []
            for ln in open(f):
                if ln.startswith('#'):
                    for t in ln[1:].split():
                        if '=' in t:
                            k, v = t.split('=', 1)
                            meta[k] = v
                    continue
                p = ln.strip().split(',')
                if cols is None:
                    cols = p
                elif len(p) == len(cols) and p[2] == 'down':
                    rows.append(p)
            if meta.get('freq') != '112.70' or 'amp_x' not in (cols or []):
                continue
            ai = [cols.index(c) for c in ('amp_x', 'amp_y', 'amp_z')]
            probe = []
            for rep in sorted(set(r[3] for r in rows)):
                r = [x for x in rows if x[3] == rep]
                zs = [float(x[0]) for x in r]
                probe.append([(zs, [float(x[i]) for x in r]) for i in ai])
            out[(corpus, meta.get('note'), meta['x'], meta['y'])].append(probe)
    return out


def test_all_axes_reproduce_the_measured_repeatability():
    # 13 point/surface/session groups at 112.7Hz.  Hard 10-snr gate: median
    # 3.22um.  15..30 fade: 2.03um.  Hold the fade to 2.2um.
    corpus = _corpus_down_ramps()
    if not corpus:
        return
    sds = []
    for key, probes in corpus.items():
        vals = []
        for probe in probes:
            v = combine_axes([[vmin_edge(zs, a, 0.10) for zs, a in rep]
                              for rep in probe], 15.)
            if v:
                vals.append(np.mean(v))
        if len(vals) >= 5:
            sds.append(float(np.std(vals, ddof=1)) * 1e3)
    assert len(sds) == 13, len(sds)
    assert float(np.median(sds)) <= 2.2, sorted(sds)


# --- contact point: air line x slope line pivoted on the half-way crossing ------
# The half-way crossing sits partway DOWN the contact slope (a median ~23um below
# where contact begins, and by a different amount per axis).  The contact point
# is where a line along that slope meets the air line.  The slope line is pivoted
# on the half-way crossing (its most precisely located point) and only its angle
# is fitted, over the 10-90% part of the slope.
tangent_knee = HaltingContactProbe._tangent_knee
guided_tangent = HaltingContactProbe._guided_tangent
contact_estimate = HaltingContactProbe._contact_estimate


def _two_line(knee=0.100, air=7.0, depth=0.4, width=0.030, climb=0.3, sign=-1,
              step=0.0033, noise=0., seed=3, air_slope=0.):
    """ln-amplitude ramp: straight air line above `knee`, a straight slope of
    `depth` over `width` below it (falling if sign<0, rising if sign>0), then a
    climb back (a V) of `climb` further down."""
    rng = np.random.default_rng(seed)
    zs = np.round(np.arange(0.24, -0.10, -step), 5)
    y = air + air_slope * (zs - knee)
    bot = knee - width
    fl = (zs < knee) & (zs >= bot)
    y = np.where(fl, air + sign * depth * (knee - zs) / width, y)
    deep = zs < bot
    y = np.where(deep, air + sign * depth - sign * climb * (bot - zs) / (bot + 0.10), y)
    if noise:
        y = y + rng.normal(0., noise, len(zs))
    return zs, np.exp(y)


def test_tangent_knee_finds_where_the_slope_meets_the_air_line():
    zs, amps = _two_line()
    f = tangent_knee(zs, amps, 0.10)
    assert abs(f['knee'] - 0.100) <= 0.002, f['knee']
    # and NOT the half-way crossing, which sits half the slope's width lower
    assert abs(f['zh'] - 0.085) <= 0.003, f['zh']


def test_tangent_knee_follows_a_sloped_air_line():
    # 0.2 ln/mm: the steepest air slope measured on this machine is 0.19.
    # (At 0.5 the half level, taken from the flat air median, biases +4um.)
    zs, amps = _two_line(air_slope=0.2)
    assert abs(tangent_knee(zs, amps, 0.10)['knee'] - 0.100) <= 0.003


def test_tangent_knee_rejects_an_air_only_ramp():
    zs = np.round(np.arange(0.24, -0.10, -0.0033), 5)
    amps = np.exp(7.0 + 0.02 * np.sin(zs * 300.))
    assert tangent_knee(zs, amps, 0.10) is None


def test_guided_fit_reads_a_RISING_axis_at_the_anchor():
    # accel y rises on contact at some points; the anchor supplies where to look
    zs, amps = _two_line(sign=+1, air=5.0, depth=1.0, width=0.040, climb=0.)
    f = guided_tangent(zs, amps, 0.100)
    assert f is not None and abs(f['knee'] - 0.100) <= 0.003, f


def test_guided_fit_uses_the_first_departure_not_a_later_fall():
    # z at (60,45): bumps UP where contact begins, then falls much further;
    # the guided fit must follow the bump that starts at the anchor
    zs = np.round(np.arange(0.24, -0.10, -0.0033), 5)
    y = np.where(zs >= 0.100, 6.8, np.where(zs >= 0.070, 6.8 + 0.2 * (0.100 - zs) / 0.030,
                 7.0 - 0.6 * (0.070 - zs) / 0.030))
    y = np.maximum(y, 6.2)
    f = guided_tangent(zs, np.exp(y), 0.100)
    assert f is not None and abs(f['knee'] - 0.100) <= 0.004, f


def test_contact_estimate_combines_axes_at_one_point():
    ramps = []
    for seed in (1, 2):
        zs, ax = _two_line(noise=0.004, seed=seed)
        _, ay = _two_line(sign=+1, air=5.0, depth=1.0, width=0.040, climb=0., noise=0.08, seed=seed + 10)
        _, az = _two_line(air=6.8, depth=0.5, width=0.020, climb=0.2, noise=0.008, seed=seed + 20)
        ramps.append((zs, [ax, ay, az]))
    r = contact_estimate(ramps, 0.10, 15.)
    assert r is not None and abs(r['z'] - 0.100) <= 0.003, r
    assert r['anchor'] in (0, 2), r
    assert len(r['per_rep']) == 2, r


def test_contact_estimate_reports_nothing_for_air_only_ramps():
    zs = np.round(np.arange(0.24, -0.10, -0.0033), 5)
    air = np.exp(7.0 + 0.02 * np.sin(zs * 300.))
    assert contact_estimate([(zs, [air, air, air])] * 2, 0.10, 15.) is None


def _manifest_groups(corpus, manifest, arm):
    import os, json
    d = os.path.join(os.path.dirname(__file__), '..', '..', corpus)
    mp = os.path.join(d, manifest)
    if not os.path.exists(mp):
        return {}
    out = {}
    for m in json.load(open(mp)):
        if m['arm'] != arm:
            continue
        probes = []
        for f in m['files']:
            cols, rows = None, []
            for ln in open(os.path.join(d, f)):
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
                probe.append(([float(x[0]) for x in r], [[float(x[i]) for x in r] for i in ai]))
            probes.append(probe)
        out[(m['x'], m['y'])] = probes
    return out


def test_contact_estimate_reproduces_the_slow_ramp_result():
    # 2026-09-23, 10 fresh smooth-PEI points, 0.15mm/s verify ramp: the
    # prototype measured median per-point sd 2.02um over the 9 points it
    # could score ((85,25) has no falling axis).  Skipped without the corpus.
    groups = _manifest_groups('probe_traces_2026-09-23b', 'fresh_run_manifest.json', 'slow')
    if not groups:
        return
    sds = []
    for probes in groups.values():
        vals = [r['z'] for r in (contact_estimate(p, 0.10, 15.) for p in probes) if r]
        if len(vals) >= 3:
            sds.append(float(np.std(vals, ddof=1)) * 1e3)
    assert len(sds) == 9, len(sds)
    assert float(np.median(sds)) <= 2.2, sorted(sds)


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
