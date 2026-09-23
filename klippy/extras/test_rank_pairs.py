#!/usr/bin/env python3
# Self-check for the (frequency, amplitude) selection a mesh calibration commits
# to.  Run from klippy/:
#     python -m extras.test_rank_pairs
#
# Two rules, deliberately different:
#
#   FAILED detection is fatal.  A pair that missed anywhere cannot be the single
#   global setting, and while hunting for one it is not worth retrying at later
#   points.  (If per-point settings are allowed it MUST still be tried
#   everywhere - a pair that misses at one location can be the best available at
#   another, which is measurably true on this bed.)
#
#   WEAK detection is not fatal, it is data.  A weak-but-detecting pair keeps
#   being measured; only the final comparison cares how weak its worst point
#   was.  Pruning on weakness would throw away the candidate that turns out to
#   be the only thing that works somewhere - at (60,100) the best available pair
#   managed 0.6-1.0x while everything else sat at 0.0-0.3x.
#
# Selection is MINIMAX: a mesh needs one setting that works at EVERY point, so
# ranking is by worst case.  On 2026-08-07 the configured mode measured 1.8-2.5x
# at (100,25) and 0.0-0.3x at (60,100); any average- or best-case rule keeps it,
# and then the mesh cannot probe (60,100) - which is what happened, ten times.
# The numbers below are the measured ones from that session.
from .resonance_probe_calibrate import ResonanceProbeCalibrate as RPC

rank = RPC.rank_candidate_pairs
failed_pairs = RPC.failed_pairs

MIN_DROP, TARGET_NOISE, MIN_MARGIN = 0.10, 0.08, 1.0

GOOD = (100, 25)      # the point that has always probed fine
DEAD = (60, 100)      # the dead zone that has never completed


# --- failure: fatal, and prunable ------------------------------------------

def test_a_recorded_miss_removes_the_pair_from_ranking():
    hist = {GOOD: {(206.5, 200): (0.44, 0.037, 1.5)},
            DEAD: {(206.5, 200): None}}
    ranked, failed = rank(hist, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert ranked == [], ranked
    assert failed == [((206.5, 200), DEAD)], failed


def test_failed_pairs_lists_what_not_to_retry():
    hist = {GOOD: {(1., 1): (0.5, 0.03, 2.0), (2., 1): None},
            DEAD: {(1., 1): (0.5, 0.03, 1.5)}}
    assert failed_pairs(hist) == {(2., 1)}


def test_untried_elsewhere_is_not_a_failure():
    # Absence of data must not be read as a miss.
    hist = {GOOD: {(206.5, 200): (0.44, 0.037, 1.5)}}
    ranked, failed = rank(hist, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert [r[0] for r in ranked] == [(206.5, 200)] and failed == []


# --- weakness: kept, measured, reported ------------------------------------

def test_weak_detection_is_ranked_not_vetoed():
    # 0.6x at the dead zone is weak but real - it must survive, flagged.
    hist = {GOOD: {(208.9, 120): (0.45, 0.061, 1.5)},
            DEAD: {(208.9, 120): (0.17, 0.029, 0.6)}}
    ranked, failed = rank(hist, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert failed == []
    pair, worst, n, decent, worst_pt = ranked[0]
    assert pair == (208.9, 120) and abs(worst - 0.6) < 1e-9
    assert decent is False and worst_pt == DEAD


def test_decent_everywhere_is_flagged_when_it_holds():
    hist = {GOOD: {(1., 1): (0.5, 0.03, 2.4)},
            DEAD: {(1., 1): (0.4, 0.04, 1.6)}}
    ranked, _ = rank(hist, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert ranked[0][3] is True, ranked


def test_the_measured_configured_mode_is_weak_not_failed():
    # 170.2Hz: 1.8x at the good point, 0.0x at the dead zone - it DETECTED at
    # both, so it is ranked last rather than dropped, and is not decent.
    hist = {GOOD: {(170.2, 120): (0.48, 0.031, 1.8)},
            DEAD: {(170.2, 120): (0.10, 0.063, 0.0)}}
    ranked, failed = rank(hist, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert failed == []
    assert ranked[0][3] is False and abs(ranked[0][1]) < 1e-9


# --- ordering ---------------------------------------------------------------

def test_ranked_by_worst_case_not_average():
    # A: 3.0x / 1.2x (avg 2.10)   B: 5.0x / 1.1x (avg 3.05, worse worst case)
    hist = {GOOD: {(1., 1): (0.5, 0.03, 3.0), (2., 1): (0.5, 0.03, 5.0)},
            DEAD: {(1., 1): (0.5, 0.03, 1.2), (2., 1): (0.5, 0.03, 1.1)}}
    ranked, _ = rank(hist, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert ranked[0][0] == (1., 1), ranked


def test_ties_prefer_the_pair_proven_at_more_points():
    hist = {GOOD: {(1., 1): (0.5, 0.03, 2.0), (2., 1): (0.5, 0.03, 2.0)},
            DEAD: {(1., 1): (0.5, 0.03, 2.0)}}
    ranked, _ = rank(hist, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert ranked[0][0] == (1., 1) and ranked[0][2] == 2, ranked


def test_amplitude_is_part_of_the_identity():
    # 148.6Hz measured 0% drop at aph=30 and 52% at aph=120 on the same spot.
    hist = {GOOD: {(148.6, 30): (0.00, 0.048, 0.0),
                   (148.6, 120): (0.52, 0.038, 1.7)}}
    ranked, _ = rank(hist, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert ranked[0][0] == (148.6, 120)
    assert dict((r[0], r[3]) for r in ranked)[(148.6, 30)] is False


def test_no_decent_pair_means_per_point_settings_are_needed():
    # The answer we actually want out of a survey: nothing covers the whole bed.
    hist = {GOOD: {(170.2, 120): (0.48, 0.031, 1.8), (208.9, 120): (0.33, 0.051, 0.7)},
            DEAD: {(170.2, 120): (0.10, 0.063, 0.0), (208.9, 120): (0.17, 0.029, 0.6)}}
    ranked, _ = rank(hist, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert ranked and not any(r[3] for r in ranked), ranked


# --- the order pairs are TRIED in -------------------------------------------
# The first point has no contact data, so prominence from the in-air sweep is
# the only prior there is.  It is a seed, not a verdict: air prominence has
# repeatedly failed to predict contact detectability here, and every point
# measured replaces more of it with real data.

order = RPC.candidate_order
PROM = [(206.5, 200), (170.2, 120), (148.6, 120), (208.9, 120)]   # prominence


def test_with_no_data_it_is_exactly_prominence_order():
    assert order({}, PROM, MIN_DROP, TARGET_NOISE, MIN_MARGIN) == PROM


def test_proven_decent_pairs_come_before_untried_ones():
    hist = {GOOD: {(208.9, 120): (0.45, 0.04, 1.9)}}
    o = order(hist, PROM, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert o[0] == (208.9, 120), o
    assert o[1:] == [(206.5, 200), (170.2, 120), (148.6, 120)], o


def test_known_weak_pairs_go_last_but_are_still_tried():
    hist = {DEAD: {(170.2, 120): (0.10, 0.063, 0.0)}}
    o = order(hist, PROM, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert o[-1] == (170.2, 120), o
    assert (170.2, 120) in o, "a weak pair must still be tried"


def test_a_failed_pair_is_demoted_but_NOT_eliminated_by_default():
    # Eliminating on one miss presumes the survey's own conclusion: if per-point
    # settings turn out to be needed, that pair may have been the best available
    # somewhere it was then never tried.
    hist = {GOOD: {(148.6, 120): None}}
    o = order(hist, PROM, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert (148.6, 120) in o, "a miss must not delete a candidate"
    assert o[-1] == (148.6, 120), o


def test_pruning_is_opt_in_for_a_global_only_search():
    hist = {GOOD: {(148.6, 120): None}}
    o = order(hist, PROM, MIN_DROP, TARGET_NOISE, MIN_MARGIN,
              prune_failed=True)
    assert (148.6, 120) not in o, o


def test_a_pair_that_missed_at_one_point_can_still_win_at_another():
    # The case pruning would have destroyed: 148.6 misses at the good point and
    # is the only thing that detects at the dead zone.
    hist = {GOOD: {(148.6, 120): None, (170.2, 120): (0.48, 0.031, 1.8)},
            DEAD: {(148.6, 120): (0.52, 0.038, 1.7), (170.2, 120): None}}
    ranked, failed = rank(hist, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert ranked == [], "neither pair works everywhere"
    assert {f[0] for f in failed} == {(148.6, 120), (170.2, 120)}
    # ...and the per-point answer is still recoverable from the history.
    assert hist[DEAD][(148.6, 120)][2] == 1.7
    assert hist[GOOD][(170.2, 120)][2] == 1.8


def test_re_ranking_promotes_what_the_data_shows():
    # Prominence puts 206.5 first; contact data says 208.9 is the better bet.
    hist = {GOOD: {(206.5, 200): (0.20, 0.05, 1.1),
                   (208.9, 120): (0.45, 0.04, 2.2)}}
    o = order(hist, PROM, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert o[0] == (208.9, 120) and o[1] == (206.5, 200), o


# --- going back after a pruned first pass -----------------------------------
# Pruning is only safe because it is reversible: run the fast global hunt with
# prune_failed=True, then revisit exactly the points nothing covered, with
# exactly the pairs they were denied.

revisit = RPC.revisit_plan
ALL = [(206.5, 200), (170.2, 120), (148.6, 120), (208.9, 120)]


def test_no_revisit_when_every_point_is_covered():
    hist = {GOOD: {(206.5, 200): (0.44, 0.04, 1.6)},
            DEAD: {(206.5, 200): (0.40, 0.05, 1.4)}}
    assert revisit(hist, ALL, MIN_DROP, TARGET_NOISE, MIN_MARGIN) == {}


def test_a_point_where_everything_missed_is_revisited():
    hist = {GOOD: {(206.5, 200): (0.44, 0.04, 1.6)},
            DEAD: {(206.5, 200): None}}
    plan = revisit(hist, ALL, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert set(plan) == {DEAD}
    assert (170.2, 120) in plan[DEAD] and (148.6, 120) in plan[DEAD]


def test_a_point_with_only_weak_results_is_revisited_too():
    # Detection happened, but nothing decent - a pruned pair may do better.
    hist = {GOOD: {(206.5, 200): (0.44, 0.04, 1.6)},
            DEAD: {(206.5, 200): (0.17, 0.03, 0.6)}}
    plan = revisit(hist, ALL, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert set(plan) == {DEAD}, plan


def test_retries_are_ordered_by_how_the_pair_did_elsewhere():
    # 148.6 was pruned after missing at GOOD but scored 1.9x at a third point;
    # 170.2 has no data at all.  The proven one is tried first.
    OTHER = (20, 25)
    hist = {GOOD: {(206.5, 200): (0.44, 0.04, 1.6), (148.6, 120): None},
            OTHER: {(148.6, 120): (0.50, 0.03, 1.9)},
            DEAD: {(206.5, 200): (0.17, 0.03, 0.6)}}
    plan = revisit(hist, ALL, MIN_DROP, TARGET_NOISE, MIN_MARGIN,
                   prominence=ALL)
    assert plan[DEAD][0] == (148.6, 120), plan[DEAD]


def test_only_untried_pairs_are_scheduled_for_a_point():
    hist = {DEAD: {(206.5, 200): (0.17, 0.03, 0.6), (170.2, 120): None}}
    plan = revisit(hist, ALL, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert (206.5, 200) not in plan[DEAD] and (170.2, 120) not in plan[DEAD]
    assert set(plan[DEAD]) == {(148.6, 120), (208.9, 120)}


def test_nothing_left_to_try_yields_no_entry():
    # Unresolved, but every pair has already been tested there - the revisit
    # cannot help, so it must not schedule an empty pass.
    hist = {DEAD: dict((p, None) for p in ALL)}
    assert revisit(hist, ALL, MIN_DROP, TARGET_NOISE, MIN_MARGIN) == {}


# --- what may enter the history at all ---------------------------------------
# A characterisation is worth exactly as much as the contact it was measured
# against.  This is the lesson of 2026-08-07: five rankings at (60,100) were
# computed at FALSE contacts, reported the configured mode at 0.0-0.3x, and a
# whole per-location frequency theory was built on them.  Re-measured against
# confirmed contacts the same mode scores 0.9-1.5x.  An unconfirmed reading must
# be no data - never a miss, because a miss is fatal to a pair.

record = RPC.record_result


def test_a_confirmed_measurement_is_recorded():
    h = {}
    assert record(h, DEAD, (170.2, 120), (0.39, 0.037, 1.2), True) is True
    assert h[DEAD][(170.2, 120)] == (0.39, 0.037, 1.2)


def test_a_confirmed_non_detection_is_recorded_as_a_miss():
    h = {}
    assert record(h, DEAD, (206.5, 120), None, True) is True
    assert h[DEAD][(206.5, 120)] is None
    assert failed_pairs(h) == {(206.5, 120)}


def test_an_unconfirmed_reading_is_not_recorded_at_all():
    h = {}
    assert record(h, DEAD, (170.2, 120), (0.01, 0.025, 0.0), False) is False
    assert h == {}, "an unverified contact must not become data"


def test_an_unconfirmed_miss_does_not_veto_a_pair():
    # The exact trap: a false contact detects nothing, and recording that as a
    # miss would kill a pair that works perfectly well at a real contact.
    h = {}
    record(h, DEAD, (170.2, 120), None, False)
    assert failed_pairs(h) == set()
    record(h, DEAD, (170.2, 120), (0.39, 0.037, 1.2), True)
    ranked, failed = rank(h, MIN_DROP, TARGET_NOISE, MIN_MARGIN)
    assert failed == [] and ranked[0][0] == (170.2, 120)


# --- reusing a contact height instead of descending again --------------------
# The descent that finds a point's contact height is the expensive step; every
# later candidate there, and every candidate on a revisit pass, ramps around
# that stored height instead.
#
# The plate is assumed FIXED for the duration of a calibration - if it moves the
# calibration is void regardless - so a stale height is not modelled.  The
# consequence for callers: a ramp finding no contact means the CANDIDATE failed
# to detect and is recorded as that pair's miss, not as a reason to re-descend.
# z_min is clamped as the configured hard floor, guarding a bug rather than a
# moving plate.

window = RPC.contact_ramp_window
UP, DOWN, ZMIN = 0.15, 0.25, -0.15


def test_normal_window_straddles_the_stored_height():
    hi, lo = window(0.09, UP, DOWN, ZMIN)
    assert abs(hi - 0.24) < 1e-9 and abs(lo - (-0.15)) < 1e-9


def test_the_down_leg_is_clamped_to_the_floor():
    # Without the clamp this would ramp to -0.26, below the configured floor.
    hi, lo = window(-0.01, UP, DOWN, ZMIN)
    assert lo == ZMIN, lo


def test_no_window_without_room_to_press_below_the_contact():
    # "Usable" means press room UNDER the surface, not total span.  A stored
    # height at the floor still leaves 0.15mm of window above it, and that window
    # is worthless: the ramp can never get under the contact to measure the
    # pressed state.  The caller must re-descend instead.
    assert window(ZMIN, UP, DOWN, ZMIN) is None          # no press room at all
    assert window(ZMIN + 0.03, UP, DOWN, ZMIN) is None   # only 0.03mm of it
    assert window(-0.20, UP, DOWN, ZMIN) is None         # already below the floor
    assert window(ZMIN + 0.06, UP, DOWN, ZMIN) is not None


def test_missing_height_is_not_an_error():
    assert window(None, UP, DOWN, ZMIN) is None


def test_a_mode_dependent_height_shift_stays_well_inside_the_window():
    # Press depth is ~0.73um/Hz; even a 60Hz mode change moves the contact by
    # ~44um, against a window of 0.15mm up / 0.25mm down.
    hi, lo = window(0.09, UP, DOWN, ZMIN)
    shifted = 0.09 - 60 * 0.00073
    assert lo < shifted < hi, (lo, shifted, hi)


def test_empty_history_is_not_an_error():
    assert rank({}, MIN_DROP, TARGET_NOISE, MIN_MARGIN) == ([], [])
    assert rank(None, MIN_DROP, TARGET_NOISE, MIN_MARGIN) == ([], [])
    assert failed_pairs(None) == set()


# --- single-point mode selection: consistency, not the loudest reading -------
#
# 2026-09-12: calibration crowned the mode with the strongest response on all
# three axes, and that mode false-halted on nearly every find - which the
# ranking never saw, because it only looked at the one characterisation that
# followed a confirmed contact.  Selection now scores every mode over several
# INTERLEAVED finds and charges it for what happened along the way.
#
# 2026-09-13 revision (user decision): a false halt or a salvage COSTS score
# (x0.8 per false halt, x0.5 per salvage) instead of rejecting; spread is a
# penalty up to a hard 100um ceiling; and heights compare RAW - the 0.73um/Hz
# press-depth correction pointed the wrong way on both days' hardware runs.

score = RPC.score_mode_trials


def trial(z, margins, false_halts=0, salvaged=False):
    return {'z': z, 'margins': margins, 'false_halts': false_halts,
            'salvaged': salvaged}


def verdict(ranked, freq):
    return [r for r in ranked if r[0] == freq][0]


def test_hardware_run_2026_09_13_picks_the_clean_mode():
    # POINT=40,40,2 after the move.  207.4 Hz was clean and tight (6um) but was
    # rejected "+50um off consensus" purely by the press-depth correction; raw,
    # it sits 1.8um from the others' median.
    trials = {
        64.4: [trial(0.0259, [4.0, 2.9, 1.7]), trial(0.0005, [3.3, 2.5, 2.4]),
               trial(-0.0147, [2.6, 2.0, 1.4])],
        122.6: [trial(0.0979, [4.0, 3.1, 1.9]), trial(-0.0390, [2.5, 2.2, 1.9]),
                trial(0.0889, [3.0, 2.9, 2.0])],
        187.0: [trial(0.0485, [4.9, 3.8, 3.8]),
                trial(0.0826, [5.4, 4.5, 3.0], 1),
                trial(0.0752, [5.3, 3.7, 3.5])],
        207.4: [trial(0.0809, [4.5, 3.2, 0.4]), trial(0.0745, [4.0, 3.2, 0.2]),
                trial(0.0770, [4.1, 2.9, 0.4])],
    }
    ranked = score(trials)
    assert ranked[0][0] == 207.4, ranked
    assert verdict(ranked, 187.0)[1] is not None, ranked   # 34um: penalised
    assert verdict(ranked, 122.6)[1] is None, ranked       # 137um: rejected
    assert verdict(ranked, 64.4)[1] is None, ranked        # ~77um off others


def test_loud_mode_that_false_halts_every_find_loses():
    trials = {
        148.0: [trial(0.030, [9., 6., 4.], 2), trial(0.031, [9., 6., 4.], 1),
                trial(0.029, [9., 6., 4.], 2)],
        172.9: [trial(0.012, [4., 2., 1.]), trial(0.013, [4., 2., 1.]),
                trial(0.011, [4., 2., 1.])],
        212.2: [trial(-0.010, [3., 2., 1.]), trial(-0.011, [3., 2., 1.]),
                trial(-0.009, [3., 2., 1.], 1)],
    }
    ranked = score(trials)
    assert ranked[0][0] == 172.9, ranked
    assert verdict(ranked, 148.0)[1] is not None, ranked


def test_each_false_halt_costs_a_fifth():
    clean = [trial(0.01, [4., 3., 1.])] * 3
    once = clean[:2] + [trial(0.01, [4., 3., 1.], 1)]
    twice = clean[:1] + [trial(0.01, [4., 3., 1.], 1)] * 2
    ranked = score({100.0: clean, 140.0: once, 180.0: twice})
    s = dict((f, v) for f, v, _w in ranked)
    assert abs(s[140.0] / s[100.0] - 0.8) < 1e-9, s
    assert abs(s[180.0] / s[100.0] - 0.64) < 1e-9, s


def test_salvage_costs_more_than_a_false_halt_but_does_not_reject():
    clean = [trial(0.01, [4., 3., 1.])] * 3
    fh = clean[:2] + [trial(0.01, [4., 3., 1.], 1)]
    sal = clean[:2] + [trial(0.01, [4., 3., 1.], salvaged=True)]
    ranked = score({100.0: clean, 140.0: fh, 180.0: sal})
    s = dict((f, v) for f, v, _w in ranked)
    assert s[180.0] is not None, ranked
    assert s[180.0] < s[140.0] < s[100.0], s


def test_mode_off_the_bed_height_consensus_is_rejected():
    # A "confirmed" contact 0.25mm above what every other mode agrees on is a
    # false halt that verify let through, however clean it looked.
    trials = {
        65.5: [trial(0.040, [5., 4., 1.])] * 3,
        148.0: [trial(0.260, [9., 8., 5.])] * 3,
        172.9: [trial(0.012, [4., 2., 1.])] * 3,
        212.2: [trial(-0.012, [3., 2., 1.])] * 3,
    }
    ranked = score(trials)
    assert verdict(ranked, 148.0)[1] is None, ranked
    assert verdict(ranked, 65.5)[1] is not None, ranked


def test_heights_compare_raw_across_frequencies():
    # No press-depth correction: equal raw heights at 65 and 212 Hz agree.
    trials = dict((f, [trial(0.08, [4., 3., 1.])] * 3)
                  for f in (65.5, 148.0, 212.2))
    assert all(r[1] is not None for r in score(trials)), score(trials)


def test_consensus_needs_two_other_modes():
    # With one other mode there is no telling which of the two is wrong.
    trials = {148.0: [trial(0.26, [9., 8., 5.])] * 3,
              172.9: [trial(0.01, [4., 2., 1.])] * 3}
    assert all(r[1] is not None for r in score(trials)), score(trials)


def test_a_missed_find_rejects():
    good = [trial(0.01, [4., 3., 1.])] * 3
    trials = {100.0: good, 120.0: good,
              180.0: good[:2] + [trial(None, None, 3)]}
    ranked = score(trials)
    assert verdict(ranked, 180.0)[1] is None, ranked
    assert verdict(ranked, 100.0)[1] is not None, ranked


def test_inconsistent_strength_loses_to_steady_strength():
    # Same average, but one round collapsed: the worst round is what a mesh
    # point will eventually meet.
    trials = {
        100.0: [trial(0.01, [8., 7., 1.]), trial(0.01, [1., 0.5, 0.]),
                trial(0.01, [8., 7., 1.])],
        140.0: [trial(0.01, [5., 4., 1.])] * 3,
        180.0: [trial(0.01, [3., 2., 1.])] * 3,
    }
    assert score(trials)[0][0] == 140.0, score(trials)


def test_moderate_height_scatter_is_penalised_not_rejected():
    tight = [trial(0.01, [4., 3., 1.])] * 3
    loose = [trial(0.00, [4., 3., 1.]), trial(0.034, [4., 3., 1.]),
             trial(0.02, [4., 3., 1.])]
    ranked = score({100.0: tight, 140.0: loose})
    assert verdict(ranked, 140.0)[1] is not None, ranked
    assert ranked[0][0] == 100.0, ranked


def test_gross_height_scatter_rejects():
    trials = {
        100.0: [trial(0.00, [6., 5., 1.]), trial(0.15, [6., 5., 1.]),
                trial(0.02, [6., 5., 1.])],
        140.0: [trial(0.01, [4., 3., 1.])] * 3,
    }
    assert verdict(score(trials), 100.0)[1] is None, score(trials)


def test_rejected_modes_say_why():
    trials = {100.0: [trial(None, None, 4)] * 3}
    (_f, s, reasons), = score(trials)
    assert s is None and reasons, reasons

def test_a_mode_that_never_detected_does_not_set_the_bed_consensus():
    # Hardware, POINT=80,25,2 cold at CONTACT_SPEED=0.2, 2026-09-21.  92.2Hz
    # never detected - 2% drop, 0.2x margin - and SALVAGED every round, landing
    # 400+um deep.  With only three candidates its heights dragged the
    # consensus (median of a PAIR is their mean), and both modes that did
    # detect at ~50% drop were rejected "off the other modes' consensus":
    # 73.4Hz -254um, 117.1Hz -181um.  A salvaged find is an over-press
    # artifact and must not vote on bed height.
    trials = {
        73.4: [trial(0.0950, [2.1, 1.5, 0.2]), trial(0.0932, [2.0, 1.5, 0.2]),
               trial(0.1013, [2.1, 1.6, 0.2])],
        92.2: [trial(0.4862, [0.2, 0.2, 0.1], salvaged=True),
               trial(0.5655, [0.2, 0.2, 0.2], salvaged=True),
               trial(0.5100, [0.2, 0.2, 0.1], salvaged=True)],
        117.1: [trial(0.1126, [2.0, 1.7, 0.4]), trial(0.1548, [2.5, 1.8, 0.2]),
                trial(0.1300, [2.2, 1.7, 0.3])],
    }
    ranked = score(trials)
    assert verdict(ranked, 117.1)[1] is not None, ranked
    assert verdict(ranked, 73.4)[1] is not None, ranked
    assert ranked[-1][0] == 92.2, ranked      # salvaged every round: last


def test_a_salvaged_height_cannot_drag_the_other_modes_consensus():
    # 100.0 salvages 2 of 3 rounds, 400um deep, so its MEDIAN height is an
    # over-press artifact.  With three candidates each mode is scored against
    # the median of the other two - which for a pair is their mean - so that
    # artifact would pull the reference ~200um and reject 140.0 and 180.0,
    # which sit 2um apart and are plainly right.  Salvaged finds must not vote.
    # (100.0 is still rejected on its own 400um spread; that is intended -
    # missing the halt is real inconsistency, and spread counts every round.)
    trials = {
        100.0: [trial(0.010, [4., 3., 1.]),
                trial(0.400, [4., 3., 1.], salvaged=True),
                trial(0.410, [4., 3., 1.], salvaged=True)],
        140.0: [trial(0.010, [4., 3., 1.])] * 3,
        180.0: [trial(0.012, [4., 3., 1.])] * 3,
    }
    ranked = score(trials)
    assert verdict(ranked, 140.0)[1] is not None, ranked
    assert verdict(ranked, 180.0)[1] is not None, ranked


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
