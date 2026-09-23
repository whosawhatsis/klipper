#!/usr/bin/env python3
"""Offline contact refinement on a descent trace, as CHANGE-POINT detection.

In air each axis's amplitude follows a smooth trajectory as Z changes; an
exponential in Z is linear in log-amplitude, which also sidesteps the sign of
the exponent (see wiki: "Model must allow either sign of exponent if an
explicit exponential is used; linear-in-log sidesteps it").  Contact makes the
axes DEPART from that trajectory, coincidently, with no fixed sign or
magnitude - so we fit air, FREEZE the model, and score how badly it predicts
the next few windows.

Differences from local/replay_changepoint.py, which this supersedes for trace
input:
  - reads descent trace CSVs (post-DFT amp per window vs mm_below_arm) rather
    than raw ACCELEROMETER_MEASURE captures, so it needs no raw samples;
  - the fit window GROWS from the start of the descent instead of trailing a
    fixed 15 windows.  Measured equal on the 2026-09-20 capture, and it cannot
    lose its air data the way a fixed window can once that window slides past
    contact.

CIRCULARITY WARNING: a live descent halts AT its trigger, so an onset found in
the last few windows of an ARMED trace proves nothing.  The informative result
is an onset comfortably ABOVE the halt - that is signal the live gradient
refine missed.
"""
import sys
import numpy as np

ARM = 20     # windows of air before the statistic may fire (5 estimated sigma
             # from 3 dof and false-fired)
FWD = 1      # windows judged against the frozen model.  The raw-capture
             # prototype used 6, but one trace window is 10um of Z and contact
             # here is a 10-20um cliff, so averaging 6 forward windows dilutes
             # it ~3x - and `range(arm, n-fwd)` then never scores the last 6
             # windows at all, which is exactly where the contact lives.
K = 4.0      # threshold as a multiple of the burst's own baseline
FLOOR = 6.0  # ...but never below this many sigma


def load(path):
    """-> (z_below_arm, amps[n,3]).  Header '# k=v' lines are returned too."""
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            for tok in line[1:].strip().split():
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    meta[k] = v
    # '#' header lines carry metadata; the first non-'#' line is the column
    # header (mm_below_arm,amp_x,...), which is NOT commented.
    rows = [ln for ln in open(path)
            if not ln.startswith('#') and not ln[:1].isalpha()]
    d = np.loadtxt(rows, delimiter=',')
    return d[:, 0], d[:, 1:4], meta


def load_verify(path, phase='down'):
    """Disarmed verify ramp -> (z, amps[n,k]) for one phase.

    This is the input the refinement actually wants: the halted descent finds
    the surface, then verify presses PAST it with no halt, so there are samples
    on the far side of the change point to fit the degradation against.  An
    armed descent trace has none - it stops at the cliff.

    Captures written before the per-axis columns existed carry a single 'amp';
    those still work, but the joint statistic degenerates to one channel.
    """
    meta, cols, rows = {}, None, []
    for line in open(path):
        if line.startswith('#'):
            for tok in line[1:].strip().split():
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    meta[k] = v
            continue
        parts = line.strip().split(',')
        if cols is None:
            cols = parts
            continue
        rows.append(parts)
    idx = ([cols.index(c) for c in ('amp_x', 'amp_y', 'amp_z')]
           if 'amp_x' in cols else [cols.index('amp')])
    zi, ti = cols.index('z'), cols.index('tag')
    sel = [r for r in rows if r[ti] == phase]
    if not sel:
        return np.empty(0), np.empty((0, len(idx))), meta
    z = np.array([float(r[zi]) for r in sel])
    a = np.array([[float(r[i]) for i in idx] for r in sel])
    return z, a, meta


def joint(z, amps, arm=ARM, fwd=FWD):
    """Frozen growing-window prediction error, in sigma, averaged over axes.

    Per axis the fit is linear in log-amplitude against z (an exponential in
    z).  sigma is that axis's own residual scatter, which normalizes away the
    ~60x difference in air predictability between axes.
    """
    ys = [np.log(np.maximum(amps[:, ax], 1.0)) for ax in range(3)]
    n = len(z)
    j = np.full(n, np.nan)
    for i in range(arm, n - fwd):
        per = []
        for y in ys:
            c = np.polyfit(z[:i], y[:i], 1)          # fit all air so far
            sd = max((y[:i] - np.polyval(c, z[:i])).std(), 1e-6)
            err = [abs(y[i + k] - np.polyval(c, z[i + k])) / sd
                   for k in range(fwd)]
            per.append(np.mean(err))
        j[i] = np.mean(per)                           # joint, not per-axis
    return j


def onset(z, amps, arm=ARM, fwd=FWD, k=K, floor=FLOOR):
    """-> (z_onset or None, j, threshold, baseline)."""
    j = joint(z, amps, arm, fwd)
    ok = ~np.isnan(j)
    if not ok.any():
        return None, j, float('nan'), float('nan')
    valid = j[ok]
    base = float(np.median(valid[:max(len(valid) // 3, 1)]))
    thr = max(k * base, floor)
    hits = np.nonzero(ok & (j > thr))[0]
    return (float(z[hits[0]]) if len(hits) else None), j, thr, base


def _selfcheck():
    """Synthetic air (exponential in z) with and without a departure."""
    z = np.arange(0, 1.9, 0.01)
    rng = np.random.default_rng(0)
    # air: each axis an exponential in z, different signs, with noise
    base = np.stack([6000 * np.exp(-0.05 * z),
                     1600 * np.exp(+0.10 * z),
                     870 * np.exp(-0.02 * z)], axis=1)
    air = base * (1 + 0.004 * rng.standard_normal(base.shape))
    zc, _, _, _ = onset(z, air)
    assert zc is None, "fired on pure air at z=%s" % (zc,)

    # contact: coincident departure in the last 0.25mm, opposite signs by axis
    hit = air.copy()
    m = z >= z[-1] - 0.25
    hit[m, 0] *= 0.45
    hit[m, 1] *= 1.8
    hit[m, 2] *= 0.6
    zc, _, _, _ = onset(z, hit)
    assert zc is not None, "missed a 55%/80%/40% coincident departure"
    assert abs(zc - (z[-1] - 0.25)) < 0.08, "onset %.3f, expected ~%.3f" % (
        zc, z[-1] - 0.25)
    print("selfcheck OK (air -> no fire; contact -> onset %.3f)" % zc)


def main(argv):
    if not argv or argv[0] == '--selfcheck':
        return _selfcheck()
    for path in argv:
        z, amps, meta = load(path)
        zc, j, thr, base = onset(z, amps)
        tail = z[-1]
        print("%-34s n=%3d freq=%s halt_at=%.3f base=%.2f thr=%.1f peak=%.1f  %s"
              % (path.split('/')[-1], len(z), meta.get('freq', '?'), tail,
                 base, thr, np.nanmax(j),
                 ("onset %.3f (%.0fum ABOVE halt)" % (zc, (tail - zc) * 1000)
                  if zc is not None else "no onset")))


if __name__ == '__main__':
    main(sys.argv[1:])


# --- joint change-point across ALL passes ------------------------------------
# The shipped estimator reduces each ramp to ONE edge (_ramp_edge) and then
# averages those numbers (down/up -> midpoint).  That throws away the shape: a
# pass whose edge estimate is bad corrupts the mean, and nothing can outvote it
# because by then each pass is a single scalar.
#
# Instead, ask every pass about every candidate depth at once.  A real contact
# is a depth where ALL passes agree something changed, even if they sit at
# different amplitudes, carry different noise, and disagree about absolute
# height (down/up hysteresis runs ~22um on this machine).  So score a candidate
# by how much a SPLIT fit beats a single fit, per pass and per axis, each
# normalized by its own single-fit error - which is what makes passes with
# different noise and bias commensurable - and sum.

def split_gain(z, y, c, min_side=4):
    """How much better a fit split at depth c is than one line, in [0,1).

    Normalized by the single-fit error, so a quiet pass and a noisy one
    contribute comparably.
    """
    hi, lo = z > c, z <= c
    if hi.sum() < min_side or lo.sum() < min_side:
        return None
    def sse(zz, yy):
        if len(zz) < 2:
            return 0.
        r = yy - np.polyval(np.polyfit(zz, yy, 1), zz)
        return float(r @ r)
    whole = sse(z, y)
    if whole <= 0:
        return None
    return max(0., 1. - (sse(z[hi], y[hi]) + sse(z[lo], y[lo])) / whole)


def joint_changepoint(passes, grid=None, min_side=4):
    """Depth at which the most passes agree something changed.

    'passes' is a list of (z, amps[n,k]).  Returns (z*, score, per_pass_gain)
    where per_pass_gain lets a caller see WHICH passes agreed - a pass that
    contributes nothing at the winning depth is the one to distrust, which a
    scalar-per-pass average can never tell you.
    """
    zs = np.concatenate([p[0] for p in passes])
    if grid is None:
        lo = max(p[0].min() for p in passes)
        hi = min(p[0].max() for p in passes)
        if not np.isfinite(lo) or hi <= lo:
            return None, 0., []
        grid = np.linspace(lo, hi, 60)
    best, best_c, best_per = -1., None, []
    for c in grid:
        per, tot = [], 0.
        for z, a in passes:
            g = [split_gain(z, np.log(np.maximum(a[:, k], 1.0)), c, min_side)
                 for k in range(a.shape[1])]
            g = [v for v in g if v is not None]
            per.append(max(g) if g else 0.)
            tot += per[-1]
        if tot > best:
            best, best_c, best_per = tot, c, per
    # score is the MEAN agreement, so it reads the same for 2 passes or 5
    return best_c, (best / max(len(passes), 1)), best_per


def joint_step_edge(passes, grid=None, min_side=4):
    """Joint edge using an ANCHORED-STEP model instead of two fitted lines.

    Same pooling idea as joint_changepoint, different per-pass model: each pass
    is described as one level above the edge and another below it, with the
    LEVELS free per pass (that is the per-pass bias/scale the passes are
    allowed to disagree about) and only the EDGE shared.  Residuals are scaled
    by each pass's own step height, so a pass with a weak contact cannot drag
    the shared edge around.

    joint_changepoint's two-line model located a piecewise-linear improvement,
    which on a flat-then-cliff signal is pulled about by the air trend and the
    post-contact floor - measured 15um across-probe scatter against 2.76um for
    the shipped per-pass crossing.
    """
    lo = max(p[0].min() for p in passes)
    hi = min(p[0].max() for p in passes)
    if hi <= lo:
        return None, 0., []
    if grid is None:
        grid = np.linspace(lo, hi, 120)
    best, best_c, best_per = None, None, []
    for c in grid:
        per, tot, ok = [], 0., True
        for z, a in passes:
            above, below = z > c, z <= c
            if above.sum() < min_side or below.sum() < min_side:
                ok = False
                break
            y = np.log(np.maximum(a[:, np.argmax(a.std(0) / (a.mean(0) + 1e-9))],
                                  1.0))
            air, con = np.median(y[above]), np.median(y[below])
            step = abs(air - con)
            if step < 1e-6:
                per.append(1.)
                tot += 1.
                continue
            r = np.concatenate([y[above] - air, y[below] - con])
            per.append(float(np.mean(np.abs(r)) / step))
            tot += per[-1]
        if not ok:
            continue
        if best is None or tot < best:
            best, best_c, best_per = tot, c, per
    return best_c, (best / max(len(passes), 1) if best is not None else 0.), best_per


def joint_step_edge_all(passes, grid=None, min_side=4, snr_floor=1.0):
    """joint_step_edge over EVERY axis, not the loudest one per pass.

    Each (pass, axis) is an independent witness to the same edge with its own
    free levels, so N passes give 3N constraints rather than N.  Picking one
    axis per pass - as joint_step_edge did, and as the shipped _ramp_edge does
    via wamps[judge] - throws two thirds of them away.  Contact is cross-axis
    anyway: it is already documented that driving x does not put the signal on
    x, and that z can fall while y rises.

    Weighting matters.  Normalizing a witness's residual by its own step height
    makes commensurable witnesses out of different amplitudes, but a witness
    with almost no step then reports a huge normalized residual and swamps the
    sum.  So weight each by its SNR (step / air noise): a clean strong axis
    dominates, a weak one still nudges, and a flat one contributes ~nothing
    instead of shouting.
    """
    lo = max(p[0].min() for p in passes)
    hi = min(p[0].max() for p in passes)
    if hi <= lo:
        return None, 0., []
    if grid is None:
        grid = np.linspace(lo, hi, 120)
    logs = [[np.log(np.maximum(a[:, k], 1.0)) for k in range(a.shape[1])]
            for _, a in passes]
    best, best_c, best_per = None, None, []
    for c in grid:
        num = den = 0.
        per = []
        for (z, _a), ys in zip(passes, logs):
            above, below = z > c, z <= c
            if above.sum() < min_side or below.sum() < min_side:
                num = None
                break
            pw = 0.
            for y in ys:
                air, con = np.median(y[above]), np.median(y[below])
                step = abs(air - con)
                noise = np.mean(np.abs(y[above] - air)) + 1e-9
                snr = step / noise
                if snr < snr_floor or step < 1e-6:
                    continue
                r = np.concatenate([y[above] - air, y[below] - con])
                num += snr * (np.mean(np.abs(r)) / step)
                den += snr
                pw += snr
            per.append(pw)
        if num is None or den <= 0:
            continue
        val = num / den
        if best is None or val < best:
            best, best_c, best_per = val, c, per
    return best_c, (best if best is not None else 0.), best_per


def joint_llr_edge(passes, detrend=True, min_side=5, grid=None):
    """Shared edge by Gaussian log-likelihood ratio, over every pass AND axis.

    The level-shift models (joint_step_edge*) ask only whether the MEAN
    amplitude moves, so an axis with a weak step contributes nothing however it
    is weighted - measured: every step-based weighting returned bit-identical
    answers to using the loudest axis alone, because y's step dwarfs x's.  But
    contact is chaotic, and the CHARACTER of the noise changes on an axis whose
    mean barely moves.

    Scoring a change in DISTRIBUTION rather than in mean captures both, and
    self-weights: the gain is a likelihood ratio, so an axis that genuinely
    changes contributes a lot and a flat one contributes ~0, with no
    hand-tuned weight.  Measured on the (80,30) 3-rep captures, per-axis gains
    at the winning edge came out 22.1/26.1/20.4 (x/y/z) - all three axes
    carrying comparable information, where the level model had y dominating
    completely.

    detrend=True fits a line per segment instead of a constant: the air
    segment has a real slope in z, and charging that slope to 'variance'
    costs a lot of accuracy (15.5um vs 3.7um across-probe sd).

    Returns (z*, total_gain, per_axis_gains) - the per-axis gains say WHICH
    witnesses agreed, which is the diagnostic a per-pass scalar cannot give.
    """
    lo = max(p[0].min() for p in passes)
    hi = min(p[0].max() for p in passes)
    if hi <= lo:
        return None, 0., []
    if grid is None:
        grid = np.linspace(lo, hi, 120)
    logs = [[np.log(np.maximum(a[:, k], 1.0)) for k in range(a.shape[1])]
            for _, a in passes]

    def seg_ll(z, y):
        n = len(y)
        if n < 3:
            return None
        if detrend:
            r = y - np.polyval(np.polyfit(z, y, 1), z)
        else:
            r = y - y.mean()
        return -0.5 * n * np.log(max(float(r @ r) / n, 1e-12))

    best, best_c, best_g = None, None, []
    for c in grid:
        tot, gains, ok = 0., [], True
        for (z, _a), ys in zip(passes, logs):
            above, below = z > c, z <= c
            if above.sum() < min_side or below.sum() < min_side:
                ok = False
                break
            for y in ys:
                null = seg_ll(z, y)
                la = seg_ll(z[above], y[above])
                lb = seg_ll(z[below], y[below])
                if None in (null, la, lb):
                    continue
                gains.append((la + lb) - null)
                tot += gains[-1]
        if not ok:
            continue
        if best is None or tot > best:
            best, best_c, best_g = tot, c, gains
    return best_c, (best if best is not None else 0.), best_g


def pass_likelihood(z, amps, grid, min_side=4):
    """One pass -> a curve of 'how likely is the surface at this height'.

    Per axis, the score at height c is how much better a two-level fit split at
    c explains the profile than a single level, normalized by the single-level
    error.  Axes are combined by their max, then the curve is scaled to [0,1]
    so passes with different amplitudes and noise are comparable as CURVES
    rather than being reduced to a scalar first.
    """
    out = np.zeros(len(grid))
    for i, c in enumerate(grid):
        above, below = z > c, z <= c
        if above.sum() < min_side or below.sum() < min_side:
            continue
        best = 0.
        for k in range(amps.shape[1]):
            y = np.log(np.maximum(amps[:, k], 1.0))
            def sse(m):
                r = y[m] - np.polyval(np.polyfit(z[m], y[m], 1), z[m])
                return float(r @ r)
            whole = sse(np.ones(len(z), bool))
            if whole <= 0:
                continue
            best = max(best, max(0., 1. - (sse(above) + sse(below)) / whole))
        out[i] = best
    m = out.max()
    return out / m if m > 0 else out


def agreeing_edge(passes, grid=None, frac=0.8, combine='geo', min_side=4):
    """Highest height at which the passes AGREE the surface is there.

    Two departures from joint_step_edge_all, both from the same insight:

    1. Passes are combined by a GEOMETRIC MEAN, not a sum.  A sum lets one
       emphatic pass carry the answer; a geometric mean lets any pass that
       says "not here" veto, which is what agreement means.
    2. The answer is the HIGHEST height clearing frac*max agreement, not the
       global maximum.  Below the surface the nozzle is pressing into the
       plate, so an artifact there can repeat at a near-constant press depth
       and therefore CORRELATES across passes - agreement alone cannot reject
       it.  Above the surface there is only air, so spurious peaks are noise
       and do not correlate.  Scanning down from the top and taking the first
       consensus is what makes the asymmetry work for us.

    Returns (z*, combined_curve, grid).
    """
    lo = max(p[0].min() for p in passes)
    hi = min(p[0].max() for p in passes)
    if hi <= lo:
        return None, None, None
    if grid is None:
        grid = np.linspace(lo, hi, 200)
    curves = np.array([pass_likelihood(z, a, grid, min_side) for z, a in passes])
    if combine == 'geo':
        comb = np.exp(np.mean(np.log(curves + 1e-6), axis=0))
    elif combine == 'min':
        comb = curves.min(axis=0)
    else:
        comb = curves.mean(axis=0)
    thr = frac * comb.max()
    hits = np.nonzero(comb >= thr)[0]
    if not len(hits):
        return None, comb, grid
    # grid ascends in z; the HIGHEST qualifying height is the last index
    return float(grid[hits[-1]]), comb, grid


def agreeing_peak_edge(passes, grid=None, frac=0.5, prominence=0.15,
                       combine='geo', min_side=4):
    """Highest distinct PEAK of the agreement curve, not the highest point
    clearing a threshold.

    The threshold form (agreeing_edge) fails when the combined curve is a broad
    shoulder rather than a set of peaks: 'highest point above frac*max' then
    walks up the flank to the top of the scan range.  Measured at (90,45): it
    returned 0.2187/0.2127/0.2170 - consistent to 2.0um, and ~130um ABOVE the
    real surface at ~0.085, with 90-97 of 200 grid points clearing the
    threshold.  Precision on the wrong feature.

    So find local maxima with a real prominence, and among those take the
    highest in z that is still a serious candidate.  The asymmetry argument is
    unchanged - a sub-surface press artifact repeats and so correlates across
    passes, while air noise above does not - but it only applies to genuine
    peaks.
    """
    zc, comb, grid = agreeing_edge(passes, grid=grid, frac=0.0,
                                   combine=combine, min_side=min_side)
    if comb is None:
        return None, None, None
    peaks = []
    for i in range(1, len(comb) - 1):
        if comb[i] >= comb[i - 1] and comb[i] > comb[i + 1]:
            # prominence: rise above the lower of the two flanking minima
            left = comb[:i].min() if i else comb[i]
            right = comb[i + 1:].min() if i + 1 < len(comb) else comb[i]
            if comb[i] - max(left, right) >= prominence * comb.max():
                peaks.append(i)
    if not peaks:
        return None, comb, grid
    good = [i for i in peaks if comb[i] >= frac * comb.max()]
    if not good:
        good = peaks
    return float(grid[max(good)]), comb, grid


def pooled_midpoint_edge(passes, grid=None, min_side=4, snr_floor=1.0):
    """Pooled split to CHOOSE the region, per-witness midpoint to report the height.

    joint_step_edge_all reported the split-gain argmax as a height, which is
    biased +7-24um high: gain peaks where a two-segment fit best explains the
    variance, and with a sloped air side and a flat contact side that is not
    the step.  Measured at (90,45) against hand-read cliff midpoints:
    79.2/90.1/95.2 actual vs 103.7/99.8/102.2 reported.

    So use the pooled gain only to locate the transition robustly - that is
    what pooling is good at - and then take each witness's own half-way
    crossing between its fitted air and contact levels, which is the quantity
    the shipped _ramp_edge computes and gets right (79.2 vs 79.2 on a clean
    probe).  Witnesses are combined by a step-weighted median so a witness with
    a weak step cannot swing the answer.
    """
    c, _score, _per = joint_step_edge_all(passes, grid=grid, min_side=min_side,
                                          snr_floor=snr_floor)
    if c is None:
        return None, []
    crossings, weights = [], []
    for z, a in passes:
        above, below = z > c, z <= c
        if above.sum() < 2 or below.sum() < 2:
            continue
        for k in range(a.shape[1]):
            y = np.log(np.maximum(a[:, k], 1.0))
            air, con = np.median(y[above]), np.median(y[below])
            step = abs(air - con)
            noise = np.mean(np.abs(y[above] - air)) + 1e-9
            if step / noise < snr_floor or step < 1e-6:
                continue
            half = 0.5 * (air + con)
            # crossing nearest the pooled split, scanning outward from it
            idx = int(np.argmin(np.abs(z - c)))
            best = None
            for i in range(len(z) - 1):
                y0, y1 = y[i] - half, y[i + 1] - half
                if y0 == 0.:
                    cand = z[i]
                elif y0 * y1 < 0:
                    cand = z[i] + (0. - y0) * (z[i + 1] - z[i]) / (y1 - y0)
                else:
                    continue
                if best is None or abs(cand - z[idx]) < abs(best - z[idx]):
                    best = cand
            if best is not None:
                crossings.append(float(best))
                weights.append(float(step))
    if not crossings:
        return float(c), []
    o = np.argsort(crossings)
    cr = np.array(crossings)[o]
    w = np.cumsum(np.array(weights)[o])
    w /= w[-1]
    return float(cr[int(np.searchsorted(w, 0.5))]), crossings


def simple_midpoint_edge(passes, edge_frac=0.25, snr_floor=1.0):
    """Each witness's own half-way crossing, combined by a step-weighted median.

    No split search at all.  Air and contact levels come from the top and
    bottom `edge_frac` of each profile, the crossing of their midpoint is read
    directly, and the witnesses are combined by a step-weighted median.

    This beat every split-gain variant on the 2026-09-22 air-rich captures
    (8.15um mean across-probe sd, against 16.8um for the pooled midpoint,
    18.5um for the gain argmax and 51.3um for shipped).  The split search was
    adding bias and complexity without buying accuracy: pooling is worth a
    great deal against a SINGLE-pass estimator, but the pooling should be over
    robust per-witness measurements, not over a fitted model's optimum.
    """
    crossings, weights = [], []
    for z, a in passes:
        n = len(z)
        m = max(3, int(n * edge_frac))
        for k in range(a.shape[1]):
            y = np.log(np.maximum(a[:, k], 1.0))
            con, air = np.median(y[:m]), np.median(y[-m:])
            step = abs(air - con)
            noise = np.mean(np.abs(y[-m:] - air)) + 1e-9
            if step / noise < snr_floor or step < 1e-6:
                continue
            half = 0.5 * (air + con)
            for i in range(n - 1):
                d0, d1 = y[i] - half, y[i + 1] - half
                if d0 * d1 <= 0 and d1 != d0:
                    crossings.append(float(z[i] + (-d0) * (z[i + 1] - z[i])
                                            / (d1 - d0)))
                    weights.append(float(step))
                    break
    if not crossings:
        return None, []
    o = np.argsort(crossings)
    cr = np.array(crossings)[o]
    w = np.cumsum(np.array(weights)[o])
    w /= w[-1]
    return float(cr[int(np.searchsorted(w, 0.5))]), crossings


def intersect_fit_edge(z, y, guess, halfwidth, min_side=4, robust=True,
                       min_above=None, min_below=None):
    """Contact height as the CROSSING of two independent fits.

    Fit the pre-touch (air) and post-touch (contact) regions separately,
    EXCLUDING a band of +-halfwidth around the guess, then intersect the two
    lines.  Excluding the band is what distinguishes this from the split-gain
    model: there, every transition-zone sample had to join one segment or the
    other, which dragged both fits and biased the answer high by 7-24um.  Here
    the uncertain samples inform neither fit.

    The two sides are not equally trustworthy.  After contact the signal is
    smooth and monotonic (damping grows steadily with press depth); in air the
    toolhead sits on a high-Q resonance and is jittery.  So the contact-side
    fit is the well-conditioned one and the air-side fit carries most of the
    error - hence robust=True, which uses a Theil-Sen style median-of-slopes
    rather than least squares on the noisy side.

    Returns (z_cross, air_slope, con_slope) or (None, ..) if either side is
    too short.
    """
    above = z > guess + halfwidth      # air
    below = z < guess - halfwidth      # contact
    # Asymmetric on purpose.  The contact side is ~1.8x smoother than air
    # (residual sd 0.029 vs 0.053 log units) and ~25x steeper, so a line
    # through 3 of its points is better determined than one through 5 air
    # points - and it is the SHORT side, so a symmetric requirement discards
    # most probes.  Measured: a symmetric min_side of 4 yielded an estimate on
    # only 5-12 of 15 probes depending on the exclusion band.
    ma = min_side if min_above is None else min_above
    mb = min_side if min_below is None else min_below
    if above.sum() < ma or below.sum() < mb:
        return None, None, None

    def fit(m):
        zz, yy = z[m], y[m]
        if robust and len(zz) >= 3:
            sl = []
            for i in range(len(zz)):
                for j in range(i + 1, len(zz)):
                    dz = zz[j] - zz[i]
                    if abs(dz) > 1e-9:
                        sl.append((yy[j] - yy[i]) / dz)
            if not sl:
                return None
            slope = float(np.median(sl))
            return slope, float(np.median(yy - slope * zz))
        c = np.polyfit(zz, yy, 1)
        return float(c[0]), float(c[1])

    fa, fb = fit(above), fit(below)
    if fa is None or fb is None:
        return None, None, None
    (sa, ia), (sb, ib) = fa, fb
    if abs(sa - sb) < 1e-9:
        return None, sa, sb
    return float((ib - ia) / (sa - sb)), sa, sb


def crossing_edge(passes, halfwidth=0.030, snr_floor=1.0, iters=2,
                  min_above=5, min_below=3):
    """simple_midpoint_edge as the seed, then per-witness two-fit intersection.

    Combined by a step-weighted median, as elsewhere.  Also returns the
    per-witness crossings so a caller can see the spread the air side causes.
    """
    seed, _ = simple_midpoint_edge(passes, snr_floor=snr_floor)
    if seed is None:
        return None, []
    crossings, weights = [], []
    for z, a in passes:
        n = len(z)
        m = max(3, int(n * 0.25))
        for k in range(a.shape[1]):
            y = np.log(np.maximum(a[:, k], 1.0))
            con, air = np.median(y[:m]), np.median(y[-m:])
            step = abs(air - con)
            noise = np.mean(np.abs(y[-m:] - air)) + 1e-9
            if step / noise < snr_floor or step < 1e-6:
                continue
            g = seed
            for _ in range(iters):
                zc, _sa, _sb = intersect_fit_edge(
                    z, y, g, halfwidth, min_above=min_above,
                    min_below=min_below)
                if zc is None or not (z.min() <= zc <= z.max()):
                    zc = None
                    break
                g = zc
            if zc is not None:
                crossings.append(float(zc))
                weights.append(float(step))
    if not crossings:
        return None, []
    o = np.argsort(crossings)
    cr = np.array(crossings)[o]
    w = np.cumsum(np.array(weights)[o]); w /= w[-1]
    return float(cr[int(np.searchsorted(w, 0.5))]), crossings


def fit_logistic(z, y, z0_grid=None, w_grid=None):
    """4-parameter logistic fit:  y = con + (air-con) / (1 + exp(-(z-z0)/w)).

    The transition looks like a SIGMOID in log-amplitude, not two straight
    lines meeting at a corner (user, from the per-pass plots).  Fitting one
    directly uses the transition samples that intersect_fit_edge has to
    exclude, and removes the arbitrary exclusion halfwidth - the midpoint z0 is
    just a fitted parameter.

    Separable least squares: for fixed (z0, w) the model is LINEAR in the two
    levels, so a 2D grid over (z0, w) with an exact linear solve inside is both
    robust and dependency-free - no scipy, and no initial-guess sensitivity.

    Returns (z0, w, air, con, rms) or None.
    """
    n = len(z)
    if n < 6:
        return None
    lo, hi = float(z.min()), float(z.max())
    span = hi - lo
    if span <= 0:
        return None
    if z0_grid is None:
        z0_grid = np.linspace(lo + 0.05 * span, hi - 0.05 * span, 60)
    if w_grid is None:
        # width from a tenth of a sample step to a quarter of the ramp
        w_grid = np.geomspace(max(span / (10. * n), 1e-5), span / 4., 18)
    best = None
    for w in w_grid:
        for z0 in z0_grid:
            s = 1.0 / (1.0 + np.exp(-np.clip((z - z0) / w, -50., 50.)))
            # y = con + (air-con)*s  ->  linear in [con, (air-con)]
            A = np.stack([np.ones_like(s), s], axis=1)
            try:
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            resid = y - A @ coef
            rms = float(np.sqrt(np.mean(resid * resid)))
            if best is None or rms < best[0]:
                best = (rms, float(z0), float(w), float(coef[0] + coef[1]),
                        float(coef[0]))
    if best is None:
        return None
    rms, z0, w, air, con = best
    return z0, w, air, con, rms


def sigmoid_edge(passes, snr_floor=1.0):
    """Per-witness logistic midpoint, combined by a step-weighted median."""
    mids, weights, widths, rmss = [], [], [], []
    for z, a in passes:
        n = len(z)
        m = max(3, int(n * 0.25))
        for k in range(a.shape[1]):
            y = np.log(np.maximum(a[:, k], 1.0))
            con, air = np.median(y[:m]), np.median(y[-m:])
            step = abs(air - con)
            noise = np.mean(np.abs(y[-m:] - air)) + 1e-9
            if step / noise < snr_floor or step < 1e-6:
                continue
            f = fit_logistic(z, y)
            if f is None:
                continue
            z0, w, _a, _c, rms = f
            if not (z.min() <= z0 <= z.max()):
                continue
            mids.append(z0)
            weights.append(step)
            widths.append(w)
            rmss.append(rms)
    if not mids:
        return None, [], []
    o = np.argsort(mids)
    cr = np.array(mids)[o]
    wt = np.cumsum(np.array(weights)[o]); wt /= wt[-1]
    return float(cr[int(np.searchsorted(wt, 0.5))]), mids, widths


def asym_edge(passes, halfwidth=0.015, snr_floor=1.0, report='cross',
              min_above=5, min_below=3):
    """Log-LINEAR air side, SATURATING sigmoid contact side, and their crossing.

    The two sides are not the same shape (user, from the per-pass plots).  In
    air the log-amplitude drifts almost linearly with z.  After contact it
    drops steeply and then FLATTENS toward a floor - a sigmoid, not a line.

    Fitting a line to a saturating region makes the slope depend on how much
    of the flat tail is included, which is why the contact-side slope measured
    1.97 +- 4.33 log/mm: scatter more than twice the mean, model error rather
    than noise.

    So: robust line above, 3-parameter saturating logistic below
        y = floor + (top - floor) / (1 + exp(-(z - zc)/w))
    and report where the fitted contact curve meets the fitted air line.
    report='mid' returns the sigmoid's own midpoint instead.
    """
    seed, _ = simple_midpoint_edge(passes, snr_floor=snr_floor)
    if seed is None:
        return None, []
    out, weights = [], []
    for z, a in passes:
        n = len(z)
        m = max(3, int(n * 0.25))
        step_z = (z.max() - z.min()) / max(n - 1, 1)
        for k in range(a.shape[1]):
            y = np.log(np.maximum(a[:, k], 1.0))
            con, air = np.median(y[:m]), np.median(y[-m:])
            step = abs(air - con)
            noise = np.mean(np.abs(y[-m:] - air)) + 1e-9
            if step / noise < snr_floor or step < 1e-6:
                continue
            above = z > seed + halfwidth
            below = z <= seed + halfwidth      # contact side KEEPS the knee
            if above.sum() < min_above or below.sum() < min_below:
                continue
            ca = np.polyfit(z[above], y[above], 1)          # air: line
            f = fit_logistic(z[below], y[below],
                             w_grid=np.geomspace(step_z, (z.max()-z.min())/4., 14))
            if f is None:
                continue
            zc, w, top, floor = f[0], f[1], f[2], f[3]
            if report == 'mid':
                val = zc
            else:
                # crossing of the air line with the contact sigmoid
                grid = np.linspace(z.min(), z.max(), 400)
                sig = floor + (top - floor) / (
                    1.0 + np.exp(-np.clip((grid - zc) / w, -50., 50.)))
                d = sig - np.polyval(ca, grid)
                s = np.nonzero(np.diff(np.sign(d)))[0]
                if not len(s):
                    continue
                i = s[np.argmin(np.abs(grid[s] - seed))]
                val = float(grid[i])
            if z.min() <= val <= z.max():
                out.append(float(val))
                weights.append(float(step))
    if not out:
        return None, []
    o = np.argsort(out)
    cr = np.array(out)[o]
    wt = np.cumsum(np.array(weights)[o]); wt /= wt[-1]
    return float(cr[int(np.searchsorted(wt, 0.5))]), out


def asym_edge_auto(passes, hw0=0.015, k=2.0, snr_floor=1.0, iters=6,
                   min_above=5, min_below=3, tol=1e-4, report='cross'):
    """asym_edge with the truncation derived from the data, not hand-picked.

    asym_edge's exclusion halfwidth was a free constant, and the absolute
    answer drifted 5-8um across reasonable values of it (diverging at 40um) -
    a systematic of the same size as the estimator's precision.  That is the
    price of escaping the line/asymptote tangency by truncating: the answer
    depends on WHERE you truncate.

    So let the sigmoid choose it: fit, set the window to k times the fitted
    width w, refit, iterate to convergence.  The window then comes from the
    transition's own scale.  Clamped to [1 sample step, span/5] so a
    degenerate sub-sample fit cannot starve the window or swallow the profile.
    """
    seed, _ = simple_midpoint_edge(passes, snr_floor=snr_floor)
    if seed is None:
        return None, [], []
    out, weights, hws = [], [], []
    for z, a in passes:
        n = len(z)
        m = max(3, int(n * 0.25))
        span = float(z.max() - z.min())
        step_z = span / max(n - 1, 1)
        for kk in range(a.shape[1]):
            y = np.log(np.maximum(a[:, kk], 1.0))
            con, air = np.median(y[:m]), np.median(y[-m:])
            stp = abs(air - con)
            if stp < 1e-6 or stp / (np.mean(np.abs(y[-m:] - air)) + 1e-9) < snr_floor:
                continue
            hw, g, val = hw0, seed, None
            for _ in range(iters):
                above = z > g + hw
                below = z <= g + hw
                if above.sum() < min_above or below.sum() < min_below:
                    break
                ca = np.polyfit(z[above], y[above], 1)
                f = fit_logistic(z[below], y[below],
                                 w_grid=np.geomspace(step_z, span / 4., 14))
                if f is None:
                    break
                zc, w, top, floor = f[0], f[1], f[2], f[3]
                grid = np.linspace(z.min(), z.max(), 400)
                sig = floor + (top - floor) / (
                    1.0 + np.exp(-np.clip((grid - zc) / w, -50., 50.)))
                d = sig - np.polyval(ca, grid)
                s = np.nonzero(np.diff(np.sign(d)))[0]
                if not len(s):
                    break
                newval = float(grid[s[np.argmin(np.abs(grid[s] - g))]])
                new_hw = float(min(max(k * w, step_z), span / 5.))
                done = (abs(new_hw - hw) < tol and val is not None
                        and abs(newval - val) < tol)
                hw, g, val = new_hw, newval, (zc if report == 'mid' else newval)
                if done:
                    break
            if val is not None and z.min() <= val <= z.max():
                out.append(float(val)); weights.append(float(stp)); hws.append(hw)
    if not out:
        return None, [], []
    o = np.argsort(out)
    cr = np.array(out)[o]
    wt = np.cumsum(np.array(weights)[o]); wt /= wt[-1]
    return float(cr[int(np.searchsorted(wt, 0.5))]), out, hws


def flank_edge(passes, lo_frac=0.20, hi_frac=0.80, air_frac=0.92,
               snr_floor=1.0, min_flank=3, min_air=5, report_offset=False,
               anchor_win=0.060):
    """Cross the sigmoid's STEEP LINEAR FLANK with the air line.

    The sigmoid's shape parameters are under-determined at this sampling (the
    transition is ~5-20um against 11.1um samples), which is why fitting one -
    or deriving a truncation window from one - keeps leaking the window choice
    into the answer.  But its MIDDLE section is very straight even at low
    resolution (user), so fit that as a line and never estimate w or top at
    all.

    Selection is by AMPLITUDE, not by z: take samples whose level lies between
    lo_frac and hi_frac of the step.  That picks the flank regardless of how
    sharp the transition is, and automatically excludes both the saturated
    tail (which curves, and which wrecked the earlier line/line fit's slope:
    1.97 +- 4.33 log/mm) and the knee near air.

    The crossing lands INSIDE the sigmoid's curvature, so it reads slightly
    low vs true first touch - for a logistic the midpoint tangent meets the
    upper asymptote at z0 + 2w exactly.  That is a fixed offset when w is
    stable, so consistency matters more than absolute placement here.
    """
    seed, _ = simple_midpoint_edge(passes, snr_floor=snr_floor)
    out, weights, slopes = [], [], []
    for z, a in passes:
        n = len(z)
        m = max(3, int(n * 0.25))
        for k in range(a.shape[1]):
            y = np.log(np.maximum(a[:, k], 1.0))
            con, air = np.median(y[:m]), np.median(y[-m:])
            step = air - con
            if abs(step) < 1e-6:
                continue
            if abs(step) / (np.mean(np.abs(y[-m:] - air)) + 1e-9) < snr_floor:
                continue
            lo = con + lo_frac * step
            hi = con + hi_frac * step
            inband = (y - lo) * (y - hi) <= 0
            # CONTIGUOUS run only.  Selecting by value alone pulls in noisy air
            # samples that happen to dip into the band from anywhere in z, so
            # the "flank" line gets fitted to a scatter spanning the whole
            # profile - measured 67.88um mean sd, worse than shipped, with the
            # answer swinging 0.12-0.34mm as the band changed.  Anchor on the
            # steepest adjacent pair and grow outward while still in band.
            if inband.sum() < 1:
                continue
            d = np.abs(np.diff(y)) / np.maximum(np.abs(np.diff(z)), 1e-9)
            # Anchor the flank NEAR THE SEED, not at the globally steepest
            # pair: where air is noisy the global maximum can be a noise spike
            # far from the transition.  Measured at (60,60), whose passes sit
            # in three different amplitude bands, the global anchor gave
            # 97.07um against ~7um at the two well-behaved points.
            if seed is not None:
                near = np.abs(0.5 * (z[:-1] + z[1:]) - seed) <= anchor_win
                c0 = int(np.argmax(np.where(near, d, -np.inf))) if near.any() \
                    else int(np.argmax(d))
            else:
                c0 = int(np.argmax(d))
            idx = [c0, c0 + 1]
            i = c0 - 1
            while i >= 0 and inband[i]:
                idx.insert(0, i); i -= 1
            j = c0 + 2
            while j < len(y) and inband[j]:
                idx.append(j); j += 1
            band = np.zeros(len(y), bool); band[idx] = True
            airm = ((y - (con + air_frac * step)) * np.sign(step) >= 0)
            airm &= (z > z[max(idx)])              # air must lie ABOVE the flank
            if band.sum() < min_flank or airm.sum() < min_air:
                continue
            cf = np.polyfit(z[band], y[band], 1)   # steep flank
            ca = np.polyfit(z[airm], y[airm], 1)   # air
            if abs(cf[0] - ca[0]) < 1e-9:
                continue
            zc = float((ca[1] - cf[1]) / (cf[0] - ca[0]))
            if not (z.min() <= zc <= z.max()):
                continue
            out.append(zc); weights.append(abs(step)); slopes.append(cf[0])
    if not out:
        return (None, [], []) if report_offset else (None, [])
    o = np.argsort(out)
    cr = np.array(out)[o]
    wt = np.cumsum(np.array(weights)[o]); wt /= wt[-1]
    est = float(cr[int(np.searchsorted(wt, 0.5))])
    return (est, out, slopes) if report_offset else (est, out)


def _flank_extent(tol):
    """Half-extent of the logistic's straight section, in units of w.

    The sigmoid tells us not just WHERE the transition is but WHICH PART of it
    is straight enough to fit a line to (user).  A logistic has zero curvature
    exactly at z0, and its deviation from the midpoint tangent depends only on
    u = (z-z0)/w - so a stated linearity tolerance fixes the extent with no
    free multiplier.  dev <= 2% of the step gives |u| <= 1.0; 5% gives 1.45.
    The hand-tuned c_flank=1.5 that measured best corresponds to ~5.7%, i.e.
    the tuning was recovering a principled value.
    """
    u = np.linspace(0., 3., 3001)
    dev = np.abs(1. / (1. + np.exp(-u)) - (0.5 + u / 4.))
    i = int(np.searchsorted(dev, tol))
    return float(u[min(i, len(u) - 1)])


def flank_edge_v2(passes, c_flank=None, c_air=3.0, snr_floor=1.0,
                  min_flank=3, min_air=4, want_detail=False, lin_tol=0.05):
    """Locate with a sigmoid, measure with a line.

    Order matters here.  Earlier attempts anchored the flank on the steepest
    adjacent pair near a mid-profile seed, which fails exactly where the middle
    is chaotic - (60,60) cost 74-97um that way.  So:

      1. Establish the contact FLOOR from the leftmost (deepest) samples.
         Those are unambiguously in contact: the press bottoms out there and
         the amplitude is at its floor.  This is the one anchor on the profile
         that does not depend on finding the transition first.
      2. Fit a sigmoid to LOCATE the transition and VALIDATE it - is there a
         real step, is its width physical, does it sit inside the profile?
         The fit need not pin down w precisely to answer those.
      3. Use the fitted z0 and w to choose WHICH samples are the straight
         flank (|z - z0| <= c_flank*w) and which are clean air (z > z0 +
         c_air*w) - levels taken from the FITTED asymptotes, not from medians
         of the profile ends.
      4. Fit lines to those two sets and cross them.  Only the line fits carry
         the measurement, so the sigmoid's under-determined shape parameters
         never enter the answer.
    """
    out, weights, detail = [], [], []
    for z, a in passes:
        n = len(z)
        m = max(3, int(n * 0.2))
        span = float(z.max() - z.min())
        step_z = span / max(n - 1, 1)
        for k in range(a.shape[1]):
            y = np.log(np.maximum(a[:, k], 1.0))
            floor = np.median(y[:m])               # deepest = certain contact
            air0 = np.median(y[-m:])
            step = air0 - floor
            noise = np.mean(np.abs(y[-m:] - air0)) + 1e-9
            if abs(step) < 1e-6 or abs(step) / noise < snr_floor:
                continue
            f = fit_logistic(z, y,
                             w_grid=np.geomspace(step_z, span / 5., 16))
            if f is None:
                continue
            z0, w, top, fl, rms = f
            # --- validation: is this the transition, or a fitted artifact?
            if not (z.min() + step_z < z0 < z.max() - step_z):
                continue
            if abs(top - fl) < 0.5 * abs(step):    # fitted step too small
                continue
            if rms > 0.6 * abs(step):              # fit explains too little
                continue
            cf = _flank_extent(lin_tol) if c_flank is None else c_flank
            flank = np.abs(z - z0) <= cf * w
            airm = z > z0 + c_air * w
            if flank.sum() < min_flank or airm.sum() < min_air:
                continue
            cf = np.polyfit(z[flank], y[flank], 1)
            ca = np.polyfit(z[airm], y[airm], 1)
            if abs(cf[0] - ca[0]) < 1e-9:
                continue
            zc = float((ca[1] - cf[1]) / (cf[0] - ca[0]))
            if not (z.min() <= zc <= z.max()):
                continue
            out.append(zc); weights.append(abs(step))
            detail.append((z0, w, rms / abs(step)))
    if not out:
        return (None, [], []) if want_detail else (None, [])
    o = np.argsort(out)
    cr = np.array(out)[o]
    wt = np.cumsum(np.array(weights)[o]); wt /= wt[-1]
    est = float(cr[int(np.searchsorted(wt, 0.5))])
    return (est, out, detail) if want_detail else (est, out)
