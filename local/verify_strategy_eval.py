#!/usr/bin/env python3
# Which verify estimator should the probe report: the DOWN ramp, the UP ramp,
# or their mean?
#
# Answered offline from saved verify traces.  Each trace holds every ramp window
# of one verify (z, amplitude, tag, rep), so all three strategies can be scored
# on the IDENTICAL physical probes - one hardware run instead of one per
# candidate, which matters because each candidate would otherwise cost a fresh
# set of contacts on a plate that wears.
#
# The number that decides it is probe-to-probe repeatability at one point, NOT
# the within-verify spread across reps: reps share a single approach, so they
# understate the error a real probe sequence sees.
import os, sys, glob, math, collections
import os as _os
_TRACES = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                       "..", "probe_traces")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'klippy', 'extras'))


def load(path):
    hdr, rows = {}, []
    for ln in open(path):
        if ln.startswith('#'):
            for kv in ln[1:].split():
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    hdr[k] = v
        elif ',' in ln and not ln.startswith('z,'):
            z, amp, tag, rep = ln.split(',')
            rows.append((float(z), float(amp), tag.strip(), int(rep)))
    return hdr, rows


def ramp_edge(zs, amps, descending, air_amp, contact_amp):
    """Mirror of resonance_probe.HaltingContactProbe._ramp_edge, without numpy."""
    order = sorted(range(len(zs)), key=lambda i: -zs[i] if descending else zs[i])
    dz = [zs[i] for i in order]
    da = [amps[i] for i in order]
    if len(da) < 3:
        return None
    steps = [(da[i + 1] - da[i]) / max(da[i], 1e-9) for i in range(len(da) - 1)]
    edge = None
    if len(steps) >= 3:
        worst = min(steps) if descending else max(steps)
        found = (worst < 0.) if descending else (worst > 0.)
        if found:
            j = steps.index(worst)
            edge = 0.5 * (dz[j] + dz[j + 1])
    if air_amp > contact_amp:
        mid = 0.5 * (air_amp + contact_amp)
        for j in range(1, len(da)):
            prev, cur = da[j - 1], da[j]
            if descending:
                if not (prev >= mid >= cur):
                    continue
                frac = (prev - mid) / max(prev - cur, 1e-9)
            else:
                if not (prev <= mid <= cur):
                    continue
                frac = (mid - prev) / max(cur - prev, 1e-9)
            edge = dz[j - 1] + (dz[j] - dz[j - 1]) * frac
            break
    return edge


def estimates(path):
    """-> (down_median, up_median) for one verify trace."""
    hdr, rows = load(path)
    air = float(hdr.get('air_amp', 0.))
    contact = float(hdr.get('contact_amp', 0.))
    reps = int(hdr.get('reps', 1))
    # A REJECTED false halt also runs verify, and its ramps never touched
    # anything - both ends read air.  Scoring an estimator on those would be
    # scoring it on noise.  The trace records how it was judged, so use the
    # verdict rather than re-deriving it; traces written before outcome tagging
    # existed fall back to the amplitude ratio.
    outcome = hdr.get('outcome')
    if outcome is None:
        outcome = 'confirmed' if (air > 0. and contact <= 0.7 * air) \
            else 'rejected'
    if outcome != 'confirmed':
        return None, None, 0, 0
    downs, ups = [], []
    for r in range(reps):
        for tag, bucket in (('down', downs), ('up', ups)):
            sel = [(z, a) for z, a, t, rr in rows if t == tag and rr == r]
            if len(sel) < 3:
                continue
            e = ramp_edge([s[0] for s in sel], [s[1] for s in sel],
                          tag == 'down', air, contact)
            if e is not None:
                bucket.append(e)
    # Must match numpy's median exactly - the live code uses np.median, and with
    # reps=2 the two disagree by the whole spread between the pair (up to ~10um
    # on a down ramp), which showed up as offline biases the machine never
    # reported.
    def med(v):
        if not v:
            return None
        s = sorted(v)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return med(downs), med(ups), len(downs), len(ups)


def raw_edges(path):
    """-> (down_edges, up_edges) for one CONFIRMED verify, unaggregated."""
    hdr, rows = load(path)
    air = float(hdr.get('air_amp', 0.))
    contact = float(hdr.get('contact_amp', 0.))
    reps = int(hdr.get('reps', 1))
    outcome = hdr.get('outcome')
    if outcome is None:
        outcome = 'confirmed' if (air > 0. and contact <= 0.7 * air) \
            else 'rejected'
    if outcome != 'confirmed':
        return [], []
    downs, ups = [], []
    for r in range(reps):
        for tag, bucket in (('down', downs), ('up', ups)):
            sel = [(z, a) for z, a, t, rr in rows if t == tag and rr == r]
            if len(sel) < 3:
                continue
            e = ramp_edge([s[0] for s in sel], [s[1] for s in sel],
                          tag == 'down', air, contact)
            if e is not None:
                bucket.append(e)
    return downs, ups


def mean(v):
    return sum(v) / len(v) if v else None


def median(v):
    if not v:
        return None
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def sd(vals):
    if len(vals) < 2:
        return float('nan')
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def main(corpus, group_by_point=True):
    paths = sorted(glob.glob(os.path.join(corpus, 'verify*.csv')))
    if not paths:
        print("no verify traces in %s - is trace_dir set and the probe rebuilt?"
              % corpus)
        return
    # Group probes that belong to the same point.  z_cand is per-probe, so
    # points are inferred from a run manifest if present, else all together.
    groups = collections.defaultdict(list)
    manifest = os.path.join(corpus, 'VERIFY_RUNS.txt')
    label_of = {}
    if os.path.exists(manifest):
        for ln in open(manifest):
            if ',' in ln:
                fname, label = ln.strip().split(',', 1)
                label_of[fname] = label
    rejected = []
    for p in paths:
        hdr, _ = load(p)
        d, u, nd, nu = estimates(p)
        if d is None or u is None:
            rejected.append((os.path.basename(p), hdr))
            continue
        groups[label_of.get(os.path.basename(p), 'all')].append((d, u))
    if rejected:
        # Kept, not discarded: these are recordings of the signal around a false
        # halt, the only direct evidence of why one fired.
        print("rejected/unusable verifies (excluded from scoring, kept for"
              " false-halt analysis): %d" % len(rejected))
        for name, hdr in rejected:
            air = float(hdr.get('air_amp', 0.))
            con = float(hdr.get('contact_amp', 0.))
            print("  %-18s z_cand=%-9s air=%-9.0f contact=%-9.0f drop=%5.1f%%"
                  "  %s" % (name, hdr.get('z_cand', '?'), air, con,
                            100. * (1. - con / air) if air > 0 else 0.,
                            hdr.get('outcome', 'untagged')))
        print()
    print("%-16s %4s | %-22s %-22s %-22s"
          % ("point", "n", "down-only", "up-only", "mean(down,up)"))
    tot = {'down': [], 'up': [], 'mean': []}
    for label, vals in sorted(groups.items()):
        if len(vals) < 2:
            continue
        downs = [v[0] for v in vals]
        ups = [v[1] for v in vals]
        means = [0.5 * (v[0] + v[1]) for v in vals]
        print("%-16s %4d | sd %6.2fum rng %5.1f  sd %6.2fum rng %5.1f  "
              "sd %6.2fum rng %5.1f"
              % (label, len(vals),
                 sd(downs) * 1000., (max(downs) - min(downs)) * 1000.,
                 sd(ups) * 1000., (max(ups) - min(ups)) * 1000.,
                 sd(means) * 1000., (max(means) - min(means)) * 1000.))
        # Pooled: each point contributes its deviations from its own mean, so
        # bed height differences between points do not inflate the result.
        for k, v in (('down', downs), ('up', ups), ('mean', means)):
            m = sum(v) / len(v)
            tot[k].extend(x - m for x in v)
    if len(tot['down']) > 1:
        print("\npooled within-point sd (the number that decides it):")
        for k in ('down', 'up', 'mean'):
            print("  %-14s %6.2f um   (n=%d)"
                  % (k, sd(tot[k]) * 1000., len(tot[k])))
        bias = [u - d for vals in groups.values() for d, u in vals]
        print("\nup-vs-down bias: mean %+.2fum, range %+.2f..%+.2fum"
              % (1000. * sum(bias) / len(bias), 1000. * min(bias),
                 1000. * max(bias)))
    aggregation_report(paths, label_of)


def aggregation_report(paths, label_of):
    """MEAN vs MEDIAN when combining the individual ramp estimates.

    Median only differs from mean once there are >=3 values to combine: with
    reps=2 the median of a two-element list IS its mean, so the per-direction
    aggregation cannot distinguish them at all.  Pooling all four ramps
    (2 down + 2 up) is the only comparison this data supports.
    """
    print("\n=== aggregating the individual ramp estimates ===")
    per_point = collections.defaultdict(list)
    spreads = []
    n_ramps = collections.Counter()
    for p in paths:
        downs, ups = raw_edges(p)
        if not downs or not ups:
            continue
        allv = downs + ups
        n_ramps[len(allv)] += 1
        spreads.append((max(allv) - min(allv), os.path.basename(p), allv))
        per_point[label_of.get(os.path.basename(p), 'all')].append({
            'down_med': median(downs), 'up_med': median(ups),
            'all_mean': mean(allv), 'all_med': median(allv),
        })
    if not spreads:
        print("  no confirmed verify traces")
        return
    print("  ramp estimates per probe: %s"
          % dict(sorted(n_ramps.items())))
    strategies = ('down_med', 'up_med', 'all_mean', 'all_med')
    pooled = {k: [] for k in strategies}
    for label, probes in sorted(per_point.items()):
        if len(probes) < 2:
            continue
        for k in strategies:
            v = [pr[k] for pr in probes]
            m = sum(v) / len(v)
            pooled[k].extend(x - m for x in v)
    print("\n  pooled within-point sd:")
    for k in strategies:
        print("    %-10s %6.2f um  (n=%d)"
              % (k, sd(pooled[k]) * 1000., len(pooled[k])))
    # Outliers are the whole reason median might win, so show whether any exist.
    spreads.sort(reverse=True)
    print("\n  widest spread between a probe's own ramp estimates:")
    for s, name, allv in spreads[:5]:
        print("    %-18s spread %6.1fum   %s" % (name, s * 1000.,
              " ".join("%+.4f" % v for v in allv)))
    worst = [s for s, _, _ in spreads]
    print("    median spread %.1fum, max %.1fum"
          % (1000. * median(worst), 1000. * max(worst)))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else _TRACES)
