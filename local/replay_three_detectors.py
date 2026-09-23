#!/usr/bin/env python3
# Would the probe be better off without the GRADIENT (threshold) halt test,
# relying on the derivative and drawdown tests alone?
#
# Live, the three tests race and the first to fire truncates the descent, so an
# armed trace cannot show what the others would have done.  Disarmed traces
# (halt_floor=1.0) run past contact with every test off, so all three can be
# replayed over the identical window stream:
#   - labelled overshoot traces (contact Z known, see LABELS.md): detection
#     depth, early fires (above truth - DETECT_ZONE) and misses
#   - never-touched disarmed descents: false alarms over the whole descent
#
# Each detector mirrors resonance_probe.py (_HostResonanceEndstop._handle_batch):
#   gradient    median of windows within smooth_z vs median of windows ref_z
#               further up; drop >= floor for 2 consecutive windows; the halt is
#               reported at the reference height (first_below_t = ref_t)
#               smooth_z = max(step_z, 3*cfg_step_z), ref_z = max(confirm_z,
#               2*win_z), win_z = win_n / SPS * speed
#   derivative  compare_detectors.fire_derivative (2 consecutive down-steps)
#   drawdown    compare_detectors.fire_drawdown (armed after 40 windows)
# Floors are compare_detectors' live 172.9 Hz values, since disarmed traces
# carry 1.0 by construction.
import os, sys, glob, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_detectors as cd

SPS = 3200.          # ADXL345 sample rate used live
CONFIRM_Z = 0.05     # detect_confirm_z
CFG_STEP_Z = 0.01    # detect_step_z


def fire_gradient(hdr, zs, amps, floors=cd.GRAD_FLOOR):
    speed = float(hdr.get('speed', 0.2))
    win_n = float(hdr.get('win_n', 148))
    step_z = float(hdr.get('step_z', CFG_STEP_Z))
    win_z = win_n / SPS * speed
    smooth_z = max(step_z, 3. * CFG_STEP_Z)
    ref_z = max(CONFIRM_Z, 2. * win_z)
    first = None
    for ax in range(3):
        run, first_below = 0, None
        for i, d in enumerate(zs):
            if d - zs[0] < ref_z + smooth_z:
                continue
            cur = [amps[j][ax] for j in range(i + 1) if zs[j] >= d - smooth_z]
            ref = [amps[j][ax] for j in range(i + 1)
                   if d - ref_z - smooth_z <= zs[j] <= d - ref_z]
            if not cur or not ref:
                continue
            r = statistics.median(ref)
            drop = (r - statistics.median(cur)) / r if r > 1e-9 else 0.
            if drop >= floors[ax]:
                if run == 0:
                    first_below = d - ref_z
                run += 1
                if run >= 2:
                    first = first_below if first is None else min(first,
                                                                  first_below)
                    break
            else:
                run = 0
    return first


def combine(*fires):
    hits = [f for f in fires if f is not None]
    return min(hits) if hits else None


def main(corpus):
    labelled = dict(cd.__dict__.get('LABELLED_OVERRIDE', {}))
    if not labelled:
        for tr, cz in ((140, -0.001267), (141, -0.001267), (143, -0.010620),
                       (144, -0.010620), (146, 0.011829), (147, 0.011829),
                       (149, -0.000344), (150, -0.000344),
                       (173, 0.002519), (174, 0.002519),
                       (180, -0.017753), (181, -0.017753),
                       (189, -0.006817), (190, -0.006817),
                       (196, -0.003029), (197, -0.003029)):
            labelled[tr] = cz
    names = ('gradient', 'derivative', 'drawdown',
             'deriv+drawdown (no gradient)', 'all three (live)')

    def fires(hdr, zs, amps):
        g = fire_gradient(hdr, zs, amps)
        v = cd.fire_derivative(zs, amps)
        w = cd.fire_drawdown(zs, amps)
        return (g, v, w, combine(v, w), combine(g, v, w))

    print("=== labelled overshoot descents: contact known ===")
    stats = {n: {'depth': [], 'early': 0, 'miss': 0} for n in names}
    n_lab = 0
    for tr in sorted(labelled):
        p = os.path.join(corpus, "descent%05d.csv" % tr)
        if not os.path.exists(p):
            continue
        n_lab += 1
        hdr, zs, amps = cd.load(p)
        truth = 2.0 - labelled[tr]
        for name, f in zip(names, fires(hdr, zs, amps)):
            s = stats[name]
            if f is None:
                s['miss'] += 1
            elif f < truth - cd.DETECT_ZONE:
                s['early'] += 1
            else:
                s['depth'].append((f - truth) * 1000.)
    for name in names:
        s = stats[name]
        d = s['depth']
        dd = ("%+.0f..%+.0fum med %+.0f" % (min(d), max(d), sorted(d)[len(d) // 2])
              if d else "-")
        print("  %-30s detected %2d/%2d  depth %-24s early %d  miss %d"
              % (name, len(d), n_lab, dd, s['early'], s['miss']))

    print("\n=== never-touched disarmed descents: any fire is a false alarm ===")
    air = []
    for p in sorted(glob.glob(os.path.join(corpus, 'descent*.csv'))):
        tr = int(os.path.basename(p)[7:12])
        if tr in labelled:
            continue
        hdr, zs, amps = cd.load(p)
        if len(zs) < cd.WARM + 5 or not hdr.get('halt_floor', '').startswith('1.0000'):
            continue
        pk, touched = [0.] * 3, False
        for a in amps:
            for ax in range(3):
                pk[ax] = max(pk[ax], a[ax])
                if pk[ax] > 0 and a[ax] <= 0.5 * pk[ax]:
                    touched = True
        if not touched:
            air.append((hdr, zs, amps))
    fa = dict((n, 0) for n in names)
    for hdr, zs, amps in air:
        for name, f in zip(names, fires(hdr, zs, amps)):
            if f is not None:
                fa[name] += 1
    print("  n=%d pure-air descents" % len(air))
    for name in names:
        print("  %-30s false alarms %3d (%.0f%%)"
              % (name, fa[name], 100. * fa[name] / max(len(air), 1)))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else cd._TRACES)
