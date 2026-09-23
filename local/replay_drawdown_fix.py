#!/usr/bin/env python3
# Can drawdown's mid-air false halts be cut without losing real contacts?
#
# 2026-09-15, gradient test off: at (80,85) two drawdown false halts fired just
# after drawdown's 40-window warm-up (63 and 85 windows into the descent, no
# real drop), and each re-armed descent then MISSED its halt because the
# corroboration floor (prior contact - 0.05mm) left too little travel below the
# surface.  Fixing the early false halt removes the trigger for both.
#
# Labels come from the verify capture that follows each live descent (paired by
# mtime, <= 60s): confirmed near the bed = REAL, rejected or confirmed above
# z_cand 0.5mm = FALSE.  A live trace ends at its trigger, so for a variant:
#   FALSE trace: fires anywhere in the trace -> still false; else avoided
#   REAL trace:  fires in the last DETECT_ZONE -> caught; fires above that ->
#                EARLY; never fires -> UNKNOWN (no data past the live trigger,
#                so it may be later or a miss - counted as risk)
# Disarmed traces (halt_floor=1.0) run past contact with detectors off, so they
# also give depth on the labelled overshoots and false alarms on pure air.
#
# The detector mirrors _HostResonanceEndstop exactly: one ContactDetector per
# armed axis, created after WARM windows of air, threshold max(SENS,
# NSIGMA * estimate_noise(air)/mean(air)), relative drawdown, persistence and
# lookback in windows (live: 2 and 6), any axis may trigger.
import glob, os, sys, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'klippy', 'extras'))
import analog_contact

CORPUS = os.path.join(HERE, '..', 'probe_traces')
DETECT_ZONE = 0.15   # mm above the live trigger that still counts as the bed
PAIR_GAP = 60.       # s between a descent and its verify capture
AIR_Z = 0.5          # verify z_cand above this is a contact in mid-air
EARLY_WIN = 60       # a false halt within this many windows of arming is "early"
LIVE = (40, 0.10, 8., 2)   # warm, sens, nsigma, persist (windows)
LOOKBACK = 6               # windows (0.06mm / 0.01mm)

LABELLED = {140: -0.001267, 141: -0.001267, 143: -0.010620, 144: -0.010620,
            146: 0.011829, 147: 0.011829, 149: -0.000344, 150: -0.000344,
            173: 0.002519, 174: 0.002519, 180: -0.017753, 181: -0.017753,
            189: -0.006817, 190: -0.006817, 196: -0.003029, 197: -0.003029}


def parse(path, data=True):
    hdr, zs, amps = {}, [], []
    with open(path, errors='replace') as fh:
        for ln in fh:
            if ln.startswith('#'):
                for kv in ln[1:].split():
                    if '=' in kv:
                        k, v = kv.split('=', 1)
                        hdr[k] = v
            elif data:
                f = ln.strip().split(',')
                if len(f) == 4 and f[0] != 'mm_below_arm':
                    zs.append(float(f[0]))
                    amps.append([float(x) for x in f[1:]])
    return hdr, zs, amps


def armed_axes(hdr):
    floors = hdr.get('halt_floor', '')
    try:
        vals = [float(v) for v in floors.split(',')]
    except ValueError:
        return [0, 1, 2]
    if len(vals) != 3 or all(v >= 0.95 for v in vals):
        return [0, 1, 2]       # disarmed trace: replay as if all armed
    return [i for i, v in enumerate(vals) if v < 0.95]


def fire(zs, amps, axes, warm, sens, nsigma, persist):
    """(depth, window index) of the first trigger on any axis, or None."""
    best = None
    if len(zs) <= warm + 1:
        return None
    for ax in axes:
        air = [a[ax] for a in amps[:warm]]
        base = sum(air) / len(air)
        if base <= 1e-9:
            continue
        sd = analog_contact.estimate_noise(air) / base
        det = analog_contact.ContactDetector(
            analog_contact.DRAWDOWN, max(sens, nsigma * sd), 1., 1.,
            persist=persist, relative=True, lookback_mm=LOOKBACK)
        for i in range(warm, len(zs)):
            if det.update(amps[i][ax], zs[i]):
                if best is None or i < best[1]:
                    best = (zs[i], i)
                break
    return best


def load_live():
    events = []
    for p in glob.glob(os.path.join(CORPUS, 'descent*.csv')):
        hdr, zs, amps = parse(p)
        if hdr.get('trigger_kind') != 'drawdown' or len(zs) < 12:
            continue
        events.append((os.path.getmtime(p), 'D', (hdr, zs, amps)))
    for p in glob.glob(os.path.join(CORPUS, 'verify*.csv')):
        hdr, _z, _a = parse(p, data=False)
        if 'outcome' in hdr and 'z_cand' in hdr:
            events.append((os.path.getmtime(p), 'V', hdr))
    events.sort(key=lambda e: e[0])
    real, false = [], []
    for i, (t, kind, d) in enumerate(events):
        if kind != 'D':
            continue
        nxt = events[i + 1] if i + 1 < len(events) else None
        if not nxt or nxt[1] != 'V' or nxt[0] - t > PAIR_GAP:
            continue
        v = nxt[2]
        if v['outcome'] == 'confirmed' and float(v['z_cand']) <= AIR_Z:
            real.append(d)
        else:
            false.append(d)
    return real, false


def load_disarmed():
    lab, air = [], []
    for p in sorted(glob.glob(os.path.join(CORPUS, 'descent*.csv'))):
        hdr, zs, amps = parse(p)
        if not hdr.get('halt_floor', '').startswith('1.0000') or len(zs) < 90:
            continue
        tr = int(os.path.basename(p)[7:12])
        if tr in LABELLED:
            lab.append((2.0 - LABELLED[tr], zs, amps))
            continue
        pk, touched = [0.] * 3, False
        for a in amps:
            for ax in range(3):
                pk[ax] = max(pk[ax], a[ax])
                if pk[ax] > 0 and a[ax] <= 0.5 * pk[ax]:
                    touched = True
        if not touched:
            air.append((zs, amps))
    return lab, air


def main():
    real, false = load_live()
    lab, air = load_disarmed()
    n_early_false = 0
    print("live drawdown descents paired with verify: %d real, %d false; "
          "disarmed: %d labelled, %d pure-air" % (len(real), len(false),
                                                  len(lab), len(air)))
    grid = list(itertools.product((40, 80), (0.10, 0.15), (8., 12.),
                                  (2, 4, 6)))
    grid.remove(LIVE)
    grid.insert(0, LIVE)
    print("\n%-26s | %-22s | %-30s | %-24s | %s"
          % ("warm sens nsig persist", "false halts still fire",
             "real: caught / EARLY / unknown", "labelled: found/early", "air FA"))
    for warm, sens, nsig, pers in grid:
        still, still_early = 0, 0
        for hdr, zs, amps in false:
            f = fire(zs, amps, armed_axes(hdr), warm, sens, nsig, pers)
            if f is not None:
                still += 1
                if f[1] <= EARLY_WIN:
                    still_early += 1
        caught = early = unknown = 0
        for hdr, zs, amps in real:
            f = fire(zs, amps, armed_axes(hdr), warm, sens, nsig, pers)
            if f is None:
                unknown += 1
            elif f[0] < zs[-1] - DETECT_ZONE:
                early += 1
            else:
                caught += 1
        found = lab_early = 0
        depths = []
        for truth, zs, amps in lab:
            f = fire(zs, amps, [0, 1, 2], warm, sens, nsig, pers)
            if f is None:
                continue
            if f[0] < truth - DETECT_ZONE:
                lab_early += 1
            else:
                found += 1
                depths.append((f[0] - truth) * 1000.)
        fa = sum(1 for zs, amps in air
                 if fire(zs, amps, [0, 1, 2], warm, sens, nsig, pers) is not None)
        dmed = sorted(depths)[len(depths) // 2] if depths else float('nan')
        tag = "LIVE " if (warm, sens, nsig, pers) == LIVE else "     "
        print("%s%3d %.2f %4.0f %2d          | %4d/%-4d (early<=%d: %3d) | %4d / %3d / %3d (of %4d)    | %2d/%-2d early %d med %+4.0fum | %3d/%d (%.0f%%)"
              % (tag, warm, sens, nsig, pers, still, len(false), EARLY_WIN,
                 still_early, caught, early, unknown, len(real), found,
                 len(lab), lab_early, dmed, fa, len(air),
                 100. * fa / max(len(air), 1)))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
