#!/usr/bin/env python3
# Replay the ramp-minimum contact statistic against every saved verify capture.
#
# The change: judge contact on the deepest damping the ramp reached, instead of
# on the dwell at the bottom of the ramp.  The dwell sits at z_cand-VERIFY_DOWN,
# which can be well past the damping minimum, where amplitude has climbed back.
#
# The risk is the mirror image of the bug: a minimum is biased low, so it could
# invent drops in air and turn genuine FALSE halts into confirmations.  That is
# what this checks, and it is why the statistic is a moving median rather than a
# bare minimum.
#
# Halts high above the bed (z_cand > HIGH_Z) are near-certainly false - the bed
# is nowhere near there - so any of those that flip to CONFIRMED is a regression,
# not a rescue.
import glob
import os
import sys

THRESH = 0.15                 # VERIFY_DROP
HIGH_Z = 0.6                  # above this, a halt cannot be real contact
K = 5                         # moving-median width


def ramp_min(pairs, k=K):
    pairs = sorted(pairs)
    if not pairs:
        return None
    if len(pairs) < k:
        return min(a for _, a in pairs)
    best = None
    for i in range(len(pairs) - k + 1):
        w = sorted(a for _, a in pairs[i:i + k])
        m = w[k // 2]
        if best is None or m < best:
            best = m
    return best


def load(path):
    hdr, rows = {}, []
    for line in open(path):
        if line.startswith('#'):
            for tok in line[1:].split():
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    hdr[k] = v
            continue
        if line.startswith('z,'):
            continue
        p = line.strip().split(',')
        if len(p) >= 4:
            try:
                rows.append((float(p[0]), float(p[1]), p[2], int(p[3])))
            except ValueError:
                pass
    return hdr, rows


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "probe_traces"
    tally = {}
    regressions, rescues = [], []
    for path in sorted(glob.glob(os.path.join(root, "verify*.csv"))):
        hdr, rows = load(path)
        air = [r[1] for r in rows if r[2] == 'air']
        dwell = [r[1] for r in rows if r[2] == 'contact']
        ramp = [(r[0], r[1]) for r in rows
                if r[2] in ('down', 'contact', 'up')]
        if len(air) < 2 or not dwell or not ramp:
            continue
        a = sorted(air)[len(air) // 2]
        d = sorted(dwell)[len(dwell) // 2]
        m = ramp_min(ramp)
        if a <= 1e-9 or m is None:
            continue
        try:
            zc = float(hdr.get('z_cand', 'nan'))
        except ValueError:
            continue
        old_ok = (a - d) / a >= THRESH
        new_ok = (a - m) / a >= THRESH
        key = ("CONFIRM" if old_ok else "reject",
               "CONFIRM" if new_ok else "reject")
        tally[key] = tally.get(key, 0) + 1
        if not old_ok and new_ok:
            (regressions if zc > HIGH_Z else rescues).append(
                (os.path.basename(path), zc, (a - d) / a, (a - m) / a))
        if old_ok and not new_ok:
            regressions.append((os.path.basename(path), zc,
                                (a - d) / a, (a - m) / a))

    total = sum(tally.values())
    print("verify captures replayed: %d\n" % total)
    print("%-22s %s" % ("dwell -> ramp-min", "count"))
    for k in sorted(tally):
        print("  %-20s %d" % ("%s -> %s" % k, tally[k]))

    print("\n--- RESCUED (was rejected, now confirmed, at a plausible height) ---")
    for name, zc, o, n in sorted(rescues, key=lambda r: -r[3])[:15]:
        print("  %-16s z_cand=%+.4f  dwell %+.0f%%  ramp-min %+.0f%%"
              % (name, zc, o * 100., n * 100.))
    print("  %d total" % len(rescues))

    print("\n--- REGRESSIONS (a high/false halt now confirms, or a confirm lost) ---")
    if not regressions:
        print("  none")
    else:
        for name, zc, o, n in regressions[:15]:
            print("  %-16s z_cand=%+.4f  dwell %+.0f%%  ramp-min %+.0f%%"
                  % (name, zc, o * 100., n * 100.))
        print("  %d total" % len(regressions))

    print("\n--- sanity: high halts (z_cand > %.1f) must stay rejected ---" % HIGH_Z)
    hi_ok = hi_bad = 0
    for path in sorted(glob.glob(os.path.join(root, "verify*.csv"))):
        hdr, rows = load(path)
        try:
            zc = float(hdr.get('z_cand', 'nan'))
        except ValueError:
            continue
        if not (zc > HIGH_Z):
            continue
        air = [r[1] for r in rows if r[2] == 'air']
        ramp = [(r[0], r[1]) for r in rows if r[2] in ('down', 'contact', 'up')]
        if len(air) < 2 or not ramp:
            continue
        a = sorted(air)[len(air) // 2]
        m = ramp_min(ramp)
        if a <= 1e-9 or m is None:
            continue
        if (a - m) / a >= THRESH:
            hi_bad += 1
        else:
            hi_ok += 1
    print("  %d stay rejected, %d now confirm" % (hi_ok, hi_bad))


if __name__ == '__main__':
    main()
