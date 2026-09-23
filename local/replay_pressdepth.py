#!/usr/bin/env python3
# Three questions against the saved verify captures, all using the ramp-minimum
# statistic that is now shipping.
#
# 1. HOW DEEP does the ramp actually need to go?  Nothing is measured below the
#    damping minimum any more, so VERIFY_DOWN only has to reach it plus margin.
#    Anything beyond that is press on the plate for nothing.
#
# 2. Is the down/up BIAS still a quality signal now that contacts are measured
#    at the right depth?  It has to earn its place by flagging bad contacts the
#    drop threshold ACCEPTS - if it only fires where the drop already fails, it
#    is redundant.
#
# 3. Does the re-excitation depth-profile depend on bed POSITION?  The platform
#    sits on a three-point mount on a cantilevered Z stage attached at the back,
#    just behind (60,100).  If pressing rotates that shelf about its attachment
#    and that is what re-excites the structure, the effect should grow toward the
#    back of the bed - i.e. with Y.  Verify captures do not record XY, so this
#    pairs them with the descent trace written immediately before.
import glob
import os
import sys


def load_verify(path):
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


def moving_min(pairs, k=5):
    pairs = sorted(pairs)
    if not pairs:
        return None, None
    if len(pairs) < k:
        z, a = min(pairs, key=lambda t: t[1])
        return a, z
    best = None
    for i in range(len(pairs) - k + 1):
        w = sorted(a for _, a in pairs[i:i + k])
        m = w[k // 2]
        if best is None or m < best[0]:
            best = (m, pairs[i + k // 2][0])
    return best


def edges(rows, tag, k=5):
    """Per-rep steepest-step edge, same rule as _ramp_edge."""
    out = []
    for rep in set(r[3] for r in rows if r[2] == tag):
        seg = sorted([r for r in rows if r[2] == tag and r[3] == rep],
                     key=lambda r: -r[0])
        if tag == 'up':
            seg = seg[::-1]
        a = [r[1] for r in seg]
        if len(a) < 4:
            continue
        steps = [(a[i + 1] - a[i]) / max(a[i], 1e-9) for i in range(len(a) - 1)]
        j = min(range(len(steps)), key=lambda i: steps[i])
        if steps[j] < 0:
            out.append(0.5 * (seg[j][0] + seg[j + 1][0]))
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "probe_traces"
    depths, quality = [], []
    for path in sorted(glob.glob(os.path.join(root, "verify*.csv"))):
        hdr, rows = load_verify(path)
        try:
            zc = float(hdr.get('z_cand', 'nan'))
        except ValueError:
            continue
        if zc != zc:
            continue
        air = [r[1] for r in rows if r[2] == 'air']
        ramp = [(r[0], r[1]) for r in rows
                if r[2] in ('down', 'contact', 'up')]
        if len(air) < 2 or len(ramp) < 8:
            continue
        a = sorted(air)[len(air) // 2]
        m, mz = moving_min(ramp)
        if a <= 1e-9 or m is None:
            continue
        drop = (a - m) / a
        deepest = min(z for z, _ in ramp)
        depths.append((zc - mz, zc - deepest, drop, os.path.basename(path)))
        d_edges, u_edges = edges(rows, 'down'), edges(rows, 'up')
        if d_edges and u_edges:
            rd = sum(d_edges) / len(d_edges)
            ru = sum(u_edges) / len(u_edges)
            quality.append((abs(ru - rd) * 1000., drop, zc,
                            os.path.basename(path)))

    print("=== 1. how deep does the ramp need to go? ===")
    print("distance from the halt DOWN to the damping minimum (mm):")
    below = sorted(d for d, _, dr, _ in depths if dr >= 0.20)
    if below:
        n = len(below)
        for label, q in (("p50", .50), ("p75", .75), ("p90", .90),
                         ("p95", .95), ("max", 1.0)):
            print("  %-4s %+.3f" % (label, below[min(n - 1, int(q * n))]))
        print("  (over %d captures whose ramp-min clears the 20%% threshold)" % n)
        print("\n  VERIFY_DOWN is 0.250.  p95 needs %.3f, so ~%.2f would keep"
              % (below[min(n - 1, int(.95 * n))],
                 below[min(n - 1, int(.95 * n))] + 0.05))
        print("  every one of these contacts with 50um of margin.")

    print("\n=== 2. does |bias| flag bad contacts the DROP accepts? ===")
    passing = [q for q in quality if q[1] >= 0.20]
    if passing:
        bs = sorted(q[0] for q in passing)
        n = len(bs)
        print("  |bias| over contacts that PASS the drop threshold:")
        print("    p50 %.1f um   p90 %.1f um   p95 %.1f um   max %.1f um"
              % (bs[n // 2], bs[min(n - 1, int(.9 * n))],
                 bs[min(n - 1, int(.95 * n))], bs[-1]))
        hi = [q for q in passing if q[0] > 50.]
        print("    %d of %d passing contacts have |bias| > 50um" % (len(hi), n))
        implausible = [q for q in hi if q[2] > 0.6]
        print("    of those, %d are at a halt height above z=0.6 (cannot be bed)"
              % len(implausible))
        for b, dr, zc, name in sorted(hi, key=lambda q: -q[0])[:8]:
            print("      %-16s |bias| %6.1f um  drop %+.0f%%  z_cand %+.4f"
                  % (name, b, dr * 100., zc))

    print("\n=== 3. does re-excitation depend on Y (cantilever test)? ===")
    print("  pairing each verify with the descent written just before it...")
    rows_by_y = {}
    for d_min, d_deep, drop, name in depths:
        num = int(''.join(c for c in name if c.isdigit()))
        # descent numbering runs ahead of verify; find the nearest descent file
        cand = None
        for off in range(0, 4):
            for guess in (num * 2 + off, num + off):
                p = os.path.join(root, "descent%05d.csv" % guess)
                if os.path.exists(p):
                    cand = p
                    break
            if cand:
                break
        if not cand:
            continue
    print("  verify captures do not record XY and the descent pairing is not")
    print("  reliable from filenames alone - deferring this to the ledger,")
    print("  which records XY per accepted contact from now on.")


if __name__ == '__main__':
    main()
