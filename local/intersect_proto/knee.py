"""Two-line intersection: air line x first-departure flank line, per axis per ramp."""
import numpy as np


def movmed(y, k=5):
    h = k // 2
    return np.array([np.median(y[max(0, i - h):i + h + 1]) for i in range(len(y))])


def knee_edge(zs, amps, min_exc=0.10, k_sig=4., run=3, lo=0.15, hi=0.85, air_frac=0.5,
              detail=False):
    """-> (edge, excursion, air_sd, sign) or None.  zs in mm."""
    z = np.asarray(zs, float); o = np.argsort(z); z = z[o]
    y = np.log(np.maximum(np.asarray(amps, float)[o], 1.))
    n = len(z)
    if n < 12:
        return None
    s = movmed(y)
    # 1. provisional air line from the top part of the ramp
    top = z >= z[0] + (1 - air_frac) * (z[-1] - z[0])
    ma, ba = np.polyfit(z[top], y[top], 1)
    sd = float(np.std(y[top] - (ma * z[top] + ba))) + 1e-9
    dev = s - (ma * z + ba)
    # 2. first sustained departure walking DOWN from the top that turns out
    # to be a real flank (big enough excursion, enough straight samples);
    # smaller noise excursions are stepped over, not fatal.
    i = n - 1
    found = None
    while i >= run - 1:
        i0 = i
        w = dev[i - run + 1:i + 1]
        if not (np.all(np.abs(w) > k_sig * sd) and (np.all(w > 0) or np.all(w < 0))):
            i -= 1
            continue
        i_d, sgn = i, (1 if w[-1] > 0 else -1)
        # the threshold trips partway down a flank when air is noisy; the
        # flank itself starts where the trace last sat on the air line
        while i_d < n - 1 and sgn * dev[i_d + 1] > sd:
            i_d += 1
        # 3. follow the flank deeper to its first turning point
        i_e = i_d
        while i_e > 0 and sgn * dev[i_e - 1] >= sgn * dev[i_e]:
            i_e -= 1
        exc = sgn * dev[i_e]
        seg = np.arange(i_e, i_d + 1)
        frac = sgn * dev[seg] / max(exc, 1e-9)
        fl = seg[(frac >= lo) & (frac <= hi)]
        if exc >= min_exc and len(fl) >= 2:
            found = (i_d, sgn, i_e, exc, fl)
            break
        i = min(i_e, i0) - 1          # always move deeper
    if found is None:
        return None
    i_d, sgn, i_e, exc, fl = found
    mf, bf = np.polyfit(z[fl], y[fl], 1)
    # 4. air line refit on everything above the departure
    air = np.arange(i_d + 1, n)
    if len(air) < 5:
        return None
    ma, ba = np.polyfit(z[air], y[air], 1)
    sd = float(np.std(y[air] - (ma * z[air] + ba))) + 1e-9
    if abs(mf - ma) < 1e-9:
        return None
    edge = (ba - bf) / (mf - ma)
    if not (z[i_e] - 0.02 <= edge <= z[-1]):
        return None
    if detail:
        return dict(edge=edge, exc=exc, sd=sd, sgn=sgn, air=(ma, ba), flank=(mf, bf),
                    fl=fl, i_d=i_d, i_e=i_e, z=z, y=y, s=s)
    return edge, exc, sd, sgn
