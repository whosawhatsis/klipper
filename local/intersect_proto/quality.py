"""Expected contact-point error for one axis's tangent fit.

knee = zh + (air(knee) - yh) / (mf - ma), so to first order
  var(knee) ~ (sd / |mf|)^2              half-way point: air-level noise over the slope
            + (d * se_mf / |mf|)^2       slope error lever-armed over d = knee - zh
            + (se_air(knee) / |mf|)^2    air line position at the knee
"""
import numpy as np
from anchor import ols, prep


def knee_sigma(z, y, f, gap=0.005, band=(0.1, 0.9)):
    """f: dict with knee, zh, yh, mf, air=(ma, ba), zt (mm units). -> sigma (mm) or inf."""
    ma, ba = f['air']; mf = f['mf']; knee = f['knee']
    am = z > knee + gap
    if am.sum() < 5 or abs(mf - ma) < 1e-9:
        return np.inf
    A = ols(z[am], y[am])
    sd = A[3]
    X = np.array([knee, 1.])
    se_air = float(np.sqrt(max(X @ A[2] @ X, 0.)))
    # slope se over the band of the flank actually used
    fl = (z >= f['zt']) & (z <= knee)
    if fl.sum() < 3:
        return np.inf
    ext = y[fl][np.argmin(z[fl])] - (A[0] * z[fl].min() + A[1])
    dev = y[fl] - (A[0] * z[fl] + A[1])
    if abs(ext) < 1e-9:
        return np.inf
    fr = dev / ext
    sel = (fr >= band[0]) & (fr <= band[1])
    if sel.sum() < 2:
        return np.inf
    if sel.sum() == 2:
        # a two-point slope has no residual; take its error from the air noise
        zz = z[fl][sel]
        se_mf = np.sqrt(2.) * sd / abs(zz[1] - zz[0])
    else:
        F = ols(z[fl][sel], y[fl][sel])
        se_mf = float(np.sqrt(max(F[2][0, 0], 0.)))
    d = abs(knee - f['zh'])
    s = abs(mf - ma)
    return float(np.sqrt((sd / s) ** 2 + (d * se_mf / s) ** 2 + (se_air / s) ** 2))
