"""Reduced homogeneous fire-spread model used for the analytic and shape-law analyses.

All quantities are SI (m, s, m s^-1). The head-fire rate follows the Rothermel-type
factorisation R_h = R0 (1 + a_w U^b_w); fire shape follows either the free-coefficient
law L = 1 + k U or the Anderson (1983) / FARSITE (Finney, 1998) length-to-breadth law
with U in m s^-1 and a hard cap L <= 8.

For spatially homogeneous forcing, a fixed wind bearing and a point ignition, the
Huygens solution of the level-set equation with the elliptical support-function speed
is an ellipse whose rear focus stays at the ignition point (Richards, 1990). The mask
operator below rasterises that exact solution through the same softened indicator as
the level-set implementation, so shape-law effects are isolated from discretisation
error.
"""
import numpy as np
from scipy.optimize import brentq

A_W = 0.35   # (s m^-1)^1.3
B_W = 1.3
L_CAP = 8.0


def wind_factor(U):
    return A_W * np.power(U, B_W)


def head_rate(R0, U):
    return R0 * (1.0 + wind_factor(U))


def gamma(U):
    """d log R_h / d log U at fixed R0 (Eq. gamma in the manuscript)."""
    phi = wind_factor(U)
    return B_W * phi / (1.0 + phi)


def L_anderson_raw(U):
    return 0.936 * np.exp(0.2566 * U) + 0.461 * np.exp(-0.1548 * U) - 0.397


def dL_anderson_raw(U):
    return 0.936 * 0.2566 * np.exp(0.2566 * U) - 0.461 * 0.1548 * np.exp(-0.1548 * U)


def L_anderson(U, s=1.0):
    return np.minimum(L_CAP, 1.0 + s * (L_anderson_raw(U) - 1.0))


def L_linear(U, k):
    return 1.0 + k * U


U_C = brentq(lambda u: L_anderson_raw(u) - L_CAP, 1.0, 20.0)


def anderson_inverse(L):
    """Wind speed giving L on the uncapped Anderson curve (L > 1)."""
    return brentq(lambda u: L_anderson_raw(u) - L, 1e-6, 40.0)


def elasticity(U):
    """E(U) = d log L_A / d log U below the cap; zero above it."""
    U = np.asarray(U, dtype=float)
    E = U * dL_anderson_raw(U) / L_anderson_raw(U)
    return np.where(U < U_C, E, 0.0)


def ellipse_axes(Rh, L):
    """Semi-axes (a, b) and centre offset c of the unit-time Huygens ellipse."""
    e = np.sqrt(1.0 - 1.0 / L ** 2)
    Rb = Rh * (1.0 - e) / (1.0 + e)
    a = 0.5 * (Rh + Rb)
    return a, a / L, 0.5 * (Rh - Rb)


def front_radius(Rh, L, t, psi):
    """Distance from the ignition point to the exact homogeneous front at time t.

    psi is the bearing relative to the wind direction. The ignition point is the rear focus
    of the Huygens ellipse, so r(psi) = a (1 - e^2) / (1 - e cos psi) with a = a_e t.
    """
    a, _, _ = ellipse_axes(Rh, L)
    e = np.sqrt(1.0 - 1.0 / L ** 2)
    return t * a * (1.0 - e ** 2) / (1.0 - e * np.cos(psi))


class BoundaryOperator:
    """Front positions sampled at fixed bearings, with Gaussian positional errors that are
    correlated along the perimeter (exponential kernel in arc length) and independent between
    frames. Residuals are whitened with the Cholesky factor of the generating covariance,
    which is built once from the reference front.
    """

    def __init__(self, times, n_bearings=72, sigma=10.0, corr_length=0.0, Rh_ref=None, L_ref=None):
        self.times = np.asarray(times, dtype=float)
        self.psi = np.linspace(0.0, 2.0 * np.pi, n_bearings, endpoint=False)
        blocks = []
        for t in self.times:
            r = front_radius(Rh_ref, L_ref, t, self.psi)
            x, y = r * np.cos(self.psi), r * np.sin(self.psi)
            seg = np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0])))
            s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
            perim = seg.sum()
            d = np.abs(s[:, None] - s[None, :])
            d = np.minimum(d, perim - d)
            if corr_length > 0:
                C = sigma ** 2 * np.exp(-d / corr_length)
            else:
                C = sigma ** 2 * np.eye(n_bearings)
            blocks.append(np.linalg.cholesky(C + 1e-9 * sigma ** 2 * np.eye(n_bearings)))
        self.chol = blocks

    def __call__(self, Rh, L):
        return np.concatenate([front_radius(Rh, L, t, self.psi) for t in self.times])

    def whiten(self, resid):
        n = len(self.psi)
        out = [np.linalg.solve(Lc, resid[i * n:(i + 1) * n]) for i, Lc in enumerate(self.chol)]
        return np.concatenate(out)

    def sample_noise(self, rng):
        n = len(self.psi)
        return np.concatenate([Lc @ rng.standard_normal(n) for Lc in self.chol])


class MaskOperator:
    """Soft burned-area masks of the exact homogeneous solution on a raster.

    Grid, cell size and softening width follow the reference synthetic experiment
    (96 x 96 cells, dx = 15 m, logistic transition of one cell). The wind blows
    towards +x; ignition is a point at `ignition` (m).
    """

    def __init__(self, times, n=96, dx=15.0, ignition=(300.0, 720.0)):
        centres = (np.arange(n) + 0.5) * dx
        self.X, self.Y = np.meshgrid(centres, centres)
        self.dx = dx
        self.times = np.asarray(times, dtype=float)
        self.ignition = ignition

    def __call__(self, Rh, L):
        a, b, c = ellipse_axes(Rh, L)
        frames = []
        for t in self.times:
            A, B = a * t, b * t
            xr = (self.X - (self.ignition[0] + c * t)) / A
            yr = (self.Y - self.ignition[1]) / B
            q = np.sqrt(xr ** 2 + yr ** 2)
            grad = np.sqrt((xr / A) ** 2 + (yr / B) ** 2) / np.maximum(q, 1e-12)
            phi = (q - 1.0) / np.maximum(grad, 1e-12)   # first-order signed distance
            frames.append(1.0 / (1.0 + np.exp(np.clip(phi / self.dx, -60.0, 60.0))))
        return np.concatenate([f.ravel() for f in frames])
