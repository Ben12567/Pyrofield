"""Checks requested for the position error model (E2).

1. Dilation bias by refitting: perimeters traced from active-fire pixels are dilated outwards by
   half a pixel (Minkowski sum with a disc of radius delta). The fixed Anderson law is fitted to
   the dilated front positions (noise-free, E2 likelihood) with the ignition point and time known
   or estimated.
2. Residual diagnostic for a misspecified error covariance: lag-1 autocorrelation of whitened
   residuals along the perimeter, averaged over frames and scaled to a z-statistic,
   z = mean(r1) * sqrt(n_frames * n_bearings); flagged if z > 2.33 (one-sided 1%).
3. Functional-form error: true linear length-to-breadth law, inversion with the Anderson law.
4. Unknown wind bearing and ignition: the graded cell sigma_s = 0.1, sigma/D_h = 0.03 refitted with
   bearing, ignition point and ignition time as additional nuisance parameters; profile-interval
   coverage and guard decisions.
Output: output/misspecification_checks.json
"""
import json
import os

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import chi2, ncx2

import pyrofield_reduced as m

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
K = 72
PSI = np.linspace(0, 2 * np.pi, K, endpoint=False)


def chol_blocks(times, Rh, L, sigma, ell):
    blocks = []
    for t in times:
        r = m.front_radius(Rh, L, t, PSI)
        x, y = r * np.cos(PSI), r * np.sin(PSI)
        seg = np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0])))
        s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        d = np.abs(s[:, None] - s[None, :])
        d = np.minimum(d, seg.sum() - d)
        C = sigma ** 2 * (np.exp(-d / ell) if ell > 0 else np.eye(K))
        blocks.append(np.linalg.cholesky(C + 1e-9 * sigma ** 2 * np.eye(K)))
    return blocks


def whiten(blocks, r):
    return np.concatenate([np.linalg.solve(Lc, r[i * K:(i + 1) * K]) for i, Lc in enumerate(blocks)])


def ray_radius(Rh, L, t, beta=0.0, x0=0.0, y0=0.0):
    """Distance from the true ignition along bearings PSI to an ellipse rotated by beta whose rear
    focus sits at (x0, y0) and which has burned for time t."""
    a, b, c = m.ellipse_axes(Rh, L)
    A, B = a * t, b * t
    cb, sb = np.cos(beta), np.sin(beta)
    dx, dy = np.cos(PSI), np.sin(PSI)
    ux, uy = cb * dx + sb * dy, -sb * dx + cb * dy                 # ray direction in ellipse frame
    ex, ey = cb * x0 + sb * y0 + c * t, -sb * x0 + cb * y0         # centre offset in ellipse frame
    qa = (ux / A) ** 2 + (uy / B) ** 2
    qb = -2 * (ux * ex / A ** 2 + uy * ey / B ** 2)
    qc = (ex / A) ** 2 + (ey / B) ** 2 - 1
    return (-qb + np.sqrt(np.maximum(qb ** 2 - 4 * qa * qc, 0.0))) / (2 * qa)


def dilated_radius(Rh, L, t, delta, n=4096):
    """Radial distance at bearings PSI of the Minkowski sum of the front with a disc of radius delta."""
    a, b, c = m.ellipse_axes(Rh, L)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    h = t * (c * np.cos(th) + np.sqrt((a * np.cos(th)) ** 2 + (b * np.sin(th)) ** 2)) + delta
    dh = np.gradient(np.concatenate([h[-2:], h, h[:2]]), th[1] - th[0])[2:-2]
    x, y = h * np.cos(th) - dh * np.sin(th), h * np.sin(th) + dh * np.cos(th)
    ang = np.mod(np.arctan2(y, x), 2 * np.pi)
    rad = np.hypot(x, y)
    order = np.argsort(ang)
    ang, rad = ang[order], rad[order]
    ang = np.concatenate([ang[-1:] - 2 * np.pi, ang, ang[:1] + 2 * np.pi])
    rad = np.concatenate([rad[-1:], rad, rad[:1]])
    return np.interp(PSI, ang, rad)


def dilation_bias():
    out = {}
    Dh, delta, R0 = 5000.0, 187.5, 0.10
    for U0 in (2.75, 4.0, 5.5):
        Rh0, L0 = m.head_rate(R0, U0), float(m.L_anderson(U0))
        T = Dh / Rh0
        times = T * np.array([0.25, 0.5, 0.75, 1.0])
        y = np.concatenate([dilated_radius(Rh0, L0, t, delta) for t in times])
        blocks = chol_blocks(times, Rh0, L0, 0.03 * Dh, 0.1 * Dh)
        res = {}
        for unknown in (False, True):
            def resid(p):
                U = np.exp(p[1])
                Rh, L = m.head_rate(np.exp(p[0]), U), m.L_anderson(U)
                x0, y0, t0 = (p[2], p[3], p[4]) if unknown else (0.0, 0.0, 0.0)
                mod = np.concatenate([ray_radius(Rh, L, t + t0, 0.0, x0, y0) for t in times])
                return whiten(blocks, mod - y)
            p0 = [np.log(R0), np.log(U0)] + ([0.0, 0.0, 0.0] if unknown else [])
            fit = least_squares(resid, p0, method="lm", xtol=1e-12, ftol=1e-12)
            res["unknown_ignition" if unknown else "known_ignition"] = {
                "wind_bias_pct": float(100 * np.expm1(fit.x[1] - np.log(U0))),
                "chi2_at_fit": float(2 * fit.cost), "N": int(fit.fun.size),
                "gof_power_5pct": float(ncx2.sf(chi2.ppf(0.95, fit.fun.size), fit.fun.size, 2 * fit.cost))}
        a, b, c = m.ellipse_axes(Rh0, L0)
        L_app = (a * T + delta) / (b * T + delta)
        res["axis_ratio_inversion_pct"] = float(100 * (m.anderson_inverse(L_app) / U0 - 1))
        out[f"U0={U0}"] = res
    return out


def residual_diagnostic(n=60):
    out = {}
    settings = {"true_ell0.3_assumed_iid": (0.3, 0.0), "true_ell0.1_assumed_0.1": (0.1, 0.1),
                "true_ell0.3_assumed_0.1": (0.3, 0.1), "true_iid_assumed_iid": (0.0, 0.0)}
    for name, (ell_t, ell_a) in settings.items():
        rng = np.random.default_rng(31337 + int(100 * ell_t) + int(1000 * ell_a))
        zs = []
        for _ in range(n):
            R0, U0 = rng.uniform(0.06, 0.16), rng.uniform(2, 7)
            Rh0, L0 = m.head_rate(R0, U0), float(m.L_anderson(U0))
            T = 1000.0 / Rh0
            times = T * np.array([0.25, 0.5, 0.75, 1.0])
            gen = chol_blocks(times, Rh0, L0, 30.0, ell_t * 1000.0)
            ass = chol_blocks(times, Rh0, L0, 30.0, ell_a * 1000.0)
            y = np.concatenate([m.front_radius(Rh0, L0, t, PSI) for t in times])
            y = y + np.concatenate([Lc @ rng.standard_normal(K) for Lc in gen])

            def resid(p):
                U = np.exp(p[1])
                mod = np.concatenate([m.front_radius(m.head_rate(np.exp(p[0]), U), m.L_anderson(U), t, PSI) for t in times])
                return whiten(ass, mod - y)
            fit = least_squares(resid, [np.log(0.1), np.log(4.0)], method="lm")
            r = fit.fun.reshape(4, K)
            r1 = [np.sum((ri - ri.mean()) * (np.roll(ri, 1) - ri.mean())) / np.sum((ri - ri.mean()) ** 2) for ri in r]
            zs.append(float(np.mean(r1) * np.sqrt(4 * K)))
        zs = np.array(zs)
        out[name] = {"n": n, "flagged_z>2.33": int(np.sum(zs > 2.33)), "median_z": float(np.median(zs))}
    return out


def unknown_geometry(n=60):
    """Graded cell sigma_s = 0.1, sigma/D_h = 0.03, ell/D_h = 0.1 with unknown bearing and ignition."""
    rng = np.random.default_rng(4242 + 99)
    sig_s, LO = 0.1, None
    lo = np.array([np.log(0.01), np.log(0.2), np.log(0.25), -np.pi / 4, -300.0, -300.0, -600.0])
    hi = np.array([np.log(1.0), np.log(20.0), np.log(4.0), np.pi / 4, 300.0, 300.0, 600.0])
    rows = []
    for _ in range(n):
        R0, U0 = rng.uniform(0.06, 0.16), rng.uniform(2, 7)
        s = float(np.exp(rng.normal(0, sig_s)))
        true = np.array([np.log(R0), np.log(U0), np.log(s), 0.0, 0.0, 0.0, 0.0])
        Rh0, L0 = m.head_rate(R0, U0), float(m.L_anderson(U0, s))
        T = 1000.0 / Rh0
        times = T * np.array([0.25, 0.5, 0.75, 1.0])
        blocks = chol_blocks(times, Rh0, L0, 30.0, 100.0)
        y = np.concatenate([m.front_radius(Rh0, L0, t, PSI) for t in times]) + \
            np.concatenate([Lc @ rng.standard_normal(K) for Lc in blocks])

        def resid(p):
            U = np.exp(p[1])
            Rh, L = m.head_rate(np.exp(p[0]), U), m.L_anderson(U, np.exp(p[2]))
            mod = np.concatenate([ray_radius(Rh, L, t + p[6], p[3], p[4], p[5]) for t in times])
            return np.append(whiten(blocks, mod - y), p[2] / sig_s)

        best = None
        for _k in range(3):
            p0 = np.array([rng.uniform(np.log(0.03), np.log(0.3)), rng.uniform(np.log(1), np.log(10)), 0, 0, 0, 0, 0])
            f = least_squares(resid, p0, bounds=(lo, hi), method="trf", max_nfev=800)
            if best is None or f.cost < best.cost:
                best = f
        p_hat, chi_min = best.x, 2 * best.cost

        def prof(z):
            idx = [0, 2, 3, 4, 5, 6]

            def g(q):
                p = p_hat.copy()
                p[1] = z
                p[idx] = q
                return resid(p)
            r = least_squares(g, p_hat[idx], bounds=(lo[idx], hi[idx]), method="trf", xtol=1e-10, ftol=1e-10, max_nfev=600)
            return 2 * r.cost - chi_min

        du = min(prof(p_hat[1] + j * np.log(1.3)) for j in (-1, 1))
        ends = []
        for sgn in (-1, 1):
            gfun = lambda dz: prof(p_hat[1] + sgn * dz) - 3.84
            grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0]
            end = np.inf
            for a_, b_ in zip(grid[:-1], grid[1:]):
                if gfun(b_) > 0:
                    end = brentq(gfun, a_, b_, xtol=1e-4)
                    break
            ends.append(end)
        covers = (p_hat[1] - ends[0]) <= true[1] <= (p_hat[1] + ends[1])
        rows.append({"det": bool(du >= 4), "covers": bool(covers), "abs_err": float(abs(np.expm1(p_hat[1] - true[1]))),
                     "half_width": float(0.5 * (ends[0] + ends[1]))})
    return {"n": n, "determined_d30": sum(r["det"] for r in rows), "coverage": sum(r["covers"] for r in rows),
            "median_abs_err_pct": float(100 * np.median([r["abs_err"] for r in rows])),
            "median_half_width_log": float(np.median([r["half_width"] for r in rows]))}


def functional_form_bias():
    """Wind bias when the true elongation follows the linear law L = 1 + k U, matched to the Anderson
    curve at 4 m/s, but the inversion fixes the Anderson law (s = 1); the fit is exact."""
    k = (m.L_anderson_raw(4.0) - 1.0) / 4.0
    return {"k_LB": float(k), **{f"U0={U}": float(100 * (m.anderson_inverse(1 + k * U) / U - 1))
                                for U in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0)}}


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {"dilation_refit": dilation_bias()}
    print(json.dumps(out["dilation_refit"], indent=1), flush=True)
    out["residual_diagnostic"] = residual_diagnostic()
    print(json.dumps(out["residual_diagnostic"], indent=1), flush=True)
    out["functional_form_linear_vs_anderson"] = functional_form_bias()
    print(json.dumps(out["functional_form_linear_vs_anderson"], indent=1), flush=True)
    out["unknown_bearing_ignition_s0.1_sig0.03"] = unknown_geometry()
    print(json.dumps(out["unknown_bearing_ignition_s0.1_sig0.03"], indent=1), flush=True)
    json.dump(out, open(os.path.join(OUT, "misspecification_checks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
