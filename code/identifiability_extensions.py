"""Extensions of the practical-identifiability analysis (position error model, Anderson law).

1. Closed form: SD(log U)^2 = SD_pos^2 + (sigma_s / E1)^2 with E1 = d log(L_A - 1) / d log U.
2. Sampling density: 36-288 bearings, independent and correlated errors.
3. Unknown ignition point and time (x0, y0, t0) as additional nuisance parameters.
4. Outward bias of perimeters derived from active-fire pixels (dilation by half a pixel).
5. Two wind periods with a known wind ratio and a common unknown multiplier alpha:
   SD(log alpha) ~ SD(log(R_b/R_a)) / |gamma(U_b) - gamma(U_a)|.
Output: output/identifiability_extensions.json
"""
import json
import os

import numpy as np

import pyrofield_reduced as m
from rothermel import FUEL_MODELS, FuelBed

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
R0, D = 0.10, 1000.0


def covariance_chol(times, Rh, L, psi, sigma, ell):
    blocks = []
    for t in times:
        r = m.front_radius(Rh, L, t, psi)
        x, y = r * np.cos(psi), r * np.sin(psi)
        seg = np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0])))
        s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        d = np.abs(s[:, None] - s[None, :])
        d = np.minimum(d, seg.sum() - d)
        C = sigma ** 2 * (np.exp(-d / ell) if ell > 0 else np.eye(len(psi)))
        blocks.append(np.linalg.cholesky(C + 1e-9 * sigma ** 2 * np.eye(len(psi))))
    return blocks


def ray_radius(Rh, L, t, psi, x0=0.0, y0=0.0):
    """Distance along bearing psi from the true ignition (origin) to an ellipse whose rear focus
    sits at (x0, y0) and which has burned for time t."""
    a, b, c = m.ellipse_axes(Rh, L)
    A, B, xc = a * t, b * t, x0 + c * t
    cx, cy = np.cos(psi), np.sin(psi)
    qa = (cx / A) ** 2 + (cy / B) ** 2
    qb = -2 * (cx * xc / A ** 2 + cy * y0 / B ** 2)
    qc = (xc / A) ** 2 + (y0 / B) ** 2 - 1
    return (-qb + np.sqrt(qb ** 2 - 4 * qa * qc)) / (2 * qa)


def sd_logU(U0, sig_rel=0.03, ell_rel=0.1, n_bearings=72, sigma_s=0.0, unknown_ignition=False):
    Rh0, L0 = m.head_rate(R0, U0), m.L_anderson(U0)
    T = D / Rh0
    times = T * np.array([0.25, 0.5, 0.75, 1.0])
    psi = np.linspace(0, 2 * np.pi, n_bearings, endpoint=False)
    chol = covariance_chol(times, Rh0, L0, psi, sig_rel * D, ell_rel * D)

    def model(th):
        lr, lu, ls, x0, y0, t0 = th
        U = np.exp(lu)
        Rh, L = m.head_rate(np.exp(lr), U), m.L_anderson(U, np.exp(ls))
        return np.concatenate([ray_radius(Rh, L, t + t0, psi, x0, y0) for t in times])

    th0 = np.array([np.log(R0), np.log(U0), 0.0, 0.0, 0.0, 0.0])
    steps = [1e-5, 1e-5, 1e-5, 1e-2, 1e-2, 1e-2]
    free = [0, 1] + ([2] if sigma_s > 0 else []) + ([3, 4, 5] if unknown_ignition else [])
    cols = []
    for k in free:
        e = np.zeros(6)
        e[k] = steps[k]
        diff = model(th0 + e) - model(th0 - e)
        n = len(psi)
        cols.append(np.concatenate([np.linalg.solve(Lc, diff[i * n:(i + 1) * n]) for i, Lc in enumerate(chol)])
                    / (2 * steps[k]))
    J = np.array(cols).T
    info = J.T @ J
    if sigma_s > 0:
        info[free.index(2), free.index(2)] += 1.0 / sigma_s ** 2
    return float(np.sqrt(np.linalg.inv(info)[1, 1]))


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    # 1 closed form
    cf = {}
    for U0 in (3.0, 5.0):
        E1 = U0 * m.dL_anderson_raw(U0) / (m.L_anderson_raw(U0) - 1)
        pos = sd_logU(U0)
        for ss in (0.1, 0.3):
            num = sd_logU(U0, sigma_s=ss)
            cf[f"U0={U0},sigma_s={ss}"] = {"E1": E1, "sd_pos_pct": 100 * pos, "floor_pct": 100 * ss / E1,
                                           "closed_form_pct": 100 * np.hypot(pos, ss / E1), "numerical_pct": 100 * num}
    out["closed_form"] = cf
    # 2 sampling density
    out["sampling_density_U0=5_sig0.03"] = {f"K={K},ell={ell}": 100 * sd_logU(5.0, ell_rel=ell, n_bearings=K)
                                            for K in (36, 72, 144, 288) for ell in (0.0, 0.1)}
    # 3 unknown ignition
    out["unknown_ignition_U0=5_sig0.03_ell0.1"] = {"known_pct": 100 * sd_logU(5.0),
                                                   "unknown_xy_t0_pct": 100 * sd_logU(5.0, unknown_ignition=True)}
    # 4 dilation bias of pixel-derived perimeters: half of a 375-m pixel on a 5-km head advance
    dil = {}
    delta, Dh = 187.5, 5000.0
    for U in (2.75, 4.0, 5.5):
        L = float(m.L_anderson(U))
        a, b, c = m.ellipse_axes(1.0, L)
        t = Dh / (a + c)
        A, B = a * t, b * t
        L_app = (A + delta) / (B + delta)
        U_app = m.anderson_inverse(L_app) if L_app > 1.0001 else float("nan")
        dil[f"U={U}"] = {"L_true": L, "L_apparent": L_app, "wind_bias_pct": 100 * (U_app / U - 1),
                         "head_advance_bias_pct": 100 * delta / Dh}
    out["dilation_half_pixel_5km"] = dil
    # 5 two-period wind with common multiplier
    fb = FuelBed(**FUEL_MODELS["FM1"])
    two = {}
    for name, phi, Ua, Ub in (("testbed", m.wind_factor, 4.0, 7.0), ("FM1", fb.phi_w, 2.0, 3.5)):
        g = lambda U: (np.log1p(phi(U * 1.0001)) - np.log1p(phi(U / 1.0001))) / (2 * np.log(1.0001))
        dg = abs(g(Ub) - g(Ua))
        two[name] = {"gamma_a": g(Ua), "gamma_b": g(Ub), "dlogratio_dlogalpha": dg,
                     "sd_logalpha_if_head_rates_known_to_2pct": float(np.hypot(0.02, 0.02) / dg)}
    out["two_period_common_multiplier"] = two
    json.dump(out, open(os.path.join(OUT, "identifiability_extensions.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
