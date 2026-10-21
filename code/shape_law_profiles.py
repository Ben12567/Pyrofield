"""Mask-only wind-speed profiles under alternative fire-shape laws.

Expected ("noise-free data") profiles: the observations are the generating masks and
the profile cost of a trial wind u is the minimum of chi^2 over the nuisance block,
with the synthetic error scale sigma = 0.05 per mask value used in the manuscript.
Nuisance parameters are re-optimised numerically (bounded trust-region least squares,
several starts), not set from the analytic compensation formulae.

Outputs: output/shape_law_profiles.csv and output/shape_law_summary.json
"""
import json
import os

import numpy as np
from scipy.optimize import least_squares

import pyrofield_reduced as m

SIGMA = 0.05
BOUNDS = {"R0": (0.01, 1.0), "U": (0.2, 20.0), "k": (0.02, 3.0), "s": (0.25, 4.0)}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

CASES = [
    # name, shape law, generating U0, generating shape coefficient, free shape coefficient?
    ("linear_free_k_U4", "linear", 4.0, 0.35, True),
    ("anderson_fixed_U3", "anderson", 3.0, 1.0, False),
    ("anderson_fixed_U5", "anderson", 5.0, 1.0, False),
    ("anderson_fixed_U10", "anderson", 10.0, 1.0, False),
    ("anderson_free_s_U5", "anderson", 5.0, 1.0, True),
]
R0_TRUE = 0.10
HEAD_ADVANCE = 840.0  # m travelled by the generating head fire at the last frame


def shape(law, U, coef):
    return m.L_linear(U, coef) if law == "linear" else m.L_anderson(U, coef)


def profile_case(name, law, U0, coef0, free_shape, ratios):
    Rh0 = m.head_rate(R0_TRUE, U0)
    T = HEAD_ADVANCE / Rh0
    op = m.MaskOperator(times=T * np.array([0.25, 0.5, 0.75, 1.0]))
    y = op(Rh0, shape(law, U0, coef0))
    key = "k" if law == "linear" else "s"

    def residual(p, u):
        R0 = np.exp(p[0])
        c = np.exp(p[1]) if free_shape else coef0
        return (op(m.head_rate(R0, u), shape(law, u, c)) - y) / SIGMA

    lo = [np.log(BOUNDS["R0"][0])] + ([np.log(BOUNDS[key][0])] if free_shape else [])
    hi = [np.log(BOUNDS["R0"][1])] + ([np.log(BOUNDS[key][1])] if free_shape else [])
    rows = []
    for r in ratios:
        u = U0 * r
        starts = [[np.log(R0_TRUE)], [np.log(Rh0 / (1.0 + m.wind_factor(u)))]]
        if free_shape:
            starts = [s0 + [np.log(c)] for s0 in starts for c in (coef0, 0.5 * coef0, 2.0 * coef0)]
        best = None
        for p0 in starts:
            p0 = np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
            fit = least_squares(residual, p0, args=(u,), bounds=(lo, hi), method="trf",
                                x_scale=1.0, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
            if best is None or fit.cost < best.cost:
                best = fit
        rows.append({
            "case": name, "U0": U0, "u": u, "ratio": r,
            "delta_chi2": 2.0 * best.cost,
            "R0_hat": float(np.exp(best.x[0])),
            "shape_coef_hat": float(np.exp(best.x[1])) if free_shape else coef0,
            "at_bound": bool(np.any(np.isclose(best.x, lo, atol=1e-6) | np.isclose(best.x, hi, atol=1e-6))),
        })
    return rows


def misspecification_bias(U0_grid, s_true):
    """Wind inferred with the fixed (s = 1) Anderson law when the truth has multiplier s_true.

    Head rate and elongation are matched exactly, so the masks are reproduced without
    residual. Returns NaN where the true elongation reaches the cap (then any u >= U_c fits).
    """
    out = []
    for U0 in U0_grid:
        L_true = 1.0 + s_true * (m.L_anderson_raw(U0) - 1.0)
        out.append(np.nan if L_true >= m.L_CAP else m.anderson_inverse(L_true) / U0 - 1.0)
    return np.array(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    ratios = np.exp(np.linspace(np.log(0.4), np.log(2.0), 49))
    ratios = np.unique(np.concatenate([ratios, [1.0], [m.U_C / 10.0]]))
    rows = []
    for case in CASES:
        rows += profile_case(*case, ratios=ratios)
        print("done", case[0])
    keys = list(rows[0].keys())
    with open(os.path.join(OUT, "shape_law_profiles.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    summary = {"U_c": m.U_C, "U_c_if_mph_misapplied": m.U_C / 2.236936}
    for name, *_ in CASES:
        sub = [r for r in rows if r["case"] == name]
        dc = np.array([r["delta_chi2"] for r in sub])
        rr = np.array([r["ratio"] for r in sub])
        summary[name] = {
            "max_delta_chi2_over_range": float(dc.max()),
            "delta_chi2_at_0.77": float(np.interp(np.log(1 / 1.3), np.log(rr), dc)),
            "delta_chi2_at_1.3": float(np.interp(np.log(1.3), np.log(rr), dc)),
            "ratio_range_with_delta_chi2_below_1e-3": [float(rr[dc < 1e-3].min()), float(rr[dc < 1e-3].max())]
            if np.any(dc < 1e-3) else None,
        }
    # Local precision of the fixed law (curvature of the expected profile) under sigma = 0.05.
    prec = {}
    for U0 in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
        h = 0.01
        sub = profile_case("tmp", "anderson", U0, 1.0, False, ratios=[np.exp(-h), np.exp(h)])
        curv = (sub[0]["delta_chi2"] + sub[1]["delta_chi2"]) / (2 * h ** 2)
        prec[str(U0)] = {"sd_logU_percent": float(100.0 / np.sqrt(curv)), "elasticity": float(m.elasticity(U0))}
    summary["fixed_law_local_sd_logU_sigma0.05"] = prec
    # Error propagation from geometry-estimator dispersion and calibration error.
    for label, Lobs in (("median_L_1.80", 1.80), ("p95_L_3.63", 3.63)):
        U = m.anderson_inverse(Lobs)
        E = float(m.elasticity(U))
        summary[label] = {"U_anderson": U, "elasticity": E,
                          "sd_logU_from_6.7pct_L_dispersion": 0.067 / E,
                          "logU_range_from_20pct_L_range": 0.20 / E}
    grid = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    summary["bias_fixed_law"] = {
        str(s): {str(U): (None if np.isnan(b) else float(b)) for U, b in zip(grid, misspecification_bias(grid, s))}
        for s in (0.67, 0.8, 0.9, 1.1, 1.2, 1.5)}
    with open(os.path.join(OUT, "shape_law_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
