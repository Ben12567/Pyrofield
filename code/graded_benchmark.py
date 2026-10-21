"""Graded benchmark of the profile guard across the decision boundary (position error model).

Each cell holds 60 independently drawn fires with the fixed Anderson law below the cap and an
uncertain multiplier s (log-normal prior with SD sigma_s, truth drawn from the prior), observed
through front positions with correlated errors (sigma/D, ell/D). Additional cells test a
near-cap wind range, a misspecified error model (true ell/D = 0.3, assumed independent errors),
miscalibrated laws fixed at s = 1, and priors whose dispersion differs from that of the true s.
Fits use three starts drawn uniformly over the full log-parameter box (no start near the truth).

For every fire we record the guard statistic D_U (converged, bound-aware refits) at 10, 30 and 50%
displacement, the reference value from the expected profile (noise-free data, same model), the
95% profile interval for log U (Delta chi^2 <= 3.84, converged refits) and whether it covers the
true wind, and the wind error of the fit.

Output: output/graded_benchmark_cases.csv, output/graded_benchmark_summary.json
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import beta

import pyrofield_reduced as m

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
LO = np.log([0.01, 0.2, 0.25])
HI = np.log([1.0, 20.0, 4.0])
D = 1000.0


def cp(k, n):
    return [float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0, float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]


class Fire:
    def __init__(self, rng, sigma_s, sig_rel, ell_true, ell_assumed, U_range, s_true_sd=None):
        """sigma_s: prior SD of log s used by the inversion (0 = s fixed at 1);
        s_true_sd: SD of the log-normal distribution of the true multiplier (default sigma_s)."""
        R0 = rng.uniform(0.06, 0.16)
        U0 = rng.uniform(*U_range)
        sd = sigma_s if s_true_sd is None else s_true_sd
        s = float(np.exp(rng.normal(0.0, sd))) if sd > 0 else 1.0
        self.sigma_s = sigma_s
        self.true = np.array([np.log(R0), np.log(U0), np.log(s)])
        # parameter vector of the inversion model closest to the truth (s fixed at 1 if not estimated)
        self.true_model = self.true.copy() if sigma_s > 0 else np.array([np.log(R0), np.log(U0), 0.0])
        Rh0, L0 = m.head_rate(R0, U0), m.L_anderson(U0, s)
        times = (D / Rh0) * np.array([0.25, 0.5, 0.75, 1.0])
        gen = m.BoundaryOperator(times, sigma=sig_rel * D, corr_length=ell_true * D, Rh_ref=Rh0, L_ref=L0)
        self.op = m.BoundaryOperator(times, sigma=sig_rel * D, corr_length=ell_assumed * D, Rh_ref=Rh0, L_ref=L0)
        self.clean = self.op(Rh0, L0)
        self.y = self.clean + gen.sample_noise(rng)

    def resid(self, p, data=None):
        """Whitened residuals; for the expected ('Asimov') data the prior pseudo-observation of
        log s sits at its true value."""
        asimov = data is not None and data is self.clean
        data = self.y if data is None else data
        U = np.exp(p[1])
        r = self.op.whiten(self.op(m.head_rate(np.exp(p[0]), U), m.L_anderson(U, np.exp(p[2]))) - data)
        if self.sigma_s > 0:
            centre = self.true[2] if asimov else 0.0
            return np.append(r, (p[2] - centre) / self.sigma_s)
        return np.append(r, 0.0)


def free_idx(f):
    return [0, 2] if f.sigma_s > 0 else [0]


def fit(f, rng):
    best = None
    idx = [0, 1, 2] if f.sigma_s > 0 else [0, 1]
    for _ in range(3):
        p0 = np.array([rng.uniform(LO[0], HI[0]), rng.uniform(LO[1], HI[1]), 0.0])
        fun = lambda q: f.resid(np.array([q[0], q[1], q[2] if len(q) > 2 else 0.0]))
        res = least_squares(fun, p0[idx], bounds=(LO[idx], HI[idx]), method="trf", max_nfev=500)
        if best is None or res.cost < best.cost:
            best = res
    p = np.array([best.x[0], best.x[1], best.x[2] if len(best.x) > 2 else 0.0])
    return p, 2 * best.cost


def profile_cost(f, p_hat, z, data=None):
    """Converged refit of the nuisance block at log-wind z from the tangent predictor; returns
    (chi2, at_bound)."""
    idx = free_idx(f)
    r0 = f.resid(p_hat, data)
    J = np.empty((r0.size, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1e-6
        J[:, k] = (f.resid(p_hat + e, data) - r0) / 1e-6
    q0 = p_hat[idx] - np.linalg.pinv(J[:, idx]) @ J[:, 1] * (z - p_hat[1])
    q0 = np.clip(q0, LO[idx] + 1e-9, HI[idx] - 1e-9)

    def fun(q):
        p = p_hat.copy()
        p[1] = z
        p[idx] = q
        return f.resid(p, data)

    best = None
    for start in (q0, np.clip(p_hat[idx], LO[idx] + 1e-9, HI[idx] - 1e-9)):
        res = least_squares(fun, start, bounds=(LO[idx], HI[idx]), method="trf", xtol=1e-12, ftol=1e-12, max_nfev=400)
        if best is None or res.cost < best.cost:
            best = res
    at_bound = bool(np.any(np.isclose(best.x, LO[idx], atol=1e-6) | np.isclose(best.x, HI[idx], atol=1e-6)))
    return 2 * best.cost, at_bound


def profile_interval(f, p_hat, chi_min, level=3.84):
    """Endpoints of {z : Delta chi^2(z) <= level} searched outwards from the fit (log U units)."""
    ends = []
    for sgn in (-1, 1):
        g = lambda dz: profile_cost(f, p_hat, p_hat[1] + sgn * dz)[0] - chi_min - level
        grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0]
        val, end = None, np.inf
        for a, b in zip(grid[:-1], grid[1:]):
            if p_hat[1] + sgn * b < LO[1] or p_hat[1] + sgn * b > HI[1]:
                break
            if g(b) > 0:
                end = brentq(g, a, b, xtol=1e-4)
                break
        ends.append(end)
    return p_hat[1] - ends[0], p_hat[1] + ends[1]


CELLS = []
for ss in (0.05, 0.1, 0.2, 0.3):
    for sg in (0.01, 0.03, 0.1):
        CELLS.append({"name": f"s{ss}_sig{sg}", "sigma_s": ss, "sig_rel": sg, "ell_true": 0.1, "ell_assumed": 0.1,
                      "U_range": (2.0, 7.0)})
CELLS.append({"name": "nearcap_exact_sig0.03", "sigma_s": 0.0, "sig_rel": 0.03, "ell_true": 0.1, "ell_assumed": 0.1,
              "U_range": (7.0, 9.5)})
CELLS.append({"name": "misspec_exact_sig0.03", "sigma_s": 0.0, "sig_rel": 0.03, "ell_true": 0.3, "ell_assumed": 0.0,
              "U_range": (2.0, 7.0)})
for sd in (0.1, 0.3):
    CELLS.append({"name": f"miscal_s{sd}_sig0.03", "sigma_s": 0.0, "s_true_sd": sd, "sig_rel": 0.03, "ell_true": 0.1,
                  "ell_assumed": 0.1, "U_range": (2.0, 7.0)})
CELLS.append({"name": "misspec_s0.1_sig0.03", "sigma_s": 0.1, "sig_rel": 0.03, "ell_true": 0.3, "ell_assumed": 0.0,
              "U_range": (2.0, 7.0)})
# prior dispersion misspecified: assumed log-SD of s differs from the spread of the true multipliers
CELLS.append({"name": "underdisp_a0.1_t0.3_sig0.03", "sigma_s": 0.1, "s_true_sd": 0.3, "sig_rel": 0.03, "ell_true": 0.1,
              "ell_assumed": 0.1, "U_range": (2.0, 7.0)})
CELLS.append({"name": "overdisp_a0.3_t0.1_sig0.03", "sigma_s": 0.3, "s_true_sd": 0.1, "sig_rel": 0.03, "ell_true": 0.1,
              "ell_assumed": 0.1, "U_range": (2.0, 7.0)})


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for ci, cell in enumerate(CELLS):
        rng = np.random.default_rng(4242 + ci)
        params = {k: v for k, v in cell.items() if k != "name"}
        for _ in range(n):
            f = Fire(rng, **params)
            p_hat, chi_min = fit(f, rng)
            row = {"cell": cell["name"], "U_true": float(np.exp(f.true[1])), "U_hat": float(np.exp(p_hat[1])),
                   "abs_err": float(abs(np.expm1(p_hat[1] - f.true[1])))}
            for d in (0.1, 0.3, 0.5):
                dz = np.log(1 + d)
                sides, ref, inc = [], [], False
                for sgn in (-1, 1):
                    z = p_hat[1] + sgn * dz
                    if LO[1] <= z <= HI[1]:
                        c, ab = profile_cost(f, p_hat, z)
                        sides.append(c - chi_min)
                        inc = inc or ab
                    else:
                        sides.append(np.inf)
                        inc = True
                    zt = f.true[1] + sgn * dz
                    ref.append(profile_cost(f, f.true_model.copy(), zt, data=f.clean)[0] if LO[1] <= zt <= HI[1] else np.inf)
                key = int(100 * d)
                row[f"DU_d{key}"] = float(max(0.0, min(sides)))
                row[f"inc_d{key}"] = bool(inc)
                row[f"ref_d{key}"] = float(min(ref))
            lo_z, hi_z = profile_interval(f, p_hat, chi_min)
            row["ci_lo"], row["ci_hi"] = float(np.exp(lo_z)), float(np.exp(hi_z))
            row["covers"] = bool(lo_z <= f.true[1] <= hi_z)
            row["s_true"] = float(np.exp(f.true[2]))
            rows.append(row)
        print(cell["name"], "done", flush=True)
    keys = list(rows[0].keys())
    with open(os.path.join(OUT, "graded_benchmark_cases.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    summ = {}
    for cell in CELLS:
        R = [r for r in rows if r["cell"] == cell["name"]]
        e = {"n": len(R), "coverage_95": [sum(r["covers"] for r in R), cp(sum(r["covers"] for r in R), len(R))],
             "median_abs_err": float(np.median([r["abs_err"] for r in R]))}
        for d in (10, 30, 50):
            det = [r for r in R if r[f"DU_d{d}"] >= 4 and not r[f"inc_d{d}"]]
            pos = [r for r in R if r[f"ref_d{d}"] >= 4]
            neg = [r for r in R if r[f"ref_d{d}"] < 4]
            fp = [r for r in neg if r in det]
            tp = [r for r in pos if r in det]
            e[f"d{d}"] = {"determined": len(det), "ref_identifiable": len(pos),
                          "false_det": [len(fp), len(neg), cp(len(fp), len(neg)) if neg else None],
                          "sensitivity": [len(tp), len(pos), cp(len(tp), len(pos)) if pos else None],
                          "err_of_determined_median_p90": [float(np.median([r["abs_err"] for r in det])) if det else None,
                                                           float(np.percentile([r["abs_err"] for r in det], 90)) if det else None]}
        summ[cell["name"]] = e
    # pooled by reference strength (30% displacement)
    strata = [(0, 1), (1, 4), (4, 9), (9, 25), (25, np.inf)]
    summ["by_reference_d30"] = []
    for lo, hi in strata:
        R = [r for r in rows if lo <= r["ref_d30"] < hi]
        det = sum(r["DU_d30"] >= 4 and not r["inc_d30"] for r in R)
        summ["by_reference_d30"].append({"ref_range": [lo, hi if np.isfinite(hi) else None], "n": len(R), "determined": det})
    json.dump(summ, open(os.path.join(OUT, "graded_benchmark_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1)[:3000])


if __name__ == "__main__":
    main()
