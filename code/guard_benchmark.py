"""Labelled benchmark of the finite-displacement profile guard in the exact homogeneous model.

Classes (100 noisy cases each, independent draws):
  C1 free length-to-breadth coefficient (structurally unidentifiable)
  C2 Anderson law with free multiplier s (structurally unidentifiable within bounds)
  C3 fixed Anderson law above the cap (one-sided: unidentifiable for larger winds)
  C4 fixed, exactly known Anderson law below the cap (identifiable)
  C5 Anderson law with uncertain multiplier, log-normal prior SD 0.3 (practically graded)
Error models:
  E1 soft burned-area masks with independent Gaussian errors (sd 0.05), as in the level-set study
  E2 front positions with Gaussian positional errors (sd 0.03 D) correlated along the perimeter
     (exponential, length 0.1 D)
Guard variants (displacement +-log(1+d), retain the cheaper side, wind 'determined' if D_U >= c):
  G4  four Levenberg-Marquardt updates from the fitted nuisance values (as in the level-set study)
  GC  tangent-predictor start, refit to convergence, 'inconclusive' if a parameter reaches a bound
Reference label: the expected profile (noise-free data, converged refit) at the same displacement;
a case is 'identifiable at d' if both displaced sides cost at least c.

Usage: python guard_benchmark.py [n_per_class]
Outputs: output/guard_benchmark_cases.csv, output/guard_benchmark_summary.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import beta

import pyrofield_reduced as m

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
LOG_BOUNDS = {"R0": np.log([0.01, 1.0]), "U": np.log([0.2, 20.0]), "k": np.log([0.02, 3.0]), "s": np.log([0.25, 4.0])}
PRIOR_S = 0.3
CUTOFFS = [1.0, 2.71, 3.84, 4.0, 6.63, 9.0, 16.0]
DISPLACEMENTS = [0.1, 0.3, 0.5]


def clopper_pearson(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return [float(lo), float(hi)]


class Case:
    """One synthetic fire: parameter layout, forward map and (whitened) residuals."""

    def __init__(self, cls, err, rng):
        self.cls, self.err = cls, err
        R0 = rng.uniform(0.06, 0.16)
        if cls == "C1":
            U0, coef = rng.uniform(2, 6), rng.uniform(0.22, 0.52)
        elif cls == "C2":
            U0, coef = rng.uniform(2.5, 6), rng.uniform(0.7, 1.4)
        elif cls == "C3":
            U0, coef = rng.uniform(9, 12), 1.0
        elif cls == "C4":
            U0, coef = rng.uniform(2, 7), 1.0
        else:
            U0, coef = rng.uniform(2, 7), float(np.exp(rng.normal(0, PRIOR_S)))
        self.law = "linear" if cls == "C1" else "anderson"
        self.free = cls in ("C1", "C2", "C5")      # shape coefficient estimated
        self.prior = cls == "C5"
        self.shape_key = "k" if cls == "C1" else "s"
        self.coef_fixed = 1.0
        self.true = np.array([np.log(R0), np.log(U0)] + ([np.log(coef)] if self.free else []))
        Rh0 = m.head_rate(R0, U0)
        L0 = self.L(U0, coef)
        if err == "E1":
            T = 840.0 / Rh0
            self.op = m.MaskOperator(times=T * np.array([0.25, 0.5, 0.75, 1.0]))
            clean = self.op(Rh0, L0)
            self.y = clean + 0.05 * rng.standard_normal(clean.size)
            self.whiten = lambda r: r / 0.05
        else:
            D = 1000.0
            T = D / Rh0
            self.op = m.BoundaryOperator(T * np.array([0.25, 0.5, 0.75, 1.0]), sigma=0.03 * D,
                                         corr_length=0.1 * D, Rh_ref=Rh0, L_ref=L0)
            clean = self.op(Rh0, L0)
            self.y = clean + self.op.sample_noise(rng)
            self.whiten = self.op.whiten
        self.y_clean = clean
        lo = [LOG_BOUNDS["R0"][0], LOG_BOUNDS["U"][0]] + ([LOG_BOUNDS[self.shape_key][0]] if self.free else [])
        hi = [LOG_BOUNDS["R0"][1], LOG_BOUNDS["U"][1]] + ([LOG_BOUNDS[self.shape_key][1]] if self.free else [])
        self.lo, self.hi = np.array(lo), np.array(hi)

    def L(self, U, coef):
        return m.L_linear(U, coef) if self.law == "linear" else m.L_anderson(U, coef)

    def resid(self, p, data=None):
        """Whitened residuals. For the noise-free (expected, 'Asimov') data set the prior
        pseudo-observation of log s is placed at the true value, so the truth fits exactly."""
        asimov = data is not None and data is self.y_clean
        data = self.y if data is None else data
        coef = np.exp(p[2]) if self.free else self.coef_fixed
        U = np.exp(p[1])
        r = self.whiten(self.op(m.head_rate(np.exp(p[0]), U), self.L(U, coef)) - data)
        if self.prior:
            centre = self.true[2] if asimov else 0.0
            r = np.append(r, (p[2] - centre) / PRIOR_S)
        return r


def fit_all(case, rng):
    """One nominal start (not class-specific) and three starts drawn over the full log box."""
    starts = [np.array([np.log(0.1), np.log(4.0)] + ([np.log(0.35 if case.cls == "C1" else 1.0)] if case.free else []))]
    for _ in range(3):
        starts.append(rng.uniform(case.lo, case.hi))
    best = None
    for p0 in starts:
        f = least_squares(case.resid, p0, bounds=(case.lo, case.hi), method="trf", x_scale=1.0, max_nfev=300)
        if best is None or f.cost < best.cost:
            best = f
    return best.x, 2 * best.cost


def fd_jac(fun, p, r0, h=1e-6):
    J = np.empty((r0.size, p.size))
    for k in range(p.size):
        e = np.zeros_like(p)
        e[k] = h
        J[:, k] = (fun(p + e) - r0) / h
    return J


def refit_four(case, p_hat, z, data=None, n_updates=4, lam0=1e-3):
    """n Levenberg-Marquardt updates on the nuisance block, started at the fitted nuisance values,
    with Marquardt damping lam * diag(J^T J) (lam0 = 1e-3, divided or multiplied by 10 after an
    accepted or rejected step)."""
    idx = [0] + ([2] if case.free else [])
    lo, hi = case.lo[idx], case.hi[idx]

    def f(q):
        p = p_hat.copy()
        p[1] = z
        p[idx] = q
        return case.resid(p, data)

    q = p_hat[idx].copy()
    r = f(q)
    cost = r @ r
    lam = lam0
    for _ in range(n_updates):
        J = fd_jac(f, q, r)
        A, g = J.T @ J, J.T @ r
        step = np.linalg.solve(A + lam * np.diag(np.diag(A)) + 1e-12 * np.eye(len(q)), -g)
        q_new = np.clip(q + step, lo, hi)
        r_new = f(q_new)
        c_new = r_new @ r_new
        if c_new < cost:
            q, r, cost, lam = q_new, r_new, c_new, lam / 10
        else:
            lam *= 10
    return cost


def refit_converged(case, p_hat, z, data=None):
    """Tangent-predictor start, refit to convergence; returns (cost, at_bound)."""
    idx = [0] + ([2] if case.free else [])
    lo, hi = case.lo[idx], case.hi[idx]

    def f(q):
        p = p_hat.copy()
        p[1] = z
        p[idx] = q
        return case.resid(p, data)

    r_hat = case.resid(p_hat, data)
    J = fd_jac(lambda p: case.resid(p, data), p_hat, r_hat)
    A, b = J[:, idx], J[:, 1]
    q_pred = p_hat[idx] - np.linalg.pinv(A) @ b * (z - p_hat[1])
    best = None
    for q0 in (np.clip(q_pred, lo + 1e-9, hi - 1e-9), np.clip(p_hat[idx], lo + 1e-9, hi - 1e-9)):
        res = least_squares(f, q0, bounds=(lo, hi), method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
        if best is None or res.cost < best.cost:
            best = res
    at_bound = bool(np.any(np.isclose(best.x, lo, atol=1e-6) | np.isclose(best.x, hi, atol=1e-6)))
    return 2 * best.cost, at_bound


def evaluate(case, rng):
    p_hat, chi_min = fit_all(case, rng)
    row = {"class": case.cls, "error_model": case.err, "U_true": float(np.exp(case.true[1])),
           "U_hat": float(np.exp(p_hat[1])), "chi2_min": chi_min}
    row["abs_log_error"] = abs(p_hat[1] - case.true[1])
    for d in DISPLACEMENTS:
        dz = np.log(1 + d)
        sides_g4, sides_gc, bound_gc, sides_ref = [], [], False, []
        trial_out = False
        for sgn in (-1, 1):
            z = p_hat[1] + sgn * dz
            if not (case.lo[1] <= z <= case.hi[1]):
                trial_out = True
                sides_g4.append(np.inf)
                sides_gc.append(np.inf)
            else:
                if d == 0.3:
                    sides_g4.append(refit_four(case, p_hat, z) - chi_min)
                c, ab = refit_converged(case, p_hat, z)
                sides_gc.append(c - chi_min)
                bound_gc = bound_gc or ab
            # reference: expected profile at the truth (noise-free data)
            zt = case.true[1] + sgn * dz
            if case.lo[1] <= zt <= case.hi[1]:
                c, _ = refit_converged(case, case.true.copy(), zt, data=case.y_clean)
                sides_ref.append(c)
            else:
                sides_ref.append(np.inf)
        key = f"d{int(100 * d)}"
        if d == 0.3:
            row[f"DU_G4_{key}"] = float(max(0.0, min(sides_g4)))
        row[f"DU_GC_{key}"] = float(max(0.0, min(sides_gc)))
        row[f"GC_inconclusive_{key}"] = bool(bound_gc or trial_out)
        row[f"ref_{key}"] = float(min(sides_ref))
    return row


def summarise(rows):
    summ = {}
    for err in ("E1", "E2"):
        R = [r for r in rows if r["error_model"] == err]
        s = {}
        for cls in ("C1", "C2", "C3", "C4", "C5"):
            C = [r for r in R if r["class"] == cls]
            n = len(C)
            entry = {"n": n}
            for var in ("G4", "GC"):
                det = sum(1 for r in C if r[f"DU_{var}_d30"] >= 4.0 and not (var == "GC" and r["GC_inconclusive_d30"]))
                entry[f"{var}_determined"] = [det, clopper_pearson(det, n)]
            entry["GC_inconclusive"] = sum(r["GC_inconclusive_d30"] for r in C)
            for key in ("d30", "d50"):
                det = sum(r[f"DU_GC_{key}"] >= 4.0 and not r[f"GC_inconclusive_{key}"] for r in C)
                inc = sum(bool(r[f"GC_inconclusive_{key}"]) for r in C)
                entry[f"GC_threeway_{key}"] = {"determined": det, "unresolved": n - det - inc, "inconclusive": inc}
                entry[f"ref_identifiable_{key}"] = 0 if cls in ("C1", "C2", "C3") else sum(r[f"ref_{key}"] >= 4.0 for r in C)

            entry["median_abs_wind_error_pct"] = float(100 * np.median([np.expm1(r["abs_log_error"]) for r in C]))
            s[cls] = entry
        # Sensitivity / specificity against the reference label, cutoff 4, d = 30%.
        # Analytic labels for C1-C3 (unidentifiable); expected-profile labels for C4-C5.
        label = lambda r, key="d30", c=4.0: (r["class"] not in ("C1", "C2", "C3")) and r[f"ref_{key}"] >= c
        for var in ("G4", "GC"):
            pos = [r for r in R if label(r)]
            neg = [r for r in R if not label(r)]
            dec = lambda r: r[f"DU_{var}_d30"] >= 4.0 and not (var == "GC" and r["GC_inconclusive_d30"])
            tp = sum(dec(r) for r in pos)
            tn = sum(not dec(r) for r in neg)
            s[f"{var}_sensitivity"] = [tp, len(pos), clopper_pearson(tp, len(pos))]
            s[f"{var}_specificity"] = [tn, len(neg), clopper_pearson(tn, len(neg))]
        # ROC over cutoffs (GC, d = 30%) and over displacements (GC, cutoff 4)
        roc = []
        for c in CUTOFFS:
            pos = [r for r in R if label(r, c=c)]
            neg = [r for r in R if not label(r, c=c)]
            tp = sum(r["DU_GC_d30"] >= c and not r["GC_inconclusive_d30"] for r in pos)
            fp = sum(r["DU_GC_d30"] >= c and not r["GC_inconclusive_d30"] for r in neg)
            roc.append({"cutoff": c, "sens": tp / max(1, len(pos)), "fpr": fp / max(1, len(neg)),
                        "n_pos": len(pos), "n_neg": len(neg)})
        s["roc_cutoff_GC_d30"] = roc
        disp = []
        for d in DISPLACEMENTS:
            key = f"d{int(100 * d)}"
            neg_struct = [r for r in R if r["class"] in ("C1", "C2", "C3")]
            fp = sum(r[f"DU_GC_{key}"] >= 4 and not r[f"GC_inconclusive_{key}"] for r in neg_struct)
            c4 = [r for r in R if r["class"] == "C4"]
            tp = sum(r[f"DU_GC_{key}"] >= 4 and not r[f"GC_inconclusive_{key}"] for r in c4)
            c5 = [r for r in R if r["class"] == "C5"]
            d5 = sum(r[f"DU_GC_{key}"] >= 4 and not r[f"GC_inconclusive_{key}"] for r in c5)
            r5 = sum(r[f"ref_{key}"] >= 4 for r in c5)
            disp.append({"displacement": d, "false_det_structural": [fp, len(neg_struct)],
                         "C4_determined": [tp, len(c4)], "C5_determined": [d5, len(c5)], "C5_ref_identifiable": r5})
        s["by_displacement_GC_cutoff4"] = disp
        summ[err] = s
    return summ


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    os.makedirs(OUT, exist_ok=True)
    rows = []
    t0 = time.time()
    for ei, err in enumerate(("E2", "E1")):
        for ci, cls in enumerate(("C1", "C2", "C3", "C4", "C5")):
            rng = np.random.default_rng(20261013 + 100 * ei + ci)
            for i in range(n):
                case = Case(cls, err, rng)
                rows.append(evaluate(case, rng))
            print(err, cls, "done", f"{time.time() - t0:.0f}s", flush=True)
    keys = list(rows[0].keys())
    with open(os.path.join(OUT, "guard_benchmark_cases.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    summ = summarise(rows)
    summ["n_per_class"] = n
    json.dump(summ, open(os.path.join(OUT, "guard_benchmark_summary.json"), "w"), indent=1)
    print(json.dumps({e: {c: summ[e][c] for c in ("C1", "C2", "C3", "C4", "C5")} for e in ("E1", "E2")}, indent=1))


if __name__ == "__main__":
    main()
