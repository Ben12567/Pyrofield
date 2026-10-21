"""Summaries of the guard benchmarks with the final decision rule, and analytic coverage predictions.

Decision rule: with displacement d and cutoff c, wind is 'determined' if both displaced sides cost at
least c; a determination is downgraded to 'inconclusive' if a nuisance parameter or the displaced
wind reached a bound on either side (the constrained refit can only overestimate the cost), and a
case is otherwise 'unresolved'.
Analytic coverage when the assumed prior SD of log s (sigma_a; 0 for a law fixed at s = 1) differs
from the spread of the true multipliers (sigma_t) (Eq. 9): the estimate of log s stays at the prior
mean, so the log-wind error has SD sqrt(SD_pos^2 + (sigma_t/E_X)^2) while the reported SD is
sqrt(SD_pos^2 + (sigma_a/E_X)^2); coverage = 2 Phi(1.96 ratio) - 1, averaged over U0 ~ U(2, 7).
Coverage of correctly specified cells is also reported by |log s_true| / sigma_s, because coverage
averaged over the prior can hide undercoverage for multipliers in its tails.
Output: output/benchmark_tables.json
"""
import csv
import json
import os

import numpy as np
from scipy.stats import beta, norm

import pyrofield_reduced as m
from identifiability_extensions import sd_logU

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


def cp(k, n):
    return [float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0, float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]


def decide(du, inc, c=4.0):
    if du >= c:
        return "inconclusive" if inc else "determined"
    return "unresolved"


def main():
    out = {}
    rows = list(csv.DictReader(open(os.path.join(OUT, "guard_benchmark_cases.csv"), encoding="utf-8")))
    lab = {}
    for err in ("E1", "E2"):
        R = [r for r in rows if r["error_model"] == err]
        lab[err] = {}
        for cls in ("C1", "C2", "C3", "C4", "C5"):
            C = [r for r in R if r["class"] == cls]
            for d in ("d30", "d50"):
                out_c = [decide(float(r[f"DU_GC_{d}"]), r[f"GC_inconclusive_{d}"] == "True") for r in C]
                lab[err][f"{cls},{d}"] = {k: out_c.count(k) for k in ("determined", "unresolved", "inconclusive")}
        struct = [r for r in R if r["class"] in ("C1", "C2", "C3")]
        fp = sum(decide(float(r["DU_GC_d30"]), r["GC_inconclusive_d30"] == "True") == "determined" for r in struct)
        lab[err]["structural_specificity_d30"] = [len(struct) - fp, len(struct), cp(len(struct) - fp, len(struct))]
        c5 = [r for r in R if r["class"] == "C5"]
        tp = sum(float(r["ref_d50"]) >= 4 and decide(float(r["DU_GC_d50"]), r["GC_inconclusive_d50"] == "True") == "determined" for r in c5)
        fpp = sum(float(r["ref_d50"]) < 4 and decide(float(r["DU_GC_d50"]), r["GC_inconclusive_d50"] == "True") == "determined" for r in c5)
        fn = sum(float(r["ref_d50"]) >= 4 and decide(float(r["DU_GC_d50"]), r["GC_inconclusive_d50"] == "True") != "determined" for r in c5)
        tn = len(c5) - tp - fpp - fn
        lab[err]["C5_confusion_d50"] = {"TP": tp, "FP": fpp, "FN": fn, "TN": tn}
    out["labelled"] = lab

    g = list(csv.DictReader(open(os.path.join(OUT, "graded_benchmark_cases.csv"), encoding="utf-8")))
    cells = sorted(set(r["cell"] for r in g), key=lambda s: [r["cell"] for r in g].index(s))
    table = []
    for cell in cells:
        C = [r for r in g if r["cell"] == cell]
        n = len(C)
        dec = [decide(float(r["DU_d30"]), r["inc_d30"] == "True") for r in C]
        det = [r for r, dd in zip(C, dec) if dd == "determined"]
        cov = sum(r["covers"] == "True" for r in C)
        err_det = [float(r["abs_err"]) for r in det]
        table.append({"cell": cell, "n": n, "coverage": [cov, cp(cov, n)],
                      "determined_d30": len(det), "ref_identifiable_d30": None if cell.startswith("miscal")
                      else sum(float(r["ref_d30"]) >= 4 for r in C),
                      "median_abs_err": float(np.median([float(r["abs_err"]) for r in C])),
                      "determined_err_median_p90_max": [float(np.median(err_det)), float(np.percentile(err_det, 90)),
                                                        float(max(err_det))] if err_det else None})
    out["graded_table"] = table
    ok = [r for r in g if r["cell"].startswith("s")]
    neg = [r for r in ok if float(r["ref_d30"]) < 4]
    pos = [r for r in ok if float(r["ref_d30"]) >= 4]
    dd = lambda r: decide(float(r["DU_d30"]), r["inc_d30"] == "True") == "determined"
    out["graded_correct_spec"] = {"det_given_ref_below4": [sum(map(dd, neg)), len(neg), cp(sum(map(dd, neg)), len(neg))],
                                  "det_given_ref_above4": [sum(map(dd, pos)), len(pos), cp(sum(map(dd, pos)), len(pos))],
                                  "coverage": [sum(r["covers"] == "True" for r in ok), len(ok)],
                                  "median_DU_over_ref": float(np.median([float(r["DU_d30"]) / float(r["ref_d30"])
                                                                         for r in ok if float(r["ref_d30"]) > 0.5]))}
    # sensitivity of the correctly specified graded cells to the cutoff c, and large errors among determined winds
    cut = {}
    for c in (2.71, 4.0, 6.63):
        dc = lambda r: decide(float(r["DU_d30"]), r["inc_d30"] == "True", c) == "determined"
        D = [r for r in ok if dc(r)]
        cut[str(c)] = {"det_given_ref_below4": [sum(map(dc, neg)), len(neg)], "det_given_ref_above4": [sum(map(dc, pos)), len(pos)],
                       "determined": len(D), "determined_err_above_30pct": sum(float(r["abs_err"]) > 0.3 for r in D)}
    out["graded_cutoff_sensitivity"] = cut
    # analytic coverage prediction for misspecified prior dispersion (sigma/D = 0.03, ell/D = 0.1)
    U = np.linspace(2.0, 7.0, 11)
    sdp = np.array([sd_logU(u) for u in U])
    EX = U * m.dL_anderson_raw(U) / (m.L_anderson_raw(U) - 1)

    def predicted(sa, st):
        return float(np.mean(2 * norm.cdf(1.96 * np.sqrt(sdp ** 2 + (sa / EX) ** 2) / np.sqrt(sdp ** 2 + (st / EX) ** 2)) - 1))
    out["coverage_predicted"] = {"miscal_s0.1_sig0.03": predicted(0.0, 0.1), "miscal_s0.3_sig0.03": predicted(0.0, 0.3),
                                 "underdisp_a0.1_t0.3_sig0.03": predicted(0.1, 0.3),
                                 "overdisp_a0.3_t0.1_sig0.03": predicted(0.3, 0.1)}
    # coverage of correctly specified cells by the standardised true multiplier
    sigma_of = lambda cell: float(cell.split("_")[0][1:])
    bins = {"<1": (0, 1), "1-2": (1, 2), ">=2": (2, np.inf)}
    by_tail = {}
    for name, (a, b) in bins.items():
        sel = [r for r in ok if a <= abs(np.log(float(r["s_true"]))) / sigma_of(r["cell"]) < b]
        k = sum(r["covers"] == "True" for r in sel)
        by_tail[name] = [k, len(sel), cp(k, len(sel)) if sel else None]
    out["coverage_by_standardised_s"] = by_tail
    # conditional (fixed-s) coverage of the prior-penalised profile interval: the estimate of log s stays at the prior
    # mean, so log U is biased by b = log s / E_X while its sampling SD is SD_pos; the interval half-width is
    # 1.96 sqrt(SD_pos^2 + (sigma_s/E_X)^2). Predicted per fire and compared with the simulated coverage by bin of
    # |log s| / sigma_s. SD_pos scales with sigma/D_h from its value at 0.03 (identifiability_extensions.sd_logU).
    edges = [0.0, 1.0, 1.5, 2.0, 2.5, np.inf]
    cond = []
    preds = []
    for r in ok:
        ss = sigma_of(r["cell"])
        sig_rel = float(r["cell"].split("_sig")[1])
        U = float(r["U_true"])
        sdp = sd_logU(U) * sig_rel / 0.03
        ex = U * m.dL_anderson_raw(U) / (m.L_anderson_raw(U) - 1)
        b = np.log(float(r["s_true"])) / ex
        tau = np.sqrt(sdp ** 2 + (ss / ex) ** 2)
        preds.append(float(norm.cdf((1.96 * tau - b) / sdp) - norm.cdf((-1.96 * tau - b) / sdp)))
    zs = np.array([abs(np.log(float(r["s_true"]))) / sigma_of(r["cell"]) for r in ok])
    covs = np.array([r["covers"] == "True" for r in ok])
    preds = np.array(preds)
    for a, b_ in zip(edges[:-1], edges[1:]):
        sel = (zs >= a) & (zs < b_)
        k = int(covs[sel].sum())
        cond.append({"bin": [a, None if np.isinf(b_) else b_], "n": int(sel.sum()), "covered": k,
                     "cp": cp(k, int(sel.sum())) if sel.any() else None, "predicted_mean": float(preds[sel].mean()) if sel.any() else None})
    rho = []
    for r in ok:
        U = float(r["U_true"])
        ex = U * m.dL_anderson_raw(U) / (m.L_anderson_raw(U) - 1)
        rho.append((sigma_of(r["cell"]) / ex) / (sd_logU(U) * float(r["cell"].split("_sig")[1]) / 0.03))
    out["conditional_coverage"] = {"bins": cond, "predicted_overall": float(preds.mean()), "rho_range": [float(min(rho)), float(max(rho))],
                                   "rho_median": float(np.median(rho))}
    json.dump(out, open(os.path.join(OUT, "benchmark_tables.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "graded_table"}, indent=1))
    for t in table:
        print(t["cell"], t["coverage"][0], t["determined_d30"], t["ref_identifiable_d30"], round(100 * t["median_abs_err"], 1),
              None if t["determined_err_median_p90_max"] is None else [round(100 * x, 1) for x in t["determined_err_median_p90_max"]])


if __name__ == "__main__":
    main()
