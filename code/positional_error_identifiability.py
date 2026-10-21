"""Practical identifiability of wind speed from perimeters with positional (boundary) errors.

Observation model: front positions at 72 bearings in each of four frames (equally spaced in
time, the last when the generating head fire has advanced D), with Gaussian positional errors
of standard deviation sigma, exponentially correlated along the perimeter with length ell and
independent between frames. All lengths scale with D, so results depend only on sigma/D and
ell/D. The Anderson-FARSITE law is either exactly known (s = 1) or has an uncertain multiplier
with a log-normal prior of standard deviation sigma_s.

The local (linearised) standard deviation of log U after nuisance adjustment is the inverse
Schur complement of the information matrix, i.e. Eq. (projected information) of the
manuscript. A nonlinear expected profile checks the linearisation for one configuration.

Outputs: output/positional_identifiability.csv, output/positional_summary.json
"""
import json
import os

import numpy as np
from scipy.optimize import least_squares

import pyrofield_reduced as m

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
D = 1000.0          # reference head advance (m); only ratios matter
R0_TRUE = 0.10


def setup(U0, sig_rel, ell_rel):
    Rh0 = m.head_rate(R0_TRUE, U0)
    T = D / Rh0
    times = T * np.array([0.25, 0.5, 0.75, 1.0])
    return m.BoundaryOperator(times, sigma=sig_rel * D, corr_length=ell_rel * D,
                              Rh_ref=Rh0, L_ref=m.L_anderson(U0))


def model(op, logR0, logU, logs):
    U = np.exp(logU)
    return op(m.head_rate(np.exp(logR0), U), m.L_anderson(U, np.exp(logs)))


def sd_logU(U0, sig_rel, ell_rel, sigma_s):
    op = setup(U0, sig_rel, ell_rel)
    th = np.array([np.log(R0_TRUE), np.log(U0), 0.0])
    h = 1e-5
    cols = []
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        cols.append(op.whiten(model(op, *(th + e)) - model(op, *(th - e))) / (2 * h))
    J = np.array(cols).T
    if sigma_s == 0:
        J = J[:, :2]
        info = J.T @ J
    else:
        info = J.T @ J
        info[2, 2] += 1.0 / sigma_s ** 2
    return float(np.sqrt(np.linalg.inv(info)[1, 1]))


def nonlinear_profile(U0, sig_rel, ell_rel, sigma_s, ratios):
    op = setup(U0, sig_rel, ell_rel)
    y = model(op, np.log(R0_TRUE), np.log(U0), 0.0)
    out = []
    for r in ratios:
        lu = np.log(U0 * r)

        def res(p):
            body = op.whiten(model(op, p[0], lu, p[1]) - y)
            return np.append(body, p[1] / sigma_s)

        best = min((least_squares(res, p0, method="lm", xtol=1e-13, ftol=1e-13, gtol=1e-13)
                    for p0 in ([np.log(R0_TRUE), 0.0],
                               [np.log(m.head_rate(R0_TRUE, U0) / (1 + m.wind_factor(U0 * r))), 0.0])),
                   key=lambda f: f.cost)
        out.append(2 * best.cost)
    return np.array(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    sig_grid = np.geomspace(1e-3, 0.2, 28)
    rows = []
    for U0 in (3.0, 5.0):
        for ell in (0.0, 0.1, 0.3):
            for ss in (0.0, 0.1, 0.3):
                for sg in sig_grid:
                    rows.append({"U0": U0, "ell_rel": ell, "sigma_s": ss, "sigma_rel": float(sg),
                                 "sd_logU": sd_logU(U0, sg, ell, ss)})
    with open(os.path.join(OUT, "positional_identifiability.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(rows[0]) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r.values()) + "\n")

    summary = {}
    for U0 in (3.0, 5.0):
        for sg in (0.01, 0.03, 0.05, 0.1):
            for ell in (0.0, 0.1, 0.3):
                for ss in (0.0, 0.1, 0.3):
                    summary[f"U0={U0},sigma/D={sg},ell/D={ell},sigma_s={ss}"] = round(100 * sd_logU(U0, sg, ell, ss), 2)
    # Nonlinear check of the linearisation (95% expected-profile interval vs +-1.96 SD)
    U0, sg, ell, ss = 5.0, 0.03, 0.1, 0.1
    ratios = np.exp(np.linspace(np.log(0.7), np.log(1.4), 141))
    prof = nonlinear_profile(U0, sg, ell, ss, ratios)
    inside = ratios[prof <= 3.84]
    lin = sd_logU(U0, sg, ell, ss)
    summary["nonlinear_check"] = {
        "config": {"U0": U0, "sigma/D": sg, "ell/D": ell, "sigma_s": ss},
        "profile_95_interval_ratio": [float(inside.min()), float(inside.max())],
        "linearised_95_interval_ratio": [float(np.exp(-1.96 * lin)), float(np.exp(1.96 * lin))],
    }
    np.savetxt(os.path.join(OUT, "positional_profile_check.csv"), np.c_[ratios, prof], delimiter=",",
               header="ratio,delta_chi2", comments="")
    json.dump(summary, open(os.path.join(OUT, "positional_summary.json"), "w"), indent=1)
    print(json.dumps(summary["nonlinear_check"], indent=1))
    for k in [k for k in summary if k.startswith("U0=5.0,sigma/D=0.03") or k.startswith("U0=3.0,sigma/D=0.05")]:
        print(k, summary[k], "%")


if __name__ == "__main__":
    main()
