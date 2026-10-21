"""Sensitivity of the profile guard to the refit: start, number of updates and damping.

Classes C1 (free length-to-breadth coefficient, unidentifiable), C4 (fixed exact Anderson law,
identifiable) and C5 (uncertain multiplier, prior SD 0.3) of guard_benchmark.py, 30 fires each,
under both error models. For every fitted fire the guard (30% displacement, cutoff 4) is recomputed
with
  start   : fitted nuisance values | tangent predictor along the compensating direction
  updates : 1, 2, 4, 8 or 16 Levenberg-Marquardt updates (Marquardt damping lam0 * diag(J^T J))
  lam0    : 1e-6, 1e-3, 1
and with a converged trust-region refit from each start.
Output: output/refit_factorial.json
"""
import json
import os

import numpy as np
from scipy.optimize import least_squares

import guard_benchmark as gb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
UPDATES = [1, 2, 4, 8, 16]
LAMS = [1e-6, 1e-3, 1.0]


def nuisance(case, p_hat, z):
    idx = [0] + ([2] if case.free else [])

    def f(q):
        p = p_hat.copy()
        p[1] = z
        p[idx] = q
        return case.resid(p)
    return f, idx


def start_point(case, p_hat, z, idx, start):
    if start == "fitted":
        q0 = p_hat[idx].copy()
    else:
        r_hat = case.resid(p_hat)
        J = gb.fd_jac(case.resid, p_hat, r_hat)
        q0 = p_hat[idx] - np.linalg.pinv(J[:, idx]) @ J[:, 1] * (z - p_hat[1])
    return np.clip(q0, case.lo[idx] + 1e-9, case.hi[idx] - 1e-9)


def lm_path(f, q0, lo, hi, n_max, lam0):
    """Costs after 1..n_max Marquardt-damped LM updates."""
    q = q0.copy()
    r = f(q)
    cost = r @ r
    lam = lam0
    out = []
    for _ in range(n_max):
        J = gb.fd_jac(f, q, r)
        A, g = J.T @ J, J.T @ r
        step = np.linalg.solve(A + lam * np.diag(np.diag(A)) + 1e-12 * np.eye(len(q)), -g)
        q_new = np.clip(q + step, lo, hi)
        r_new = f(q_new)
        c_new = r_new @ r_new
        if c_new < cost:
            q, r, cost, lam = q_new, r_new, c_new, lam / 10
        else:
            lam *= 10
        out.append(cost)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    n = 30
    out = {}
    for ei, err in enumerate(("E2", "E1")):
        for ci, cls in enumerate(("C1", "C4", "C5")):
            rng = np.random.default_rng(9000 + 10 * ei + ci)
            counts = {}
            for _ in range(n):
                case = gb.Case(cls, err, rng)
                p_hat, chi_min = gb.fit_all(case, rng)
                for start in ("fitted", "predictor"):
                    sides = {}
                    for j in (-1, 1):
                        z = p_hat[1] + j * np.log(1.3)
                        f, idx = nuisance(case, p_hat, z)
                        q0 = start_point(case, p_hat, z, idx, start)
                        lo, hi = case.lo[idx], case.hi[idx]
                        for lam0 in LAMS:
                            path = lm_path(f, q0, lo, hi, max(UPDATES), lam0)
                            for k in UPDATES:
                                sides.setdefault(f"{start}_lam{lam0:g}_n{k}", []).append(path[k - 1])
                        res = least_squares(f, q0, bounds=(lo, hi), method="trf", xtol=1e-12, ftol=1e-12, max_nfev=400)
                        sides.setdefault(f"{start}_converged", []).append(2 * res.cost)
                    for key, v in sides.items():
                        counts[key] = counts.get(key, 0) + int(min(v) - chi_min >= 4.0)
            out[f"{err}_{cls}"] = {"n": n, "determined": counts}
            print(err, cls, {k: v for k, v in counts.items() if "lam0.001" in k or "converged" in k}, flush=True)
    json.dump(out, open(os.path.join(OUT, "refit_factorial.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
