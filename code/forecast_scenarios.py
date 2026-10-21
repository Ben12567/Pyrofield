"""Forecast consequences of an unresolved historical wind.

For a homogeneous fire whose post-shift arrival is measured on a line normal to the new wind, the
delay after the shift is inversely proportional to the post-shift head rate (Eq. 7). A member with
historical wind error e0 (u = U0 (1 + e0)) receives the future wind U1 (1 + kappa e0): kappa = 0
when the future wind is known independently of the source implicit in the calibration, kappa = 1
when one multiplicative bias affects both periods. (kappa is the regression of the future on the
historical error; residual future-wind uncertainty adds a variance gamma(U1)^2 sigma_{1|0}^2 that
is common to all members and not evaluated here.) To first order the relative delay error is
[gamma(U0) - kappa gamma(U1)] e0 when members share phi_w; cancellation at kappa = 1 requires
gamma to be constant, i.e. 1 + phi_w to be a power law of U.

Spread laws: the reduced testbed, and Rothermel members compensated by dead fuel moisture or by
load at fixed packing ratio (phi_w then unchanged), under three wind-limit treatments and shift
magnitudes U1/U0 of 1.25-2.5. A flank target (line parallel to the new wind) is evaluated for the
testbed free law.
Output: output/forecast_scenarios.json
"""
import json
import os

import numpy as np

import pyrofield_reduced as m
from fire_behaviour import bed, member, ros

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
E0 = np.round(np.arange(-0.4, 0.6001, 0.02), 3)
KAPPAS = [0.0, 0.5, 0.8, 1.0]


def testbed_delay(e0, kappa, U0=4.0, shift=1.75):
    phi = m.wind_factor
    U1 = shift * U0
    return (1 + phi(U1)) / (1 + phi(U0)) * (1 + phi(U0 * (1 + e0))) / (1 + phi(U1 * (1 + kappa * e0)))


def rothermel_delay(key, mode, e0, kappa, limit, shift, U0=2.0, M0=0.06):
    mem = member(key, mode, M0, U0, U0 * (1 + e0), limit)
    if mem is None:
        return None
    fb, lam, M, _ = mem
    U1 = shift * U0
    return ros(bed(key), M0, U1, limit) / ros(fb, M, U1 * (1 + kappa * e0), limit)


def flank_delay(e0, U0=4.0, R0=0.10, k0=0.35, shift=1.75):
    Rh_star, L_star = m.head_rate(R0, U0), m.L_linear(U0, k0)
    U1 = shift * U0

    def b1(u):
        Rh1 = Rh_star * (1 + m.wind_factor(U1)) / (1 + m.wind_factor(u))
        a, b, _ = m.ellipse_axes(Rh1, 1 + (L_star - 1) * U1 / u)
        return b
    return b1(U0) / b1(U0 * (1 + e0))


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {"e0": E0.tolist()}
    out["testbed"] = {f"kappa={k}": [float(testbed_delay(e, k) - 1) for e in E0] for k in KAPPAS}
    out["testbed_flank_kappa0"] = [float(flank_delay(e) - 1) for e in E0]
    for key, mode in (("FM1", "moisture"), ("FM10", "load_beta")):
        out[f"{key}_{mode}_none_shift1.75"] = {
            f"kappa={k}": [None if (v := rothermel_delay(key, mode, e, k, "none", 1.75)) is None else float(v - 1) for e in E0]
            for k in KAPPAS}
    sweep = {}
    for key, mode in (("FM1", "moisture"), ("FM1", "load_beta"), ("FM4", "load_beta"), ("FM10", "moisture"), ("FM10", "load_beta")):
        for limit in ("none", "revised", "original"):
            for shift in (1.25, 1.5, 1.75, 2.0, 2.5):
                for e in (-0.2, 0.2):
                    for k in (0.0, 1.0):
                        v = rothermel_delay(key, mode, e, k, limit, shift)
                        sweep[f"{key},{mode},{limit},shift={shift},e0={e},kappa={k}"] = None if v is None else float(v - 1)
    out["sweep"] = sweep
    out["reference_fire_arrivals_min"] = {
        "kappa0": {str(e): 20 + 11.25 * float(testbed_delay(e, 0.0)) for e in (-0.4, 0.0, 0.6)},
        "kappa1": {str(e): 20 + 11.25 * float(testbed_delay(e, 1.0)) for e in (-0.4, 0.0, 0.6)}}
    bias = {}
    for s in (0.8, 1.2, 1.5):
        Uh = m.anderson_inverse(1 + s * (m.L_anderson_raw(4.0) - 1))
        bias[str(s)] = {"wind_ratio": Uh / 4.0, "delay_error_kappa0": float(testbed_delay(Uh / 4.0 - 1, 0.0) - 1)}
    out["miscalibrated_fixed_law_U0=4"] = bias
    json.dump(out, open(os.path.join(OUT, "forecast_scenarios.json"), "w"), indent=1)
    i = {e: list(E0).index(e) for e in (-0.2, 0.2)}
    print("testbed", {k: [round(100 * out['testbed'][k][j], 1) for j in i.values()] for k in out["testbed"]})
    print("flank", [round(100 * out["testbed_flank_kappa0"][j], 1) for j in i.values()])
    for key in ("FM1_moisture_none_shift1.75", "FM10_load_beta_none_shift1.75"):
        print(key, {k: [None if out[key][k][j] is None else round(100 * out[key][k][j], 1) for j in i.values()] for k in out[key]})
    for k, v in sweep.items():
        if "e0=0.2" in k:
            print(k, None if v is None else round(100 * v, 1))


if __name__ == "__main__":
    main()
