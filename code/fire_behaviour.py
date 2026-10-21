"""Fireline intensity, flame length and scorch height of fit-equivalent fires in Rothermel's model.

For a reference fire (dead moisture M0, live moisture 100 %, midflame wind U0) with head-fire rate
R*, a trial wind u is matched by one compensating attribute that restores R*:
  'moisture'     all dead size classes shifted together (2 % <= M <= 0.98 Mx);
  'load_beta'    all loads and the depth scaled by mu in [0.25, 4] (packing ratio fixed);
  'ros_factor'   a rate-of-spread adjustment factor R*/R(u) applied to the unchanged fuel bed.
Trial winds span the window implied by Eq. (5) for a shape-law calibration uncertainty sigma_s:
U0 exp(+-1.96 sigma_s / E_X(U0)).
Fireline intensity (Byram's quantity) is estimated in two ways:
  'reaction'     I = I_R t_r R, with t_r = 384/sigma (BEHAVE convention; Albini, 1976);
  'consumption'  I = H_y w_1 R with complete flaming consumption of the 1-h dead load w_1 and heat
                 yield H_y = 18,608 - 2,442 M kJ/kg (the standard heat content of 8,000 Btu/lb less
                 the latent heat of vaporisation of the fuel moisture).
Flame length L_f = 0.0775 I^0.46 (Byram, 1959). Scorch height follows Van Wagner (1973) in the
English-unit form used by BEHAVE: h_s [ft] = 63/(140 - T) I^(7/6) / (I + U^3)^(1/2), with I in
Btu/ft/s, U the midflame wind in mi/h and T = 77 F. Wind-limit treatments: 'none' (spread rate
capped at the midflame wind speed), 'revised' (96.8 I_R^(1/3) ft/min) and 'original' (0.9 I_R ft/min).

Load scaled at fixed depth (packing ratio changes) is not used as a compensator: R is nearly flat or
non-monotone in the load multiplier, so no unique compensating load exists; the diagnostic
'load_depth_response' records R(mu)/R(1) at the reference state.

Outputs: output/fire_behaviour.json, output/fire_behaviour_members.csv
"""
import json
import os

import numpy as np
from scipy.optimize import brentq

import pyrofield_reduced as m
from rothermel import BTU_FT_S_TO_KW_M, FT_PER_M, FUEL_MODELS, FuelBed, T_AC_TO_LB_FT2, flame_length_m, validate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FUELS = ["FM1", "FM2", "FM3", "FM4", "FM8", "FM9", "FM10"]
MODES = ("moisture", "load_beta", "ros_factor")
M_LIVE = 1.0
MU_BOUNDS = (0.25, 4.0)
LB_FT2_TO_KG_M2 = 4.88243
H_C = 18608.0          # kJ/kg (8000 Btu/lb)
T_AMBIENT_F = 77.0
MS_TO_MPH = 2.23694


def bed(key, mu=1.0, mode="load_depth"):
    p = dict(FUEL_MODELS[key])
    p["load"] = tuple(mu * w for w in p["load"])
    if mode == "load_beta":
        p["depth"] = p["depth"] * mu
    return FuelBed(**p)


def wind_cap(fb, M, u, limit):
    if limit == "revised":
        return min(u, 96.8 * fb.I_R(M, M_LIVE) ** (1 / 3) / (FT_PER_M * 60.0))
    if limit == "original":
        return min(u, 0.9 * fb.I_R(M, M_LIVE) / (FT_PER_M * 60.0))
    return u


def ros(fb, M, u, limit="none"):
    """Head-fire rate in m/min."""
    R = fb.ros(M, wind_cap(fb, M, u, limit), M_LIVE) / FT_PER_M
    return min(R, u * 60.0)


def intensity(fb, key, mu, M, R, convention):
    if convention == "reaction":
        return fb.I_R(M, M_LIVE) * (384.0 / fb.sigma) * (R * FT_PER_M) / 60.0 * BTU_FT_S_TO_KW_M
    w1 = mu * FUEL_MODELS[key]["load"][0] * T_AC_TO_LB_FT2 * LB_FT2_TO_KG_M2
    return (H_C - 2442.0 * M) * w1 * R / 60.0


def scorch_height_m(I_kw_m, u_ms):
    I = I_kw_m / BTU_FT_S_TO_KW_M
    U = u_ms * MS_TO_MPH
    return 63.0 / (140.0 - T_AMBIENT_F) * I ** (7.0 / 6.0) / np.sqrt(I + U ** 3) / FT_PER_M


def E_X(U):
    return U * m.dL_anderson_raw(U) / (m.L_anderson_raw(U) - 1.0)


def window(U0, sigma_s):
    h = 1.96 * sigma_s / E_X(U0)
    return float(np.exp(-h)), float(np.exp(h))


def member(key, mode, M0, U0, u, limit="none"):
    """Compensating attribute restoring R*(M0, U0) at wind u; returns (fuel bed, mu, M, R*) or None."""
    fb0 = bed(key)
    Rstar = ros(fb0, M0, U0, limit)
    if mode == "ros_factor":
        return fb0, 1.0, M0, Rstar
    if mode == "moisture":
        f = lambda M: ros(fb0, M, u, limit) - Rstar
        lo, hi = 0.02, 0.98 * fb0.mx_dead
    else:
        f = lambda mu: ros(bed(key, mu, mode), M0, u, limit) - Rstar
        lo, hi = MU_BOUNDS
    try:
        if f(lo) * f(hi) > 0:
            return None
        x = brentq(f, lo, hi, xtol=1e-10)
    except ValueError:
        return None
    mu, M = (1.0, x) if mode == "moisture" else (x, M0)
    fb = bed(key, mu, "load_beta") if mode == "load_beta" else fb0
    if abs(ros(fb, M, u, limit) - Rstar) > 1e-6 * Rstar:
        return None
    return fb, mu, M, Rstar


def window_summary(key, mode, M0, U0, sigma_s, limit="none", n=41):
    lo_r, hi_r = window(U0, sigma_s)
    rows = []
    for r in np.linspace(lo_r, hi_r, n):
        mem = member(key, mode, M0, U0, U0 * r, limit)
        if mem is None:
            continue
        fb, mu, M, R = mem
        row = {"ratio": float(r), "mu": mu, "M": M}
        for c in ("reaction", "consumption"):
            I = intensity(fb, key, mu, M, R, c)
            row[f"I_{c}"] = I
            row[f"Lf_{c}"] = float(flame_length_m(I))
            row[f"hs_{c}"] = float(scorch_height_m(I, U0 * r))
        rows.append(row)
    if len(rows) < 2:
        return None, rows
    s = {"u_ratio": [rows[0]["ratio"], rows[-1]["ratio"]], "window": [lo_r, hi_r],
         "mu": [min(r["mu"] for r in rows), max(r["mu"] for r in rows)],
         "M": [min(r["M"] for r in rows), max(r["M"] for r in rows)]}
    for c in ("reaction", "consumption"):
        I = [r[f"I_{c}"] for r in rows]
        s[f"{c}_maxmin"] = float(max(I) / min(I))
        for q in ("Lf", "hs"):
            v = [r[f"{q}_{c}"] for r in rows]
            s[f"{q}_{c}_m"] = [float(min(v)), float(max(v))]
        s[f"hs_{c}_maxmin"] = s[f"hs_{c}_m"][1] / s[f"hs_{c}_m"][0]
    return s, rows


def load_depth_response(key, M0=0.06, U0=2.0):
    R1 = ros(bed(key), M0, U0)
    return {str(mu): float(ros(bed(key, mu, "load_depth"), M0, U0) / R1) for mu in (0.25, 0.5, 1.0, 2.0, 4.0)}


def limit_sensitivity(n=9):
    """Member evaluations (sigma_s = 0.3 windows, 9 trial winds) whose compensating attribute or
    admissibility changes under a wind limit, and the intensity ratios under each limit, over all
    fuels and reference states."""
    out = {}
    for limit in ("revised", "original"):
        changed, total = 0, 0
        for key in FUELS:
            for M0 in (0.04, 0.06, 0.08):
                for U0 in (1.0, 2.0, 3.0):
                    lo, hi = window(U0, 0.3)
                    for r in np.linspace(lo, hi, n):
                        for mode in ("moisture", "load_beta"):
                            a = member(key, mode, M0, U0, U0 * r, "none")
                            b = member(key, mode, M0, U0, U0 * r, limit)
                            if a is None and b is None:
                                continue
                            total += 1
                            if (a is None) != (b is None) or abs(a[1] - b[1]) > 1e-6 or abs(a[2] - b[2]) > 1e-6:
                                changed += 1
        ratios = {}
        for mode in ("moisture", "load_beta"):
            for ss in (0.1, 0.3):
                v = [window_summary(k, mode, M0, U0, ss, limit)[0] for k in FUELS for M0 in (0.04, 0.06, 0.08)
                     for U0 in (1.0, 2.0, 3.0)]
                v = [x for x in v if x is not None]
                ratios[f"{mode},sigma_s={ss}"] = {"reaction_maxmin": [min(x["reaction_maxmin"] for x in v),
                                                                      max(x["reaction_maxmin"] for x in v)],
                                                  "n_compensable": len(v)}
        out[limit] = {"members_changed": changed, "members_admissible": total, "ratios": ratios}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {"validation_anderson1982": [{"fuel": k, "ros_ch_h": round(r, 3), "ref_ros": rr, "flame_ft": round(fl, 2),
                                        "ref_flame": fr} for k, r, rr, fl, fr in validate()],
           "E_X": {str(U): E_X(U) for U in (1.0, 2.0, 3.0)},
           "windows": {f"U0={U},sigma_s={s}": window(U, s) for U in (1.0, 2.0, 3.0) for s in (0.1, 0.3)}}
    members_out = []
    for key in FUELS:
        fb = bed(key)
        out[key] = {"name": FUEL_MODELS[key]["name"], "phi_w_2ms": float(fb.phi_w(2.0)), "gamma_2ms": float(fb.gamma(2.0)),
                    "beta_ratio": float(fb.ratio), "load_depth_response": load_depth_response(key)}
        for M0 in (0.04, 0.06, 0.08):
            for U0 in (1.0, 2.0, 3.0):
                for ss in (0.1, 0.3):
                    for mode in MODES:
                        s, rows = window_summary(key, mode, M0, U0, ss)
                        out[key][f"M0={M0},U0={U0},sigma_s={ss},{mode}"] = s
                        if M0 == 0.06 and U0 == 2.0:
                            members_out += [dict(fuel=key, mode=mode, sigma_s=ss, **r) for r in rows]
        print(key, "done", flush=True)
    groups = {"grass_shrub": ["FM1", "FM2", "FM3", "FM4"], "litter": ["FM8", "FM9", "FM10"], "all": FUELS}
    summ = {}
    for g, fuels in groups.items():
        for ss in (0.1, 0.3):
            for mode in MODES:
                vals = [out[k][f"M0={M0},U0={U0},sigma_s={ss},{mode}"] for k in fuels for M0 in (0.04, 0.06, 0.08)
                        for U0 in (1.0, 2.0, 3.0)]
                ok = [v for v in vals if v is not None]

                def rng(key):
                    return [float(min(v[key] for v in ok)), float(max(v[key] for v in ok))] if ok else None
                summ[f"{g},sigma_s={ss},{mode}"] = {
                    "n_states": len(vals), "n_compensable": len(ok),
                    "n_full_window": int(sum(abs(v["u_ratio"][0] - v["window"][0]) < 1e-9 and
                                             abs(v["u_ratio"][1] - v["window"][1]) < 1e-9 for v in ok)),
                    "reaction_maxmin": rng("reaction_maxmin"), "consumption_maxmin": rng("consumption_maxmin"),
                    "hs_reaction_maxmin": rng("hs_reaction_maxmin"), "hs_consumption_maxmin": rng("hs_consumption_maxmin"),
                    "mu": [float(min(v["mu"][0] for v in ok)), float(max(v["mu"][1] for v in ok))] if ok else None}
    out["summary"] = summ
    out["wind_limit_sensitivity"] = limit_sensitivity()
    json.dump(out, open(os.path.join(OUT, "fire_behaviour.json"), "w"), indent=1)
    with open(os.path.join(OUT, "fire_behaviour_members.csv"), "w", encoding="utf-8") as f:
        keys = list(members_out[0].keys())
        f.write(",".join(keys) + "\n")
        for r in members_out:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    for k, v in summ.items():
        print(k, v)
    print(json.dumps(out["wind_limit_sensitivity"], indent=1))


if __name__ == "__main__":
    main()
