"""Demonstration with experimental grassland fires: the 1986 Annaburroo point-ignition fires.

Data: CSIRO Data Access Portal, doi:10.25919/ycd3-w209 (CC BY 4.0). Download the dataset zip, extract it
to data_external/annaburroo/raw (or pass the folder as the first argument) and run
    python annaburroo_demo.py [raw_dir]
For every fire listed in fires.json the hand-placed anchors (anchors/<fire>.json) are snapped to the
drawn isochrone lines (snap_isochrones.py), converted to metres about the ignition point and fitted
with the elliptical Huygens wavelet of the main text:
  * per isochrone, with the rear focus at the ignition point (primary): elongation L, bearing, size.
    Uncertainty from a fixed-design residual circular block bootstrap along each digitised line
    (2,000 replicates; block length = smallest lag at which the residual autocorrelation falls
    below 0.2, at least 25 points); spread summarised robustly (IQR/1.349) with its Monte Carlo
    error. Because the residuals are systematic departures of the drawn line from an ellipse, this
    SD measures shape discrepancy as well as digitising noise;
  * a model-free elongation, (head distance + fitted backing distance) / maximum width, for
    isochrones digitised on both flanks; fits with the focus free;
  * jointly, under constant forcing (one head rate, one L, an effective ignition time).
The summary isochrone of a fire is its latest isochrone digitised on both flanks.
Wind is the mean of the four plot-corner anemometers from the 5-s logs from ignition to the isochrone
time (primary), or over the preceding burn period, weighted by the head-fire advance of each burn
period, or at the windiest or calmest corner; used at 2 m or converted with the dataset's median
ratios (line-ignition plots of the same fuel treatment, or of the same plot) to midflame and 10-m
wind, or to a FARSITE-style midflame wind (Albini-Baughman unsheltered adjustment factor for the
recorded fuel height, as given by Andrews (2012), applied to the 6.1-m wind taken as the 10-m wind
divided by an assumed 1.15). The multiplier s = (L - 1)/(L_A(U) - 1) of the Anderson (1983) law is
summarised by its mean log (bias) and SD over fires and over plots.
Outputs: ../output/annaburroo_demo.json, ../output/annaburroo_isochrones.csv, ../output/annaburroo_snapped.json
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import beta, chi2, norm, t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import pyrofield_reduced as m  # noqa: E402
from snap_isochrones import red_mask, snap_curve  # noqa: E402

OUT = os.path.join(HERE, "..", "output")
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "data_external", "annaburroo", "raw")
META = json.load(open(os.path.join(HERE, "fires.json"), encoding="utf-8"))
CORNERS = ("NE2mWind.ms", "NW2mWind.ms", "SW2mWind.ms", "SE2mWind.ms")
NBOOT = 2000
GUARD_SD = np.log(1.3) / 2.0
U10_TO_U61 = 1.0 / 1.15
FT_PER_M = 3.28084


def EX(U):
    return U * m.dL_anderson_raw(U) / (m.L_anderson_raw(U) - 1)


def waf_unsheltered(depth_m):
    """Albini and Baughman (1979) unsheltered wind adjustment factor, as given by Andrews (2012)."""
    H = depth_m * FT_PER_M
    return 1.83 / np.log((20.0 + 0.36 * H) / (0.13 * H))


def hms(s):
    s = s.strip().zfill(6)
    return int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])


def cp(k, n):
    return [float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0, float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]


def sd_ci(x):
    """SD with its normal-theory chi-square 95% interval."""
    n, sd = len(x), float(np.std(x, ddof=1))
    return [sd, float(sd * np.sqrt((n - 1) / chi2.ppf(0.975, n - 1))), float(sd * np.sqrt((n - 1) / chi2.ppf(0.025, n - 1)))]


def robust_sd(v):
    q1, q3 = np.percentile(v, [25, 75])
    return float((q3 - q1) / 1.349)


def r_model(phi, a, L, th):
    e = np.sqrt(1 - 1 / L ** 2)
    return a * (1 - e ** 2) / (1 - e * np.cos(phi - th))


def radial(x, y, a, L, th, x0=0.0, y0=0.0):
    dx, dy = x - x0, y - y0
    return np.hypot(dx, dy) - r_model(np.arctan2(dy, dx), a, L, th)


def acf(r, k):
    return float(np.corrcoef(r[:-k], r[k:])[0, 1]) if len(r) > k + 3 else np.nan


def fit_fixed(x, y, th0, q0=None):
    q0 = [np.log(np.max(np.hypot(x, y)) / 2), np.log(1.5), th0] if q0 is None else q0
    return least_squares(lambda q: radial(x, y, np.exp(q[0]), 1 + np.exp(q[1]), q[2]), q0, x_scale=[0.3, 0.3, 0.1])


def fit_free(x, y, q0):
    return least_squares(lambda q: radial(x, y, np.exp(q[0]), 1 + np.exp(q[1]), q[2], q[3], q[4]),
                         list(q0) + [0.0, 0.0], x_scale=[0.3, 0.3, 0.1, 2.0, 2.0])


def block_length(curves, q):
    res = [radial(x, y, np.exp(q[0]), 1 + np.exp(q[1]), q[2]) for x, y in curves]
    nmax = max(len(r) for r in res)
    for k in range(1, nmax // 3):
        vals = [acf(r, k) for r in res if len(r) > k + 3]
        if vals and np.nanmean(vals) < 0.2:
            return max(25, k)
    return max(25, nmax // 3)


def residual_block_bootstrap(curves, q, rng, block):
    """Fixed-design residual bootstrap: residuals along each line are resampled in circular blocks
    and added to the fitted front at the observed bearings; returns robust SD of log(L-1) and its MC error."""
    a, L, th = np.exp(q[0]), 1 + np.exp(q[1]), q[2]
    parts = []
    for x, y in curves:
        phi = np.arctan2(y, x)
        rm = r_model(phi, a, L, th)
        parts.append((phi, rm, np.hypot(x, y) - rm))
    est = []
    for _ in range(NBOOT):
        bx, by = [], []
        for phi, rm, e in parts:
            n = len(e)
            b = min(block, n)
            starts = rng.integers(0, n, int(np.ceil(n / b)))
            idx = np.concatenate([(s0 + np.arange(b)) % n for s0 in starts])[:n]
            rs = rm + e[idx]
            bx.append(rs * np.cos(phi))
            by.append(rs * np.sin(phi))
        f = fit_fixed(np.concatenate(bx), np.concatenate(by), th, q)
        est.append(f.x[1])
    est = np.array(est)
    batches = [robust_sd(b) for b in np.array_split(est, 4)]
    return robust_sd(est), float(np.std(est, ddof=1)), float(np.std(batches, ddof=1) / 2.0)


def model_free_L(x, y, a, L, th):
    u = x * np.cos(th) + y * np.sin(th)
    v = -x * np.sin(th) + y * np.cos(th)
    e = np.sqrt(1 - 1 / L ** 2)
    back = max(-np.min(u), a * (1 - e))
    return float((np.max(u) + back) / (np.max(v) - np.min(v)))


def fit_joint(groups, th0):
    def res(q):
        Rh, L, th, t0 = np.exp(q[0]), 1 + np.exp(q[1]), q[2], q[3]
        e = np.sqrt(1 - 1 / L ** 2)
        a_rate = (Rh + Rh * (1 - e) / (1 + e)) / 2
        return np.concatenate([radial(x, y, a_rate * (t - t0), L, th) for t, (x, y) in groups.items()])
    f = least_squares(res, [np.log(0.5), np.log(1.5), th0, 0.0], x_scale=[0.3, 0.3, 0.1, 20])
    return {"Rh": float(np.exp(f.x[0])), "L": float(1 + np.exp(f.x[1])), "bearing_rad": float(f.x[2]), "t0": float(f.x[3]),
            "resid_sd_m": float(np.std(f.fun)), "n_par": 4}


def period_rows(fid):
    rows = [x for x in csv.DictReader(open(os.path.join(RAW, "Burn_Period_Data.csv"), encoding="utf-8-sig")) if x["FireID"] == fid]
    for x in rows:
        x["t"] = float(x["TimeSinceIgn"])
    return rows


def plot_midflame_ratio(ids):
    rows = list(csv.DictReader(open(os.path.join(RAW, "Plot_Average_Data.csv"), encoding="utf-8-sig")))
    q = [float(r["MidFlwind.ms"]) / float(r["2mWind.ms"]) for r in rows if r["FireID"] in ids]
    return float(np.mean(q)) if q else None


def wind_log(path):
    rows = list(csv.DictReader(open(os.path.join(RAW, path), encoding="utf-8-sig")))
    t, corner = [], []
    for r in rows:
        keys = {k.strip(): v for k, v in r.items()}
        w = []
        for c in CORNERS:
            try:
                w.append(float(keys.get(c, "")))
            except (TypeError, ValueError):
                w.append(np.nan)
        if np.isfinite(w).any():
            t.append(hms(keys["Time.hhmmss"]))
            corner.append(w)
    return np.array(t), np.array(corner)


def window(log, t_ign, t0, t1):
    tt, cc = log
    sel = (tt > t_ign + t0) & (tt <= t_ign + t1 + 1e-6)
    return cc[sel]


def summarise(logs, plots):
    logs = np.asarray(logs)
    pm = {}
    for v, p in zip(logs, plots):
        pm.setdefault(p, []).append(v)
    pmeans = np.array([np.mean(v) for v in pm.values()])
    n = len(logs)
    return {"n_fires": int(n), "mean_log_s": float(np.mean(logs)), "se_mean": float(np.std(logs, ddof=1) / np.sqrt(n)),
            "sd_fires": sd_ci(logs), "n_plots": int(len(pmeans)), "mean_log_s_plots": float(np.mean(pmeans)),
            "sd_plots": sd_ci(pmeans), "n_s_gt_1": int(np.sum(logs > 0))}


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1986)
    masks, snapped, iso_rows, fires = {}, {}, [], {}
    for fid, F in META["fires"].items():
        path = os.path.join(RAW, F["map"])
        if path not in masks:
            masks[path] = red_mask(path)
        anchors = json.load(open(os.path.join(HERE, "anchors", fid + ".json"), encoding="utf-8"))["curves"]
        ix, iy = F["ignition_px"]
        ppm = F["px_per_m"]
        curves, one_flank = {}, set()
        snapped[fid] = {}
        for name, t in F["curves"].items():
            pts = snap_curve(masks[path], anchors[name], F["search_px"], F["jump_px"])
            snapped[fid][name] = pts
            P = np.array(pts)
            curves.setdefault(float(t), []).append(((P[:, 0] - ix) / ppm, -(P[:, 1] - iy) / ppm))
            if name in F.get("one_flank", []):
                one_flank.add(float(t))
        rows = period_rows(fid)
        fuel = rows[0]["FuelTreat"]
        depth = float(rows[0]["FuelHt.m"])
        t_ign = hms(rows[0]["IgnTime.hhmmss"])
        wlog = wind_log(F["wind_log"])
        ratios = {"mf": META["midflame_ratio"][fuel], "2m": 1.0, "10m": META["ten_m_ratio"][fuel],
                  "farsite": float(waf_unsheltered(depth) * U10_TO_U61 * META["ten_m_ratio"][fuel])}
        r_plot = plot_midflame_ratio(F["plot_line_fires"])
        plot_ratio_available = r_plot is not None
        r_plot = r_plot if plot_ratio_available else ratios["mf"]
        groups = {t: (np.concatenate([c[0] for c in cs]), np.concatenate([c[1] for c in cs])) for t, cs in curves.items()}
        allx = np.concatenate([g[0] for g in groups.values()])
        ally = np.concatenate([g[1] for g in groups.values()])
        k = int(np.argmax(np.hypot(allx, ally)))
        th0 = float(np.arctan2(ally[k], allx[k]))
        per = {}
        times = sorted(float(r["t"]) for r in rows)
        for t, (x, y) in sorted(groups.items()):
            f = fit_fixed(x, y, th0)
            a, L, th = np.exp(f.x[0]), 1 + np.exp(f.x[1]), f.x[2]
            e = np.sqrt(1 - 1 / L ** 2)
            cov = np.linalg.pinv(f.jac.T @ f.jac) * np.mean(f.fun ** 2)
            ff = fit_free(x, y, f.x)
            blk = block_length(curves[t], f.x)
            sd_r, sd_c, sd_mc = residual_block_bootstrap(curves[t], f.x, rng, blk)
            sd50, _, _ = residual_block_bootstrap(curves[t], f.x, rng, 50)
            head = float(a * (1 + e))
            flanks = 1 if t in one_flank else 2
            Lmf = model_free_L(x, y, a, L, th) if flanks == 2 else None
            t_prev = max([u for u in times if u < t - 1e-6], default=0.0)
            cw = window(wlog, t_ign, 0.0, t)
            corner_means = np.nanmean(cw, axis=0)
            U2 = float(np.nanmean(np.nanmean(cw, axis=1)))
            U2_last = float(np.nanmean(np.nanmean(window(wlog, t_ign, t_prev, t), axis=1)))
            num = den = 0.0
            tp = 0.0
            for r in rows:
                if r["t"] > t + 1e-6:
                    break
                wp = window(wlog, t_ign, tp, r["t"])
                if len(wp):
                    num += float(r["IncDist.m"]) * float(np.nanmean(np.nanmean(wp, axis=1)))
                    den += float(r["IncDist.m"])
                tp = r["t"]
            U2_dist = num / den
            cum = [float(r["CumDist.m"]) for r in rows if abs(r["t"] - t) < 1e-6][0]
            fr = {"t_s": t, "flanks": flanks, "L": float(L), "L_model_free": Lmf, "sd_logLm1_naive": float(np.sqrt(cov[1, 1])),
                  "block_points": int(blk), "sd_logLm1_boot": sd_r, "sd_logLm1_boot_classical": sd_c, "sd_logLm1_boot_mcerr": sd_mc,
                  "sd_logLm1_boot50": sd50, "a_m": float(a), "head_m": head, "cumdist_m": cum, "bearing_rad": float(th),
                  "resid_sd_m": float(np.std(f.fun)), "lag1": acf(f.fun, 1), "n": int(len(x)),
                  "nearest_point_frac_of_head": float(np.min(np.hypot(x, y)) / head),
                  "L_free_focus": float(1 + np.exp(ff.x[1])), "focus_offset_m": float(np.hypot(ff.x[3], ff.x[4])),
                  "U2_ms": U2, "U2_lastperiod_ms": U2_last, "U2_distweighted_ms": float(U2_dist),
                  "U2_windiest_corner_ms": float(np.nanmax(corner_means)), "U2_calmest_corner_ms": float(np.nanmin(corner_means)),
                  "corner_cv": float(np.nanstd(corner_means, ddof=1) / np.nanmean(corner_means)),
                  "ratio_farsite": ratios["farsite"], "ratio_plot_midflame": r_plot, "plot_ratio_available": plot_ratio_available}
            winds = {"ign": U2, "last": U2_last, "dist": U2_dist, "windiest": fr["U2_windiest_corner_ms"],
                     "calmest": fr["U2_calmest_corner_ms"]}
            for tag, ratio in ratios.items():
                for win, Uw in winds.items():
                    U = ratio * Uw
                    fr[f"s_{tag}_{win}"] = float((L - 1) / (m.L_anderson_raw(U) - 1))
                    fr[f"U_{tag}_{win}_ms"] = float(U)
            fr["s_plotmf_ign"] = float((L - 1) / (m.L_anderson_raw(r_plot * U2) - 1))
            fr["s_mf_ign_modelfree"] = float((Lmf - 1) / (m.L_anderson_raw(fr["U_mf_ign_ms"]) - 1)) if Lmf else None
            fr["s_mf_ign_freefocus"] = float((fr["L_free_focus"] - 1) / (m.L_anderson_raw(fr["U_mf_ign_ms"]) - 1))
            fr["U_anderson_ms"] = float(m.anderson_inverse(L)) if L < m.L_CAP else None
            fr["L_anderson_farsite"] = float(m.L_anderson_raw(fr["U_farsite_ign_ms"]))
            per[str(int(t))] = fr
            iso_rows.append({"fire": fid, "plot": F["plot"], "fuel": fuel, **{k2: v for k2, v in fr.items() if k2 != "bearing_rad"}})
        joint = fit_joint(dict(sorted(groups.items())), th0) if len(groups) > 1 else None
        two = [k for k, v in per.items() if v["flanks"] == 2]
        skey = max(two, key=float)
        fires[fid] = {"fuel": fuel, "plot": F["plot"], "neighbours": F.get("neighbours"), "isochrones": per,
                      "summary_isochrone": skey, "joint_constant_forcing": joint, "scale_source": F["scale_source"]}
        print(fid, "; ".join(f"t={v['t_s']:.0f}{'*' if k == skey else ''} L={v['L']:.2f} mfree={v['L_model_free']} "
                             f"SD {v['sd_logLm1_boot']:.3f}+-{v['sd_logLm1_boot_mcerr']:.3f} (classical {v['sd_logLm1_boot_classical']:.3f}, "
                             f"50pt {v['sd_logLm1_boot50']:.3f}) s_mf={v['s_mf_ign']:.2f}" for k, v in per.items()), flush=True)

    fids = list(fires)
    plots = [fires[f]["plot"] for f in fids]
    last = {f: fires[f]["isochrones"][fires[f]["summary_isochrone"]] for f in fids}
    S = {}
    for tag in ("mf", "2m", "10m", "farsite"):
        for win in ("ign", "last", "dist", "windiest", "calmest"):
            logs = [np.log(last[f][f"s_{tag}_{win}"]) for f in fids]
            S[f"{tag}_{win}"] = summarise(logs, plots)
            S[f"{tag}_{win}"]["median_UA_over_U"] = float(np.median([last[f]["U_anderson_ms"] / last[f][f"U_{tag}_{win}_ms"] for f in fids]))
    sens = {}
    allrows = [(f, v) for f in fids for v in fires[f]["isochrones"].values()]
    sens["all_isochrones_mf_ign"] = summarise([np.log(v["s_mf_ign"]) for f, v in allrows], [fires[f]["plot"] for f, v in allrows])
    sens["model_free_L_mf_ign"] = summarise([np.log(last[f]["s_mf_ign_modelfree"]) for f in fids], plots)
    sens["model_free_L_median_UA_over_U"] = float(np.median([m.anderson_inverse(last[f]["L_model_free"]) / last[f]["U_mf_ign_ms"] for f in fids]))
    sens["model_free_minus_fitted_L"] = [float(min(last[f]["L_model_free"] / last[f]["L"] - 1 for f in fids)),
                                         float(max(last[f]["L_model_free"] / last[f]["L"] - 1 for f in fids))]
    sens["free_focus_mf_ign"] = summarise([np.log(last[f]["s_mf_ign_freefocus"]) for f in fids], plots)
    sens["plot_midflame_ratio_ign"] = summarise([np.log(last[f]["s_plotmf_ign"]) for f in fids], plots)
    robust = [f for f in fids if last[f]["nearest_point_frac_of_head"] <= 0.2]
    sens["focus_robust_fires"] = robust
    sens["focus_robust_mf_ign"] = summarise([np.log(last[f]["s_mf_ign"]) for f in robust], [fires[f]["plot"] for f in robust])
    keep = [f for f in fids if f not in ("F213", "ES12")]
    sens["without_F213_ES12_mf_ign"] = summarise([np.log(last[f]["s_mf_ign"]) for f in keep], [fires[f]["plot"] for f in keep])
    keep = [f for f in fids if fires[f]["plot"] != "F25"]
    sens["without_plot_F25_mf_ign"] = summarise([np.log(last[f]["s_mf_ign"]) for f in keep], [fires[f]["plot"] for f in keep])
    sens["within_plot_F25_sd"] = float(np.std([np.log(last[f]["s_mf_ign"]) for f in fids if fires[f]["plot"] == "F25"], ddof=1))
    sig = S["mf_ign"]["sd_fires"][0]
    sens["noise_budget"] = {"sigma_s_hat": sig,
                            "median_shape_sd": float(np.median([last[f]["sd_logLm1_boot"] for f in fids])),
                            "rms_shape_sd": float(np.sqrt(np.mean([last[f]["sd_logLm1_boot"] ** 2 for f in fids]))),
                            "rms_corner_cv_times_EX": float(np.sqrt(np.mean([(EX(last[f]["U_mf_ign_ms"]) * last[f]["corner_cv"]) ** 2 for f in fids]))),
                            "corner_cv_range": [float(min(last[f]["corner_cv"] for f in fids)), float(max(last[f]["corner_cv"] for f in fids))]}
    sens["farsite_waf_range"] = [float(min(last[f]["ratio_farsite"] for f in fids)), float(max(last[f]["ratio_farsite"] for f in fids))]
    sens["farsite_L_predicted_range"] = [float(min(last[f]["L_anderson_farsite"] for f in fids)), float(max(last[f]["L_anderson_farsite"] for f in fids))]
    mu = S["mf_ign"]["mean_log_s"]
    sens["effective_floor_logU"] = [float(min(np.hypot(mu, sig) / EX(last[f]["U_mf_ign_ms"]) for f in fids)),
                                    float(max(np.hypot(mu, sig) / EX(last[f]["U_mf_ign_ms"]) for f in fids))]

    # plot-level (cluster) analysis: one mean of log s per plot
    def plot_level(key):
        pm = {}
        for f in fids:
            pm.setdefault(fires[f]["plot"], []).append(np.log(last[f][key]))
        means = np.array([np.mean(v) for v in pm.values()])
        k = len(means)
        se = float(np.std(means, ddof=1) / np.sqrt(k))
        q = float(tdist.ppf(0.975, k - 1))
        bs = [np.mean(rng.choice(means, k)) for _ in range(4000)]
        return {"plots": {p: float(np.mean(v)) for p, v in pm.items()}, "mean": float(np.mean(means)), "se": se,
                "t95": [float(np.mean(means) - q * se), float(np.mean(means) + q * se)],
                "cluster_boot95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "n_plots_log_s_positive": int(np.sum(means > 0)), "n_plots": k}
    PL = {}
    for tag in ("mf", "2m", "10m", "farsite"):
        PL[tag] = plot_level(f"s_{tag}_ign")
    PL["mf_modelfree"] = plot_level("s_mf_ign_modelfree")
    PL["mf_freefocus"] = plot_level("s_mf_ign_freefocus")
    PL["mf_distweighted"] = plot_level("s_mf_dist")

    checks = []
    for f in fids:
        v = last[f]
        others = [np.log(last[g]["s_mf_ign"]) for g in fids if fires[g]["plot"] != fires[f]["plot"]]
        kk = len(others)
        sg, mo = float(np.std(others, ddof=1)), float(np.mean(others))
        ex = EX(v["U_anderson_ms"])
        zU, zA = np.log(v["U_mf_ign_ms"]), np.log(v["U_anderson_ms"])
        zB = np.log(m.anderson_inverse(1 + (v["L"] - 1) / np.exp(mo)))
        sd_fix = v["sd_logLm1_boot"] / ex
        q_t = float(tdist.ppf(0.975, kk - 1))
        q_tp = float(q_t * np.sqrt(1 + 1 / kk))
        checks.append({
            "fire": f, "sd_logU_fixed": float(sd_fix), "sigma_lopo": sg, "mu_lopo": mo, "k": kk,
            "covers_fixed": bool(abs(zA - zU) <= 1.96 * sd_fix),
            "covers_fixed_block50": bool(abs(zA - zU) <= 1.96 * v["sd_logLm1_boot50"] / ex),
            **{f"covers_fixed_{tag}": bool(abs(zA - np.log(v[f"U_{tag}_ign_ms"])) <= 1.96 * sd_fix) for tag in ("mf", "2m", "10m", "farsite")},
            "determined_fixed": bool(sd_fix <= GUARD_SD), "determined_fixed_block50": bool(v["sd_logLm1_boot50"] / ex <= GUARD_SD),
            "covers_prior_z": bool(abs(zA - zU) <= 1.96 * sg / ex), "covers_prior_t": bool(abs(zA - zU) <= q_t * sg / ex),
            "covers_biascorr_z": bool(abs(zB - zU) <= 1.96 * sg / ex), "covers_biascorr_tpred": bool(abs(zB - zU) <= q_tp * sg / ex),
            "determined_prior": bool(sg / ex <= GUARD_SD),
            "halfwidth_factor_prior_t": float(np.exp(q_t * sg / ex)), "halfwidth_factor_biascorr_tpred": float(np.exp(q_tp * sg / ex))})
    cov = {}
    for key in checks[0]:
        if key.startswith(("covers", "determined")):
            k_ = int(sum(c[key] for c in checks))
            cov[key] = [k_, len(checks), cp(k_, len(checks))]
    pred = []
    for f in fids:
        ex = EX(last[f]["U_anderson_ms"])
        b, sa = mu / ex, sig / ex
        pred.append(norm.cdf((1.96 * sa - b) / sa) - norm.cdf((-1.96 * sa - b) / sa))
    gaps = []
    for f in fids:
        ts = sorted(float(r["t"]) for r in period_rows(f))
        used = sorted(v["t_s"] for v in fires[f]["isochrones"].values())
        gaps += [b - a for a, b in zip(ts[:-1], ts[1:]) if b in used]
    summary = {
        "n_fires": len(fids), "n_plots": len(set(plots)), "n_isochrones": len(iso_rows),
        "isochrone_gap_s_range": [float(min(gaps)), float(max(gaps))],
        "head_distance_rel_error_independent_scale": [float(r["head_m"] / r["cumdist_m"] - 1) for r in iso_rows
                                                      if META["fires"][r["fire"]]["scale_source"].startswith("plot-width")],
        "lag1_range": [float(min(r["lag1"] for r in iso_rows)), float(max(r["lag1"] for r in iso_rows))],
        "block_points_range": [int(min(r["block_points"] for r in iso_rows)), int(max(r["block_points"] for r in iso_rows))],
        "sd_logLm1_naive_range": [float(min(r["sd_logLm1_naive"] for r in iso_rows)), float(max(r["sd_logLm1_naive"] for r in iso_rows))],
        "sd_logLm1_boot_range": [float(min(r["sd_logLm1_boot"] for r in iso_rows)), float(max(r["sd_logLm1_boot"] for r in iso_rows))],
        "sd_logLm1_boot_mcerr_max": float(max(r["sd_logLm1_boot_mcerr"] for r in iso_rows)),
        "free_focus_change_two_flank": [float(min(r["L_free_focus"] / r["L"] - 1 for r in iso_rows if r["flanks"] == 2)),
                                        float(max(r["L_free_focus"] / r["L"] - 1 for r in iso_rows if r["flanks"] == 2))],
        "free_focus_offset_m_range": [float(min(r["focus_offset_m"] for r in iso_rows)), float(max(r["focus_offset_m"] for r in iso_rows))],
        "by_wind": S, "sensitivity": sens, "plot_level": PL,
        "elongation_growth_L_sdboot": {f: [[round(v["L"], 3), round(v["sd_logLm1_boot"], 3)] for k, v in
                                           sorted(fires[f]["isochrones"].items(), key=lambda kv: float(kv[0]))]
                                       for f in fids if len(fires[f]["isochrones"]) > 1},
        "interval_checks": checks, "coverage": cov, "predicted_coverage_prior_centred_s1_with_offset": float(np.mean(pred)),
        "joint_fits": {f: fires[f]["joint_constant_forcing"] for f in fids if fires[f]["joint_constant_forcing"]},
    }
    json.dump({"meta": {k: META[k] for k in ("dataset", "selection", "midflame_ratio", "ten_m_ratio", "excluded")},
               "fires": fires, "summary": summary}, open(os.path.join(OUT, "annaburroo_demo.json"), "w"), indent=1)
    json.dump(snapped, open(os.path.join(OUT, "annaburroo_snapped.json"), "w"))
    with open(os.path.join(OUT, "annaburroo_isochrones.csv"), "w", encoding="utf-8") as fh:
        keys = list(iso_rows[0].keys())
        fh.write(",".join(keys) + "\n")
        for r in iso_rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    for k, v in S.items():
        print(k, round(v["mean_log_s"], 3), [round(x, 3) for x in v["sd_fires"]], round(v["median_UA_over_U"], 3), v["n_s_gt_1"])
    for k, v in sens.items():
        print(k, json.dumps(v)[:300])
    print("coverage", json.dumps({k: v[:2] for k, v in cov.items()}))
    for k, v in PL.items():
        print("plot-level", k, round(v["mean"], 3), [round(x, 3) for x in v["t95"]], [round(x, 3) for x in v["cluster_boot95"]],
              v["n_plots_log_s_positive"], "/", v["n_plots"])
    print("predicted (prior centred on s=1, with offset)", round(summary["predicted_coverage_prior_centred_s1_with_offset"], 3))
    print({k: summary[k] for k in ("isochrone_gap_s_range", "lag1_range", "block_points_range", "sd_logLm1_naive_range",
                                    "sd_logLm1_boot_range", "sd_logLm1_boot_mcerr_max", "free_focus_change_two_flank",
                                    "free_focus_offset_m_range")})


if __name__ == "__main__":
    main()
