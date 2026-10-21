"""Figures of the manuscript and the graphical abstract.

Main figures are computed from the model equations and the outputs in output/ of
shape_law_profiles.py, positional_error_identifiability.py, guard_benchmark.py, graded_benchmark.py,
benchmark_tables.py, misspecification_checks.py, fire_behaviour.py, forecast_scenarios.py and
annaburroo/annaburroo_demo.py.
Figures are sized to the 390-pt text width of the Elsevier preprint layout.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as _beta
from scipy.stats import norm

import pyrofield_reduced as m

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "output")
FIG = os.path.join(HERE, "..", "manuscript", "figures")
W = 390.0 / 72.0
BLUE, ORANGE, GREEN, GREY, PURPLE, RED = "#2b6a99", "#d0663f", "#2a7a5f", "#6f7782", "#7b5aa6", "#b03a2e"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "pdf.fonttype": 42, "savefig.dpi": 600,
    "legend.handlelength": 1.8,
})


def panel(ax, letter, x=-0.2):
    ax.text(x, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def load_json(*p):
    return json.load(open(os.path.join(*p), encoding="utf-8"))


def read_csv(*p):
    return list(csv.DictReader(open(os.path.join(*p), encoding="utf-8")))


def cp(k, n):
    return (_beta.ppf(0.025, k, n - k + 1) if k else 0.0, _beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


# ------------------------------------------------------------------------------- Fig. 1
def fig_shape_law():
    fig, axs = plt.subplots(2, 2, figsize=(W, 5.4), constrained_layout=True)
    U = np.linspace(0.5, 12.0, 800)
    ax = axs[0, 0]
    ax.plot(U, m.L_anderson_raw(U), color=GREY, ls=":", label=r"uncapped $L_A(U)$")
    ax.plot(U, m.L_anderson(U), color=BLUE, label=r"$\min\{8,L_A(U)\}$")
    ax.axvline(m.U_C, color="k", lw=0.5, ls="--")
    ax.text(m.U_C + 0.25, 4.6, r"$U_c=8.49$", fontsize=7)
    ax.set_ylim(1, 12)
    ax.set_xlabel(r"Effective midflame wind $U$ (m s$^{-1}$)")
    ax.set_ylabel("Length-to-breadth ratio $L$")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    below, above = U[U < m.U_C], U[U > m.U_C]
    EL = m.elasticity(below)
    EX = below * m.dL_anderson_raw(below) / (m.L_anderson_raw(below) - 1)
    ax2.plot(below, EL, color=ORANGE, ls="--", lw=1.0)
    ax2.plot(below, EX, color=PURPLE, ls="-.", lw=1.0)
    ax2.plot(above, np.zeros_like(above), color=ORANGE, ls="--", lw=1.4)
    ax2.set_ylim(-0.15, 3)
    ax2.set_ylabel("Elasticity")
    ax.plot([], [], color=ORANGE, ls="--", lw=1.0, label=r"$E_L$ (right axis)")
    ax.plot([], [], color=PURPLE, ls="-.", lw=1.0, label=r"$E_X$ (right axis)")
    ax.legend(frameon=False, loc="upper left")
    panel(ax, "a")

    ax = axs[0, 1]
    prof = {}
    for r in read_csv(OUTD, "shape_law_profiles.csv"):
        prof.setdefault(r["case"], []).append((float(r["ratio"]), float(r["delta_chi2"])))
    spec = [("anderson_fixed_U3", r"fixed $L_A$, $U_0=3$", ORANGE, "--"),
            ("anderson_fixed_U5", r"fixed $L_A$, $U_0=5$", BLUE, "-"),
            ("anderson_fixed_U10", r"fixed, capped, $U_0=10$", "k", ":"),
            ("anderson_free_s_U5", r"free $s$, $U_0=5$", PURPLE, "-."),
            ("linear_free_k_U4", r"free $k_{\mathrm{LB}}$, $U_0=4$", GREEN, "-")]
    for i, (key, lab, col, ls) in enumerate(spec):
        r, d = np.array(sorted(prof[key])).T
        ax.plot(r, np.maximum(d, 0.0) + (0.12 if "linear" in key else 0.0), color=col, ls=ls, label=lab,
                lw=1.6 if "linear" in key else 1.1, zorder=5 - i)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-0.15, 1e6)
    ax.set_xticks([0.4, 0.5, 0.7, 1, 1.4, 2])
    ax.set_xticklabels(["0.4", "0.5", "0.7", "1", "1.4", "2"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"Trial wind / generating wind, $u/U_0$")
    ax.set_ylabel(r"Expected profile $\Delta\chi^2$ (symlog)")
    ax.legend(frameon=False, loc="upper center", fontsize=6.4, ncol=2, bbox_to_anchor=(0.5, -0.19))
    panel(ax, "b")

    ax = axs[1, 0]
    U0 = np.linspace(2.0, 8.4, 200)
    for s_true, col, ls in ((0.67, BLUE, ":"), (0.8, BLUE, "-"), (0.9, BLUE, "--"),
                            (1.1, ORANGE, "--"), (1.2, ORANGE, "-"), (1.5, ORANGE, ":")):
        Lt = 1 + s_true * (m.L_anderson_raw(U0) - 1)
        ok = Lt < m.L_CAP
        b = np.full_like(U0, np.nan)
        b[ok] = [m.anderson_inverse(L) for L in Lt[ok]]
        ax.plot(U0, 100 * (b / U0 - 1), color=col, ls=ls, label=f"$s={s_true}$")
    ax.axhline(0, color=GREY, lw=0.5)
    ax.set_xlabel(r"Generating wind $U_0$ (m s$^{-1}$)")
    ax.set_ylabel(r"Wind error assuming $s=1$ (%)")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=6.6, columnspacing=0.8)
    ax.set_ylim(-30, 44)
    panel(ax, "c")

    ax = axs[1, 1]
    rows = read_csv(OUTD, "positional_identifiability.csv")
    EX5 = 5.0 * m.dL_anderson_raw(5.0) / (m.L_anderson_raw(5.0) - 1)
    sel = [(0.0, 0.0, r"exact law, independent errors", BLUE, ":"),
           (0.1, 0.0, r"exact law, correlated ($\ell/D_h=0.1$)", BLUE, "-"),
           (0.1, 0.1, r"$\sigma_s=0.1$, correlated", ORANGE, "--"),
           (0.1, 0.3, r"$\sigma_s=0.3$, correlated", RED, "-")]
    for ell, ss, lab, col, ls in sel:
        rr = [r for r in rows if float(r["U0"]) == 5.0 and float(r["ell_rel"]) == ell and float(r["sigma_s"]) == ss]
        ax.plot([float(r["sigma_rel"]) for r in rr], [100 * float(r["sd_logU"]) for r in rr], color=col, ls=ls, label=lab)
    for ss, col in ((0.1, ORANGE), (0.3, RED)):
        ax.axhline(100 * ss / EX5, color=col, lw=0.6, ls=":")
    ax.text(1.1e-3, 100 * 0.3 / EX5 * 1.15, r"floor $\sigma_s/E_X$", fontsize=6.6, color=RED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Positional error / head advance, $\sigma/D_h$")
    ax.set_ylabel(r"SD of $\log U$ (%), $U_0=5$ m s$^{-1}$")
    ax.legend(frameon=False, loc="lower right", fontsize=6.3)
    ax.set_ylim(0.03, 80)
    panel(ax, "d")
    save(fig, os.path.join(FIG, "Fig1.pdf"))


# ------------------------------------------------------------------------------- Fig. 2
def fig_guard():
    T = load_json(OUTD, "benchmark_tables.json")
    M = load_json(OUTD, "misspecification_checks.json")
    g = read_csv(OUTD, "graded_benchmark_cases.csv")
    fig = plt.figure(figsize=(W, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax = axs[0]
    groups = [("C1", "d30"), ("C2", "d30"), ("C3", "d30"), ("C4", "d30"), ("C5", "d30"), ("C5", "d50")]
    width = 0.38
    for j, (err, lab, col) in enumerate((("E1", "masks", BLUE), ("E2", "positions", ORANGE))):
        xs = np.arange(len(groups)) + (j - 0.5) * width
        tw = [T["labelled"][err][f"{c},{d}"] for c, d in groups]
        det = np.array([t["determined"] for t in tw]) / 100
        inc = np.array([t["inconclusive"] for t in tw]) / 100
        ax.bar(xs, det, width, color=col, label=f"{lab}: determined")
        ax.bar(xs, inc, width, bottom=det, color="white", edgecolor=GREY, hatch="///", lw=0.5)
        ax.bar(xs, 1 - det - inc, width, bottom=det + inc, color="#e9ecef", edgecolor="white", lw=0.3,
               label="unresolved" if j == 0 else None)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["C1", "C2", "C3", "C4", "C5", "C5\n(50%)"], fontsize=6.8)
    ax.set_ylabel("Fraction of fires (30% displacement)")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=6.2, loc="lower center", bbox_to_anchor=(0.62, 1.0), ncol=2, columnspacing=0.6,
              handlelength=1.2)
    panel(ax, "a", x=-0.28)

    ax = axs[1]
    ok = [r for r in g if r["cell"].startswith("s")]
    lam = np.array([float(r["ref_d30"]) for r in ok])
    det = np.array([float(r["DU_d30"]) >= 4 for r in ok])
    edges = np.array([0, 1, 2, 3, 3.5, 4, 4.5, 5, 6, 8, 12, 25, 1e9])
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (lam >= a) & (lam < b)
        n, k = sel.sum(), det[sel].sum()
        if n < 5:
            continue
        x = np.sqrt(max(a, 0.5) * min(b, 50))
        lo, hi = cp(k, n)
        ax.errorbar(x, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt="o", color=PURPLE, ms=3, elinewidth=0.6, capsize=1.5)
    xx = np.geomspace(0.5, 50, 200)
    ax.plot(xx, (xx >= 4).astype(float), color=GREY, lw=0.8, ls="--", label="theory: step at 4")
    ax.set_xscale("log")
    ax.set_xlabel(r"Expected non-centrality $\lambda$ at 30%")
    ax.set_ylabel("Fraction determined")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, loc="center right", fontsize=6.2)
    panel(ax, "b")

    ax = axs[2]
    tab = {t["cell"]: t for t in T["graded_table"]}
    rows = [("s0.05_sig0.03", r"$\sigma_s$=0.05", BLUE), ("s0.1_sig0.03", r"$\sigma_s$=0.1", BLUE),
            ("s0.2_sig0.03", r"$\sigma_s$=0.2", BLUE), ("s0.3_sig0.03", r"$\sigma_s$=0.3", BLUE),
            ("nearcap_exact_sig0.03", "near cap", BLUE), ("unknown", "unknown bearing, ignition", BLUE),
            ("misspec_exact_sig0.03", r"wrong $\Sigma$, exact", ORANGE), ("misspec_s0.1_sig0.03", r"wrong $\Sigma$, $\sigma_s$=0.1", ORANGE),
            ("overdisp_a0.3_t0.1_sig0.03", "prior 0.3, true 0.1", RED), ("underdisp_a0.1_t0.3_sig0.03", "prior 0.1, true 0.3", RED),
            ("miscal_s0.1_sig0.03", "fixed s, true 0.1", RED), ("miscal_s0.3_sig0.03", "fixed s, true 0.3", RED)]
    pred = T["coverage_predicted"]
    for i, (key, lab, col) in enumerate(rows):
        if key == "unknown":
            u = M["unknown_bearing_ignition_s0.1_sig0.03"]
            k, n = u["coverage"], u["n"]
        else:
            k, n = tab[key]["coverage"][0], 60
        lo, hi = cp(k, n)
        ax.errorbar(k / n, i, xerr=[[k / n - lo], [hi - k / n]], fmt="o", color=col, ms=3, elinewidth=0.6, capsize=1.5)
        if key in pred:
            ax.plot(pred[key], i, "x", color="k", ms=4)
    ax.plot([], [], "x", color="k", ms=4, label="predicted (Eq. 9)")
    ax.axvline(0.95, color=GREY, lw=0.6, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.06)
    ax.set_xticks([0, 0.5, 0.95])
    ax.set_xlabel("95% interval coverage")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=6.0, handletextpad=0.3)
    panel(ax, "c", x=-0.6)

    ax = axs[3]
    C = T["conditional_coverage"]
    zz = np.linspace(0, 3.2, 300)
    for rho, ls in ((1.0, ":"), (3.0, "-"), (10.0, "--")):
        tau = np.sqrt(1 + rho ** 2)
        cov = norm.cdf(1.96 * tau - zz * rho) - norm.cdf(-1.96 * tau - zz * rho)
        ax.plot(zz, cov, color=GREY if rho != 3.0 else BLUE, ls=ls, lw=1.0, label=fr"theory, $\rho={rho:g}$")
    for b in C["bins"]:
        lo_b, hi_b = b["bin"][0], (b["bin"][1] if b["bin"][1] is not None else 3.0)
        x = 0.5 * (lo_b + hi_b)
        k, n = b["covered"], b["n"]
        lo, hi = b["cp"]
        ax.errorbar(x, k / n, yerr=[[k / n - lo], [hi - k / n]], xerr=[[x - lo_b], [hi_b - x]], fmt="o", color=ORANGE, ms=3.5,
                    elinewidth=0.6, capsize=1.5, label="simulated" if b is C["bins"][0] else None)
        ax.plot(x, b["predicted_mean"], "x", color="k", ms=4, label="predicted for these fires" if b is C["bins"][0] else None)
    ax.axhline(0.95, color=GREY, lw=0.6, ls="--")
    ax.set_xlabel(r"True multiplier, $|\log s|/\sigma_s$")
    ax.set_ylabel("Coverage at fixed $s$")
    ax.set_xlim(0, 3.2)
    ax.set_ylim(-0.02, 1.04)
    ax.legend(frameon=False, fontsize=5.8, loc="lower left")
    panel(ax, "d")
    save(fig, os.path.join(FIG, "Fig2.pdf"))


# ------------------------------------------------------------------------------- Fig. 3
def fig_fire_behaviour():
    F = load_json(OUTD, "fire_behaviour.json")
    fuels = ["FM1", "FM2", "FM3", "FM4", "FM8", "FM9", "FM10"]
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.75), constrained_layout=True)
    spec = [("load_beta", "reaction", "load (fixed packing)", GREEN, "o"),
            ("moisture", "reaction", "moisture, reaction", PURPLE, "^"),
            ("moisture", "consumption", "moisture, consumption", ORANGE, "v"),
            ("ros_factor", "reaction", "ROS adjustment factor", GREY, "s")]

    def dumbbells(ax, quantity):
        for j, (mode, conv, lab, col, mk) in enumerate(spec):
            for i, f in enumerate(fuels):
                v = [F[f][f"M0=0.06,U0=2.0,sigma_s={ss},{mode}"][quantity.format(conv)] for ss in (0.1, 0.3)]
                x = i + (j - 1.5) * 0.17
                ax.plot([x, x], v, color=col, lw=0.8)
                ax.plot(x, v[0], mk, color=col, ms=3.0, mfc="white", mew=0.8)
                ax.plot(x, v[1], mk, color=col, ms=3.0, label=lab if i == 0 else None)
        ax.set_yscale("log")
        ax.set_xticks(range(len(fuels)))
        ax.set_xticklabels(fuels, fontsize=6.2, rotation=90)
        ax.axhline(1.0, color=GREY, lw=0.5)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax = axs[0]
    dumbbells(ax, "{}_maxmin")
    ax.set_ylim(0.95, 7)
    ax.set_yticks([1, 1.5, 2, 3, 5])
    ax.set_yticklabels(["1", "1.5", "2", "3", "5"])
    ax.set_ylabel("Fireline intensity, max/min")
    ax.legend(frameon=False, fontsize=5.6, loc="upper right", handlelength=1.0, bbox_to_anchor=(1.02, 1.0))
    panel(ax, "a", x=-0.3)
    ax = axs[1]
    dumbbells(ax, "hs_{}_maxmin")
    ax.set_ylim(0.95, 14)
    ax.set_yticks([1, 1.5, 2, 3, 5, 10])
    ax.set_yticklabels(["1", "1.5", "2", "3", "5", "10"])
    ax.set_ylabel("Scorch height, max/min")
    panel(ax, "b", x=-0.3)
    ax = axs[2]
    for i, f in enumerate(fuels):
        for conv, col, off in (("reaction", GREEN, -0.13), ("consumption", ORANGE, 0.13)):
            lo, hi = F[f]["M0=0.06,U0=2.0,sigma_s=0.3,load_beta"][f"Lf_{conv}_m"]
            ax.plot([i + off, i + off], [lo, hi], color=col, lw=2.0, solid_capstyle="butt",
                    label=(("reaction (BEHAVE)" if conv == "reaction" else "consumption") if i == 0 else None))
    for y, lab in ((1.2, "1.2 m"), (2.4, "2.4 m"), (3.4, "3.4 m")):
        ax.axhline(y, color=GREY, lw=0.6, ls=":")
        ax.text(6.45, y * 1.04, lab, fontsize=6.0, color=GREY, ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_yticks([0.3, 0.5, 1, 2, 3, 5])
    ax.set_yticklabels(["0.3", "0.5", "1", "2", "3", "5"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(-0.5, 7.3)
    ax.set_xticks(range(len(fuels)))
    ax.set_xticklabels(fuels, fontsize=6.2, rotation=90)
    ax.set_ylabel(r"Flame length, load (fixed packing), $\sigma_s=0.3$ (m)")
    ax.legend(frameon=False, fontsize=5.6, loc="lower right", handlelength=1.0)
    panel(ax, "c", x=-0.3)
    save(fig, os.path.join(FIG, "Fig3.pdf"))


# ------------------------------------------------------------------------------- Fig. 4
def fig_forecast():
    F = load_json(OUTD, "forecast_scenarios.json")
    e0 = 100 * np.array(F["e0"])
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.7), constrained_layout=True)
    ax = axs[0]
    for k, ls in (("0.0", "-"), ("0.5", "--"), ("0.8", "-."), ("1.0", ":")):
        ax.plot(e0, 100 * np.array(F["testbed"][f"kappa={k}"]), color=BLUE, ls=ls, label=fr"testbed, $\kappa={float(k):g}$")
    for key, col, lab in (("FM1_moisture_none_shift1.75", ORANGE, "FM1"), ("FM10_load_beta_none_shift1.75", GREEN, "FM10")):
        for k, ls in (("0.0", "-"), ("1.0", ":")):
            y = np.array([np.nan if v is None else 100 * v for v in F[key][f"kappa={k}"]])
            ax.plot(e0, y, color=col, ls=ls, label=fr"{lab}, $\kappa={float(k):g}$")
    ax.plot(e0, 100 * np.array(F["testbed_flank_kappa0"]), color="k", lw=2.4, alpha=0.25, label=r"flank, $\kappa=0$")
    ax.axhline(0, color=GREY, lw=0.4)
    ax.set_xlabel(r"Historical wind error $\varepsilon_0$ (%)")
    ax.set_ylabel("Post-shift arrival-delay error (%)")
    ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    ax.set_ylim(-45, 185)
    panel(ax, "a")
    ax = axs[1]
    S = F["sweep"]
    shifts = [1.25, 1.5, 1.75, 2.0, 2.5]
    for key, mode, col, lab in (("FM1", "moisture", ORANGE, "FM1 moisture"), ("FM1", "load_beta", PURPLE, "FM1 load")):
        for limit, ls, mk in (("revised", "--", "s"), ("original", ":", "^")):
            y = [S[f"{key},{mode},{limit},shift={s},e0=0.2,kappa=1.0"] for s in shifts]
            y = [np.nan if v is None else 100 * v for v in y]
            ax.plot(shifts, y, ls=ls, marker=mk, color=col, ms=3, label=f"{lab}, {limit} limit")
    env = [max(abs(S[f"{k},{md},{lim},shift={s},e0={e},kappa=1.0"] or 0.0) for k, md, lim, e in
               [(a, b, c, d) for a, b in (("FM1", "moisture"), ("FM1", "load_beta"), ("FM4", "load_beta"),
                                          ("FM10", "moisture"), ("FM10", "load_beta"))
                for c in ("none",) for d in (-0.2, 0.2)]) for s in shifts]
    ax.fill_between(shifts, [-100 * v for v in env], [100 * v for v in env], color=GREY, alpha=0.35, lw=0,
                    label="no wind limit, all members")
    ax.axhline(0, color=GREY, lw=0.4)
    ax.set_xlabel(r"Wind shift $U_1/U_0$")
    ax.set_ylabel(r"Delay error, $\varepsilon_0=+20\%$, $\kappa=1$ (%)")
    ax.legend(frameon=False, fontsize=6.0, loc="upper left", ncol=1)
    ax.set_ylim(-10, 210)
    panel(ax, "b")
    save(fig, os.path.join(FIG, "Fig4.pdf"))



# ------------------------------------------------------------------------------- Fig. 5
def fig_annaburroo():
    """Digitised Annaburroo point-fire isochrones (annaburroo/annaburroo_demo.py) and elongation against wind."""
    D = load_json(OUTD, "annaburroo_demo.json")
    S = load_json(OUTD, "annaburroo_snapped.json")
    meta = json.load(open(os.path.join(HERE, "annaburroo", "fires.json"), encoding="utf-8"))["fires"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.9), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    pal_iso = [ORANGE, RED, BLUE]
    for fid, dy in (("F223", 55.0), ("F252", -45.0)):
        F, d = meta[fid], D["fires"][fid]
        ix, iy = F["ignition_px"]
        ppm = F["px_per_m"]
        th_ref = max(d["isochrones"].values(), key=lambda v: v["t_s"])["bearing_rad"]
        c, sn = np.cos(-th_ref), np.sin(-th_ref)
        keys = sorted(d["isochrones"], key=float)
        col_of = {k: pal_iso[i] for i, k in enumerate(keys)}
        for name, t in F["curves"].items():
            P = np.array(S[fid][name])
            x, y = (P[:, 0] - ix) / ppm, -(P[:, 1] - iy) / ppm
            ax.plot(c * x - sn * y, sn * x + c * y + dy, ".", ms=0.8, color=col_of[str(int(t))])
        for key in keys:
            v = d["isochrones"][key]
            L, a, th = v["L"], v["a_m"], v["bearing_rad"] - th_ref
            e = np.sqrt(1 - 1 / L ** 2)
            psi = np.linspace(0, 2 * np.pi, 400)
            r = a * (1 - e ** 2) / (1 - e * np.cos(psi))
            ax.plot(r * np.cos(psi + th), r * np.sin(psi + th) + dy, color=col_of[key], lw=0.7,
                    label=f"{fid}, t = {int(v['t_s'])} s")
        j = d["joint_constant_forcing"]
        if j:
            L, th = j["L"], j["bearing_rad"] - th_ref
            e = np.sqrt(1 - 1 / L ** 2)
            arate = (j["Rh"] + j["Rh"] * (1 - e) / (1 + e)) / 2
            for v in d["isochrones"].values():
                a = arate * (v["t_s"] - j["t0"])
                psi = np.linspace(0, 2 * np.pi, 400)
                r = a * (1 - e ** 2) / (1 - e * np.cos(psi))
                ax.plot(r * np.cos(psi + th), r * np.sin(psi + th) + dy, color="k", lw=0.6, ls="--")
            ax.plot([], [], color="k", lw=0.6, ls="--", label=f"{fid}, constant forcing")
        ax.plot(0, dy, "*", color="k", ms=5)
        ax.text(-20, dy + 12, fid, fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlabel("Distance along the head direction (m)")
    ax.set_ylabel("Cross-wind distance (m)")
    ax.legend(frameon=False, fontsize=5.4, loc="upper right", ncol=1, handlelength=1.4)
    ax.set_xlim(-25, 140)
    ax.set_ylim(-75, 115)
    panel(ax, "a", x=-0.14)

    ax = axs[1]
    U2 = np.linspace(1.5, 6.0, 200)
    for c, lab, ls in ((0.40, r"FARSITE-style midflame $=0.40\,U_2$", "-."), (0.91, r"midflame wind $=0.91\,U_2$", "-"),
                       (1.0, r"wind at 2 m", "--"), (1.23, r"10-m wind $=1.23\,U_2$", ":")):
        ax.plot(U2, m.L_anderson_raw(c * U2), color="k", lw=0.9, ls=ls, label=f"Anderson law, {lab}")
    pal = [BLUE, ORANGE, GREEN, PURPLE, RED, GREY, "#8c6d1f"]
    for (fid, d), col in zip(D["fires"].items(), pal):
        v = sorted(d["isochrones"].values(), key=lambda q: q["t_s"])
        x = [q["U2_ms"] for q in v]
        y = [q["L"] for q in v]
        lo = [1 + (q["L"] - 1) * np.exp(-q["sd_logLm1_boot"]) for q in v]
        hi = [1 + (q["L"] - 1) * np.exp(q["sd_logLm1_boot"]) for q in v]
        ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="o-", color=col, ms=3, lw=0.8, elinewidth=0.5,
                    capsize=1.2, label=fid)
    ax.set_xlabel(r"Measured 2-m wind $U_2$ (m s$^{-1}$)")
    ax.set_ylabel("Length-to-breadth ratio $L$")
    ax.set_xlim(1.5, 6.0)
    ax.set_ylim(1, 10.5)
    ax.legend(frameon=False, fontsize=5.0, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.6)
    panel(ax, "b")
    save(fig, os.path.join(FIG, "Fig5.pdf"))

# ------------------------------------------------------------------------ graphical abstract
def support_xy(h_fun, n=720):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    h = h_fun(th)
    dh = np.gradient(np.concatenate([h[-2:], h, h[:2]]), th[1] - th[0])[2:-2]
    return h * np.cos(th) - dh * np.sin(th), h * np.sin(th) + dh * np.cos(th)


def wavelet_support(Rh, L, bearing):
    a, b, c = m.ellipse_axes(Rh, L)
    return lambda th: c * np.cos(th - bearing) + np.sqrt((a * np.cos(th - bearing)) ** 2 + (b * np.sin(th - bearing)) ** 2)


def graphical_abstract():
    R0s, U0, k0, U1 = 0.10, 4.0, 0.35, 7.0
    Rh_star, L_star = m.head_rate(R0s, U0), m.L_linear(U0, k0)
    t0, dt, veer = 20 * 60.0, 20 * 60.0, np.deg2rad(45)
    members = [(2.4, ORANGE), (4.0, "k"), (6.4, BLUE)]
    fig, axs = plt.subplots(1, 3, figsize=(6.64, 2.655), gridspec_kw=dict(width_ratios=[1, 1.15, 1.05]))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.84, bottom=0.05, wspace=0.12)
    hist = wavelet_support(Rh_star, L_star, 0.0)
    ax = axs[0]
    x, y = support_xy(lambda th: t0 * hist(th))
    ax.fill(x, y, color="#c9a27e", alpha=0.6, lw=0)
    for (u, col), lw in zip(members, (2.6, 1.6, 0.8)):
        ax.plot(x, y, color=col, lw=lw)
    ax.plot(0, 0, "*", color="k", ms=6)
    ax.set_title("Same past\nthree wind-fuel pairs, one perimeter", fontsize=7.5)
    ax.text(0.5, 0.05, "u = 2.4, 4.0, 6.4 m/s\nfit equally well", transform=ax.transAxes, ha="center", fontsize=6.8)
    ax = axs[1]
    for u, col in members:
        R0u = Rh_star / (1 + m.wind_factor(u))
        fut = wavelet_support(m.head_rate(R0u, U1), m.L_linear(U1, (L_star - 1) / u), veer)
        x, y = support_xy(lambda th: t0 * hist(th) + dt * fut(th))
        ax.plot(x, y, color=col, lw=1.2, label=f"u = {u} m/s")
    x, y = support_xy(lambda th: t0 * hist(th))
    ax.fill(x, y, color="#c9a27e", alpha=0.6, lw=0)
    ax.annotate("", (420 * np.cos(veer), 420 * np.sin(veer)), (0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.0, color=GREY))
    ax.legend(frameon=False, loc="lower right", fontsize=6.4)
    ax.set_title("Different futures\nindependently known new wind", fontsize=7.5)
    for a in axs[:2]:
        a.set_aspect("equal")
        a.axis("off")
    axs[0].set_xlim(-160, 480)
    axs[0].set_ylim(-235, 235)
    ax = axs[2]
    ax.axis("off")
    ax.set_title("What to report", fontsize=7.5)
    lines = ["1  Fit under a stated error model", "2  Converged nuisance-refitted",
             "    profile intervals", "3  Compatible fires: intensity and",
             "    forecasts, with the shape-law", "    and wind-error assumptions",
             "", "A good fit is not enough."]
    for i, s in enumerate(lines):
        ax.text(0.02, 0.88 - 0.115 * i, s, transform=ax.transAxes, fontsize=7.0,
                fontweight="bold" if s.startswith("A good") else "normal")
    fig.savefig(os.path.join(FIG, "..", "graphical_abstract.pdf"))
    fig.savefig(os.path.join(FIG, "..", "graphical_abstract.png"), dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_shape_law()
    fig_guard()
    fig_fire_behaviour()
    fig_forecast()
    fig_annaburroo()
    graphical_abstract()
    print("figures written to", os.path.abspath(FIG))
