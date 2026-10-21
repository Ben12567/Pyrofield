"""Multi-size-class Rothermel (1972) surface fire spread model with the Albini (1976) revisions.

Size classes are weighted by surface area within the dead and live categories (Rothermel, 1972,
Eqs. 53-60); the live moisture of extinction follows Albini (1976). Internal units are those of the
original model (lb, ft, min, Btu); helpers convert to SI. Fuel loads are in tons per acre.

Standard fuel models are those of Albini (1976) / Anderson (1982):
  (load 1h, 10h, 100h, live) t/ac, SAV 1h, SAV live (1/ft), depth (ft), dead moisture of extinction
"""
import numpy as np

FT_PER_M = 3.28084
T_AC_TO_LB_FT2 = 0.0459137
BTU_FT_S_TO_KW_M = 3.46165
H, ST, SE, RHO_P = 8000.0, 0.0555, 0.010, 32.0
SAV_10H, SAV_100H = 109.0, 30.0

FUEL_MODELS = {
    "FM1": dict(load=(0.74, 0.0, 0.0, 0.0), sav1=3500.0, savl=1500.0, depth=1.0, mx=0.12, name="short grass"),
    "FM2": dict(load=(2.00, 1.00, 0.50, 0.50), sav1=3000.0, savl=1500.0, depth=1.0, mx=0.15, name="timber grass and understory"),
    "FM3": dict(load=(3.01, 0.0, 0.0, 0.0), sav1=1500.0, savl=1500.0, depth=2.5, mx=0.25, name="tall grass"),
    "FM4": dict(load=(5.01, 4.01, 2.00, 5.01), sav1=2000.0, savl=1500.0, depth=6.0, mx=0.20, name="chaparral"),
    "FM8": dict(load=(1.50, 1.00, 2.50, 0.0), sav1=2000.0, savl=1500.0, depth=0.2, mx=0.30, name="compact timber litter"),
    "FM9": dict(load=(2.92, 0.41, 0.15, 0.0), sav1=2500.0, savl=1500.0, depth=0.2, mx=0.25, name="hardwood litter"),
    "FM10": dict(load=(3.01, 2.00, 5.01, 2.00), sav1=2000.0, savl=1500.0, depth=1.0, mx=0.25, name="timber litter and understory"),
}
# Reference behaviour tabulated by Anderson (1982) at 8% dead and 100% live moisture, 5 mi/h midflame wind
ANDERSON_1982 = {"FM1": (78, 4), "FM2": (35, 6), "FM3": (104, 12), "FM4": (75, 19),
                 "FM8": (1.6, 1.0), "FM9": (7.5, 2.6), "FM10": (7.9, 4.8)}


class FuelBed:
    def __init__(self, load, sav1, savl, depth, mx, **_):
        w = np.array(load) * T_AC_TO_LB_FT2
        sav = np.array([sav1, SAV_10H, SAV_100H, savl])
        keep = w > 0
        self.w, self.sav, self.depth, self.mx_dead = w[keep], sav[keep], depth, mx
        self.live = np.array([False, False, False, True])[keep]
        area = self.sav * self.w / RHO_P
        self.f_ij = np.empty_like(area)
        cats = [~self.live, self.live]
        self.A_j = np.array([area[c].sum() for c in cats])
        for c, Aj in zip(cats, self.A_j):
            if Aj > 0:
                self.f_ij[c] = area[c] / Aj
        self.f_j = self.A_j / self.A_j.sum()
        self.sigma_j = np.array([(self.f_ij[c] * self.sav[c]).sum() if c.any() else 0.0 for c in cats])
        self.sigma = float((self.f_j * self.sigma_j).sum())
        self.rho_b = self.w.sum() / depth
        self.beta = self.rho_b / RHO_P
        beta_op = 3.348 * self.sigma ** -0.8189
        self.ratio = self.beta / beta_op
        A = 133.0 * self.sigma ** -0.7913
        gmax = self.sigma ** 1.5 / (495.0 + 0.0594 * self.sigma ** 1.5)
        self.gamma_r = gmax * self.ratio ** A * np.exp(A * (1.0 - self.ratio))
        self.wn_j = np.array([(self.f_ij[c] * self.w[c]).sum() * (1 - ST) if c.any() else 0.0 for c in cats])
        self.eta_s = min(1.0, 0.174 * SE ** -0.19)
        self.xi = np.exp((0.792 + 0.681 * self.sigma ** 0.5) * (self.beta + 0.1)) / (192.0 + 0.2595 * self.sigma)
        self.C = 7.47 * np.exp(-0.133 * self.sigma ** 0.55)
        self.B = 0.02526 * self.sigma ** 0.54
        self.E = 0.715 * np.exp(-3.59e-4 * self.sigma)
        self.cats = cats

    @staticmethod
    def _eta_m(Mf, Mx):
        r = min(Mf / Mx, 1.0) if Mx > 0 else 1.0
        return max(0.0, 1.0 - 2.59 * r + 5.11 * r ** 2 - 3.52 * r ** 3)

    def _moist(self, M_dead, M_live):
        return np.where(self.live, M_live, M_dead)

    def I_R(self, M_dead, M_live=1.0):
        M = self._moist(M_dead, M_live)
        dead, live = self.cats
        Mf_dead = (self.f_ij[dead] * M[dead]).sum()
        eta_dead = self._eta_m(Mf_dead, self.mx_dead)
        eta_live = 0.0
        if live.any():
            wd = (self.w[dead] * np.exp(-138.0 / self.sav[dead])).sum()
            wl = (self.w[live] * np.exp(-500.0 / self.sav[live])).sum()
            Md_fine = (self.w[dead] * np.exp(-138.0 / self.sav[dead]) * M[dead]).sum() / wd
            mx_live = max(self.mx_dead, 2.9 * (wd / wl) * (1 - Md_fine / self.mx_dead) - 0.226)
            Mf_live = (self.f_ij[live] * M[live]).sum()
            eta_live = self._eta_m(Mf_live, mx_live)
        return self.gamma_r * H * self.eta_s * (self.wn_j[0] * eta_dead + self.wn_j[1] * eta_live)

    def phi_w(self, U_ms):
        return self.C * (U_ms * FT_PER_M * 60.0) ** self.B * self.ratio ** -self.E

    def ros(self, M_dead, U_ms, M_live=1.0):                       # ft/min
        M = self._moist(M_dead, M_live)
        eps = np.exp(-138.0 / self.sav)
        qig = 250.0 + 1116.0 * M
        sink = self.rho_b * sum(self.f_j[k] * (self.f_ij[c] * eps[c] * qig[c]).sum()
                                for k, c in enumerate(self.cats) if c.any())
        return self.I_R(M_dead, M_live) * self.xi * (1.0 + self.phi_w(U_ms)) / sink

    def byram(self, M_dead, U_ms, M_live=1.0):                    # kW/m
        return self.I_R(M_dead, M_live) * (384.0 / self.sigma) * self.ros(M_dead, U_ms, M_live) / 60.0 * BTU_FT_S_TO_KW_M

    def wind_limit_ms(self, M_dead, M_live=1.0):
        return 96.8 * self.I_R(M_dead, M_live) ** (1.0 / 3.0) / (FT_PER_M * 60.0)

    def gamma(self, U_ms):
        phi = self.phi_w(U_ms)
        return self.B * phi / (1.0 + phi)


def flame_length_m(I_kw_m):
    return 0.0775 * I_kw_m ** 0.46


def validate():
    """Compare with the reference behaviour tabulated by Anderson (1982)."""
    rows = []
    for key, (ros_ref, fl_ref) in ANDERSON_1982.items():
        fb = FuelBed(**FUEL_MODELS[key])
        U = 5 * 0.44704
        ros = fb.ros(0.08, U) * 60.0 / 66.0
        fl = flame_length_m(fb.byram(0.08, U)) * FT_PER_M
        rows.append((key, ros, ros_ref, fl, fl_ref))
    return rows


if __name__ == "__main__":
    for key, ros, rr, fl, fr in validate():
        print(f"{key:5s} ROS {ros:7.1f} ch/h (Anderson 1982: {rr})   flame {fl:5.1f} ft ({fr})")
