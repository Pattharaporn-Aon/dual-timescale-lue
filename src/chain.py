"""Dual-timescale light-use-efficiency canopy model for an evergreen durian orchard.
Weather (ERA5) -> stress scalars -> GPP -> leaf pool L (fast) + crown cover c (slow) -> NDVI.
"""
import numpy as np, pandas as pd
from numba import njit

# ---------------- fixed values from literature ----------------
EPS = 1.268          # g C MJ-1, MOD17 BPLUT evergreen broadleaf forest
VPD_MIN, VPD_MAX = 0.80, 3.10   # kPa, MOD17 BPLUT EBF (daytime mean VPD)
T_OPT, SIG_T = 28.0, 5.59       # degC; sigma from f(40 degC)=0.1
K_EXT = 0.5                      # Beer-Lambert extinction, broadleaf
NDVI_FULL = 0.90
KC = 0.90                        # crop coefficient (assumed, evergreen fruit tree)
P_DEPL = 0.50                    # FAO-56 depletion fraction
ZR = 1000.0                      # root-zone depth (mm)
SOIL = dict(wp=0.181, fc=0.336, sat=0.493)   # SoilGrids, 0-100 cm
TAW = (SOIL["fc"] - SOIL["wp"]) * ZR          # mm


def f_T(T):
    return np.exp(-(T - T_OPT) ** 2 / (2 * SIG_T ** 2))


def f_VPD_ramp(v):
    return np.clip((VPD_MAX - v) / (VPD_MAX - VPD_MIN), 0, 1)


def f_VPD_gauss(v, vopt=1.4, sig=0.8):
    return np.exp(-(v - vopt) ** 2 / (2 * sig ** 2))


def f_SM_trapezoid(theta):
    wp, fc, sat = SOIL["wp"], SOIL["fc"], SOIL["sat"]
    c = fc - P_DEPL * (fc - wp)
    w = sat - 0.10
    return np.where(theta <= wp, 0, np.where(theta < c, (theta - wp) / (c - wp),
                    np.where(theta <= w, 1, np.clip((sat - theta) / (sat - w), 0, 1))))


@njit(cache=True)
def _bucket(P, ET0, phi, taw, p, kc):
    raw = p * taw
    Dr = 0.0
    Ks = np.empty(len(P))
    for i in range(len(P)):
        k = 1.0 if Dr <= raw else max(0.0, (taw - Dr) / ((1 - p) * taw))
        Ks[i] = k
        Dr = Dr - P[i] + k * kc * ET0[i]
        Dr = min(taw, max(0.0, Dr))
        if Dr > raw:
            Dr -= phi * (Dr - raw)
    return Ks


def bucket(P, ET0, phi=0.0):
    """FAO-56 root-zone water balance with latent irrigation fraction phi
    (phi=0 rainfed; phi=1 depletion restored to RAW whenever exceeded)."""
    return _bucket(np.asarray(P, float), np.asarray(ET0, float), float(phi), TAW, P_DEPL, KC)


@njit(cache=True)
def _canopy(PAR, stress, fS, L0, c0, alpha, b0, b1, r, cmax, dual, eps, kext, gbar):
    n = len(PAR)
    L = np.empty(n); c = np.empty(n); gpp = np.empty(n)
    L[0] = L0; c[0] = c0
    for i in range(n - 1):
        fpar = 1 - np.exp(-kext * L[i])
        gpp[i] = eps * fpar * PAR[i] * stress[i]
        beta = b0 + b1 * (1 - fS[i])
        L[i + 1] = max(1e-3, L[i] + alpha * gpp[i] - beta * L[i])
        if dual:
            c[i + 1] = min(0.999, c[i] + r * (gpp[i] / gbar) * c[i] * (1 - c[i] / cmax))
        else:
            c[i + 1] = c[i]
    gpp[n - 1] = gpp[n - 2]
    return L, c, gpp


def simulate(drv, th, variant="dual", vpd="ramp", sm="bucket"):
    """drv: dict of daily arrays PAR, T, VPD, SM, P, ET0. th: parameter dict."""
    PAR = np.asarray(drv["PAR"], float)
    fT = f_T(drv["T"])
    fV = f_VPD_ramp(drv["VPD"]) if vpd == "ramp" else f_VPD_gauss(drv["VPD"])
    if sm == "bucket":
        fS = bucket(drv["P"], drv["ET0"], th.get("phi", 0.0))
    elif sm == "era5":
        fS = f_SM_trapezoid(drv["SM"])
    else:
        fS = np.ones(len(PAR))
    fS = np.asarray(fS, float)
    stress = np.asarray(fT * fV * fS, float)
    gbar = EPS * np.mean(PAR * stress) * 0.5
    L, c, gpp = _canopy(PAR, stress, fS, th["L0"], th.get("c0", 1.0), th["alpha"], th["b0"],
                        th.get("b1", 0.0), th.get("r", 0.0), th.get("cmax", 1.0),
                        variant == "dual", EPS, K_EXT, gbar)
    fpar = 1 - np.exp(-K_EXT * L)
    if "uwet" in th:   # two-layer mixing: tree crowns (cover c) + inter-row understory
        g = pd.Series(bucket(drv["P"], drv["ET0"], 0.0)).ewm(alpha=1.0 / th["tau_u"]).mean().values
        U = th["bg"] + (th["uwet"] - th["bg"]) * g
        ndvi = c * (th["bg"] + (NDVI_FULL - th["bg"]) * fpar) + (1 - c) * U
    else:
        U = np.full(len(PAR), th["bg"])
        ndvi = th["bg"] + (NDVI_FULL - th["bg"]) * c * fpar
    return dict(ndvi=ndvi, L=L, c=c, fpar=fpar, U=U, gpp_crown=gpp, gpp_ground=gpp * c,
                fT=fT, fV=fV, fS=fS, stress=stress)


def nectec_logistic(drv, th):
    """Annual-crop chain (NECTEC style): Gaussian VPD, LAI logistic in cumulative GPP."""
    fT = f_T(drv["T"]); fV = f_VPD_gauss(drv["VPD"]); fS = f_SM_trapezoid(drv["SM"])
    gpp = EPS * drv["PAR"] * fT * fV * fS
    cg = np.cumsum(gpp)
    lai = th["Lmax"] / (1 + np.exp(-th["a"] * (cg - th["b"])))
    fpar = 1 - np.exp(-K_EXT * lai)
    return dict(ndvi=th["bg"] + (NDVI_FULL - th["bg"]) * fpar)
