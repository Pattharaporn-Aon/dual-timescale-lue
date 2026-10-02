"""Final comparison on cleaned NDVI. B1r = annual-crop chain applied as intended to a perennial:
logistic LAI in GPP accumulated since an annual re-initialisation date, same stress inputs as M3."""
import numpy as np, pandas as pd, json, chain
from fit import d, drv, metrics, harmonic_X, differential_evolution, least_squares
from chain import simulate
import fit3
from fit3 import g, flag, simulate_given, fS_era5_scaled

obs = g[~flag].reset_index(drop=True)
idx = d.index.get_indexer(obs.date); y = obs.NDVI.values; yrs = obs.date.dt.year.values
doy = d.index.dayofyear.values

def b1r(x):
    Lmax, a, b, bg, d0, phi = x
    Ks = chain.bucket(drv["P"], drv["ET0"], phi)
    gpp = chain.EPS * drv["PAR"] * chain.f_T(drv["T"]) * chain.f_VPD_ramp(drv["VPD"]) * Ks
    reset = doy == int(round(d0))
    cg = np.empty_like(gpp); acc = 0.0
    for i in range(len(gpp)):
        if reset[i]: acc = 0.0
        acc += gpp[i]; cg[i] = acc
    lai = Lmax / (1 + np.exp(-a * (cg - b)))
    return (bg + (chain.NDVI_FULL - bg) * (1 - np.exp(-chain.K_EXT * lai)))[idx]
B1B = [(0.1, 8), (1e-5, 0.05), (0, 2500), (0, 0.6), (1, 365), (0, 1)]

B = dict(alpha=(1e-4, 0.1), b0=(1e-3, 0.1), bg=(0.0, 0.6), L0=(0.1, 8), phi=(0, 1), r=(0, 0.005), c0=(0.2, 1.0))
SPECS = {"M3": ("bucket", "ramp", ["alpha", "b0", "bg", "L0", "phi", "r", "c0"]),
         "M3g": ("bucket", "gauss", ["alpha", "b0", "bg", "L0", "phi", "r", "c0"]),
         "M3n": ("none", "ramp", ["alpha", "b0", "bg", "L0", "r", "c0"]),
         "M3s": ("given", "ramp", ["alpha", "b0", "bg", "L0", "r", "c0"])}
def pm(nm, x):
    sm, vpd, names = SPECS[nm]; th = dict(zip(names, x)); th["cmax"] = 1.0
    if sm == "given": return simulate_given(drv, th, fS_era5_scaled)["ndvi"][idx]
    return simulate(drv, th, variant="dual", vpd=vpd, sm=sm)["ndvi"][idx]
def fitf(f, bounds, m, seed):
    obj = lambda x: float(np.mean((f(x)[m] - y[m]) ** 2))
    de = differential_evolution(obj, bounds, seed=seed, maxiter=300, popsize=20, tol=1e-8, polish=False)
    lo, hi = np.array(bounds).T
    r = least_squares(lambda x: f(x)[m] - y[m], np.clip(de.x, lo + 1e-12, hi - 1e-12), bounds=(lo, hi))
    return r.x if 2 * r.cost / m.sum() <= de.fun else de.x

cv = {k: np.full(len(y), np.nan) for k in ["B0", "B1r"] + list(SPECS)}; fold = {}
