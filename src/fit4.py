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
X = harmonic_X(pd.DatetimeIndex(obs.date))
for yr in np.unique(yrs):
    tr, te = yrs != yr, yrs == yr
    b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; cv["B0"][te] = (X @ b)[te]
    x = fitf(b1r, B1B, tr, int(yr)); cv["B1r"][te] = b1r(x)[te]; fold.setdefault("B1r", {})[int(yr)] = x.tolist()
    for nm in SPECS:
        x = fitf(lambda z: pm(nm, z), [B[n] for n in SPECS[nm][2]], tr, int(yr)); cv[nm][te] = pm(nm, x)[te]
        fold.setdefault(nm, {})[int(yr)] = dict(zip(SPECS[nm][2], x.tolist()))
    print(yr, flush=True)
res = {k: metrics(y, v) for k, v in cv.items()}
full = {nm: dict(zip(SPECS[nm][2], fitf(lambda z: pm(nm, z), [B[n] for n in SPECS[nm][2]], np.ones(len(y), bool), 42).tolist())) for nm in SPECS}
xb = fitf(b1r, B1B, np.ones(len(y), bool), 42); full["B1r"] = xb.tolist()
for nm in SPECS: res[nm]["insample"] = metrics(y, pm(nm, list(full[nm].values())))
res["B1r"]["insample"] = metrics(y, b1r(xb))
json.dump(dict(cv=res, fold=fold, full=full, n=int(len(y))), open("results/results4.json", "w"), indent=1)
pd.DataFrame({"date": obs.date, "NDVI_obs": y, **cv}).to_csv("results/cv_predictions4.csv", index=False)
for k, v in res.items(): print(k, {m: round(v[m], 4) for m in ["RMSE", "MAE", "R2"]}, round(v.get("insample", {}).get("R2", np.nan), 3))
print("tau", [round(1 / v["b0"], 1) for v in fold["M3"].values()], "B1r d0", [round(v[4]) for v in fold["B1r"].values()])
