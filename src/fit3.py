"""Fixes for negative R2: (1) temporal outlier filter on NDVI, (2) ERA5-Land soil moisture used with
its own internal wetness scale instead of SoilGrids thresholds. Leave-one-year-out CV."""
import numpy as np, pandas as pd, json
import chain
from fit import d, drv, metrics, harmonic_X, nectec_fit, differential_evolution, least_squares
from chain import simulate, nectec_logistic, P_DEPL

s = pd.read_csv("data/raw/S2_orchard_indices_2020-2026.csv", parse_dates=["date"])
g = s[s.clear_frac >= 0.9].reset_index(drop=True)
flag = np.zeros(len(g), bool)
for i, r in g.iterrows():
    nb = g[(abs((g.date - r.date).dt.days) <= 30) & (g.index != i)]
    if len(nb) >= 2 and r.NDVI < nb.NDVI.median() - 0.08:
        flag[i] = True

# ERA5-Land relative wetness on its own scale (5th-95th percentile), dry-side FAO-style ramp only
sm = drv["SM"]; lo, hi = np.percentile(sm, [5, 95])
w = np.clip((sm - lo) / (hi - lo), 0, 1)
fS_era5_scaled = np.clip(w / (1 - P_DEPL), 0, 1)

B = dict(alpha=(1e-4, 0.1), b0=(1e-3, 0.1), bg=(0.0, 0.6), L0=(0.1, 8), phi=(0, 1), r=(0, 0.005), c0=(0.2, 1.0))
SPECS = {
 "M3_dual_bucket": (dict(variant="dual", vpd="ramp", sm="bucket"), ["alpha", "b0", "bg", "L0", "phi", "r", "c0"]),
 "M3e_dual_ERA5SM_SoilGrids": (dict(variant="dual", vpd="ramp", sm="era5"), ["alpha", "b0", "bg", "L0", "r", "c0"]),
 "M3s_dual_ERA5SM_ownscale": (dict(variant="dual", vpd="ramp", sm="given"), ["alpha", "b0", "bg", "L0", "r", "c0"]),
}


def sim(nm, x):
    th = dict(zip(SPECS[nm][1], x)); th["cmax"] = 1.0
    kw = SPECS[nm][0]
    if kw["sm"] == "given":
        # reuse simulate with sm="none", then rebuild with the scaled scalar
        d2 = dict(drv); out = simulate_given(d2, th, fS_era5_scaled)
        return out
    return simulate(drv, th, **kw)


def simulate_given(drv_, th, fS):
    PAR = np.asarray(drv_["PAR"], float)
    fT = chain.f_T(drv_["T"]); fV = chain.f_VPD_ramp(drv_["VPD"])
    stress = np.asarray(fT * fV * fS, float)
    gbar = chain.EPS * np.mean(PAR * stress) * 0.5
    L, c, gpp = chain._canopy(PAR, stress, np.asarray(fS, float), th["L0"], th["c0"], th["alpha"], th["b0"], 0.0,
                              th["r"], 1.0, True, chain.EPS, chain.K_EXT, gbar)
    fpar = 1 - np.exp(-chain.K_EXT * L)
    return dict(ndvi=th["bg"] + (chain.NDVI_FULL - th["bg"]) * c * fpar)


def run(tag, obs):
    idx = d.index.get_indexer(obs.date); y = obs.NDVI.values; yrs = obs.date.dt.year.values
    pred = lambda nm, x: sim(nm, x)["ndvi"][idx]
    def fit(nm, m, seed):
        bounds = [B[n] for n in SPECS[nm][1]]
        f = lambda x: float(np.mean((pred(nm, x)[m] - y[m]) ** 2))
        de = differential_evolution(f, bounds, seed=seed, maxiter=300, popsize=20, tol=1e-8, polish=False)
        lo_, hi_ = np.array(bounds).T
        r = least_squares(lambda x: pred(nm, x)[m] - y[m], np.clip(de.x, lo_ + 1e-12, hi_ - 1e-12), bounds=(lo_, hi_))
        return r.x if 2 * r.cost / m.sum() <= de.fun else de.x
    cv = {k: np.full(len(y), np.nan) for k in ["B0", "B1_annual_crop"] + list(SPECS)}
    tau = []
    X = harmonic_X(pd.DatetimeIndex(obs.date))
    import fit as F
    for yr in np.unique(yrs):
        tr, te = yrs != yr, yrs == yr
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; cv["B0"][te] = (X @ b)[te]
        F.idx, F.y = idx, y
        xn = nectec_fit(tr)
        cv["B1_annual_crop"][te] = nectec_logistic(drv, dict(Lmax=xn[0], a=xn[1], b=xn[2], bg=xn[3]))["ndvi"][idx][te]
        for nm in SPECS:
            x = fit(nm, tr, int(yr)); cv[nm][te] = pred(nm, x)[te]
            if nm == "M3_dual_bucket": tau.append(1 / x[1])
        print(tag, yr, flush=True)
    res = {k: metrics(y, v) for k, v in cv.items()}
    res["tau_folds"] = tau
    return res


if __name__ == "__main__":
    out = {"n_flagged": int(flag.sum()), "flagged_dates": g.date[flag].dt.strftime("%Y-%m-%d").tolist(),
           "era5_scale": [float(lo), float(hi)]}
    out["clean"] = run("clean", g[~flag].reset_index(drop=True))
    json.dump(out, open("results/results3.json", "w"), indent=1)
    for k, v in out["clean"].items():
        if isinstance(v, dict): print(k, {m: round(v[m], 4) for m in ["RMSE", "MAE", "R2"]})
    print("tau", np.round(out["clean"]["tau_folds"], 1))
