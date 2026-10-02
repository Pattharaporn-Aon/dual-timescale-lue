"""Second analysis: does the weather-driven chain carry information beyond a seasonal+trend statistical baseline?"""
import numpy as np, pandas as pd, json
from fit import *

SPECS["M3_dual_bucket"] = (dict(variant="dual", vpd="ramp", sm="bucket"), ["alpha", "b0", "bg", "L0", "phi", "r", "c0"])
SPECS["M3g_dual_gaussVPD"] = (dict(variant="dual", vpd="gauss", sm="bucket"), ["alpha", "b0", "bg", "L0", "phi", "r", "c0"])
SPECS["M3e_dual_ERA5SM"] = (dict(variant="dual", vpd="ramp", sm="era5"), ["alpha", "b0", "bg", "L0", "r", "c0"])
SPECS["M3n_dual_nowater"] = (dict(variant="dual", vpd="ramp", sm="none"), ["alpha", "b0", "bg", "L0", "r", "c0"])
MODELS = ["M3_dual_bucket", "M3g_dual_gaussVPD", "M3e_dual_ERA5SM", "M3n_dual_nowater"]

X = harmonic_X(obs.index)          # 1, t, s1, c1, s2, c2
Xs = X[:, [0, 2, 3, 4, 5]]         # seasonal only (no trend)

if __name__ == "__main__":
    out = {k: np.full(len(y), np.nan) for k in
           ["B0"] + MODELS + ["H2_chain_plus_seasonal_discrepancy", "H3_harmonic_plus_chain"]}
    gam, fold = [], {}
    for yr in np.unique(years):
        tr, te = years != yr, years == yr
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; out["B0"][te] = (X @ b)[te]
        chains = {}
        for nm in MODELS:
            x = fit_chain(nm, tr, seed=int(yr)); p = pred_chain(nm, x); chains[nm] = p
            out[nm][te] = p[te]
            fold.setdefault(nm, {})[int(yr)] = th_of(nm, x)
        p = chains["M3_dual_bucket"]
        bd = np.linalg.lstsq(Xs[tr], (y - p)[tr], rcond=None)[0]
        out["H2_chain_plus_seasonal_discrepancy"][te] = (p + Xs @ bd)[te]
        Z = np.c_[X, p]; bz = np.linalg.lstsq(Z[tr], y[tr], rcond=None)[0]
        out["H3_harmonic_plus_chain"][te] = (Z @ bz)[te]; gam.append(float(bz[-1]))
        print("fold", yr, round(bz[-1], 3), flush=True)
    res = {k: metrics(y, v) for k, v in out.items()}
    # anomaly skill: observed departure from seasonal+trend baseline vs chain departure
    an_obs = y - out["B0"]; an_ch = out["M3_dual_bucket"] - out["B0"]
    res["anomaly_corr_M3"] = float(np.corrcoef(an_obs, an_ch)[0, 1])
    res["gamma_per_fold"] = gam
    full = {nm: th_of(nm, fit_chain(nm, np.ones(len(y), bool), seed=42)) for nm in MODELS}
    for nm in MODELS:
        res[nm]["insample"] = metrics(y, simulate(drv, full[nm], **SPECS[nm][0])["ndvi"][idx])
    json.dump(dict(cv=res, full=full, fold=fold), open("results/results2.json", "w"), indent=1)
    pd.DataFrame({"date": obs.index, "NDVI_obs": y, **out}).to_csv("results/cv_predictions2.csv", index=False)
    print(pd.DataFrame({k: v for k, v in res.items() if isinstance(v, dict)}).T[["RMSE", "MAE", "R2", "bias"]].round(4))
    print("anomaly corr", res["anomaly_corr_M3"], "gamma", np.round(gam, 3))
    print(json.dumps(full, indent=1))
