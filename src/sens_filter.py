"""Sensitivity of LOYO skill and canopy memory to the temporal NDVI screening rule.
usage: python sens_filter.py DELTA WINDOW   (DELTA='none' disables the filter)"""
import sys, json, numpy as np, pandas as pd
import fit4lib as fit4
from fit4lib import b1r, B1B, B, SPECS, pm, fitf, harmonic_X, metrics, d

delta = None if sys.argv[1] == "none" else float(sys.argv[1]); win = int(sys.argv[2])
s = pd.read_csv("data/raw/S2_orchard_indices_2020-2026.csv", parse_dates=["date"])
g = s[s.clear_frac >= 0.9].reset_index(drop=True)
flag = np.zeros(len(g), bool)
if delta is not None:
    for i, r in g.iterrows():
        nb = g[(abs((g.date - r.date).dt.days) <= win) & (g.index != i)]
        if len(nb) >= 2 and r.NDVI < nb.NDVI.median() - delta: flag[i] = True
obs = g[~flag].reset_index(drop=True)
# rebind module-level data used by fit4 helpers
fit4.idx = d.index.get_indexer(obs.date); fit4.y = obs.NDVI.values
y = fit4.y; yrs = obs.date.dt.year.values
X = harmonic_X(pd.DatetimeIndex(obs.date))
models = ["B0", "B1r", "M3", "M3s"]
cv = {k: np.full(len(y), np.nan) for k in models}; taus = []
for yr in np.unique(yrs):
    tr, te = yrs != yr, yrs == yr
    b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; cv["B0"][te] = (X @ b)[te]
    x = fitf(b1r, B1B, tr, int(yr)); cv["B1r"][te] = b1r(x)[te]
    for nm in ["M3", "M3s"]:
        x = fitf(lambda z: pm(nm, z), [B[n] for n in SPECS[nm][2]], tr, int(yr)); cv[nm][te] = pm(nm, x)[te]
        if nm == "M3": taus.append(1 / dict(zip(SPECS[nm][2], x))["b0"])
xf = fitf(lambda z: pm("M3", z), [B[n] for n in SPECS["M3"][2]], np.ones(len(y), bool), 42)
thf = dict(zip(SPECS["M3"][2], xf))
out = dict(delta=delta, window=win, n_removed=int(flag.sum()), n=int(len(y)),
           removed_dates=[str(t.date()) for t in g.date[flag]],
           cv={k: metrics(y, v) for k, v in cv.items()}, tau_full=1 / thf["b0"], tau_folds=taus,
           theta_full=thf, insample_M3=metrics(y, pm("M3", xf)))
json.dump(out, open(f"results/sens_filter_{sys.argv[1]}_{win}.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ["delta", "window", "n_removed", "n", "tau_full"]}),
      {k: round(v["R2"], 3) for k, v in out["cv"].items()})
