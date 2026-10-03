"""Ablation of the slow pool: M3 with the crown-expansion rate fixed at r = 0 (constant crown cover c0).
Leave-one-year-out comparison with the same procedure as fit4.py. Writes results/ablation_slowpool.json."""
import json, numpy as np, pandas as pd
import fit4lib as F
from fit4lib import pm, fitf, metrics, B, SPECS, d
from fit3 import g, flag

obs = g[~flag].reset_index(drop=True)
F.idx = d.index.get_indexer(obs.date); F.y = obs.NDVI.values
y = F.y; yrs = obs.date.dt.year.values
SPECS["M3c"] = ("bucket", "ramp", ["alpha", "b0", "bg", "L0", "phi", "c0"])
cv = np.full(len(y), np.nan); taus = []
for yr in np.unique(yrs):
    tr, te = yrs != yr, yrs == yr
    x = fitf(lambda z: pm("M3c", z), [B[n] for n in SPECS["M3c"][2]], tr, int(yr)); cv[te] = pm("M3c", x)[te]
    taus.append(1 / dict(zip(SPECS["M3c"][2], x))["b0"]); print(yr, flush=True)
xf = fitf(lambda z: pm("M3c", z), [B[n] for n in SPECS["M3c"][2]], np.ones(len(y), bool), 42)
th = dict(zip(SPECS["M3c"][2], xf))
out = dict(cv=metrics(y, cv), insample=metrics(y, pm("M3c", xf)), tau_full=1 / th["b0"], tau_folds=taus, theta_full=th)
json.dump(out, open("results/ablation_slowpool.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ["cv", "insample", "tau_full"]}, indent=1))
