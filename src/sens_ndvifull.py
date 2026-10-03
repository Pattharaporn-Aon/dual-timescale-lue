"""Sensitivity of the calibrated chain (M3) to the assumed closed-canopy NDVI (NDVI_full = 0.85, 0.95).
Same procedure as sens.py (full-data refit, seed 7). Writes results/sensitivity_ndvifull.json."""
import json, chain
from fit2 import *

ALL = np.ones(len(y), bool); NAMES = SPECS["M3_dual_bucket"][1]; out = {}
for v in [0.85, 0.90, 0.95]:
    chain.NDVI_FULL = v
    x = fit_chain("M3_dual_bucket", ALL, seed=7)
    m = metrics(y, pred_chain("M3_dual_bucket", x)); th = dict(zip(NAMES, x))
    out[f"NDVI_FULL={v}"] = dict(R2=m["R2"], RMSE=m["RMSE"], tau=1 / th["b0"], phi=th["phi"], c0=th["c0"], r=th["r"], bg=th["bg"])
    print(v, {k: round(float(w), 4) for k, w in out[f"NDVI_FULL={v}"].items()}, flush=True)
chain.NDVI_FULL = 0.90
json.dump(out, open("results/sensitivity_ndvifull.json", "w"), indent=1)
