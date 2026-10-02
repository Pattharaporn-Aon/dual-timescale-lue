"""Sensitivity of the calibrated chain (M3) to the fixed literature values."""
import numpy as np, json, chain
from fit2 import *

ALL = np.ones(len(y), bool)
NAMES = SPECS["M3_dual_bucket"][1]
cases = [("base", {}), ("T_OPT=26", {"T_OPT": 26.0}), ("T_OPT=30", {"T_OPT": 30.0}),
         ("VPD_MAX=2.5", {"VPD_MAX": 2.5}), ("VPD_MAX=3.7", {"VPD_MAX": 3.7}),
         ("VPD_MIN=0.65", {"VPD_MIN": 0.65}), ("VPD_MIN=1.0", {"VPD_MIN": 1.0}),
         ("EPS=1.044(crop)", {"EPS": 1.044}), ("K_EXT=0.4", {"K_EXT": 0.4}), ("K_EXT=0.6", {"K_EXT": 0.6}),
         ("KC=0.8", {"KC": 0.8}), ("KC=1.0", {"KC": 1.0}), ("ZR=600", {"ZR": 600.0}), ("ZR=1400", {"ZR": 1400.0})]
base = {k: getattr(chain, k) for k in ["T_OPT", "VPD_MAX", "VPD_MIN", "EPS", "K_EXT", "KC", "ZR", "TAW"]}
out = {}
for lab, ch in cases:
    for k, v in base.items(): setattr(chain, k, v)
    for k, v in ch.items(): setattr(chain, k, v)
    chain.TAW = (chain.SOIL["fc"] - chain.SOIL["wp"]) * chain.ZR
    x = fit_chain("M3_dual_bucket", ALL, seed=7)
    m = metrics(y, pred_chain("M3_dual_bucket", x)); th = dict(zip(NAMES, x))
    out[lab] = dict(R2=m["R2"], RMSE=m["RMSE"], tau=1 / th["b0"], phi=th["phi"], c0=th["c0"], r=th["r"], alpha=th["alpha"])
    print(lab, {k: round(v, 4) for k, v in out[lab].items()}, flush=True)
json.dump(out, open("results/sensitivity.json", "w"), indent=1)
