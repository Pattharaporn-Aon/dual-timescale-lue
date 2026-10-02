"""Memory scan repeated with the soil-water scalar removed or with phi fixed, so that no estimated
parameter enters the scan. Writes results/scan_robustness.json."""
import json, numpy as np, pandas as pd
import chain
from fit import d, drv, obs, y

r = json.load(open("results/results4.json")); phi_hat = r["full"]["M3"]["phi"]
t0 = (obs.index - obs.index[0]).days.values
A = np.c_[np.ones_like(t0), t0]; res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]   # linearly detrended NDVI
base = drv["PAR"] * chain.f_T(drv["T"]) * chain.f_VPD_ramp(drv["VPD"])
cases = {"calibrated_phi": chain.bucket(drv["P"], drv["ET0"], phi_hat),
         "phi_0": chain.bucket(drv["P"], drv["ET0"], 0.0),
         "phi_1": chain.bucket(drv["P"], drv["ET0"], 1.0),
         "no_soil_water_scalar": np.ones(len(d))}
out = {}
for k, fs in cases.items():
    pot = pd.Series(base * fs, d.index)
    rr = np.array([np.corrcoef(pot.ewm(alpha=1 / t).mean().reindex(obs.index).values, res)[0, 1] for t in range(1, 181)])
    out[k] = dict(peak_days=int(rr.argmax()) + 1, r=float(rr.max()))
    print(k, out[k])
json.dump(out, open("results/scan_robustness.json", "w"), indent=1)
