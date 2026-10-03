"""Check every number reported in the article against the outputs of this repository.
Run from the repository root after run_all.sh:  python src/verify_paper.py
Each line prints the article value, the recomputed value and PASS/FAIL (agreement at the reported precision)."""
import sys, json, numpy as np, pandas as pd
from scipy.stats import chi2
sys.path.insert(0, "src")
import chain
from fit import d, drv, obs, harmonic_X

J = lambda f: json.load(open(f"results/{f}"))
R4, S, SEN, SUM = J("results4.json"), J("stats_results.json"), J("sensitivity.json"), J("summary.json")
PRO, B0P = J("stats_profiles_full.json"), J("stats_b0profile.json")
EPS, K = chain.EPS, chain.K_EXT
rows, fails = [], 0


def check(label, paper, value, ndp):
    global fails
    ok = abs(round(value, ndp) - paper) < 10 ** (-ndp) / 2 + 1e-12
    fails += (not ok)
    rows.append((("PASS" if ok else "FAIL"), label, paper, round(value, ndp + 2)))


# ---------------- data ----------------
h = pd.read_csv("data/raw/ERA5_orchard_12.6510N_102.2128E_2020-2026_hourly.csv", parse_dates=["time_local"])
check("hours of weather", 59016, len(h), 0)
rain = h.set_index("time_local").precipitation.resample("YE").sum()
check("annual rainfall min 2020-2025 (mm)", 2550, rain.iloc[:6].min(), 0)
check("annual rainfall max 2020-2025 (mm)", 3118, rain.iloc[:6].max(), 0)
day = h.shortwave_radiation > 0
check("daylight hours with VPD > Vmin (%)", 48, 100 * (h.loc[day, "vapour_pressure_deficit"] > chain.VPD_MIN).mean(), 0)
check("daylight hours with VPD > Vmax (%)", 1.6, 100 * (h.loc[day, "vapour_pressure_deficit"] > chain.VPD_MAX).mean(), 1)
s2 = pd.read_csv("data/raw/S2_orchard_indices_2020-2026.csv")
check("pixels in polygon", 745, s2.n_pix.max(), 0)
check("dates with clear pixels", 186, (s2.n_clear > 0).sum(), 0)
check("dates with >= 90% clear pixels", 137, (s2.clear_frac >= 0.9).sum(), 0)
check("scenes after temporal screening", 132, len(obs), 0)
sg = pd.read_csv("data/raw/SoilGrids_orchard_12.6510N_102.2128E.csv")
w = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15, "30-60cm": 30, "60-100cm": 40}
sg = sg[sg.depth.isin(w)].assign(w=lambda x: x.depth.map(w))
m = sg.groupby("property").apply(lambda g: (g["mean"] * g.w).sum() / g.w.sum(), include_groups=False)
check("theta_wp", 0.181, m["wv1500"], 3); check("theta_fc", 0.336, m["wv0033"], 3)
check("theta_sat = 1 - bd/2.65", 0.493, 1 - m["bdod"] / 2.65, 3)
check("sand (%)", 42, m["sand"], 0); check("clay (%)", 30, m["clay"], 0)
check("sigma_T for f(40)=0.1", 5.59, 12 / np.sqrt(2 * np.log(10)), 2)
sm = pd.Series(drv["SM"])
check("ERA5-Land root-zone SM min", 0.29, sm.min(), 2); check("ERA5-Land root-zone SM max", 0.51, sm.max(), 2)
check("days above theta_w (%)", 54, 100 * SUM["share_ERA5_above_theta_w"], 0)
check("days below theta_c (%)", 0, 100 * SUM["share_ERA5_below_theta_c"], 0)

# ---------------- Table 1 ----------------
F, Fo = R4["full"]["M3"], R4["fold"]["M3"].values()
fr = lambda f: (min(f(x) for x in Fo), max(f(x) for x in Fo))
check("psi", 0.0066, F["alpha"] * EPS * K, 4)
lo, hi = fr(lambda x: x["alpha"] * EPS * K); check("psi fold min", 0.0059, lo, 4); check("psi fold max", 0.0085, hi, 4)
check("1/beta0 (d)", 50.9, 1 / F["b0"], 1)
lo, hi = fr(lambda x: 1 / x["b0"]); check("1/beta0 fold min", 33.0, lo, 1); check("1/beta0 fold max", 56.3, hi, 1)
taus = sorted(1 / x["b0"] for x in Fo); check("1/beta0 second-lowest fold", 47.5, taus[1], 1)
check("phi", 0.109, F["phi"], 3)
lo, hi = fr(lambda x: x["phi"]); check("phi fold min", 0.074, lo, 3); check("phi fold max", 0.156, hi, 3)
check("c0", 0.67, F["c0"], 2)
lo, hi = fr(lambda x: x["c0"]); check("c0 fold min", 0.53, lo, 2); check("c0 fold max", 0.72, hi, 2)
check("r (1e-4)", 3.9, F["r"] * 1e4, 1)
lo, hi = fr(lambda x: x["r"] * 1e4); check("r fold min (1e-4)", 3.6, lo, 1); check("r fold max (1e-4)", 5.8, hi, 1)
check("l0 = k L0", 1.61, F["L0"] * K, 2)
lo, hi = fr(lambda x: x["L0"] * K); check("l0 fold min", 0.96, lo, 2); check("l0 fold max", 4.00, hi, 2)
check("rho_bg", 0.02, F["bg"], 2)
lo, hi = fr(lambda x: x["bg"]); check("rho_bg fold min", 0.00, lo, 2); check("rho_bg fold max", 0.29, hi, 2)

# ---------------- stress scalars ----------------
out = chain.simulate(drv, {**F, "cmax": 1.0}, variant="dual", vpd="ramp", sm="bucket")
D = d.index
roll = lambda a: pd.Series(a, D).rolling(15, center=True).mean()
fv, ft, fs = roll(out["fV"]), roll(out["fT"]), roll(out["fS"])
check("15-d f(VPD) dry-season minimum, lowest year", 0.28, fv.groupby(D.year).min().min(), 2)
check("15-d f(VPD) dry-season minimum, highest year", 0.44, fv.groupby(D.year).min().max(), 2)
check("15-d f(T) minimum", 0.70, ft.min(), 2)
check("15-d Ks dry-season minimum, lowest year", 0.68, fs.groupby(D.year).min().min(), 2)
check("15-d Ks dry-season minimum, highest year", 0.73, fs.groupby(D.year).min().max(), 2)
check("15-d Ks wet-season (Jun-Oct) minimum", 1.0, fs[(D.month >= 6) & (D.month <= 10)].min(), 2)

# ---------------- Table 2 and skill ----------------
T2 = {"B0": (0.063, 0.046, 0.53), "B1r": (0.082, 0.067, 0.20), "M3": (0.070, 0.054, 0.42),
      "M3g": (0.074, 0.055, 0.35), "M3s": (0.074, 0.057, 0.35), "M3n": (0.071, 0.056, 0.40)}
for k, (rm, ma, r2) in T2.items():
    c = R4["cv"][k]; check(f"{k} RMSE", rm, c["RMSE"], 3); check(f"{k} MAE", ma, c["MAE"], 3); check(f"{k} R2", r2, c["R2"], 2)
check("M3 R2 as % of B0 R2", 79, 100 * R4["cv"]["M3"]["R2"] / R4["cv"]["B0"]["R2"], 0)
cv = pd.read_csv("results/cv_predictions4.csv")
check("corr of departures from B0", 0.10, np.corrcoef(cv.NDVI_obs - cv.B0, cv.M3 - cv.B0)[0, 1], 2)
d0 = [round(x[4]) for x in R4["fold"]["B1r"].values()]
check("B1 folds with reset on day 8", 5, sum(v == 8 for v in d0), 0)
b1 = list(R4["fold"]["B1r"].values())
check("B1 folds with slope at upper bound", 4, sum(x[1] > 0.0499 for x in b1), 0)
check("B1 background min", 0.51, min(x[3] for x in b1), 2); check("B1 background max", 0.55, max(x[3] for x in b1), 2)
check("B1 phi min", 0, min(x[5] for x in b1), 0); check("B1 phi max", 1, max(x[5] for x in b1), 0)
X = harmonic_X(obs.index); y = obs.NDVI.values; e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
check("B0 in-sample R2", 0.62, 1 - e @ e / ((y - y.mean()) @ (y - y.mean())), 2)
check("B1 in-sample R2", 0.41, R4["cv"]["B1r"]["insample"]["R2"], 2)
check("M3 in-sample R2", 0.58, R4["cv"]["M3"]["insample"]["R2"], 2)

# ---------------- latent states ----------------
check("L min", 2.0, SUM["L_range"][0], 1); check("L max", 4.8, SUM["L_range"][1], 1)
check("c start", 0.67, SUM["c_start"], 2); check("c end", 0.91, SUM["c_end"], 2)
ny = obs.NDVI.groupby(obs.index.year).mean()
check("observed annual NDVI 2020", 0.52, ny[2020], 2); check("observed annual NDVI 2025", 0.65, ny[2025], 2)
check("mean ground GPP (g C m-2 d-1)", 5.5, SUM["GPP_daily_mean"], 1)
g = [v for k, v in SUM["GPP_annual_gC"].items() if k != "2026"]
check("annual GPP min (kg C)", 1.7, min(g) / 1000, 1); check("annual GPP max (kg C)", 2.2, max(g) / 1000, 1)

# ---------------- memory and sensitivity ----------------
check("scan peak (d)", 58, SUM["tau_emp"], 0); check("scan r", 0.62, SUM["r_emp"], 2)
check("tau_eff (d)", 81, SUM["tau_eff"], 0)
SR = J("scan_robustness.json")
check("scan peak without soil-water scalar (d)", 61, SR["no_soil_water_scalar"]["peak_days"], 0)
check("scan r without soil-water scalar", 0.52, SR["no_soil_water_scalar"]["r"], 2)
check("scan peak with phi = 0 (d)", 47, SR["phi_0"]["peak_days"], 0)
check("scan peak with calibrated phi (d)", 58, SR["calibrated_phi"]["peak_days"], 0)
T3 = {"base": (0.575, 50.9, 0.109), "T_OPT=26": (0.586, 36.1, 0.225), "T_OPT=30": (0.552, 56.1, 0.041),
      "VPD_MAX=2.5": (0.576, 43.7, 0.145), "VPD_MAX=3.7": (0.568, 50.1, 0.076), "VPD_MIN=0.65": (0.584, 50.8, 0.145),
      "VPD_MIN=1.0": (0.559, 46.9, 0.069), "EPS=1.044(crop)": (0.575, 50.9, 0.109), "K_EXT=0.4": (0.575, 50.9, 0.109),
      "K_EXT=0.6": (0.575, 50.9, 0.109), "KC=0.8": (0.569, 51.6, 0.117), "KC=1.0": (0.581, 47.1, 0.100),
      "ZR=600": (0.578, 52.4, 0.135), "ZR=1400": (0.573, 49.2, 0.098)}
for k, (r2, tau, phi) in T3.items():
    v = SEN[k]; check(f"sens {k} R2", r2, v["R2"], 3); check(f"sens {k} tau", tau, v["tau"], 1); check(f"sens {k} phi", phi, v["phi"], 3)
SN = J("sensitivity_ndvifull.json")
for k, (r2, tau, phi) in {"NDVI_FULL=0.85": (0.569, 51.2, 0.094), "NDVI_FULL=0.95": (0.580, 49.4, 0.118)}.items():
    v = SN[k]; check(f"sens {k} R2", r2, v["R2"], 3); check(f"sens {k} tau", tau, v["tau"], 1); check(f"sens {k} phi", phi, v["phi"], 3)
check("NDVI_full: max change of turnover time below 2 d", 1, float(max(abs(SN[k]["tau"] - SN["NDVI_FULL=0.9"]["tau"]) for k in SN) < 2), 0)
allt = [v["tau"] for k, v in SEN.items() if k != "base"] + [SN[k]["tau"] for k in ["NDVI_FULL=0.85", "NDVI_FULL=0.95"]]
allr = [v["R2"] for k, v in SEN.items() if k != "base"] + [SN[k]["R2"] for k in ["NDVI_FULL=0.85", "NDVI_FULL=0.95"]]
check("sensitivity cases (13 + NDVI_full 2)", 15, len(allt), 0)
check("sensitivity tau min (all 15)", 36.1, min(allt), 1); check("sensitivity tau max (all 15)", 56.1, max(allt), 1)
check("sensitivity R2 min (all 15)", 0.55, min(allr), 2); check("sensitivity R2 max (all 15)", 0.59, max(allr), 2)
AB = J("ablation_slowpool.json")
check("no slow pool (M3c): out-of-year RMSE", 0.146, AB["cv"]["RMSE"], 3)
check("no slow pool: RMSE ratio to M3 (doubled)", 2.0, AB["cv"]["RMSE"] / R4["cv"]["M3"]["RMSE"], 0)
check("no slow pool: in-sample R2", 0.38, AB["insample"]["R2"], 2)
check("no slow pool: turnover time (d)", 24, AB["tau_full"], 0)

# ---------------- discrepancy ----------------
md = SUM["monthly_discrepancy"]
check("discrepancy June", 0.10, md["6"], 2); check("discrepancy July", 0.10, md["7"], 2)
neg = [md["4"], md["8"], md["9"]]
check("discrepancy Apr/Aug/Sep, smallest magnitude", 0.05, -max(neg), 2); check("discrepancy Apr/Aug/Sep, largest magnitude", 0.06, -min(neg), 2)

# ---------------- statistical layer ----------------
acf = [a["r"] for a in S["acf"]]
for lab, pv, v in zip(["1-15", "15-30", "30-60", "60-120", "120-240"], [0.73, 0.50, 0.02, 0.02, -0.09], acf):
    check(f"residual correlation {lab} d", pv, v, 2)
check("OU correlation length (d)", 34.4, S["ell_gls"], 1); check("sigma", 0.063, S["sigma"], 3)
check("LR statistic", 90.0, S["LR_ou_vs_iid"], 1); check("p < 1e-20", 1, float(S["p_LR"] < 1e-20), 0)
check("whitened lag-1 correlation", -0.04, S["whitened_resid_lag1_corr"], 2)
grid = np.array(B0P["grid"]); c95 = chi2.ppf(0.95, 1)
for lab, key, lo_p, hi_p in [("OU", "dev_ou", 14.3, 81.0), ("iid", "dev_iid", 25.5, 65.2)]:
    ins = grid[np.array(B0P[key]) <= c95]
    check(f"tau 95% lower ({lab})", lo_p, 1 / ins.max(), 1); check(f"tau 95% upper ({lab})", hi_p, 1 / ins.min(), 1)
tg = S["theta_gls"]
check("GLS 1/beta0", 50.3, 1 / tg["b0"], 1); check("GLS psi", 0.0064, tg["alpha"] * EPS * K, 4)
check("GLS r (1e-4)", 3.4, tg["r"] * 1e4, 1); check("GLS c0", 0.73, tg["c0"], 2); check("GLS phi", 0.17, tg["phi"], 2)
check("GLS rho_bg", 0.01, tg["bg"], 2); check("GLS l0", 1.39, tg["L0"] * K, 2)
ci = lambda k: PRO[k]["ci"]
check("psi CI low", 0.0036, ci("alpha")[0] * EPS * K, 4); check("psi CI high", 0.0167, ci("alpha")[1] * EPS * K, 4)
check("r CI low (1e-4)", 2.0, ci("r")[0] * 1e4, 1); check("r CI high (1e-4)", 26, ci("r")[1] * 1e4, 0)
check("c0 CI low", 0.26, ci("c0")[0], 2); check("c0 CI high", 0.87, ci("c0")[1], 2)
check("phi CI low", 0.04, ci("phi")[0], 2); check("phi CI reaches 1", 1, float(PRO["phi"]["hits_upper"]), 0)
check("rho_bg CI high", 0.50, ci("bg")[1], 2); check("rho_bg CI reaches 0", 1, float(PRO["bg"]["hits_lower"]), 0)
check("l0 CI low", 0.05, ci("L0")[0] * K, 2); check("l0 CI high", 2.78, ci("L0")[1] * K, 2)

# ---------------- screening-rule sensitivity ----------------
sf = {f"{a}_{w}": J(f"sens_filter_{a}_{w}.json") for a, w in
      [("none", 30), ("0.06", 30), ("0.08", 30), ("0.10", 30), ("0.08", 20), ("0.08", 45)]}
m3 = [v["cv"]["M3"]["R2"] for k, v in sf.items() if k != "none_30"]; tf = [v["tau_full"] for k, v in sf.items() if k != "none_30"]
rem = [v["n_removed"] for k, v in sf.items() if k != "none_30"]
check("screening: M3 R2 min", 0.40, min(m3), 2); check("screening: M3 R2 max", 0.42, max(m3), 2)
check("screening: tau min", 48.8, min(tf), 1); check("screening: tau max", 51.2, max(tf), 1)
check("screening: scenes removed min", 1, min(rem), 0); check("screening: scenes removed max", 11, max(rem), 0)
check("no filter: M3 R2", 0.41, sf["none_30"]["cv"]["M3"]["R2"], 2)

w = max(len(r[1]) for r in rows)
for st, lab, pv, v in rows:
    print(f"{st}  {lab:<{w}}  article {pv:<8}  code {v}")
print(f"\n{len(rows) - fails} of {len(rows)} checks passed")
sys.exit(1 if fails else 0)
