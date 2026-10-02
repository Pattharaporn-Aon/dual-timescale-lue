import numpy as np, pandas as pd, json, warnings
from scipy.optimize import least_squares, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF, WhiteKernel, ConstantKernel as C
from chain import *
warnings.filterwarnings("ignore")

d = pd.read_csv("data/processed/ERA5_orchard_2020-2026_daily.csv", parse_dates=["time_local"]).set_index("time_local")
s = pd.read_csv("data/raw/S2_orchard_indices_2020-2026.csv", parse_dates=["date"])
_g = s[s.clear_frac >= 0.9].reset_index(drop=True)
_flag = [(len(nb := _g[(abs((_g.date - r.date).dt.days) <= 30) & (_g.index != i)]) >= 2) and (r.NDVI < nb.NDVI.median() - 0.08)
         for i, r in _g.iterrows()]
obs = _g[~pd.Series(_flag)].set_index("date")   # cloud-screened and temporally filtered
drv = dict(PAR=d.PAR_MJ_m2_d.values, T=d.T_daytime_C.values, VPD=d.VPD_daytime_kPa.values,
           SM=d.SM_rootzone_0_100cm.values, P=d.P_mm_d.values, ET0=d.ET0_mm_d.values)
idx = d.index.get_indexer(obs.index)
y = obs.NDVI.values
years = obs.index.year.values
tdays = (obs.index - pd.Timestamp("2020-01-01")).days.values.astype(float)

# name: (simulator kwargs, {param: (lower, upper)})
B = dict(alpha=(1e-4, 0.1), b0=(1e-3, 0.1), b1=(0, 0.3), bg=(0.0, 0.6), L0=(0.1, 8), phi=(0, 1),
         r=(0, 0.005), c0=(0.2, 1.0), uwet=(0.2, 0.9), tau_u=(2, 120))
SPECS = {
 "M1_single_ERA5SM": (dict(variant="single", vpd="ramp", sm="era5"), ["alpha", "b0", "b1", "bg", "L0"]),
 "M2_single_bucket": (dict(variant="single", vpd="ramp", sm="bucket"), ["alpha", "b0", "b1", "bg", "L0", "phi"]),
 "M3_dual_bucket": (dict(variant="dual", vpd="ramp", sm="bucket"), ["alpha", "b0", "b1", "bg", "L0", "phi", "r", "c0"]),
 "M3g_dual_gaussVPD": (dict(variant="dual", vpd="gauss", sm="bucket"), ["alpha", "b0", "b1", "bg", "L0", "phi", "r", "c0"]),
 "M4_dual_understory": (dict(variant="dual", vpd="ramp", sm="bucket"), ["alpha", "b0", "b1", "bg", "L0", "phi", "r", "c0", "uwet", "tau_u"]),
}


def th_of(name, x):
    th = dict(zip(SPECS[name][1], x)); th["cmax"] = 1.0
    return th


def sim(name, x):
    return simulate(drv, th_of(name, x), **SPECS[name][0])


def pred_chain(name, x):
    return sim(name, x)["ndvi"][idx]


def fit_chain(name, mask, seed=0):
    names = SPECS[name][1]; bounds = [B[n] for n in names]
    f = lambda x: float(np.mean((pred_chain(name, x)[mask] - y[mask]) ** 2))
    de = differential_evolution(f, bounds, seed=seed, maxiter=300, popsize=20, tol=1e-8, polish=False)
    lo, hi = np.array(bounds).T
    r = least_squares(lambda x: pred_chain(name, x)[mask] - y[mask], np.clip(de.x, lo + 1e-12, hi - 1e-12), bounds=(lo, hi))
    return r.x if r.cost * 2 / mask.sum() <= de.fun else de.x


def nectec_fit(mask):
    def res(x):
        return nectec_logistic(drv, dict(Lmax=x[0], a=x[1], b=x[2], bg=x[3]))["ndvi"][idx][mask] - y[mask]
    f = lambda x: float(np.mean(res(x) ** 2))
    de = differential_evolution(f, [(0.1, 8), (1e-6, 0.05), (0, 20000), (0, 0.6)], seed=1, maxiter=200, polish=False)
    return de.x


def harmonic_X(t):
    doy = 2 * np.pi * t.dayofyear.values / 365.25
    tt = (t - pd.Timestamp("2020-01-01")).days.values / 365.25
    return np.c_[np.ones(len(t)), tt, np.sin(doy), np.cos(doy), np.sin(2 * doy), np.cos(2 * doy)]


def gp_residual(tr_t, tr_r, te_t):
    k = C(0.003, (1e-5, 0.1)) * ExpSineSquared(length_scale=60, periodicity=365.25, periodicity_bounds="fixed",
                                                  length_scale_bounds=(5, 1000)) * RBF(1500, (365, 1e5)) \
        + WhiteKernel(0.002, (1e-5, 0.05))
    gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=2, random_state=0)
    gp.fit(tr_t[:, None], tr_r)
    m, sd = gp.predict(te_t[:, None], return_std=True)
    return m, sd, gp


def metrics(yt, yp):
    e = yp - yt
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                R2=float(1 - np.sum(e ** 2) / np.sum((yt - yt.mean()) ** 2)), bias=float(e.mean()))


if __name__ == "__main__":
    names = ["B0_harmonic_trend", "B1_NECTEC_logistic"] + list(SPECS) + ["M5_hybrid_M4_GP", "B2_GP_only"]
    cvpred = {nm: np.full(len(y), np.nan) for nm in names}
    cvsd = np.full(len(y), np.nan)
    foldpar = {}
    for yr in np.unique(years):
        tr, te = years != yr, years == yr
        X = harmonic_X(obs.index); b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        cvpred["B0_harmonic_trend"][te] = (X @ b)[te]
        xn = nectec_fit(tr)
        cvpred["B1_NECTEC_logistic"][te] = nectec_logistic(drv, dict(Lmax=xn[0], a=xn[1], b=xn[2], bg=xn[3]))["ndvi"][idx][te]
        for nm in SPECS:
            x = fit_chain(nm, tr, seed=int(yr))
            p = pred_chain(nm, x); cvpred[nm][te] = p[te]
            if nm == "M4_dual_understory":
                foldpar[int(yr)] = th_of(nm, x)
                m, sd, _ = gp_residual(tdays[tr], (y - p)[tr], tdays[te])
                cvpred["M5_hybrid_M4_GP"][te] = p[te] + m; cvsd[te] = sd
        mu = y[tr].mean()
        m, sd, _ = gp_residual(tdays[tr], y[tr] - mu, tdays[te]); cvpred["B2_GP_only"][te] = mu + m
        print("fold", yr, "done", flush=True)
    results = {nm: metrics(y, cvpred[nm]) for nm in names}
    # coverage of 95% interval for hybrid
    z = np.abs(cvpred["M5_hybrid_M4_GP"] - y) / cvsd
    results["M5_hybrid_M4_GP"]["cover95"] = float(np.mean(z <= 1.96))
    full = {}
    for nm in SPECS:
        x = fit_chain(nm, np.ones(len(y), bool), seed=42)
        full[nm] = th_of(nm, x); results[nm]["insample"] = metrics(y, pred_chain(nm, x))
    json.dump(dict(cv=results, full=full, foldpar=foldpar), open("results/results.json", "w"), indent=1)
    pd.DataFrame({"date": obs.index, "NDVI_obs": y, **cvpred, "M5_sd": cvsd}).to_csv("results/cv_predictions.csv", index=False)
    print(pd.DataFrame({k: {m: v[m] for m in ["RMSE", "MAE", "R2", "bias"]} for k, v in results.items()}).T.round(4))
    print(json.dumps({k: {a: round(b, 5) for a, b in v.items()} for k, v in full.items()}, indent=1))
