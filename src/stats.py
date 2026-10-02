"""Statistical layer for the dual-timescale chain (M3):
observation model NDVI_obs = NDVI(theta) + eps, eps Gaussian with exponential (OU) temporal correlation,
residual autocorrelation diagnostics, feasible GLS, profile likelihood CIs and practical identifiability."""
import numpy as np, pandas as pd, json
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import chi2
import fit
from fit import drv, y, idx, obs
from chain import simulate, EPS, K_EXT

t = (obs.index - obs.index[0]).days.values.astype(float)
n = len(y)
NAMES = ["alpha", "b0", "bg", "L0", "phi", "r", "c0"]
LO = np.array([1e-4, 1e-3, 0.0, 0.1, 0.0, 0.0, 0.2]); HI = np.array([0.1, 0.1, 0.6, 8.0, 1.0, 0.005, 1.0])
th0 = json.load(open("results/results4.json"))["full"]["M3"]
x0 = np.array([th0[k] for k in NAMES])
D = np.abs(t[:, None] - t[None, :])


def pred(x):
    th = dict(zip(NAMES, x)); th["cmax"] = 1.0
    return simulate(drv, th, variant="dual", vpd="ramp", sm="bucket")["ndvi"][idx]


def ou_negloglik(ell, r):
    """-2 log L profiled over sigma^2 for eps ~ N(0, sigma^2 R), R_ij = exp(-|ti-tj|/ell)."""
    R = np.exp(-D / ell) if ell > 1e-6 else np.eye(n)
    Lc = np.linalg.cholesky(R + 1e-10 * np.eye(n))
    w = np.linalg.solve(Lc, r)
    return n * np.log(w @ w / n) + 2 * np.sum(np.log(np.diag(Lc)))


def fit_ell(r):
    res = minimize_scalar(lambda le: ou_negloglik(np.exp(le), r), bounds=(np.log(0.5), np.log(365)), method="bounded")
    return float(np.exp(res.x)), float(res.fun)


def whitener(ell):
    R = np.exp(-D / ell); Lc = np.linalg.cholesky(R + 1e-10 * np.eye(n)); return Lc, 2 * np.sum(np.log(np.diag(Lc)))


def gls_fit(xstart, Lc, free=None, fixed_val=None):
    free = np.arange(len(NAMES)) if free is None else free
    def res(z):
        x = xstart.copy(); x[free] = z
        return np.linalg.solve(Lc, pred(x) - y)
    z0 = np.clip(xstart[free], LO[free] + 1e-9, HI[free] - 1e-9)
    r = least_squares(res, z0, bounds=(LO[free], HI[free]), x_scale="jac", max_nfev=2000)
    x = xstart.copy(); x[free] = r.x
    return x, 2 * r.cost


if __name__ == "__main__":
    out = {}
    r0 = y - pred(x0)
    # 1. residual autocorrelation diagnostics
    lagbins = [(1, 15), (15, 30), (30, 60), (60, 120), (120, 240)]
    acf = []
    for a, b in lagbins:
        m = (D > a) & (D <= b)
        ii, jj = np.where(np.triu(m))
        acf.append(dict(lag=f"{a}-{b}", pairs=int(len(ii)), r=float(np.corrcoef(r0[ii], r0[jj])[0, 1])))
    ell_hat, m2ll_ou = fit_ell(r0)
    m2ll_iid = n * np.log(r0 @ r0 / n)
    lr = m2ll_iid - m2ll_ou
    out["acf"] = acf; out["ell_ols"] = ell_hat; out["LR_ou_vs_iid"] = float(lr); out["p_LR"] = float(0.5 * chi2.sf(lr, 1))
    print("residual correlogram", acf); print("OU range (d)", ell_hat, "LR", lr, "p", out["p_LR"])
    # 2. feasible GLS: iterate theta | ell and ell | theta
    x = x0.copy(); ell = ell_hat
    for it in range(3):
        Lc, logdet = whitener(ell)
        x, rss_w = gls_fit(x, Lc)
        ell, _ = fit_ell(y - pred(x))
        print("GLS iter", it, "ell", round(ell, 2), dict(zip(NAMES, np.round(x, 5))))
    Lc, logdet = whitener(ell)
    xg, rss_w = gls_fit(x, Lc)
    sig2 = rss_w / n
    m2ll_best = n * np.log(rss_w / n) + logdet
    out.update(ell_gls=ell, sigma=float(np.sqrt(sig2)), theta_gls=dict(zip(NAMES, xg.tolist())), m2ll=float(m2ll_best))
    rw = np.linalg.solve(Lc, y - pred(xg)) / np.sqrt(sig2)
    out["whitened_resid_lag1_corr"] = float(np.corrcoef(rw[:-1], rw[1:])[0, 1])
    out["n_eff_ratio"] = float((y - pred(xg)) @ (y - pred(xg)) / n / sig2)
    # 3. profile likelihood for each parameter (theta_k fixed on grid, others re-optimised, ell fixed)
    crit = chi2.ppf(0.95, 1)
    grids = {"b0": np.geomspace(0.008, 0.06, 31), "alpha": np.geomspace(0.004, 0.03, 25), "phi": np.linspace(0, 0.6, 25),
             "r": np.linspace(0, 0.0015, 25), "c0": np.linspace(0.3, 1.0, 25), "bg": np.linspace(0, 0.5, 21),
             "L0": np.geomspace(0.3, 8, 21)}
    prof = {}
    for k, grid in grids.items():
        ki = NAMES.index(k); free = np.array([i for i in range(len(NAMES)) if i != ki]); vals = []
        xw = xg.copy()
        # walk outward from the optimum in both directions for warm starts
        order = np.argsort(np.abs(np.log(grid + 1e-9) - np.log(xg[ki] + 1e-9)))
        res_k = {}
        for gi in order:
            xs = xg.copy(); xs[ki] = grid[gi]
            best = None
            for start in [xs, np.r_[xw[:ki], grid[gi], xw[ki + 1:]]]:
                xf, rs = gls_fit(start.copy(), Lc, free)
                if best is None or rs < best[1]: best = (xf, rs)
            res_k[gi] = n * np.log(best[1] / n) + logdet - m2ll_best
            xw = best[0]
        dev = np.array([res_k[i] for i in range(len(grid))])
        dev = dev - min(0, dev.min())
        inside = grid[dev <= crit]
        lo_open = dev[0] <= crit; hi_open = dev[-1] <= crit
        prof[k] = dict(grid=grid.tolist(), dev=dev.tolist(), mle=float(xg[ki]), ci=[float(inside.min()), float(inside.max())],
                       lower_open=bool(lo_open), upper_open=bool(hi_open))
        print(k, "MLE", round(xg[ki], 5), "CI", np.round(prof[k]["ci"], 5), "open", lo_open, hi_open, flush=True)
    out["profiles"] = prof
    b = prof["b0"]; out["tau_mle"] = 1 / b["mle"]; out["tau_ci"] = [1 / b["ci"][1], 1 / b["ci"][0]]
    out["psi_mle"] = prof["alpha"]["mle"] * EPS * K_EXT; out["psi_ci"] = [v * EPS * K_EXT for v in prof["alpha"]["ci"]]
    print("tau", out["tau_mle"], out["tau_ci"], "psi", out["psi_mle"], out["psi_ci"])
    json.dump(out, open("results/stats_results.json", "w"), indent=1)
