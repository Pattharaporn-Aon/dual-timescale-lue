import numpy as np, pandas as pd, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import chain
from fit2 import *

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"axes.axisbelow": True, "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})

R2 = json.load(open("results/results_final.json"))
cv = pd.read_csv("results/cv_predictions_final.csv", parse_dates=["date"])
th = R2["full"]["M3_dual_bucket"]
out = simulate(drv, th, **SPECS["M3_dual_bucket"][0])
days = d.index
summary = {}

# ---------- Fig 2: stress scalars ----------
fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
fv = pd.Series(out["fV"], days).rolling(15, center=True).mean()
ft = pd.Series(out["fT"], days).rolling(15, center=True).mean()
ax[0].plot(days, fv, color=BLUE, label=r"$f(\mathrm{VPD})$, ramp")
ax[0].plot(days, ft, color=ORANGE, label=r"$f(T)$")
ax[0].set_ylabel("Scalar (15-d mean)"); ax[0].set_ylim(0, 1.25); ax[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax[0].legend(loc="upper left", ncol=2)
ax[0].set_title("(a) Temperature and vapour-pressure-deficit scalars", loc="left", fontsize=9)
fs_b = pd.Series(out["fS"], days).rolling(15, center=True).mean()
fs_e = pd.Series(f_SM_trapezoid(drv["SM"]), days).rolling(15, center=True).mean()
ax[1].plot(days, fs_b, color=BLUE, label=r"$f(\mathrm{SM})$, water balance ($\hat\phi$=%.2f)" % th["phi"])
ax[1].plot(days, fs_e, color=ORANGE, label=r"$f(\mathrm{SM})$, ERA5-Land soil moisture")
ax[1].set_ylabel("Scalar (15-d mean)"); ax[1].set_ylim(0, 1.32); ax[1].set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax[1].legend(loc="upper left", ncol=2, fontsize=7.5)
ax[1].set_title("(b) Soil-water scalar from two sources", loc="left", fontsize=9)
sm = pd.Series(drv["SM"], days)
ax[2].plot(days, sm, color=INK2, lw=1.0, label="ERA5-Land root-zone soil moisture")
wp, fc, sat = chain.SOIL["wp"], chain.SOIL["fc"], chain.SOIL["sat"]
for v, lab in [(fc - 0.5 * (fc - wp), r"$\theta_c$"), (fc, r"$\theta_{fc}$"), (sat - 0.10, r"$\theta_w$"), (sat, r"$\theta_{sat}$")]:
    ax[2].axhline(v, color=INK, lw=0.7, ls="--"); ax[2].text(days[-1] + pd.Timedelta(days=45), v, lab, va="center", fontsize=8, color=INK)
ax[2].set_ylabel(r"$\theta$ (m$^3$ m$^{-3}$)"); ax[2].legend(loc="lower left")
ax[2].set_title("(c) Reanalysis soil moisture against SoilGrids thresholds", loc="left", fontsize=9)
fig.savefig("figures/fig2_scalars.pdf"); fig.savefig("figures/fig2_scalars.png"); plt.close(fig)
summary["share_ERA5_above_theta_w"] = float((drv["SM"] > sat - 0.10).mean())
summary["share_ERA5_below_theta_c"] = float((drv["SM"] < fc - 0.5 * (fc - wp)).mean())

# ---------- Fig 3: NDVI obs vs out-of-year predictions ----------
fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
for a, series, title in [(ax[0], [("B0", "Harmonic + trend (statistical)", ORANGE), ("M3_dual_bucket", "Dual-timescale chain (this study)", BLUE)],
                          "(a) Leave-one-year-out predictions"),
                         (ax[1], [("B1r", "Annual-crop chain, re-initialised each year", ORANGE), ("M3_dual_bucket", "Dual-timescale chain (this study)", BLUE)],
                          "(b) Annual-crop closure against the dual-timescale chain")]:
    a.scatter(cv.date, cv.NDVI_obs, s=10, color=INK, zorder=3, label="Sentinel-2 NDVI (screened)")
    for col, lab, c in series:
        a.plot(cv.date, cv[col], color=c, marker="o", ms=0, lw=1.4, label=lab)
    a.set_ylabel("NDVI"); a.set_title(title, loc="left", fontsize=9); a.set_ylim(0.38, 0.93); a.legend(loc="upper left", ncol=3, fontsize=7)

fig.savefig("figures/fig3_ndvi_cv.pdf"); fig.savefig("figures/fig3_ndvi_cv.png"); plt.close(fig)

# ---------- Fig 4: latent states ----------
fig, ax = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True)
ax[0].plot(days, out["c"], color=BLUE); ax[0].set_ylabel("Crown cover $c$")
ax[0].set_title("(a) Slow pool: crown cover", loc="left", fontsize=9)
ax[1].plot(days, out["L"], color=AQUA); ax[1].set_ylabel(r"Leaf area $L$ (m$^2$ m$^{-2}$ crown)")
ax[1].set_title("(b) Fast pool: within-crown leaf area", loc="left", fontsize=9)
g = pd.Series(out["gpp_ground"], days).rolling(15, center=True).mean()
ax[2].plot(days, g, color=ORANGE); ax[2].set_ylabel(r"GPP (g C m$^{-2}$ d$^{-1}$)")
ax[2].set_title("(c) Ground-area GPP (15-d mean)", loc="left", fontsize=9)
fig.savefig("figures/fig4_states.pdf"); fig.savefig("figures/fig4_states.png"); plt.close(fig)
summary["c_start"] = float(out["c"][0]); summary["c_end"] = float(out["c"][-1])
summary["L_mean"] = float(out["L"].mean()); summary["L_range"] = [float(out["L"].min()), float(out["L"].max())]
gy = pd.Series(out["gpp_ground"], days); summary["GPP_annual_gC"] = {str(k.year): v for k, v in gy.resample("YE").sum().round(0).items()}
summary["GPP_daily_mean"] = float(gy.mean())

# ---------- Fig 5: canopy memory + structured discrepancy ----------
t0 = (obs.index - obs.index[0]).days.values
A = np.c_[np.ones_like(t0), t0]; res_detr = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
pot = pd.Series(drv["PAR"] * out["fT"] * out["fV"] * out["fS"], days)
taus = np.arange(1, 181, 1); rr = []
for tau in taus:
    e = pot.ewm(alpha=1 / tau).mean().reindex(obs.index).values
    rr.append(np.corrcoef(e, res_detr)[0, 1])
rr = np.array(rr); tau_emp = int(taus[rr.argmax()])
# effective memory from linearisation
kk, eps = chain.K_EXT, chain.EPS
drive = drv["PAR"] * out["stress"]
lam = th["b0"] - th["alpha"] * eps * kk * np.mean(drive * np.exp(-kk * out["L"]))
tau_b = 1 / th["b0"]; tau_eff = 1 / lam
fold_tau = [1 / v["b0"] for v in R2["fold"]["M3_dual_bucket"].values()]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8)); fig.subplots_adjust(wspace=0.38)
ax[0].plot(taus, rr, color=BLUE)
ax[0].axvspan(min(fold_tau), max(fold_tau), color=AQUA, alpha=0.18, lw=0)
ax[0].axvline(tau_eff, color=ORANGE, lw=1.2, ls="--")
ax[0].text(tau_eff + 3, rr.min() + 0.02, r"$\tau_{\mathrm{eff}}$=%.0f d" % tau_eff, color=INK, fontsize=8)
ax[0].text(np.mean(fold_tau), rr.min() + 0.005, r"$1/\beta_0$" + "\n(folds)", color=INK, fontsize=7.5, ha="center")
ax[0].set_xlabel(r"Memory time scale $\tau$ (days)"); ax[0].set_ylabel("Correlation with detrended NDVI")
ax[0].set_title("(a) Memory scan", loc="left", fontsize=9)
mres = (cv.NDVI_obs - cv.M3_dual_bucket).groupby(cv.date.dt.month)
mm, se = mres.mean(), mres.std() / np.sqrt(mres.count())
ax[1].bar(mm.index, mm.values, color=[BLUE if v >= 0 else ORANGE for v in mm.values], width=0.7)
ax[1].errorbar(mm.index, mm.values, yerr=1.96 * se.values, fmt="none", ecolor=INK2, lw=0.8)
ax[1].axhline(0, color=INK2, lw=0.8); ax[1].set_xticks(range(1, 13)); ax[1].set_xticklabels("JFMAMJJASOND")
ax[1].set_ylabel("Observed − chain NDVI"); ax[1].set_title("(b) Seasonal structure of the discrepancy", loc="left", fontsize=9)
fig.savefig("figures/fig5_memory.pdf"); fig.savefig("figures/fig5_memory.png"); plt.close(fig)
summary.update(tau_emp=tau_emp, r_emp=float(rr.max()), tau_b=tau_b, tau_eff=tau_eff, fold_tau=fold_tau,
               monthly_discrepancy=mm.round(3).to_dict())
json.dump(summary, open("results/summary.json", "w"), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
