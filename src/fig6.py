import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import chi2
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
  "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
  "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
S = json.load(open("results/stats_results.json")); P = json.load(open("results/stats_b0profile.json"))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9)); fig.subplots_adjust(wspace=0.35)
mid = [8, 22.5, 45, 90, 180]; r = [a["r"] for a in S["acf"]]
h = np.linspace(0, 240, 200)
ax[0].plot(h, np.exp(-h / S["ell_gls"]), color=BLUE, label=r"OU model, $\ell$=%.0f d" % S["ell_gls"])
ax[0].scatter(mid, r, color=INK, s=18, zorder=3, label="Empirical (binned lags)")
ax[0].axhline(0, color=INK2, lw=0.8); ax[0].set_xlabel("Time lag (days)"); ax[0].set_ylabel("Residual correlation")
ax[0].set_title("(a) Temporal correlation of residuals", loc="left", fontsize=9); ax[0].legend(loc="upper right", fontsize=7.5)
tau = 1 / np.array(P["grid"]); c = chi2.ppf(0.95, 1)
o = np.argsort(tau)
ax[1].plot(tau[o], np.array(P["dev_ou"])[o], color=BLUE, label="Correlated errors (OU)")
ax[1].plot(tau[o], np.array(P["dev_iid"])[o], color=ORANGE, label="Independent errors")
ax[1].axhline(c, color=INK2, lw=0.8, ls="--"); ax[1].text(105, c + 0.4, "95% threshold", fontsize=7.5, color=INK)
ax[1].set_xlim(8, 170); ax[1].set_ylim(0, 14); ax[1].set_xlabel(r"Leaf turnover time $1/\beta_0$ (days)")
ax[1].set_ylabel(r"Profile deviance $\Delta$"); ax[1].set_title("(b) Profile likelihood of canopy memory", loc="left", fontsize=9)
ax[1].legend(loc="upper center", fontsize=7.5)
fig.savefig("figures/fig6_inference.pdf"); fig.savefig("figures/fig6_inference.png")
