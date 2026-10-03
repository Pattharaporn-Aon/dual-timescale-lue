# Dual-timescale light-use-efficiency model of perennial canopy memory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23112752.svg)](https://doi.org/10.5281/zenodo.23112752)

Code and data for the article

> Thongnim, P. *Profile-likelihood inference for a satellite-constrained dual-timescale light-use-efficiency model of perennial canopy memory.* 

The model converts daily weather into the greenness (NDVI) of an evergreen orchard observed by Sentinel-2:

```
ERA5 / ERA5-Land weather → stress scalars f(T), f(VPD), f(SM) → GPP (Monteith LUE)
   → fast leaf pool L (first-order turnover β0) + slow crown-cover pool c (logistic)
   → NDVI = ρ_bg + (0.90 − ρ_bg) · c · (1 − e^(−kL))
```

It is calibrated against 132 cloud-screened Sentinel-2 scenes of a 7.2-ha durian orchard in Chanthaburi, Thailand (2020–2026), evaluated by leave-one-year-out cross-validation, and embedded in a statistical observation model with temporally correlated (Ornstein–Uhlenbeck) errors, fitted by feasible GLS with profile-likelihood intervals.

## Main results reproduced by this repository

| Quantity | Value |
|---|---|
| Out-of-year R-Squared, dual-timescale chain (M3) | 0.42 |
| Out-of-year R-Squared, annual-crop chain (B1) | 0.20 |
| Out-of-year R-Squared, harmonic baseline (B0) | 0.53 |
| Leaf turnover time 1/β0 | 50.9 d (folds 33–56 d) |
| 95% profile-likelihood interval for the turnover time (correlated errors) | 14–81 d |

## Repository structure

```
data/
  raw/          hourly ERA5/ERA5-Land, Sentinel-2 indices, SoilGrids, orchard polygon (KML)
  processed/    daily model drivers
  README.md     column descriptions, sources and licences
scripts/
  download_era5.py          download hourly weather from the Open-Meteo archive
  gee_sentinel2_indices.js  Google Earth Engine script for the Sentinel-2 indices
  make_s2_table.py          GEE export (one row per image) → one row per date
  make_daily.py             hourly → daily drivers
src/
  chain.py        model: stress scalars, FAO-56 water balance, dual-timescale canopy (numba)
  fit.py, fit2.py, fit3.py   data loading, NDVI screening, calibration utilities
  fit4.py         leave-one-year-out comparison of B0, B1, M3, M3g, M3n, M3s
  fit4lib.py      shared functions for fit4.py and sens_filter.py
  make_final.py   final calibration files for the figure scripts
  sens.py         sensitivity to the fixed literature values
  sens_ndvifull.py  sensitivity to the assumed closed-canopy NDVI
  ablation_slowpool.py  chain without crown expansion (M3c)
  stats.py, stats2.py, stats3.py   correlated-error model, feasible GLS, profile likelihood
  sens_filter.py  sensitivity to the temporal NDVI screening rule
  scan_robustness.py  memory scan without estimated parameters
  figs.py, fig6.py               figures
  verify_paper.py checks every number in the article against the outputs
notebooks/
  dual_timescale_lue_chain.ipynb   self-contained Google Colab version of the whole analysis
results/        output JSON/CSV files (created by run_all.sh)
figures/        output figures (created by run_all.sh)
run_all.sh      reproduces every result and figure
```

## Quick start

```bash
git clone https://github.com/Pattharaporn-Aon/dual-timescale-lue.git
cd dual-timescale-lue
pip install -r requirements.txt
./run_all.sh            # about 1–2 h on a laptop; all scripts are run from the repository root
python src/verify_paper.py   # 194 checks of the article's numbers (also the last step of run_all.sh)
```

Python 3.10 or later is required. The analysis was run with Python 3.11, NumPy 2.4, pandas 3.0, SciPy 1.17, numba 0.67, matplotlib 3.10 and scikit-learn 1.8.

The notebook in `notebooks/` runs the same analysis in Google Colab. It downloads the weather data from Open-Meteo and contains the Sentinel-2 table inline.

## Where each result comes from

| Article element | Script | Output |
|---|---|---|
| Parameter estimates and fold ranges | `src/fit4.py` | `results/results4.json` |
| Leave-one-year-out skill of the six models | `src/fit4.py` | `results/results4.json`, `results/cv_predictions4.csv` |
| Structural comparison of B0, B1 and M3 | `src/fit4.py` | `results/results4.json` |
| Sensitivity to fixed values | `src/sens.py`, `src/sens_ndvifull.py` | `results/sensitivity.json`, `results/sensitivity_ndvifull.json` |
| Chain without crown expansion (M3c, r = 0) | `src/ablation_slowpool.py` | `results/ablation_slowpool.json` |
| GLS estimates, profile-likelihood intervals, identifiability | `src/stats.py`, `src/stats2.py`, `src/stats3.py` | `results/stats_*.json` |
| Sensitivity to the NDVI screening rule | `src/sens_filter.py` | `results/sens_filter_*.json` |
| Stress scalars, NDVI predictions, latent states, canopy memory | `src/figs.py` | `figures/fig2_*.pdf` – `figures/fig5_*.pdf`, `results/summary.json` |
| Residual correlogram and profile likelihood | `src/fig6.py` | `figures/fig6_inference.pdf` |
| Memory scan without the soil-water scalar or with φ fixed | `src/scan_robustness.py` | `results/scan_robustness.json` |
| All other numbers in the text (VPD shares, rainfall, soil texture, departures from B0, B1 bounds) | `src/verify_paper.py` | printed table |

The study-area map is based on Google Earth imagery and is not redistributed here. The orchard boundary is provided in `data/raw/study_orchard.kml` and can be opened in Google Earth.

## Implementation notes

- The leaf pool is floored at 10⁻³ and crown cover is capped at 0.999 for numerical safety. Neither bound is reached in the reported runs (L ≥ 2.0, c ≤ 0.91).
- Calibration uses differential evolution (population 20 × number of parameters, at most 300 generations, `tol=1e-8`) followed by bounded least squares, which is kept only if it lowers the mean squared error. Seeds: the held-out year for each fold, 42 for the full-data fit and 7 for the sensitivity refits.
- The GLS fit and the profiles use local least squares from two warm starts. Profile intervals reported in the article come from `stats2.py` (turnover time) and `stats3.py` (other parameters). The `profiles`, `tau_ci` and `psi_ci` entries in `results/stats_results.json` come from coarser preliminary grids in `stats.py` and are not the reported values.
- The 5th and 95th percentiles used to rescale ERA5-Land soil moisture (variant M3s) are taken over the full 2020–2026 weather series. No NDVI enters the rescaling.
- `src/fit.py` and `src/fit2.py` also contain exploratory variants that are not part of the article (single-pool chains, an understory term and Gaussian-process hybrids). Only their data-loading and helper functions are used by the scripts in `run_all.sh`.

## Data sources

- ERA5 (Hersbach et al., 2020) and ERA5-Land (Muñoz-Sabater et al., 2021), Copernicus Climate Change Service, accessed through the Open-Meteo historical archive.
- Sentinel-2 Level-2A surface reflectance, Copernicus programme, with Cloud Score+ (Pasquarella et al., 2023), processed in Google Earth Engine.
- SoilGrids 2.0 (Poggio et al., 2021), ISRIC – World Soil Information.
- Fixed model parameters from the MOD17 BPLUT (Running & Zhao, 2015) and FAO-56 (Allen et al., 1998).

See `data/README.md` for licences and attribution.

## Citation

Please cite the article and the archived code and data:

> Thongnim, P., & Trithaveesak, O. (2026). *Code and data for: Profile-likelihood inference for a satellite-constrained dual-timescale light-use-efficiency model of perennial canopy memory*. Zenodo. https://doi.org/10.5281/zenodo.23112752

GitHub's "Cite this repository" button (from `CITATION.cff`) gives the same reference in APA and BibTeX.

## Licence

Code: MIT (`LICENSE`). Derived data: CC BY 4.0 (`data/README.md`).

## Contact

Pattharaporn Thongnim, Department of Mathematics, Faculty of Science, Burapha University, Thailand.
ORCID [0000-0001-8904-3979](https://orcid.org/0000-0001-8904-3979)
