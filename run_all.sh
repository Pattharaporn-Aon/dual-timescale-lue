#!/usr/bin/env bash
# Reproduce all results and figures of the paper. Run from the repository root.
# Total run time is roughly 1-2 hours on a laptop (the cross-validation steps dominate).
set -euo pipefail
mkdir -p results figures

echo "[1/8] Daily drivers from hourly ERA5/ERA5-Land"
python scripts/make_daily.py

echo "[2/8] Leave-one-year-out comparison of B0, B1, M3, M3g, M3n, M3s"
python src/fit4.py

echo "[3/8] Final calibration files for the figure scripts"
python src/make_final.py

echo "[4/8] Sensitivity to the fixed values (including NDVI_full)"
python src/sens.py
python src/sens_ndvifull.py

echo "[5/8] Correlated-error observation model, GLS and profile likelihood"
python src/stats.py
python src/stats2.py
python src/stats3.py

echo "[6/8] Figures 2-6 and robustness of the memory scan"
python src/figs.py
python src/fig6.py
python src/scan_robustness.py
python src/ablation_slowpool.py

echo "[7/8] Sensitivity to the temporal NDVI screening rule"
for a in "none 30" "0.06 30" "0.08 30" "0.10 30" "0.08 20" "0.08 45"; do
  set -- $a; python src/sens_filter.py "$1" "$2"
done
echo "[8/8] Check every number in the article against these outputs"
python src/verify_paper.py
echo "Done. See results/ and figures/."
