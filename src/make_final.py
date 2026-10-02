"""Collect the final M3 calibration and leave-one-year-out predictions for the figure scripts."""
import json, pandas as pd

r = json.load(open("results/results4.json"))
add_cmax = lambda th: {**th, "cmax": 1.0}
final = dict(full={"M3_dual_bucket": add_cmax(r["full"]["M3"])},
             fold={"M3_dual_bucket": {yr: add_cmax(th) for yr, th in r["fold"]["M3"].items()}},
             cv=r["cv"])
json.dump(final, open("results/results_final.json", "w"), indent=1)
pd.read_csv("results/cv_predictions4.csv").rename(columns={"M3": "M3_dual_bucket"}).to_csv(
    "results/cv_predictions_final.csv", index=False)
print("written results/results_final.json and results/cv_predictions_final.csv")
