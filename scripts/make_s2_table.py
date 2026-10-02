"""Convert the Google Earth Engine export (one row per Sentinel-2 image, from gee_sentinel2_indices.js)
into data/raw/S2_orchard_indices_2020-2026.csv (one row per acquisition date).

usage: python scripts/make_s2_table.py S2_orchard_indices_2020_2026.csv   (the file exported to Google Drive)

Steps: print the number of acquisitions, keep images with index values over the polygon, average images
acquired on the same date (adjacent tiles or orbits), and label each date by its clear-pixel fraction.
The repository already contains the output used in the article."""
import sys, pandas as pd

g = pd.read_csv(sys.argv[1])
print("Sentinel-2 acquisitions in the export:", len(g))
g["date"] = g["date"].str[:10]
g = g.dropna(subset=["NDVI"])
cols = ["scene_cloud_pct", "n_clear", "NDVI", "EVI", "NDRE", "NDMI", "n_pix"]
s = g.groupby("date", as_index=False)[cols].mean()
print("dates with index values:", len(s), "| of which with two images averaged:", int((g.groupby("date").size() > 1).sum()))
s[["NDVI", "EVI", "NDRE", "NDMI"]] = s[["NDVI", "EVI", "NDRE", "NDMI"]].round(4)
s["n_pix"] = s["n_pix"].round(0); s["n_clear"] = s["n_clear"].round(0)
s["clear_frac"] = (s["n_clear"] / s["n_pix"]).round(3)
s["quality"] = pd.cut(s["clear_frac"], [-1, 0.5, 0.9, 2], right=False, labels=["poor(<50%)", "partial", "good(>=90%)"])
s.to_csv("data/raw/S2_orchard_indices_2020-2026.csv", index=False)
print("written data/raw/S2_orchard_indices_2020-2026.csv")
