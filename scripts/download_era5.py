"""Download hourly ERA5-Land and ERA5 data for the orchard centroid from the Open-Meteo historical archive
and write data/raw/ERA5_orchard_12.6510N_102.2128E_2020-2026_hourly.csv (local time, UTC+7).
Land-surface and near-surface variables come from ERA5-Land, and the remaining variables from ERA5."""
import time, requests, numpy as np, pandas as pd

LAT, LON = 12.651007, 102.212826
START, END = "2020-01-01", "2026-09-24"
BASE = "https://archive-api.open-meteo.com/v1/archive"
V_LAND = ("temperature_2m,dew_point_2m,relative_humidity_2m,vapour_pressure_deficit,"
          "soil_temperature_0_to_7cm,soil_temperature_7_to_28cm,soil_temperature_28_to_100cm,soil_temperature_100_to_255cm,"
          "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,soil_moisture_28_to_100cm,soil_moisture_100_to_255cm")
V_ERA5 = ("precipitation,surface_pressure,cloud_cover,shortwave_radiation,direct_radiation,diffuse_radiation,"
          "wind_speed_10m,wind_direction_10m,et0_fao_evapotranspiration")


def get(hourly, model):
    q = dict(latitude=LAT, longitude=LON, start_date=START, end_date=END, hourly=hourly, models=model,
             timezone="Asia/Bangkok", wind_speed_unit="ms")
    for _ in range(5):
        r = requests.get(BASE, params=q, timeout=180)
        if r.status_code == 200:
            return r.json()
        print("retry", r.status_code, r.text[:120]); time.sleep(65)
    raise RuntimeError("Open-Meteo request failed")


land = get(V_LAND, "era5_land"); time.sleep(5); era5 = get(V_ERA5, "era5")
h = pd.DataFrame(land["hourly"]).merge(pd.DataFrame(era5["hourly"]), on="time").rename(columns={"time": "time_local"})
rad = np.deg2rad(h["wind_direction_10m"])
h["u10_ms"] = (-h["wind_speed_10m"] * np.sin(rad)).round(3)
h["v10_ms"] = (-h["wind_speed_10m"] * np.cos(rad)).round(3)
h["PAR_MJ_m2_h"] = (h["shortwave_radiation"] * 0.0036 * 0.48).round(4)   # W m-2 -> MJ m-2 h-1, PAR fraction 0.48
out = "data/raw/ERA5_orchard_12.6510N_102.2128E_2020-2026_hourly.csv"
h.to_csv(out, index=False)
print("ERA5-Land grid", land["latitude"], land["longitude"], "| ERA5 grid", era5["latitude"], era5["longitude"],
      "| hours", len(h), "->", out)
