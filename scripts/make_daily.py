import pandas as pd, numpy as np
f="data/raw/ERA5_orchard_12.6510N_102.2128E_2020-2026_hourly.csv"
h=pd.read_csv(f,parse_dates=["time_local"]).set_index("time_local")
print(h.shape, h.index.min(), h.index.max(), "missing cells:", int(h.isna().sum().sum()))
day=h["shortwave_radiation"]>0
g=h.resample("D")
d=pd.DataFrame({
 "T_mean_C":g["temperature_2m"].mean(),"T_min_C":g["temperature_2m"].min(),"T_max_C":g["temperature_2m"].max(),
 "Td_mean_C":g["dew_point_2m"].mean(),"RH_mean_pct":g["relative_humidity_2m"].mean(),
 "VPD_mean_kPa":g["vapour_pressure_deficit"].mean(),
 "VPD_daytime_kPa":h.loc[day,"vapour_pressure_deficit"].resample("D").mean(),
 "T_daytime_C":h.loc[day,"temperature_2m"].resample("D").mean(),
 "Rs_MJ_m2_d":g["shortwave_radiation"].sum()*0.0036,
 "PAR_MJ_m2_d":g["PAR_MJ_m2_h"].sum(),
 "P_mm_d":g["precipitation"].sum(),
 "ET0_mm_d":g["et0_fao_evapotranspiration"].sum(),
 "wind_mean_ms":g["wind_speed_10m"].mean(),"u10_mean_ms":g["u10_ms"].mean(),"v10_mean_ms":g["v10_ms"].mean(),
 "Ps_mean_kPa":g["surface_pressure"].mean()/10,"cloud_mean_pct":g["cloud_cover"].mean(),
})
for c in [c for c in h.columns if c.startswith("soil_")]: d[c]=g[c].mean()
# root-zone SM 0-100 cm weighted by layer thickness (7,21,72 cm)
d["SM_rootzone_0_100cm"]=(7*d.soil_moisture_0_to_7cm+21*d.soil_moisture_7_to_28cm+72*d.soil_moisture_28_to_100cm)/100
d.round(4).to_csv("data/processed/ERA5_orchard_2020-2026_daily.csv")
print(d.describe().T[["mean","min","max"]].round(2).to_string())
