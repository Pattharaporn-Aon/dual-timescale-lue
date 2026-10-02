# Data

All files describe one durian orchard in Chanthaburi Province, Eastern Thailand
(polygon centroid 12.6510 N, 102.2128 E, area 7.2 ha), for 1 January 2020 to 24 September 2026.

## raw/

| File | Content | Source |
|---|---|---|
| `ERA5_orchard_12.6510N_102.2128E_2020-2026_hourly.csv` | Hourly weather at the orchard centroid, local time (UTC+7). 59,016 rows. Temperature, dew point, humidity, VPD, soil temperature and soil moisture (4 layers) from ERA5-Land. Precipitation, pressure, cloud, radiation, wind and FAO-56 ET0 from ERA5. `PAR_MJ_m2_h` = 0.48 × shortwave × 0.0036. | Open-Meteo historical archive (`scripts/download_era5.py`) |
| `S2_orchard_indices_2020-2026.csv` | Sentinel-2 L2A indices (NDVI, EVI, NDRE, NDMI) averaged over the clear 10-m pixels of the 745 pixels in the polygon, with clear-pixel counts. One row per date with index values (187 rows, 186 of them with at least one clear pixel). Of 517 Sentinel-2 acquisitions over the orchard in 2020–2026, the others were fully masked. On three dates (2022-11-29, 2024-11-18, 2026-01-27) two images were available and were averaged. | Google Earth Engine (`scripts/gee_sentinel2_indices.js`, then `scripts/make_s2_table.py`) |
| `SoilGrids_orchard_12.6510N_102.2128E.csv` | SoilGrids 2.0 properties (water retention, bulk density, texture) by depth, mean and Q05/Q50/Q95. | ISRIC SoilGrids REST API |
| `fSM_parameters_rootzone_0-100cm.csv` | Root-zone (0–100 cm) wilting point, field capacity, saturation and the derived stress thresholds used in the model. | Derived from the SoilGrids file |
| `study_orchard.kml` | Orchard boundary polygon (10 vertices, WGS 84). | Digitised on Google Earth imagery |

## processed/

| File | Content |
|---|---|
| `ERA5_orchard_2020-2026_daily.csv` | Daily drivers produced by `scripts/make_daily.py`: daytime-mean T and VPD, daily PAR, P, ET0, and thickness-weighted root-zone soil moisture (0–100 cm). |

## Column notes for the Sentinel-2 file

- `n_pix`: pixels in the polygon (745). `n_clear`: pixels passing Cloud Score+ (`cs_cdf` ≥ 0.6) and the SCL mask (classes 3, 8, 9, 10, 11 removed).
- `clear_frac` = `n_clear` / `n_pix`. The analysis keeps dates with `clear_frac` ≥ 0.9 (137 dates) and then applies the temporal screening rule in the code (132 dates).

## Licences and attribution

- ERA5 and ERA5-Land: Contains modified Copernicus Climate Change Service information 2020–2026. Neither the European Commission nor ECMWF is responsible for any use of this information. Accessed through Open-Meteo (CC BY 4.0).
- Sentinel-2: Contains modified Copernicus Sentinel data 2020–2026, processed in Google Earth Engine.
- SoilGrids 2.0: ISRIC – World Soil Information, CC BY 4.0.
- The derived data files in this folder are released under CC BY 4.0. Please cite the article and the original data providers.
