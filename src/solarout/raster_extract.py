"""Zonal-mean extraction of Global Solar Atlas rasters per weather station.

For each city we approximate "the area covered by the city" as a 10km-radius
buffer around its station coordinate (CITY_BUFFER_RADIUS_M in config.py) and
take the mean of all valid raster pixels inside that buffer. This is a
simplification in place of true city/town boundary polygons, which are not
uniformly available for all 49 station locations (several are airports,
military bases, or small towns).

Source rasters (Australia_GISdata_LTAy_AvgDailyTotals.../GEOTIFF):
  - PVOUT_01..12.tif  (monthly/, kWh/kWp/day) -- the PV power output potential,
    the only parameter provided at monthly resolution
  - PVOUT.tif, GHI.tif, DNI.tif, TEMP.tif (top-level, annual averages, kWh/m^2/day
    for GHI/DNI, deg C for TEMP)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from shapely.ops import transform as shp_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solarout.config import (
    CITY_BUFFER_RADIUS_M,
    CITY_COORDINATES_CSV,
    GSA_EXTRACT_DIR,
    SOLAR_BASELINE_CSV,
)

GSA_DIR = next(GSA_EXTRACT_DIR.glob("Australia_GISdata_LTAy_AvgDailyTotals*"))
MONTHLY_PVOUT_DIR = GSA_DIR / "monthly"

ANNUAL_RASTERS = {
    "PVOUT_annual_avg_daily": GSA_DIR / "PVOUT.tif",
    "GHI_annual_avg_daily": GSA_DIR / "GHI.tif",
    "DNI_annual_avg_daily": GSA_DIR / "DNI.tif",
    "TEMP_annual_avg": GSA_DIR / "TEMP.tif",
}

# Buffer geometries in a metric Australian Albers CRS, then reproject to
# whatever the raster's CRS is (EPSG:4326 for all GSA layers here).
METRIC_CRS = "EPSG:3577"


def make_buffer_geom(lat: float, lon: float, raster_crs: str, radius_m: float = CITY_BUFFER_RADIUS_M):
    to_metric = pyproj.Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True).transform
    to_raster = pyproj.Transformer.from_crs(METRIC_CRS, raster_crs, always_xy=True).transform

    from shapely.geometry import Point

    point_metric = shp_transform(to_metric, Point(lon, lat))
    buffer_metric = point_metric.buffer(radius_m)
    return shp_transform(to_raster, buffer_metric)


def zonal_mean(raster_path: Path, geom) -> float:
    with rasterio.open(raster_path) as src:
        try:
            out_image, _ = mask(src, [mapping(geom)], crop=True, nodata=np.nan)
        except ValueError:
            # geometry falls entirely outside this raster's extent
            return float("nan")
        data = out_image[0]
        valid = data[~np.isnan(data)]
        if valid.size == 0:
            return float("nan")
        return float(valid.mean())


def main() -> None:
    cities = pd.read_csv(CITY_COORDINATES_CSV)

    with rasterio.open(ANNUAL_RASTERS["PVOUT_annual_avg_daily"]) as src:
        raster_crs = str(src.crs)

    monthly_files = {m: MONTHLY_PVOUT_DIR / f"PVOUT_{m:02d}.tif" for m in range(1, 13)}

    rows = []
    for _, city in cities.iterrows():
        loc, lat, lon = city["Location"], city["lat"], city["lon"]
        if pd.isna(lat) or pd.isna(lon):
            print(f"skipping {loc}: no coordinates")
            continue

        geom = make_buffer_geom(lat, lon, raster_crs)

        annual_values = {name: zonal_mean(path, geom) for name, path in ANNUAL_RASTERS.items()}
        if np.isnan(annual_values["PVOUT_annual_avg_daily"]):
            print(f"{loc}: OUTSIDE the GSA Australia raster extent (e.g. Norfolk Island) -- excluding")
            continue

        for month in range(1, 13):
            pvout_month = zonal_mean(monthly_files[month], geom)
            rows.append(
                {
                    "Location": loc,
                    "Month": month,
                    "PVOUT_avg_daily": pvout_month,
                    **annual_values,
                }
            )
        print(f"{loc}: PVOUT annual={annual_values['PVOUT_annual_avg_daily']:.3f} kWh/kWp/day, "
              f"GHI annual={annual_values['GHI_annual_avg_daily']:.3f} kWh/m2/day")

    df = pd.DataFrame(rows)
    SOLAR_BASELINE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SOLAR_BASELINE_CSV, index=False)
    print(f"\nsaved {len(df)} rows -> {SOLAR_BASELINE_CSV}")


if __name__ == "__main__":
    main()
