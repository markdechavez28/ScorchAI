"""Pick the training year, merge weather + solar baseline, engineer features,
construct the target, and save the modeling table.

Year selection is data-driven: for every calendar year present in
weatherAUS.csv we score how many of the 48 supported locations report data,
how many days per location on average (full-year coverage), and what
fraction of rows have the Sunshine/Cloud fields needed to build the target.
We pick the best-scoring year and record the comparison table for the
writeup.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solarout.config import (
    CITY_COORDINATES_CSV,
    DAILY_DATASET_CSV,
    SOLAR_BASELINE_CSV,
    WEATHER_CSV_PATH,
    YEAR_SELECTION_REPORT,
)
from solarout.target import build_target

MIN_LOCATIONS_FRACTION = 0.8  # require at least 80% of supported locations reporting
MIN_AVG_DAYS_FRACTION = 0.8   # require at least 80% of days in the year present, on average


def score_years(weather: pd.DataFrame, n_supported_locations: int) -> pd.DataFrame:
    weather = weather.copy()
    weather["Year"] = pd.to_datetime(weather["Date"]).dt.year
    weather["has_sun_or_cloud"] = (
        weather["Sunshine"].notna() | weather["Cloud9am"].notna() | weather["Cloud3pm"].notna()
    )

    rows = []
    for year, grp in weather.groupby("Year"):
        n_locations = grp["Location"].nunique()
        n_rows = len(grp)
        days_in_year = 366 if pd.Timestamp(year, 12, 31).is_leap_year else 365
        avg_days_per_location = n_rows / n_locations if n_locations else 0
        frac_usable = grp["has_sun_or_cloud"].mean()
        location_fraction = n_locations / n_supported_locations
        days_fraction = min(avg_days_per_location / days_in_year, 1.0)

        eligible = (
            location_fraction >= MIN_LOCATIONS_FRACTION
            and days_fraction >= MIN_AVG_DAYS_FRACTION
        )
        score = location_fraction * days_fraction * frac_usable if eligible else 0.0

        rows.append(
            {
                "Year": year,
                "n_locations": n_locations,
                "n_rows": n_rows,
                "avg_days_per_location": round(avg_days_per_location, 1),
                "frac_usable_sun_or_cloud": round(frac_usable, 3),
                "eligible": eligible,
                "score": round(score, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    return df


def main() -> None:
    weather = pd.read_csv(WEATHER_CSV_PATH)
    cities = pd.read_csv(CITY_COORDINATES_CSV).dropna(subset=["lat", "lon"])
    baseline = pd.read_csv(SOLAR_BASELINE_CSV)

    supported_locations = set(baseline["Location"].unique())
    n_supported = len(supported_locations)
    weather = weather[weather["Location"].isin(supported_locations)]

    report = score_years(weather, n_supported)
    YEAR_SELECTION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(YEAR_SELECTION_REPORT, index=False)
    print(report.head(10).to_string(index=False))

    best = report.iloc[0]
    if best["score"] == 0:
        raise RuntimeError("no eligible year found -- relax MIN_LOCATIONS_FRACTION/MIN_AVG_DAYS_FRACTION")
    year = int(best["Year"])
    print(f"\nSelected YEAR = {year} (score={best['score']}, "
          f"{int(best['n_locations'])}/{n_supported} locations, "
          f"{best['avg_days_per_location']} avg days/location, "
          f"{best['frac_usable_sun_or_cloud']*100:.1f}% rows with Sunshine/Cloud)")

    weather_year = weather[pd.to_datetime(weather["Date"]).dt.year == year].copy()
    weather_year = weather_year.merge(cities[["Location", "lat", "lon"]], on="Location", how="left")
    weather_year = engineer_features(weather_year)
    weather_year = weather_year.merge(baseline, on=["Location", "Month"], how="left")

    weather_year["SolarOutput_kWh_per_kWp"] = build_target(weather_year)

    before = len(weather_year)
    weather_year = weather_year.dropna(subset=["SolarOutput_kWh_per_kWp", "lat", "lon"])
    print(f"\ndropped {before - len(weather_year)} rows with no usable target/location "
          f"({len(weather_year)} rows remain)")

    weather_year = weather_year.drop(columns=["RISK_MM"], errors="ignore")

    DAILY_DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    weather_year.to_csv(DAILY_DATASET_CSV, index=False)
    print(f"saved -> {DAILY_DATASET_CSV}")
    print(weather_year[["Date", "Location", "PVOUT_avg_daily", "SolarOutput_kWh_per_kWp"]].head())


if __name__ == "__main__":
    main()
