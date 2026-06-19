"""Construct the daily solar output regression target.

Global Solar Atlas only gives long-term *climatological* monthly/annual
averages, not actual historical daily output. To get a day-by-day target we
combine that climatological baseline with a daily "clear-sky factor"
estimated from the day's actual weather:

    daily_output (kWh/kWp/day) = PVOUT_baseline(city, month) * clear_sky_factor(day)

clear_sky_factor is primarily the ratio of observed bright-sunshine hours to
the maximum possible daylight hours for that latitude/day-of-year (the most
direct physical proxy for atmospheric clearness). When Sunshine wasn't
recorded, we fall back to cloud-cover oktas (Cloud9am/Cloud3pm).
"""
import numpy as np
import pandas as pd

MIN_CLEAR_SKY_FACTOR = 0.05
MAX_CLEAR_SKY_FACTOR = 1.0
CLOUD_OKTAS_MAX = 8.0
CLOUD_OUTPUT_PENALTY = 0.7  # full (8 okta) cloud cover cuts clear-sky output by ~70%


def day_length_hours(lat_deg: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    """Astronomical day length (sunrise to sunset) in hours."""
    lat = np.radians(lat_deg)
    declination = np.radians(23.45) * np.sin(np.radians(360 / 365 * (day_of_year - 81)))
    cos_hour_angle = -np.tan(lat) * np.tan(declination)
    cos_hour_angle = np.clip(cos_hour_angle, -1.0, 1.0)
    hour_angle_deg = np.degrees(np.arccos(cos_hour_angle))
    return 2 * hour_angle_deg / 15.0


def clear_sky_factor(df: pd.DataFrame, lat_col="lat", date_col="Date") -> pd.Series:
    day_of_year = pd.to_datetime(df[date_col]).dt.dayofyear.to_numpy()
    daylight_hours = day_length_hours(df[lat_col].to_numpy(), day_of_year)

    sunshine_ratio = df["Sunshine"] / daylight_hours
    cloud_avg = df[["Cloud9am", "Cloud3pm"]].mean(axis=1)
    cloud_ratio = 1.0 - CLOUD_OUTPUT_PENALTY * (cloud_avg / CLOUD_OKTAS_MAX)

    factor = sunshine_ratio.where(df["Sunshine"].notna(), cloud_ratio)
    return factor.clip(MIN_CLEAR_SKY_FACTOR, MAX_CLEAR_SKY_FACTOR)


def build_target(df: pd.DataFrame) -> pd.Series:
    """df must have PVOUT_avg_daily (monthly baseline already merged in), lat, Date,
    Sunshine, Cloud9am, Cloud3pm columns."""
    factor = clear_sky_factor(df)
    return df["PVOUT_avg_daily"] * factor
