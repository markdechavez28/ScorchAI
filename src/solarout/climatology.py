"""Historical weather/solar normals and location lookup helpers used by the agent.

This is how we handle "missing requirements" (task 3): when a question
doesn't specify expected weather, we fall back to the long-run (all-years)
historical average conditions for that city and month rather than asking the
user or guessing -- a defensible, statistically grounded default that's
explicitly disclosed to the user in the agent's answer.
"""
import difflib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solarout.config import CITY_COORDINATES_CSV, SOLAR_BASELINE_CSV, WEATHER_CSV_PATH

WEATHER_NORMAL_COLUMNS = [
    "MinTemp", "MaxTemp", "Rainfall", "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm", "WindSpeed9am", "WindSpeed3pm",
    "Sunshine", "Cloud9am", "Cloud3pm",
]


class Climatology:
    def __init__(self):
        baseline = pd.read_csv(SOLAR_BASELINE_CSV)
        self.solar_baseline = baseline.set_index(["Location", "Month"])
        self.solar_annual = baseline.drop_duplicates("Location").set_index("Location")[
            [c for c in baseline.columns if c.endswith("_annual_avg_daily") or c == "TEMP_annual_avg"]
        ]

        self.cities = pd.read_csv(CITY_COORDINATES_CSV).dropna(subset=["lat", "lon"])
        self.cities = self.cities[self.cities["Location"].isin(baseline["Location"].unique())]
        self.locations = sorted(self.cities["Location"].unique())

        weather = pd.read_csv(WEATHER_CSV_PATH, usecols=["Date", "Location"] + WEATHER_NORMAL_COLUMNS)
        weather = weather[weather["Location"].isin(self.locations)]
        weather["Month"] = pd.to_datetime(weather["Date"]).dt.month
        self.weather_normals = (
            weather.groupby(["Location", "Month"])[WEATHER_NORMAL_COLUMNS].mean()
        )

    def resolve_exact(self, name: str) -> str | None:
        """Exact (case/space-insensitive) match only -- safe to run on arbitrary
        multi-word phrases without risking spurious fuzzy matches."""
        normalized = name.replace(" ", "").lower()
        for loc in self.locations:
            if loc.lower() == normalized:
                return loc
        return None

    def resolve_fuzzy(self, word: str, cutoff: float = 0.8) -> str | None:
        """Fuzzy-match a single word (e.g. a likely-misspelled city name) to a
        supported station name. Only intended for single tokens -- matching
        arbitrary multi-word phrases produces false positives."""
        normalized = word.replace(" ", "").lower()
        matches = difflib.get_close_matches(normalized, [l.lower() for l in self.locations], n=1, cutoff=cutoff)
        if matches:
            idx = [l.lower() for l in self.locations].index(matches[0])
            return self.locations[idx]
        return None

    def resolve_location(self, name: str) -> str | None:
        """Resolve a single free-text candidate: exact match, else fuzzy."""
        return self.resolve_exact(name) or self.resolve_fuzzy(name)

    def nearest_location(self, lat: float, lon: float) -> tuple[str, float]:
        """Haversine nearest supported station to an arbitrary coordinate."""
        lat1, lon1 = np.radians(lat), np.radians(lon)
        lat2 = np.radians(self.cities["lat"].to_numpy())
        lon2 = np.radians(self.cities["lon"].to_numpy())
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        dist_km = 2 * 6371 * np.arcsin(np.sqrt(a))
        idx = int(np.argmin(dist_km))
        return self.cities["Location"].iloc[idx], float(dist_km[idx])

    def weather_normal(self, location: str, month: int) -> dict:
        try:
            row = self.weather_normals.loc[(location, month)]
        except KeyError:
            return {}
        return row.dropna().to_dict()

    def solar_baseline_for(self, location: str, month: int | None = None) -> dict:
        if month is None:
            return self.solar_annual.loc[location].to_dict()
        return self.solar_baseline.loc[(location, month)].to_dict()

    def list_locations(self) -> list[str]:
        return self.locations
