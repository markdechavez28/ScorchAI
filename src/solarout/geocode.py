"""Resolve each BoM weather station Location name in weatherAUS.csv to a (lat, lon).

Uses the free OSM Nominatim geocoder (1 req/sec, no API key). A handful of
station names are BoM-specific abbreviations or ambiguous nationally, so we
override those with manually verified coordinates.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solarout.config import CITY_COORDINATES_CSV, WEATHER_CSV_PATH

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "SolarOutput-research-project/1.0 (educational coursework)"

# Manually verified overrides for stations that geocode poorly or ambiguously
# by name alone (military bases, abbreviations, multiple matches nationally).
MANUAL_OVERRIDES = {
    "PearceRAAF": (-31.6676, 116.0153),       # RAAF Base Pearce, WA
    "SydneyAirport": (-33.9461, 151.1772),
    "MelbourneAirport": (-37.6690, 144.8410),
    "PerthAirport": (-31.9385, 115.9672),
    "Tuggeranong": (-35.4194, 149.0890),
    "MountGinini": (-35.5294, 148.7733),       # Mt Ginini, ACT
    "NorfolkIsland": (-29.0408, 167.9547),
    "SalmonGums": (-32.9815, 121.6438),
    "Witchcliffe": (-34.0261, 115.1006),
    "Uluru": (-25.3444, 131.0369),
    "Nhil": (-36.3328, 141.6503),
    "Watsonia": (-37.7080, 145.0830),
    "Dartmoor": (-37.9242, 141.2806),
    "Richmond": (-33.6, 150.7758),             # Richmond NSW (RAAF Base), not Richmond elsewhere
    "BadgerysCreek": (-33.8911, 150.7444),
    "CoffsHarbour": (-30.2963, 153.1135),
    "GoldCoast": (-28.0167, 153.4000),
    "AliceSprings": (-23.6980, 133.8807),
    "NorahHead": (-33.2833, 151.5833),
    "MountGambier": (-37.8284, 140.7833),
    "WaggaWagga": (-35.1082, 147.3598),
}


def geocode_one(location: str) -> tuple[float, float] | None:
    if location in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[location]
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": f"{location}, Australia", "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return None
    return float(results[0]["lat"]), float(results[0]["lon"])


def main() -> None:
    if CITY_COORDINATES_CSV.exists():
        print(f"already exists, skipping: {CITY_COORDINATES_CSV}")
        return

    weather = pd.read_csv(WEATHER_CSV_PATH, usecols=["Location"])
    locations = sorted(weather["Location"].dropna().unique())
    print(f"{len(locations)} unique locations to geocode")

    rows = []
    for i, loc in enumerate(locations, 1):
        try:
            coords = geocode_one(loc)
        except requests.RequestException as exc:
            print(f"  [{i}/{len(locations)}] {loc}: request failed ({exc})")
            coords = None
        if coords is None:
            print(f"  [{i}/{len(locations)}] {loc}: NOT FOUND")
            rows.append({"Location": loc, "lat": None, "lon": None})
        else:
            lat, lon = coords
            source = "override" if loc in MANUAL_OVERRIDES else "nominatim"
            print(f"  [{i}/{len(locations)}] {loc}: ({lat:.4f}, {lon:.4f}) [{source}]")
            rows.append({"Location": loc, "lat": lat, "lon": lon})
        if loc not in MANUAL_OVERRIDES:
            time.sleep(1.0)  # Nominatim usage policy: max 1 req/sec

    df = pd.DataFrame(rows)
    missing = df["lat"].isna().sum()
    if missing:
        print(f"\nWARNING: {missing} locations could not be geocoded, add manual overrides for them.")

    CITY_COORDINATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CITY_COORDINATES_CSV, index=False)
    print(f"\nsaved -> {CITY_COORDINATES_CSV}")


if __name__ == "__main__":
    main()
