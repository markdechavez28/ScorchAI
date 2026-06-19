"""Fetch the two raw data sources for the project and extract the GSA bundle.

- Kaggle "Rain in Australia" weather data, via its canonical upstream mirror
  (the same BoM-derived data Kaggle's weatherAUS.csv is sourced from).
- Global Solar Atlas long-term-average daily solar resource rasters for Australia.

Both are public, unauthenticated downloads.
"""
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solarout.config import (
    DATA_RAW,
    GSA_AUSTRALIA_ZIP_URL,
    GSA_EXTRACT_DIR,
    GSA_ZIP_PATH,
    WEATHER_CSV_PATH,
    WEATHER_CSV_URL,
)


def download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    if dest.exists():
        print(f"already exists, skipping: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:8.1f} MB / {total / 1e6:8.1f} MB ({pct:5.1f}%)", end="")
        print()


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"already extracted, skipping: {dest_dir}")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    download(WEATHER_CSV_URL, WEATHER_CSV_PATH)
    download(GSA_AUSTRALIA_ZIP_URL, GSA_ZIP_PATH)
    extract_zip(GSA_ZIP_PATH, GSA_EXTRACT_DIR)

    print("\nExtracted GSA file listing:")
    for p in sorted(GSA_EXTRACT_DIR.rglob("*")):
        if p.is_file():
            print(" ", p.relative_to(GSA_EXTRACT_DIR))


if __name__ == "__main__":
    main()
