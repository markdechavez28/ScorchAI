from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
APP_DB_PATH = ROOT / "data" / "app.db"

WEATHER_CSV_URL = "https://rattle.togaware.com/weatherAUS.csv"
GSA_AUSTRALIA_ZIP_URL = (
    "https://api.globalsolaratlas.info/download/Australia/"
    "Australia_GISdata_LTAym_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF.zip"
)

WEATHER_CSV_PATH = DATA_RAW / "weatherAUS.csv"
GSA_ZIP_PATH = DATA_RAW / "Australia_GISdata_LTAym_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF.zip"
GSA_EXTRACT_DIR = DATA_RAW / "gsa_australia"

CITY_COORDINATES_CSV = DATA_PROCESSED / "city_coordinates.csv"
SOLAR_BASELINE_CSV = DATA_PROCESSED / "solar_baseline_by_city_month.csv"
DAILY_DATASET_CSV = DATA_PROCESSED / "daily_dataset.csv"
YEAR_SELECTION_REPORT = DATA_PROCESSED / "year_selection_report.csv"

CITY_BUFFER_RADIUS_M = 10_000  # 10km buffer approximates "area covered by the city"
