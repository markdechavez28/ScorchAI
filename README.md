# Predicting Solar Output (Australia)

Solves the "Predicting Solar Output" challenge: build a tabular solar-output
dataset from Global Solar Atlas rasters, train a model that predicts daily
solar output from weather regressors (Kaggle "Rain in Australia" /
weatherAUS data), and expose it through a Q&A agent.

## Data sources

| Data | Source | Notes |
|---|---|---|
| Weather | `https://rattle.togaware.com/weatherAUS.csv` | Canonical upstream mirror of the Kaggle "Rain in Australia" dataset (same BoM-derived data). Public, no auth. |
| Solar resource | `https://api.globalsolaratlas.info/download/Australia/Australia_GISdata_LTAym_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF.zip` | Global Solar Atlas long-term-average GeoTIFFs for Australia: GHI, DNI, DIF, GTI, OPTA, TEMP (annual only) and **PVOUT (annual + 12 monthly)**. Public, no auth. |

Both are fetched by `scripts/download_data.py` (no manual download or API key
needed).

## Pipeline

```
scripts/download_data.py   -> data/raw/weatherAUS.csv, data/raw/gsa_australia/*.tif
src/solarout/geocode.py    -> data/processed/city_coordinates.csv      (49 BoM stations -> lat/lon)
src/solarout/raster_extract.py -> data/processed/solar_baseline_by_city_month.csv (rasterio zonal means)
src/solarout/build_dataset.py  -> data/processed/daily_dataset.csv      (year selection + target)
src/solarout/model.py          -> models/solar_output_models.joblib    (train + evaluate)
src/solarout/{climatology,tools,agent}.py -> the Q&A agent
```

Run in order (from the repo root, with the venv active):

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\download_data.py
.venv\Scripts\python src\solarout\geocode.py
.venv\Scripts\python src\solarout\raster_extract.py
.venv\Scripts\python src\solarout\build_dataset.py
.venv\Scripts\python src\solarout\model.py
:: create a .env file in the repo root with ANTHROPIC_API_KEY=sk-ant-... first (see Task 3 below)
.venv\Scripts\python src\solarout\agent.py -q "What is my expected solar output tomorrow in Darwin if it's cloudy and rainy?"
```

Notebooks (`notebooks/01_data_prep.ipynb`, `02_modeling.ipynb`,
`03_agent_questions.ipynb`) walk through the same steps with plots and
narrative, generated/executed via `scripts/build_notebooks.py`.

## Task 1: Tabular solar data via Rasterio

`raster_extract.py` geocodes each of the 49 BoM weather stations in
weatherAUS.csv (OSM Nominatim + manual overrides for ambiguous/abbreviated
names), then for each station:

1. Builds a **10km-radius buffer** around the station's coordinate in a
   metric CRS (EPSG:3577, Australian Albers), reprojected back to the
   raster's CRS (EPSG:4326).
2. Uses `rasterio.mask.mask` to clip the GSA raster to that buffer and takes
   the **mean of valid pixels** -- this is our operationalization of "the
   area covered by the city," since authoritative city/town boundary
   polygons aren't uniformly available for all 49 stations (several are
   airports, military bases, or small towns/islands).
3. Repeats this for the monthly PVOUT rasters (the only parameter GSA
   provides at monthly resolution) plus the annual GHI/DNI/TEMP/PVOUT
   rasters.

One station, **Norfolk Island**, falls entirely outside the GSA "Australia"
raster extent (it's a remote external territory ~1400km east of the
mainland) and is excluded from the solar dataset -- 48 of 49 stations are
supported.

Sanity check (see `notebooks/01_data_prep.ipynb`): tropical Darwin's annual
PVOUT (4.66 kWh/kWp/day) is higher than temperate Hobart's (3.59), as
expected.

## Task 2: Predictive model

### Year used: **2015**

Global Solar Atlas only provides **climatological** monthly/annual averages,
not actual historical daily values, so the year choice doesn't affect the
solar side -- it only affects which year of *weather* data to pair it with.
Year selection is data-driven (`build_dataset.py`): for every year we score
`(locations reporting / 48) x (avg days per location / days in year) x
(fraction of rows with Sunshine or Cloud data)`, restricted to years with
>=80% location and day coverage. 2015 scored highest (48/48 locations, full
365-day coverage, 69.1% of rows with usable Sunshine/Cloud data) -- see
`data/processed/year_selection_report.csv` for the full comparison.

### Target construction (the key methodological choice)

Since there's no historical daily PV output to train against, the target is
a **constructed physical proxy**, built in `src/solarout/target.py`:

```
SolarOutput (kWh/kWp/day) = PVOUT_baseline(city, month) x clear_sky_factor(day)
```

`clear_sky_factor` is, in priority order:
1. `Sunshine_hours / day_length_hours(latitude, day_of_year)` (day length from
   the standard solar-declination formula) when Sunshine was recorded --
   the most direct physical clearness proxy.
2. `1 - 0.7 x (avg(Cloud9am, Cloud3pm) / 8)` when Sunshine is missing but
   cloud oktas are available.

clipped to `[0.05, 1.0]`. Rows with neither Sunshine nor Cloud data are
dropped (no usable target) -- this drops 5,414 of 17,520 candidate rows,
leaving 12,106.

### Two model variants

Both are `HistGradientBoostingRegressor` pipelines (one-hot Location +
numeric weather, native NaN handling), evaluated on a **temporal holdout**
(train Jan-Oct, test Nov-Dec -- mimicking real forecasting of unseen future
dates rather than randomly held-out days):

| Variant | Features | Holdout RMSE | Holdout MAE | Holdout R² |
|---|---|---|---|---|
| **full** | + Sunshine, Cloud9am, Cloud3pm | 0.235 | 0.185 | **0.973** |
| **basic** | Temp/Rain/Humidity/Pressure/Wind/Month/Location only | 1.012 | 0.787 | **0.491** |

The gap is intentional, not a bug: Sunshine/Cloud are the literal physical
drivers used to construct the target (permutation importance for Sunshine
is ~7x every other feature combined in the full model). The **basic** model
represents the realistic accuracy achievable from a generic weather
forecast that doesn't include an irradiance/cloudiness estimate, which is
exactly the accuracy the agent should (and does) disclose when a user's
question doesn't supply one.

**Known limitation**: 10 of the 48 stations (e.g. GoldCoast, Witchcliffe,
Penrith) never record Sunshine or Cloud in the entire dataset history (not
just 2015) -- likely automatic stations without that sensor. `tools.py`
detects this (no irradiance field available from either the caller or
climatology) and always falls back to the **basic** model for them, since
the **full** model was never trained on rows missing all three irradiance
fields simultaneously and would otherwise extrapolate far outside its
training distribution (verified during development: without this guard,
predicted output for these stations could fall *below* the fully-overcast
floor, which is physically impossible).

## Task 3: The Q&A agent

**LLM choice**: the agent (`src/solarout/agent.py`) is a real LLM-powered
tool-calling agent using **Claude Haiku 4.5** (`claude-haiku-4-5`) via the
official `anthropic` Python SDK's `tool_runner`. The six functions in
`src/solarout/tools.py` are exposed to the model as tools (via the
`@beta_tool` decorator, which generates each tool's JSON schema straight from
the function's type hints and docstring); the model decides which tool(s) to
call, translates qualitative weather language ("cloudy and rainy") into the
tools' numeric arguments itself, does unit conversions (MW -> kW, hectares ->
m²) itself, and writes the final natural-language answer from the tool
results. `tools.py` itself is unchanged -- the agent is purely the dispatcher
in front of it. Haiku 4.5 was chosen over a larger model because the tool
surface is small and well-specified (6 functions, clear docstrings), so the
cheapest current Claude model is sufficient; swapping `MODEL` in `agent.py`
to `claude-sonnet-4-6` or `claude-opus-4-8` requires no other code changes.

**Setup**: requires an Anthropic API key. Get one at
[console.anthropic.com](https://console.anthropic.com), then create a file
named `.env` in the repo root (copy `.env.example`) containing:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is in `.gitignore` and is never committed. `agent.py` loads it
automatically (via `python-dotenv`) regardless of your current directory, so
no shell environment variable needs to be set.

Multi-turn context is supported: the CLI keeps the conversation in memory for
the session, and the web app persists it per-conversation in `data/app.db`
and replays it to Claude on every turn, so follow-up questions ("what about
for a 5 MW farm there instead?") work without repeating earlier context.

### Handling missing weather (explicitly stated, as required)

`predict_daily_output` in `tools.py` implements three cases:

1. **No weather given at all** -> fall back to the city's full long-run
   (all-years) historical average for that month (climatology), using the
   **full** model. Every input used is disclosed to the user as assumed.
2. **A real irradiance/cloudiness proxy given** (Sunshine or Cloud
   oktas) -> use the **full** model with the supplied value(s); only the
   *other* gaps (e.g. humidity, pressure) are backfilled from climatology.
3. **Weather given but no irradiance proxy** (e.g. just temperature/rain
   from a generic forecast) -> use the **basic** model, honestly reflecting
   the lower achievable accuracy (R² 0.49 vs 0.97) rather than silently
   borrowing a climatological sunshine value that the user didn't ask for.

This is a blend of the task's suggested strategies #2 (assume typical
conditions) and a third "answer at reduced confidence" mode. The tool
functions return the disclosure fields (`assumed_climatological_inputs`,
`model_variant`, `model_holdout_r2`); the agent's system prompt instructs
Claude to always surface them in plain language -- never a silent guess.

### The 5 required questions

(Full transcripts and accuracy justification in
`notebooks/03_agent_questions.ipynb`.)

1. **"What is my expected solar output tomorrow in Darwin if it's cloudy and rainy?"**
   Uses the **full** model (a real cloud/rain signal was supplied) -- holdout R²=0.97.
2. **"If I set up a solar farm in Alice Springs of size 2 MW, what's my expected daily output?"**
   Same per-kWp prediction as Q1, scaled linearly by capacity -- linear scaling adds no error beyond the per-kWp figure.
3. **"Which month is Hobart likely to have the highest solar output?"**
   Answered directly from the Global Solar Atlas climatology baseline (no regression needed) -- the most reliable possible answer for this question type.
4. **"How much will cloud cover reduce my expected output compared to a clear day in Perth?"**
   Computed analytically from the same clear-sky-factor formula used to build the target (not via model extrapolation, since cloud-only training rows are sparse) -- exact by construction.
5. **"Given typical conditions in Brisbane in October, what's my expected output?"**
   No weather given -> climatology fallback (case 1 above), with a 1-sigma confidence range from the full model's holdout residual std.

### Try it yourself

```
.venv\Scripts\python src\solarout\agent.py
> What is my expected solar output tomorrow in Darwin if it's cloudy and rainy?
> If I set up a solar farm in Alice Springs of size 2 MW, what's my expected daily output?
> list locations
```

or one-shot: `python src\solarout\agent.py -q "<question>"`.

## Web UI

A small FastAPI + vanilla JS/HTML/CSS website wraps every feature above for
live demoing (no Node/npm/build step -- just the existing venv):

```
.venv\Scripts\python scripts\run_web.py
```

Then open **http://127.0.0.1:8000**. The app is branded **SolarMate**. Tabs:
**Chat Agent** (free-text Q&A, with one-click buttons for the 5 required
questions), **Predict Output**, **Farm Sizing** (by capacity or area),
**Best Month** (chart), **Cloud Sensitivity** (chart), and **City Explorer**
(a Leaflet map of all 48 stations colored by annual PV output potential).
`web/server.py` is a thin wrapper -- every route calls directly into
`SolarTools`/`agent.interpret`, with no duplicated logic.

### Accounts & chat history

Every tab requires logging in (sign up with just a username + password --
no email/verification, by design, since this is a local demo). Accounts,
sessions, and per-user chat history are stored in a small SQLite database
at `data/app.db` (`web/accounts.py`), created automatically on first run.
Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib `hashlib`, 200k
iterations, random per-user salt) -- never stored in plaintext. Session
identity is a random token in an HTTP-only cookie; there's no `secure` flag
since the app runs over plain HTTP on `localhost` -- fine for a local demo,
not hardened for a real deployment. On login, your past chat messages
reload into the Chat tab automatically.

## Assumptions summary

- City "area" = 10km-radius buffer around the BoM station coordinate.
- Farm area -> capacity conversion: 150 W/m² site power density (mid-range
  of published 100-200 W/m² figures for utility-scale ground-mount solar).
- Daily output target is a constructed physical proxy (baseline x clear-sky
  factor), not a measured value -- documented and justified above.
- Training year: 2015, chosen by a data-driven completeness score, not by assumption.
- Missing weather: climatology fallback (all-years average) or reduced-confidence
  basic model, always disclosed -- never a silent guess, never blocking the user.
