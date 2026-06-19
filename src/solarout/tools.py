"""Agent tool functions: the callable building blocks behind the Q&A agent.

Each function here is deliberately self-contained with a clear signature and
docstring so it could be wired up unchanged as a tool/function-calling
definition for a real LLM agent (Claude/OpenAI tool-use) later -- the
deterministic dispatcher in agent.py calls the exact same functions a
function-calling LLM loop would.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solarout.climatology import Climatology
from solarout.config import MODELS_DIR
from solarout.target import CLOUD_OKTAS_MAX, CLOUD_OUTPUT_PENALTY

# Assumed power density for converting a ground area into installed PV
# capacity for utility-scale farms (accounts for row spacing, access roads,
# inverter/substation footprint -- not 100% panel coverage). Typical published
# figures for utility solar farms are 100-200 W/m^2 of site area; we use the
# middle of that range.
PANEL_POWER_DENSITY_W_PER_M2 = 150


class SolarTools:
    def __init__(self):
        bundle = joblib.load(MODELS_DIR / "solar_output_models.joblib")
        self.pipelines = bundle["pipelines"]
        self.feature_sets = bundle["feature_sets"]
        self.residual_std = bundle["residual_std"]
        self.test_metrics = bundle["test_metrics"]
        self.clim = Climatology()

    def _month_features(self, month: int) -> dict:
        return {"month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12)}

    def predict_daily_output(self, location: str, month: int, weather: dict | None = None) -> dict:
        """Predict expected daily solar output (kWh per kWp installed) for a
        city and month. Three cases, reflecting how missing weather inputs
        are handled (see README "Missing weather" section):

        1. weather is None / empty -> no forecast available at all. Fall
           back to the city's full long-run historical average conditions
           for that month (climatology), using the "full" model. Every
           input is disclosed as assumed.
        2. weather includes Sunshine or Cloud9am/Cloud3pm -> the caller has
           a real irradiance/cloudiness proxy. Use the "full" model with the
           supplied values, filling any other gaps from climatology.
        3. weather is given but has no irradiance proxy (e.g. just
           temperature/humidity/rain from a generic forecast) -> use the
           "basic" model, which doesn't need Sunshine/Cloud. This is honest
           about the lower achievable accuracy (holdout R^2 ~0.49 vs ~0.97).
        """
        loc = self.clim.resolve_location(location)
        if loc is None:
            return {"error": f"'{location}' is not a recognized/supported location."}

        normals = self.clim.weather_normal(loc, month)
        weather = weather or {}
        irradiance_fields = ("Sunshine", "Cloud9am", "Cloud3pm")
        has_irradiance_proxy = any(k in weather for k in irradiance_fields)

        if not weather:
            # No forecast at all: fall back to full climatological normals, including irradiance.
            variant = "full"
            backfill = normals
        elif has_irradiance_proxy:
            # A real irradiance/cloudiness proxy was given. Backfill non-irradiance gaps from
            # climatology, but leave any *other* irradiance field genuinely missing (NaN) rather
            # than overwriting it with a stale climatological value that would contradict the
            # caller's input and mute its effect on the prediction.
            variant = "full"
            backfill = {k: v for k, v in normals.items() if k not in irradiance_fields}
        else:
            variant = "basic"
            backfill = normals

        row = {**backfill, **weather}
        if variant == "full" and not any(f in row for f in irradiance_fields):
            # Safety fallback: some stations (e.g. GoldCoast, Witchcliffe) never record
            # Sunshine/Cloud at all, so climatology can't backfill them either. The "full"
            # model was never trained on rows missing all three irradiance fields at once
            # (such rows have no usable target -- see build_dataset.py), so it would be
            # extrapolating far outside its training distribution. Use "basic" instead.
            variant = "basic"
            backfill = normals
            row = {**backfill, **weather}

        assumed_fields = [f for f in self.feature_sets[variant] if f not in weather and f in backfill]
        numeric_features = self.feature_sets[variant]
        row = {**row, **self._month_features(month), "Location": loc}
        X = pd.DataFrame([row]).reindex(columns=["Location"] + numeric_features)
        pred = float(self.pipelines[variant].predict(X)[0])
        std = self.residual_std[variant]

        return {
            "location": loc,
            "month": month,
            "model_variant": variant,
            "predicted_kWh_per_kWp_per_day": round(pred, 3),
            "confidence_range_kWh_per_kWp_per_day": [round(pred - std, 3), round(pred + std, 3)],
            "model_holdout_r2": round(self.test_metrics[variant]["R2"], 3),
            "assumed_climatological_inputs": assumed_fields,
            "weather_used": {k: row[k] for k in numeric_features if k in row},
        }

    def estimate_farm_output(
        self,
        location: str,
        month: int,
        capacity_kw: float | None = None,
        area_m2: float | None = None,
        weather: dict | None = None,
        days: int = 1,
    ) -> dict:
        """Estimate total daily (or multi-day) energy output for a farm of a
        given size. Provide either capacity_kw directly, or area_m2 (converted
        via the assumed PANEL_POWER_DENSITY_W_PER_M2 site power density).
        """
        if capacity_kw is None and area_m2 is None:
            return {"error": "must provide capacity_kw or area_m2"}
        if capacity_kw is None:
            capacity_kw = area_m2 * PANEL_POWER_DENSITY_W_PER_M2 / 1000.0

        result = self.predict_daily_output(location, month, weather)
        if "error" in result:
            return result

        per_day = result["predicted_kWh_per_kWp_per_day"] * capacity_kw
        low, high = result["confidence_range_kWh_per_kWp_per_day"]
        result.update(
            {
                "installed_capacity_kW": round(capacity_kw, 1),
                "estimated_output_kWh": round(per_day * days, 1),
                "estimated_output_range_kWh": [round(low * capacity_kw * days, 1), round(high * capacity_kw * days, 1)],
                "days": days,
            }
        )
        return result

    def get_climatology(self, location: str, month: int | None = None) -> dict:
        """Direct climatological lookup (no model): historical weather
        normals and Global Solar Atlas baseline for a city, optionally for a
        specific month (else annual)."""
        loc = self.clim.resolve_location(location)
        if loc is None:
            return {"error": f"'{location}' is not a recognized/supported location."}
        result = {"location": loc, "solar_baseline": self.clim.solar_baseline_for(loc, month)}
        if month is not None:
            result["weather_normal"] = self.clim.weather_normal(loc, month)
        return result

    def best_month(self, location: str) -> dict:
        """Rank calendar months by climatological solar output potential for a city."""
        loc = self.clim.resolve_location(location)
        if loc is None:
            return {"error": f"'{location}' is not a recognized/supported location."}
        rows = [
            {"month": m, "PVOUT_avg_daily": self.clim.solar_baseline_for(loc, m)["PVOUT_avg_daily"]}
            for m in range(1, 13)
        ]
        rows.sort(key=lambda r: r["PVOUT_avg_daily"], reverse=True)
        return {"location": loc, "ranking": rows}

    def cloud_sensitivity(self, location: str, month: int) -> dict:
        """How much cloud cover reduces expected output vs a clear day for a
        city/month.

        This applies the same clear-sky-factor formula used to construct the
        training target (target.py) directly to the climatological PVOUT
        baseline, rather than asking the regression model to extrapolate
        along a cloud-only axis: rows where Sunshine wasn't recorded (so the
        target was built purely from cloud oktas) are a minority of the
        training data, so the model's marginal response to cloud alone,
        holding Sunshine unknown, is not well resolved. The formula is exact
        by construction and is the more reliable answer to this question.
        """
        loc = self.clim.resolve_location(location)
        if loc is None:
            return {"error": f"'{location}' is not a recognized/supported location."}
        baseline = self.clim.solar_baseline_for(loc, month)["PVOUT_avg_daily"]
        outputs = {
            okta: round(baseline * (1 - CLOUD_OUTPUT_PENALTY * (okta / CLOUD_OKTAS_MAX)), 3)
            for okta in (0, 2, 4, 6, 8)
        }
        clear = outputs[0]
        return {
            "location": loc,
            "month": month,
            "clear_sky_baseline_kWh_per_kWp_per_day": baseline,
            "output_by_cloud_oktas": outputs,
            "pct_reduction_vs_clear": {
                okta: round(100 * (1 - v / clear), 1) for okta, v in outputs.items()
            },
        }

    def list_supported_locations(self) -> list[str]:
        return self.clim.list_locations()
