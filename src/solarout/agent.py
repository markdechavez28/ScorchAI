"""Deterministic question-answering agent over the SolarTools functions.

No external LLM call is used (by design choice -- see README "LLM agent"
section): this is a small rule-based natural-language-understanding layer
(regex/keyword entity extraction + intent classification) that dispatches to
the same tool functions a function-calling LLM loop would call. Swapping in
a real Claude/OpenAI tool-use loop later just means replacing `interpret()`
below with an LLM call that picks the same tools -- `tools.py` doesn't change.
"""
import argparse
import calendar
import datetime as dt
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solarout.tools import SolarTools

MONTH_NAMES = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTH_ABBR = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
# Southern-hemisphere season -> a representative month
SEASON_MONTHS = {"summer": 1, "autumn": 4, "fall": 4, "winter": 7, "spring": 10}

QUALITATIVE_WEATHER = {
    "sunny": {"Cloud9am": 1, "Cloud3pm": 1},
    "clear": {"Cloud9am": 0, "Cloud3pm": 0},
    "fine": {"Cloud9am": 1, "Cloud3pm": 1},
    "partly cloudy": {"Cloud9am": 4, "Cloud3pm": 4},
    "overcast": {"Cloud9am": 8, "Cloud3pm": 8},
    "cloudy": {"Cloud9am": 7, "Cloud3pm": 7},
    "rainy": {"Cloud9am": 7, "Cloud3pm": 7, "Rainfall": 12},
    "raining": {"Cloud9am": 7, "Cloud3pm": 7, "Rainfall": 12},
    "stormy": {"Cloud9am": 8, "Cloud3pm": 8, "Rainfall": 25},
    "hot": {"MaxTemp": 36},
    "cold": {"MaxTemp": 13},
    "humid": {"Humidity3pm": 80},
}

UNIT_TO_KW = {"kw": 1, "kwp": 1, "mw": 1000, "mwp": 1000}
UNIT_TO_M2 = {"m2": 1, "m^2": 1, "sqm": 1, "sq.m": 1, "hectare": 10_000, "hectares": 10_000, "ha": 10_000}


def display_name(location: str) -> str:
    """Render a CamelCase station code (e.g. AliceSprings) as 'Alice Springs'."""
    return re.sub(r"(?<!^)(?=[A-Z][a-z])", " ", location)


def parse_location(text: str, tools: SolarTools) -> str | None:
    words = text.split()
    # Exact matches first (safe across arbitrary multi-word windows).
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            candidate = " ".join(words[i : i + n]).strip(",.?!")
            loc = tools.clim.resolve_exact(candidate)
            if loc:
                return loc
    # Only fall back to fuzzy matching on individual words (likely typos),
    # with a strict cutoff -- fuzzy-matching whole phrase fragments against
    # city names produces false positives (e.g. "which month is" ~ "Richmond").
    for word in words:
        loc = tools.clim.resolve_fuzzy(word.strip(",.?!"), cutoff=0.8)
        if loc:
            return loc
    return None


def parse_month(text: str, reference_date: dt.date | None = None) -> int:
    text_l = text.lower()
    reference_date = reference_date or dt.date.today()

    if "tomorrow" in text_l:
        return (reference_date + dt.timedelta(days=1)).month
    if "today" in text_l:
        return reference_date.month
    if "next month" in text_l:
        return (reference_date.month % 12) + 1

    for name, num in MONTH_NAMES.items():
        if name in text_l:
            return num
    for abbr, num in MONTH_ABBR.items():
        if re.search(rf"\b{abbr}\b", text_l):
            return num
    for season, num in SEASON_MONTHS.items():
        if season in text_l:
            return num
    return reference_date.month  # default: current month


def parse_capacity_kw(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kwp|kw|mwp|mw)\b", text, re.IGNORECASE)
    if m:
        value, unit = float(m.group(1)), m.group(2).lower()
        return value * UNIT_TO_KW[unit]
    return None


def parse_area_m2(text: str) -> float | None:
    m = re.search(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(m2|m\^2|sq\.?\s?m|hectares?|ha)\b", text, re.IGNORECASE)
    if m:
        value = float(m.group(1).replace(",", ""))
        unit = m.group(2).lower().replace(" ", "").rstrip(".")
        unit = "hectare" if unit.startswith("hectare") else unit
        return value * UNIT_TO_M2.get(unit, 1)
    return None


def parse_weather(text: str) -> dict:
    weather = {}
    text_l = text.lower()
    for phrase, fields in QUALITATIVE_WEATHER.items():
        if phrase in text_l:
            weather.update(fields)

    patterns = [
        (r"(\d+(?:\.\d+)?)\s*mm\b", "Rainfall"),
        (r"(\d+(?:\.\d+)?)\s*(?:°|deg(?:rees)?)\s*c?\b", "MaxTemp"),
        (r"(\d+(?:\.\d+)?)\s*%\s*humidity", "Humidity3pm"),
        (r"(\d+(?:\.\d+)?)\s*oktas?\s*(?:of\s*)?cloud", "Cloud3pm"),
    ]
    for pattern, field in patterns:
        m = re.search(pattern, text_l)
        if m:
            weather[field] = float(m.group(1))
    if "Cloud3pm" in weather and "Cloud9am" not in weather:
        weather["Cloud9am"] = weather["Cloud3pm"]
    return weather


def interpret(question: str, tools: SolarTools) -> str:
    text_l = question.lower()
    loc = parse_location(question, tools)

    if re.search(r"\b(list|which|what)\s+(cities|towns|locations)\b", text_l):
        return "Supported locations: " + ", ".join(display_name(l) for l in tools.list_supported_locations())

    if loc is None:
        return ("I couldn't identify a supported city/town in that question. "
                f"Supported locations: {', '.join(display_name(l) for l in tools.list_supported_locations())}")
    loc_disp = display_name(loc)

    if re.search(r"best month|which month|highest output|peak (output|season)", text_l):
        r = tools.best_month(loc)
        top = r["ranking"][0]
        month_name = calendar.month_name[top["month"]]
        return (f"For {loc_disp}, the best month is historically {month_name}, with an average "
                f"baseline output of {top['PVOUT_avg_daily']:.2f} kWh/kWp/day "
                f"(Global Solar Atlas long-term climatology). Full ranking: " +
                ", ".join(f"{calendar.month_name[row['month']]}={row['PVOUT_avg_daily']:.2f}"
                          for row in r["ranking"]))

    if re.search(r"cloud|overcast", text_l) and re.search(r"reduce|compared|versus|vs\.?|impact|effect", text_l):
        month = parse_month(question)
        r = tools.cloud_sensitivity(loc, month)
        return (f"For {loc_disp} in {calendar.month_name[month]}, expected output falls from "
                f"{r['output_by_cloud_oktas'][0]:.2f} kWh/kWp/day on a clear day (0 oktas) to "
                f"{r['output_by_cloud_oktas'][8]:.2f} kWh/kWp/day fully overcast (8 oktas) -- "
                f"a {r['pct_reduction_vs_clear'][8]:.0f}% reduction. "
                f"At moderate cloud (4 oktas) the reduction is {r['pct_reduction_vs_clear'][4]:.0f}%. "
                "(Derived directly from the clear-sky-factor formula used to build the training target.)")

    capacity_kw = parse_capacity_kw(question)
    area_m2 = parse_area_m2(question)
    if capacity_kw is not None or area_m2 is not None:
        month = parse_month(question)
        weather = parse_weather(question)
        r = tools.estimate_farm_output(loc, month, capacity_kw=capacity_kw, area_m2=area_m2, weather=weather)
        if "error" in r:
            return r["error"]
        disclosure = (f" (assumed {', '.join(r['assumed_climatological_inputs'])} from {loc_disp}'s "
                       f"historical average for {calendar.month_name[month]}, since you didn't specify them)"
                       if r["assumed_climatological_inputs"] else "")
        return (f"A {r['installed_capacity_kW']:.0f} kW farm in {loc_disp} in {calendar.month_name[month]} "
                f"is expected to produce about {r['estimated_output_kWh']:.0f} kWh/day "
                f"(range {r['estimated_output_range_kWh'][0]:.0f}-{r['estimated_output_range_kWh'][1]:.0f} kWh), "
                f"using the '{r['model_variant']}' model (holdout R^2={r['model_holdout_r2']:.2f})." + disclosure)

    month = parse_month(question)
    weather = parse_weather(question)
    r = tools.predict_daily_output(loc, month, weather)
    if "error" in r:
        return r["error"]
    disclosure = (f" I assumed {', '.join(r['assumed_climatological_inputs'])} from {loc_disp}'s historical "
                   f"average for {calendar.month_name[month]}, since you didn't specify them."
                   if r["assumed_climatological_inputs"] else "")
    return (f"Expected solar output for {loc_disp} in {calendar.month_name[month]}: "
            f"{r['predicted_kWh_per_kWp_per_day']:.2f} kWh/kWp/day "
            f"(range {r['confidence_range_kWh_per_kWp_per_day'][0]:.2f}-"
            f"{r['confidence_range_kWh_per_kWp_per_day'][1]:.2f}), "
            f"using the '{r['model_variant']}' model (holdout R^2={r['model_holdout_r2']:.2f})." + disclosure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solar output Q&A agent")
    parser.add_argument("--question", "-q", type=str, help="ask a single question and exit")
    args = parser.parse_args()

    tools = SolarTools()

    if args.question:
        print(interpret(args.question, tools))
        return

    print("Solar Output Agent (deterministic tool-calling, no external LLM). Type 'quit' to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit"):
            break
        print(interpret(question, tools))


if __name__ == "__main__":
    main()
