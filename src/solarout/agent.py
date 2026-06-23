"""LLM-powered Q&A agent over the SolarTools functions.

Uses the Claude API tool-use loop (`anthropic` SDK's `tool_runner`): each
function in `src/solarout/tools.py` is exposed as a tool, the model decides
which to call and with what arguments (including translating qualitative
weather language like "cloudy and rainy" into numeric tool inputs and doing
unit conversions like MW -> kW itself), and the model's final text response
is returned. `tools.py` does not change -- this only replaces the dispatcher
that used to sit in front of it (see README "Task 3" for the history: this
used to be a deterministic regex/keyword dispatcher; it's now a real LLM).
"""
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

import anthropic
from anthropic import beta_tool
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solarout.tools import SolarTools

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 2048

_client: anthropic.Anthropic | None = None


def display_name(location: str) -> str:
    """Render a CamelCase station code (e.g. AliceSprings) as 'Alice Springs'."""
    return re.sub(r"(?<!^)(?=[A-Z][a-z])", " ", location)


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _weather_dict(
    sunshine_hours: float | None,
    cloud_9am_oktas: float | None,
    cloud_3pm_oktas: float | None,
    rainfall_mm: float | None,
    min_temp_c: float | None,
    max_temp_c: float | None,
    humidity_9am_pct: float | None,
    humidity_3pm_pct: float | None,
    pressure_9am_hpa: float | None,
    pressure_3pm_hpa: float | None,
    wind_speed_9am_kmh: float | None,
    wind_speed_3pm_kmh: float | None,
) -> dict:
    mapping = {
        "Sunshine": sunshine_hours,
        "Cloud9am": cloud_9am_oktas,
        "Cloud3pm": cloud_3pm_oktas,
        "Rainfall": rainfall_mm,
        "MinTemp": min_temp_c,
        "MaxTemp": max_temp_c,
        "Humidity9am": humidity_9am_pct,
        "Humidity3pm": humidity_3pm_pct,
        "Pressure9am": pressure_9am_hpa,
        "Pressure3pm": pressure_3pm_hpa,
        "WindSpeed9am": wind_speed_9am_kmh,
        "WindSpeed3pm": wind_speed_3pm_kmh,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _build_tools(tools: SolarTools) -> list:
    """Build the @beta_tool-decorated functions for one request, closing over
    the caller's already-loaded SolarTools instance (avoids reloading the
    model bundle on every chat turn)."""

    @beta_tool
    def predict_daily_output(
        location: str,
        month: int,
        sunshine_hours: float | None = None,
        cloud_9am_oktas: float | None = None,
        cloud_3pm_oktas: float | None = None,
        rainfall_mm: float | None = None,
        min_temp_c: float | None = None,
        max_temp_c: float | None = None,
        humidity_9am_pct: float | None = None,
        humidity_3pm_pct: float | None = None,
        pressure_9am_hpa: float | None = None,
        pressure_3pm_hpa: float | None = None,
        wind_speed_9am_kmh: float | None = None,
        wind_speed_3pm_kmh: float | None = None,
    ) -> str:
        """Predict expected daily solar output (kWh per kWp installed) for a city and month.

        Only pass the weather fields you actually know (from the user's question or a
        forecast); omit the rest and they will be backfilled from that city's historical
        climatology for the month, with every assumed field disclosed in the result's
        `assumed_climatological_inputs`. Supplying any of sunshine_hours/cloud_9am_oktas/
        cloud_3pm_oktas (a real irradiance or cloudiness signal) unlocks the more accurate
        "full" model (holdout R^2 ~0.97); without one of those the "basic" model is used
        (holdout R^2 ~0.49) since sunshine/cloud are the dominant physical drivers.

        Args:
            location: City or town name (one of the supported Australian BoM stations).
            month: Calendar month, 1-12.
            sunshine_hours: Bright sunshine hours for the day (0 to ~13).
            cloud_9am_oktas: Cloud cover at 9am in oktas, 0 (clear) to 8 (overcast).
            cloud_3pm_oktas: Cloud cover at 3pm in oktas, 0 (clear) to 8 (overcast).
            rainfall_mm: Rainfall in mm for the day.
            min_temp_c: Minimum temperature in Celsius.
            max_temp_c: Maximum temperature in Celsius.
            humidity_9am_pct: Relative humidity at 9am, percent.
            humidity_3pm_pct: Relative humidity at 3pm, percent.
            pressure_9am_hpa: Atmospheric pressure at 9am, hPa.
            pressure_3pm_hpa: Atmospheric pressure at 3pm, hPa.
            wind_speed_9am_kmh: Wind speed at 9am, km/h.
            wind_speed_3pm_kmh: Wind speed at 3pm, km/h.
        """
        weather = _weather_dict(
            sunshine_hours, cloud_9am_oktas, cloud_3pm_oktas, rainfall_mm,
            min_temp_c, max_temp_c, humidity_9am_pct, humidity_3pm_pct,
            pressure_9am_hpa, pressure_3pm_hpa, wind_speed_9am_kmh, wind_speed_3pm_kmh,
        )
        return json.dumps(tools.predict_daily_output(location, month, weather or None), default=str)

    @beta_tool
    def estimate_farm_output(
        location: str,
        month: int,
        capacity_kw: float | None = None,
        area_m2: float | None = None,
        days: int = 1,
        sunshine_hours: float | None = None,
        cloud_9am_oktas: float | None = None,
        cloud_3pm_oktas: float | None = None,
        rainfall_mm: float | None = None,
        min_temp_c: float | None = None,
        max_temp_c: float | None = None,
        humidity_9am_pct: float | None = None,
        humidity_3pm_pct: float | None = None,
        pressure_9am_hpa: float | None = None,
        pressure_3pm_hpa: float | None = None,
        wind_speed_9am_kmh: float | None = None,
        wind_speed_3pm_kmh: float | None = None,
    ) -> str:
        """Estimate total energy output (kWh) for a solar farm of a given size in a city/month.

        Provide either capacity_kw (convert MW to kW yourself: 1 MW = 1000 kW) or area_m2
        (convert hectares to m^2 yourself: 1 hectare = 10000 m^2) -- not both. Weather
        fields behave exactly as in predict_daily_output: omit what you don't know and it
        is backfilled from climatology, disclosed in the result.

        Args:
            location: City or town name (one of the supported Australian BoM stations).
            month: Calendar month, 1-12.
            capacity_kw: Installed capacity in kW. Convert MW to kW before calling.
            area_m2: Ground area of the farm in square meters. Convert hectares before calling.
            days: Number of days to total the output over (default 1, i.e. a single day).
            sunshine_hours: Bright sunshine hours for the day (0 to ~13).
            cloud_9am_oktas: Cloud cover at 9am in oktas, 0 (clear) to 8 (overcast).
            cloud_3pm_oktas: Cloud cover at 3pm in oktas, 0 (clear) to 8 (overcast).
            rainfall_mm: Rainfall in mm for the day.
            min_temp_c: Minimum temperature in Celsius.
            max_temp_c: Maximum temperature in Celsius.
            humidity_9am_pct: Relative humidity at 9am, percent.
            humidity_3pm_pct: Relative humidity at 3pm, percent.
            pressure_9am_hpa: Atmospheric pressure at 9am, hPa.
            pressure_3pm_hpa: Atmospheric pressure at 3pm, hPa.
            wind_speed_9am_kmh: Wind speed at 9am, km/h.
            wind_speed_3pm_kmh: Wind speed at 3pm, km/h.
        """
        weather = _weather_dict(
            sunshine_hours, cloud_9am_oktas, cloud_3pm_oktas, rainfall_mm,
            min_temp_c, max_temp_c, humidity_9am_pct, humidity_3pm_pct,
            pressure_9am_hpa, pressure_3pm_hpa, wind_speed_9am_kmh, wind_speed_3pm_kmh,
        )
        return json.dumps(
            tools.estimate_farm_output(
                location, month, capacity_kw=capacity_kw, area_m2=area_m2,
                weather=weather or None, days=days,
            ),
            default=str,
        )

    @beta_tool
    def get_climatology(location: str, month: int | None = None) -> str:
        """Look up historical weather normals and the Global Solar Atlas solar baseline
        for a city, with no model involved -- use this for "what's typical/average"
        questions rather than predict_daily_output.

        Args:
            location: City or town name (one of the supported Australian BoM stations).
            month: Calendar month, 1-12. Omit for annual figures.
        """
        return json.dumps(tools.get_climatology(location, month), default=str)

    @beta_tool
    def best_month(location: str) -> str:
        """Rank all 12 calendar months by climatological solar output potential for a city.
        Use this for "which month is best/highest output" questions.

        Args:
            location: City or town name (one of the supported Australian BoM stations).
        """
        return json.dumps(tools.best_month(location), default=str)

    @beta_tool
    def cloud_sensitivity(location: str, month: int) -> str:
        """Compute how much cloud cover reduces expected output vs. a clear day, for a
        city/month, at oktas 0/2/4/6/8. Use this for "how much does cloud/overcast
        reduce my output" questions -- it's an exact analytical answer, more reliable
        than predict_daily_output for isolating the effect of cloud alone.

        Args:
            location: City or town name (one of the supported Australian BoM stations).
            month: Calendar month, 1-12.
        """
        return json.dumps(tools.cloud_sensitivity(location, month), default=str)

    @beta_tool
    def list_supported_locations() -> str:
        """List every city/town this system has data for. Call this when the user names
        a city that the other tools report as unrecognized, so you can suggest the
        closest valid alternative."""
        return json.dumps(tools.list_supported_locations(), default=str)

    return [
        predict_daily_output,
        estimate_farm_output,
        get_climatology,
        best_month,
        cloud_sensitivity,
        list_supported_locations,
    ]


def _system_prompt() -> str:
    today = dt.date.today()
    return f"""You are a solar-power-output Q&A assistant for an Australian solar yield \
prediction system covering 48 Australian Bureau of Meteorology weather station cities. \
Today's date is {today.isoformat()} ({today.strftime('%B %Y')}) -- use it to resolve \
relative time references like "tomorrow" or "next month" into a calendar month.

You answer by calling the provided tools, which wrap a trained regression model and a \
Global Solar Atlas climatology baseline. Never compute or estimate solar output yourself \
without calling a tool -- the numbers must come from the tools.

When the user describes weather qualitatively, translate it into the tools' numeric \
arguments yourself using your judgment, e.g.: clear/sunny -> cloud oktas ~0-1; partly \
cloudy -> ~4; cloudy/overcast -> ~7-8; rainy -> cloud ~7-8 plus rainfall ~10-20mm; stormy \
-> cloud ~8 plus rainfall ~25mm+; hot -> max_temp_c ~36; cold -> max_temp_c ~13; humid -> \
humidity_3pm_pct ~80. Only pass fields you have actual evidence for from the question; \
leave the rest unset so the tool backfills them from climatology.

For farm-sizing questions, convert units yourself before calling estimate_farm_output \
(MW -> kW, hectares -> m^2).

If a city name isn't recognized (a tool returns an error), call list_supported_locations \
and suggest the closest match rather than just reporting failure.

Never ask the user a clarifying question when you can proceed with a reasonable default -- \
always answer in one turn. If a question doesn't name a month, default to the current \
calendar month (from today's date above) and say you've done so; do the same for any other \
unspecified-but-required tool argument. This matches the system's policy of always giving a \
best-effort, disclosed-assumption answer rather than blocking on missing information.

Every tool result that includes assumed_climatological_inputs, model_variant, or \
model_holdout_r2 is telling you what was assumed and how trustworthy the prediction is -- \
always surface that to the user in plain language (e.g. "using the full model, R^2=0.97" \
or "I assumed typical October humidity and pressure for Brisbane since you didn't specify \
them"), so the answer is never presented as more certain than it is.

Output formatting: your answer is rendered as markdown, so use it deliberately, not \
decoratively -- **bold** for the key number(s), a markdown table only when comparing 3+ \
rows (e.g. a month ranking or cloud-oktas sensitivity), short prose otherwise. Do not use \
emoji. Do not use exclamation points or sales-y/enthusiastic language ("Great question!", \
"I'd be happy to..."). Lead with the concrete number or result as a normal paragraph -- \
that's the part the user actually asked for. Then put every disclosure (model_variant, \
model_holdout_r2, assumed_climatological_inputs, or any other caveat about reliability or \
assumed inputs) in a markdown blockquote (a line starting with "> ") immediately after, so \
it renders visually set apart from the headline answer instead of blending into the same \
paragraph. Never merge the two -- the headline must be able to stand alone as the direct \
answer, with the blockquote purely additive context. Keep both parts tight: a few sentences \
for the headline, a few for the blockquote, plus a table/list only when one is genuinely \
useful."""


def _history_to_messages(history: list[dict] | None) -> list[dict]:
    if not history:
        return []
    role_map = {"user": "user", "agent": "assistant", "assistant": "assistant"}
    return [
        {"role": role_map.get(m["role"], "user"), "content": m["content"]}
        for m in history
        if m.get("content")
    ]


def interpret(question: str, tools: SolarTools, history: list[dict] | None = None) -> str:
    """Answer one question, optionally continuing a prior conversation.

    `history` is a list of {"role": "user"|"agent"|"assistant", "content": str} dicts in
    chronological order (the shape already stored by web/accounts.py chat history),
    excluding the new `question` itself.
    """
    if "ANTHROPIC_API_KEY" not in os.environ and "ANTHROPIC_AUTH_TOKEN" not in os.environ:
        return (
            "ANTHROPIC_API_KEY is not set, so the LLM agent can't run. Set it (e.g. "
            "`set ANTHROPIC_API_KEY=sk-ant-...` on Windows) and try again."
        )

    client = _get_client()
    messages = _history_to_messages(history) + [{"role": "user", "content": question}]

    try:
        runner = client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=_system_prompt(),
            tools=_build_tools(tools),
            messages=messages,
        )
        final_message = None
        for message in runner:
            final_message = message
    except anthropic.APIError as e:
        return f"The LLM agent hit an API error: {e}"

    if final_message is None:
        return "(no response from the model)"
    text = "".join(b.text for b in final_message.content if b.type == "text")
    return text or "(the model returned no text -- it may have only made tool calls)"


def main() -> None:
    import argparse

    # Claude's answers routinely include characters (en-dash, deg/squared signs, minus
    # sign) outside the default Windows console codepage (cp1252) -- without this,
    # printing them raises UnicodeEncodeError and crashes the CLI.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Solar output Q&A agent (Claude tool-use)")
    parser.add_argument("--question", "-q", type=str, help="ask a single question and exit")
    args = parser.parse_args()

    tools = SolarTools()

    if args.question:
        print(interpret(args.question, tools))
        return

    print(f"Solar Output Agent ({MODEL}, Claude tool-use). Type 'quit' to exit.")
    history: list[dict] = []
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit"):
            break
        answer = interpret(question, tools, history)
        print(answer)
        history.append({"role": "user", "content": question})
        history.append({"role": "agent", "content": answer})


if __name__ == "__main__":
    main()
