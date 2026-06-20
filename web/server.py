"""FastAPI backend for the Solar Output web UI.

Thin wrapper around the existing src/solarout package -- every route just
calls into SolarTools / agent.interpret, which already do all the real
work (model inference, climatology lookups, the Claude tool-use agent).
No new business logic lives here.
"""
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.types import Scope


class NoCacheStaticFiles(StaticFiles):
    """Force the browser to revalidate (not blindly reuse a stale cached copy)
    on every request -- this is an actively-edited local demo, not a CDN asset;
    a plain refresh should always pick up the latest app.js/index.html/style.css
    instead of requiring a hard refresh after every change."""

    async def get_response(self, path: str, scope: Scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solarout.agent import display_name, interpret
from solarout.tools import SolarTools

import web.accounts as accounts

app = FastAPI(title="ScorchAI API")
tools = SolarTools()
accounts.init_db()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Keep error responses shaped like {"error": ...} everywhere, matching
    # the plain dicts SolarTools/agent.interpret already return on bad input.
    body = exc.detail if isinstance(exc.detail, dict) else {"error": exc.detail}
    return JSONResponse(status_code=exc.status_code, content=body)

SESSION_COOKIE = "session_token"
SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def get_current_user_id(request: Request) -> int:
    user_id = accounts.get_user_id_for_token(request.cookies.get(SESSION_COOKIE))
    if user_id is None:
        raise HTTPException(status_code=401, detail={"error": "Not logged in"})
    return user_id


class SignupRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class PredictRequest(BaseModel):
    location: str
    month: int
    weather: dict[str, float] | None = None


class FarmRequest(BaseModel):
    location: str
    month: int
    capacity_kw: float | None = None
    area_m2: float | None = None
    weather: dict[str, float] | None = None
    days: int = 1


class ChatRequest(BaseModel):
    question: str
    conversation_id: int | None = None


@app.post("/api/signup")
def signup(req: SignupRequest, response: Response):
    if not req.username.strip() or not req.password:
        raise HTTPException(status_code=400, detail={"error": "Username and password are required"})
    try:
        user_id = accounts.create_user(req.username.strip(), req.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
    token = accounts.create_session(user_id)
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax", max_age=SESSION_MAX_AGE)
    return {"username": accounts.get_username(user_id)}


@app.post("/api/login")
def login(req: LoginRequest, response: Response):
    user_id = accounts.authenticate(req.username.strip(), req.password)
    if user_id is None:
        raise HTTPException(status_code=401, detail={"error": "Invalid username or password"})
    token = accounts.create_session(user_id)
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax", max_age=SESSION_MAX_AGE)
    return {"username": accounts.get_username(user_id)}


@app.post("/api/logout")
def logout(request: Request, response: Response):
    accounts.delete_session(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@app.get("/api/me")
def me(user_id: int = Depends(get_current_user_id)):
    return {"username": accounts.get_username(user_id)}


@app.get("/api/conversations")
def list_conversations(user_id: int = Depends(get_current_user_id)):
    return accounts.list_conversations(user_id)


@app.get("/api/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: int, user_id: int = Depends(get_current_user_id)):
    return accounts.get_conversation_messages(conversation_id, user_id)


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: int, user_id: int = Depends(get_current_user_id)):
    accounts.delete_conversation(conversation_id, user_id)
    return {"ok": True}


@app.get("/api/locations")
def locations(user_id: int = Depends(get_current_user_id)):
    cities = tools.clim.cities
    rows = []
    for _, row in cities.iterrows():
        loc = row["Location"]
        annual = tools.clim.solar_annual.loc[loc]
        rows.append(
            {
                "code": loc,
                "display_name": display_name(loc),
                "lat": row["lat"],
                "lon": row["lon"],
                "pvout_annual_avg_daily": round(float(annual["PVOUT_annual_avg_daily"]), 3),
            }
        )
    return sorted(rows, key=lambda r: r["display_name"])


@app.post("/api/predict")
def predict(req: PredictRequest, user_id: int = Depends(get_current_user_id)):
    return tools.predict_daily_output(req.location, req.month, req.weather)


@app.post("/api/farm")
def farm(req: FarmRequest, user_id: int = Depends(get_current_user_id)):
    return tools.estimate_farm_output(
        req.location, req.month, capacity_kw=req.capacity_kw, area_m2=req.area_m2,
        weather=req.weather, days=req.days,
    )


@app.get("/api/climatology")
def climatology(location: str, month: int | None = None, user_id: int = Depends(get_current_user_id)):
    return tools.get_climatology(location, month)


@app.get("/api/best_month")
def best_month(location: str, user_id: int = Depends(get_current_user_id)):
    return tools.best_month(location)


@app.get("/api/cloud_sensitivity")
def cloud_sensitivity(location: str, month: int, user_id: int = Depends(get_current_user_id)):
    return tools.cloud_sensitivity(location, month)


@app.post("/api/chat")
def chat(req: ChatRequest, user_id: int = Depends(get_current_user_id)):
    conversation_id = req.conversation_id
    if conversation_id is None:
        conversation_id = accounts.create_conversation(user_id, accounts.make_title(req.question))
        history = []
    else:
        history = accounts.get_conversation_messages(conversation_id, user_id)
    accounts.add_message(conversation_id, user_id, "user", req.question)
    answer = interpret(req.question, tools, history)
    accounts.add_message(conversation_id, user_id, "agent", answer)
    return {"answer": answer, "conversation_id": conversation_id}


app.mount("/", NoCacheStaticFiles(directory=ROOT / "web" / "static", html=True), name="static")
