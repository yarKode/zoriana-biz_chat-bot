"""Вебчат: FastAPI-сервер над контролером діалогу.

    .venv/bin/uvicorn zoriana_bot.web:app --port 8000

Змінні середовища: IDLE_SECONDS – пауза до нагадування С17 (типово 600 с, для демо можна 20),
DATA_DIR – тека для handoffs.jsonl і unrecognized.jsonl (типово ./data),
NLU_UNAVAILABLE=1 – замість класифікатора заглушка, яка завжди падає (демо збою сервісу).
"""

from __future__ import annotations

import html
import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .controller import Controller, Inbox, Session
from .nlu.model import IntentClassifier
from .nlu.stub import FailingClassifier
from .tree import name

STATIC = Path(__file__).parent / "static"
IDLE_SECONDS = int(os.environ.get("IDLE_SECONDS", "600"))

app = FastAPI(title="Чат-бот zoriana.biz")
inbox = Inbox(Path(os.environ.get("DATA_DIR", "data")))
nlu = FailingClassifier() if os.environ.get("NLU_UNAVAILABLE") == "1" else IntentClassifier.load()
bot = Controller(nlu, inbox)
sessions: dict[str, Session] = {}


class EventIn(BaseModel):
    session_id: str
    event: dict


def _payload(session: Session, reply) -> dict:
    data = asdict(reply)
    data["session_id"] = session.id
    data["state_name"] = name(reply.state)
    data["buttons"] = [{"id": b, "label": label} for b, label in reply.buttons]
    return data


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config() -> dict:
    return {"idle_seconds": IDLE_SECONDS}


@app.post("/api/start")
def start() -> dict:
    session = Session()
    sessions[session.id] = session
    return _payload(session, bot.handle(session, {"type": "start"}))


@app.post("/api/event")
def event(body: EventIn) -> dict:
    session = sessions.get(body.session_id)
    if session is None:
        raise HTTPException(404, "Сесію не знайдено")
    event = {k: v for k, v in body.event.items() if k in ("type", "id", "text")}
    if event.get("type") not in ("button", "text", "idle"):
        raise HTTPException(400, "Невідомий тип події")
    return _payload(session, bot.handle(session, event))


@app.get("/inbox", response_class=HTMLResponse)
def inbox_page() -> str:
    """Імітація робочого місця адвоката: передані звернення і журнал нерозпізнаних фраз."""
    def cell(value) -> str:
        return html.escape("" if value is None else str(value))

    rows = "".join(
        f"<tr><td>{cell(h['time'].replace('T', ' '))}</td><td>{cell(h['name'])}</td><td>{cell(h['phone'])}</td>"
        f"<td>{cell(h['area'])}</td><td>{cell(h['urgent'] or '')}</td><td>{cell(h['question'] or '')}</td>"
        f"<td class='path'>{cell(' → '.join(h['path']))}</td></tr>"
        for h in reversed(inbox.handoffs)
    ) or "<tr><td colspan='7'>Звернень ще немає</td></tr>"
    unknown = "".join(
        f"<tr><td>{cell(u['time'].replace('T', ' '))}</td><td>{cell(u['state'])}</td><td>{cell(u['text'])}</td></tr>"
        for u in reversed(inbox.unrecognized)
    ) or "<tr><td colspan='3'>Нерозпізнаних фраз немає</td></tr>"
    return (STATIC / "inbox.html").read_text(encoding="utf-8").replace("{{rows}}", rows).replace(
        "{{unknown}}", unknown)
