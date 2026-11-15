"""Вебінтерфейс: HTTP API і скринька адвоката."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import importlib

    import zoriana_bot.web as web

    importlib.reload(web)
    return TestClient(web.app), tmp_path


def send(client, sid, **event):
    res = client.post("/api/event", json={"session_id": sid, "event": event})
    assert res.status_code == 200, res.text
    return res.json()


def test_page_and_config(client):
    c, _ = client
    assert "Помічник адвоката" in c.get("/").text
    assert c.get("/api/config").json()["idle_seconds"] > 0


def test_full_dialog_reaches_inbox(client):
    c, data = client
    reply = c.post("/api/start").json()
    sid = reply["session_id"]
    assert reply["state"] == "С1" and {b["id"] for b in reply["buttons"]} >= {"services", "urgent"}
    reply = send(c, sid, type="text", text="Здравствуйте, муж подал на развод, нужны алименты на ребёнка")
    assert reply["nlu"]["intent"] == "service"
    if reply["state"] == "С14":
        reply = send(c, sid, type="button", id="yes")
    assert reply["state"] == "С3"
    send(c, sid, type="button", id="apply")
    send(c, sid, type="text", text="Ірина")
    send(c, sid, type="text", text="067 123 45 67")
    reply = send(c, sid, type="button", id="agree")
    assert reply["state"] == "К2" and reply["ended"]
    page = c.get("/inbox").text
    assert "Ірина" in page and "Сімейні справи" in page
    assert (data / "handoffs.jsonl").exists()


def test_errors(client):
    c, _ = client
    assert c.post("/api/event", json={"session_id": "nope", "event": {"type": "start"}}).status_code == 404
    sid = c.post("/api/start").json()["session_id"]
    assert c.post("/api/event", json={"session_id": sid, "event": {"type": "hack"}}).status_code == 400


def test_html_is_escaped_in_inbox(client):
    c, _ = client
    sid = c.post("/api/start").json()["session_id"]
    send(c, sid, type="button", id="apply")
    send(c, sid, type="text", text="<script>alert(1)</script>Ім'я")
    send(c, sid, type="text", text="0671234567")
    send(c, sid, type="button", id="agree")
    page = c.get("/inbox").text
    assert "<script>alert(1)</script>" not in page
