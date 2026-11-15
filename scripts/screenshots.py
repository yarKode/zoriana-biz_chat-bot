"""Скріншоти діалогів у вебчаті: запускає сервер, проходить сценарії в Chrome (Playwright), знімає чат.

    .venv/bin/python scripts/screenshots.py

Потрібен встановлений Google Chrome (Playwright запускає його через channel="chrome").
Результат – artifacts/screenshots/*.png.
"""

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "screenshots"

# Чат на всю висоту розмови, без декоративної колонки сайту: так у кадр потрапляє весь діалог.
FULL_HEIGHT = ".site{display:none} .page{justify-content:center} .chat{height:auto} .log{overflow:visible}"

S3 = "Здравствуйте, муж подал на развод, нужны алименты на ребёнка"

# (файл, демо-режим, кроки). Крок: ("b", підпис кнопки) | ("t", текст) | ("idle", секунди).
SCENARIOS = [
    ("01_s1_main", False, [("b", "Послуги"), ("b", "Господарські спори"), ("b", "Вартість"), ("b", "Ні"),
                           ("t", "Андрій"), ("t", "067 123 45 67"), ("b", "Погоджуюсь")]),
    ("02_s2_urgent", False, [("b", "Терміново"), ("b", "ДТП або позбавлення права керування"),
                             ("t", "098 111 22 33"), ("b", "Погоджуюсь")]),
    ("03_s3_hybrid", True, [("t", S3), ("b", "Онлайн-консультація"), ("b", "Залишити заявку"), ("t", "Ірина"),
                            ("t", "+380671234567"), ("b", "Погоджуюсь")]),
    ("04_medium_confirm", True, [("t", "потрібна порада щодо розлучення"), ("b", "Так"), ("b", "Вартість"),
                                 ("b", "Так, допомогло")]),
    ("05_s4_unknown", True, [("t", "Потрібна допомога з документами"), ("t", "не знаю, як сформулювати"),
                             ("t", "050 123 45 67"), ("b", "Погоджуюсь")]),
    ("06_s5_phone", False, [("b", "Зв’язатися з адвокатом"), ("t", "067 12"), ("t", "не хочу давати номер"),
                            ("b", "Ввести номер"), ("t", "0671234567"), ("b", "Погоджуюсь")]),
    ("07_s6_back_menu", False, [("b", "Послуги"), ("b", "Цивільні справи"), ("b", "Назад"),
                                ("b", "Сімейні справи"), ("b", "Залишити заявку"), ("t", "Ірина"),
                                ("b", "Головне меню"), ("b", "Головне меню")]),
    ("08_s7_no_answer", True, [("t", "Чи ведете ви справи в Харкові?"), ("b", "Залишити заявку"),
                               ("t", "Марина"), ("t", "067 123 45 67"), ("b", "Погоджуюсь")]),
    ("09_s8_refusal", False, [("b", "Залишити заявку"), ("t", "Андрій"), ("t", "067 123 45 67"),
                              ("b", "Не погоджуюсь"), ("b", "Назад до згоди"), ("b", "Не погоджуюсь"),
                              ("b", "Завершити")]),
    ("10_context_cost", True, [("b", "Послуги"), ("b", "Сімейні справи"), ("t", "А скільки коштує?"),
                               ("b", "Так, допомогло")]),
    ("11_urgent_text", True, [("t", "зараз іде обшук, потрібен адвокат"), ("t", "067 765 43 21"),
                              ("b", "Погоджуюсь")]),
    ("12_idle", True, [("b", "Послуги"), ("idle", 2), ("b", "Продовжити"), ("idle", 2), ("idle", 2)]),
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_server(url: str) -> None:
    for _ in range(100):
        try:
            urllib.request.urlopen(url + "/api/config", timeout=1)
            return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("сервер не запустився")


def run_steps(page, steps) -> None:
    for kind, value in steps:
        count = page.locator(".msg.bot").count()
        if kind == "b":
            page.locator("#buttons").get_by_role("button", name=value, exact=True).click()
        elif kind == "t":
            page.fill("#input", value)
            page.press("#input", "Enter")
        elif kind == "idle":
            page.evaluate(f"idleSeconds = {value}; resetIdle(false)")
        page.wait_for_function(f"document.querySelectorAll('.msg.bot').length > {count}",
                               timeout=(value + 5) * 1000 if kind == "idle" else 5000)
        page.wait_for_timeout(150)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    data = tempfile.mkdtemp(prefix="zoriana-data-")
    env = os.environ | {"DATA_DIR": data, "IDLE_SECONDS": "600"}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "zoriana_bot.web:app", "--port", str(port)],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_server(url)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            page = browser.new_page(viewport={"width": 900, "height": 900}, device_scale_factor=2)
            for name, demo, steps in SCENARIOS:
                page.goto(url + ("/?demo" if demo else "/"))
                page.add_style_tag(content=FULL_HEIGHT)
                page.wait_for_selector(".msg.bot")
                run_steps(page, steps)
                page.locator("#chat").screenshot(path=str(OUT / f"{name}.png"))
                print("✓", name)
            inbox = browser.new_page(viewport={"width": 1300, "height": 700}, device_scale_factor=2)
            inbox.goto(url + "/inbox")
            inbox.screenshot(path=str(OUT / "13_inbox.png"), full_page=True)
            print("✓ 13_inbox")
            browser.close()
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
