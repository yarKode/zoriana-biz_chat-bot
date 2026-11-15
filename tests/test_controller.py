"""Окремі правила контролера, які не видно з повних сценаріїв."""

import pytest

from zoriana_bot.cases import CASES, N, run_case
from zoriana_bot.controller import Controller, Session, normalize_phone, valid_name
from zoriana_bot.nlu.stub import ScriptedClassifier
from zoriana_bot.tree import EDGES, ENDS, STATES


@pytest.mark.parametrize("text, expected", [
    ("067 123 45 67", "+380671234567"),
    ("(067) 123-45-67", "+380671234567"),
    ("+380671234567", "+380671234567"),
    ("380671234567", "+380671234567"),
    ("067 12", None),
    ("не хочу давати номер", None),
    ("1671234567", None),
])
def test_normalize_phone(text, expected):
    assert normalize_phone(text) == expected


def test_valid_name():
    assert valid_name("Ірина")
    assert not valid_name("123")
    assert not valid_name("x" * 61)


def bot_with(results):
    nlu = ScriptedClassifier(results)
    return Controller(nlu), Session(), nlu


def test_back_after_classifier_returns_to_menu():
    bot, s, _ = bot_with({"потрібен адвокат": N("service", "high")})
    bot.handle(s, {"type": "start"})
    assert bot.handle(s, {"type": "text", "text": "потрібен адвокат"}).state == "С2"
    assert bot.handle(s, {"type": "button", "id": "back"}).state == "С1"


def test_menu_after_refusal_clears_contacts():
    bot, s, _ = bot_with({})
    bot.handle(s, {"type": "start"})
    for event in [{"type": "button", "id": "apply"}, {"type": "text", "text": "Олег"},
                  {"type": "text", "text": "0671234567"}, {"type": "button", "id": "disagree"}]:
        bot.handle(s, event)
    assert (s.name, s.phone) == ("Олег", "+380671234567")  # "Назад до згоди" ще може повернути заявку
    bot.handle(s, {"type": "button", "id": "menu"})
    assert (s.name, s.phone) == (None, None)


def test_unknown_button_keeps_state():
    bot, s, _ = bot_with({})
    bot.handle(s, {"type": "start"})
    reply = bot.handle(s, {"type": "button", "id": "no-such-button"})
    assert reply.state == "С1" and reply.buttons


def test_end_state_restarts_dialog():
    bot, s, _ = bot_with({})
    bot.handle(s, {"type": "start"})
    for button in ["services", "area:family", "frag:cost", "helped"]:
        reply = bot.handle(s, {"type": "button", "id": button})
    assert reply.state == "К1" and reply.ended
    assert bot.handle(s, {"type": "button", "id": "menu"}).state == "С1"


def test_every_state_has_exit():
    """Жоден стан, крім кінцевих, не лишає відвідувача без кнопок."""
    bot, s, _ = bot_with({})
    for code in STATES:
        if code in ("С12", "С13"):
            continue  # стани обробки, у них діалог не зупиняється
        s.state, s.area = code, "family"
        s.resume = "С1"
        _, buttons, _ = bot._view(s)
        assert buttons, code


def test_edge_coverage():
    """Тест-кейси проходять усі ребра дерева рішень лаби 3."""
    covered = {(a, b) for case in CASES for a, b, _ in run_case(case)["trace"] if a}
    missing = EDGES - covered
    assert not missing, sorted(missing)
    assert {b for a, b in covered} & set(ENDS) == set(ENDS)
