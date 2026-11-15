"""Справжній класифікатор С13: фрази сценаріїв лаби 2, яких немає в навчальних даних, і мінімальна якість."""

import pytest

from zoriana_bot.nlu import NluContext
from zoriana_bot.nlu.model import IntentClassifier, load_dataset


@pytest.fixture(scope="module")
def model():
    return IntentClassifier.load()


def test_scenario_phrases_not_in_training_data():
    texts = {e.text.lower() for e in load_dataset()}
    for phrase in ["Здравствуйте, муж подал на развод, нужны алименты на ребёнка",
                   "Потрібна допомога з документами", "Чи ведете ви справи в Харкові?"]:
        assert phrase.lower() not in texts


@pytest.mark.parametrize("text, state, intent, area, fragment", [
    ("Здравствуйте, муж подал на развод, нужны алименты на ребёнка", "С1", "service", "family", None),  # S3
    ("Потрібна допомога з документами", "С1", "unknown", None, None),  # S4
    ("Чи ведете ви справи в Харкові?", "С1", "service", None, None),  # S7
    ("А скільки коштує?", "С3", "service", None, "cost"),
])
def test_scenario_phrases(model, text, state, intent, area, fragment):
    r = model.classify(text, NluContext(state=state))
    assert (r.intent, r.area, r.fragment) == (intent, area, fragment), r


def test_exploratory_regression(model):
    """Помилка П2: фрази дослідницького тестування після доповнення датасету."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_nlu import EXPLORATORY, KNOWN_LIMITS

    texts = {e.text.lower() for e in load_dataset()}
    for text, intent, area, _ in EXPLORATORY:
        assert text.lower() not in texts
        if text in KNOWN_LIMITS:
            continue
        r = model.classify(text, NluContext(state="С1"))
        assert r.intent == intent and (area is None or r.area == area), (text, r)


def test_result_format(model):
    r = model.classify("хочу розлучитися", NluContext(state="С1"))
    assert r.status == "ok"
    assert r.zone in ("high", "medium", "low")
    assert 0 <= r.confidence <= 1
    assert "intent_probs" in r.details
