"""Класифікатор наміру, напряму і фрагмента сайту: TF-IDF (символьні n-грами) + логістична регресія.

Три окремі моделі навчаються на одному датасеті (dataset.txt). Символьні n-грами переносять схожість
між близькими українськими й російськими формами слів ("аліменти" / "алименты") і стійкі до одруківок. Модель лише обирає гілку і фрагмент: тексту відповіді вона не складає.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from . import NluContext, NluResult

HERE = Path(__file__).parent
DATASET = HERE / "dataset.txt"
MODEL_FILE = HERE / "model.joblib"

# Межі зон упевненості (лаба 3 фіксувала лише зони, межі підібрано на валідації, див. scripts/evaluate_nlu.py).
HIGH = 0.55
LOW = 0.35
AREA_MIN = 0.45
FRAGMENT_MIN = 0.45

QUESTION_WORDS = {
    "чи", "як", "скільки", "коли", "де", "куди", "хто", "який", "яка", "яке", "які", "чому", "можна",
    "ли", "сколько", "как", "где", "куда", "когда", "кто", "какой", "какая", "какие", "почему", "можно",
}


@dataclass
class Example:
    text: str
    intent: str
    area: str
    fragment: str


def load_dataset(path: Path = DATASET) -> list[Example]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        labels, text = line.split(" | ", 1)
        intent, area, fragment = labels.split()
        examples.append(Example(text.strip(), intent, area, fragment))
    return examples


def _pipeline() -> Pipeline:
    features = FeatureUnion([
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ("words", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(C=10, max_iter=5000, class_weight="balanced")),
    ])


def is_question(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    return "?" in text or bool(words and words[0] in QUESTION_WORDS)


class IntentClassifier:
    def __init__(self, intent: Pipeline, area: Pipeline, fragment: Pipeline):
        self.intent, self.area, self.fragment = intent, area, fragment

    @classmethod
    def train(cls, examples: list[Example]) -> IntentClassifier:
        texts = [e.text for e in examples]
        models = []
        for attr in ("intent", "area", "fragment"):
            model = _pipeline()
            model.fit(texts, [getattr(e, attr) for e in examples])
            models.append(model)
        return cls(*models)

    @classmethod
    def load(cls, path: Path = MODEL_FILE) -> IntentClassifier:
        """Завантажує збережену модель або навчає нову, якщо файлу немає чи датасет новіший."""
        if path.exists() and path.stat().st_mtime >= DATASET.stat().st_mtime:
            return cls(*joblib.load(path))
        model = cls.train(load_dataset())
        joblib.dump((model.intent, model.area, model.fragment), path)
        return model

    @staticmethod
    def _top(model: Pipeline, text: str) -> tuple[str, float, dict[str, float]]:
        probs = model.predict_proba([text])[0]
        ranked = dict(sorted(zip(map(str, model.classes_), probs.round(3).tolist()), key=lambda kv: -kv[1]))
        label = next(iter(ranked))
        return label, ranked[label], ranked

    def classify(self, text: str, context: NluContext) -> NluResult:
        if len(set(re.findall(r"[^\W\d_]", text.lower()))) < 3:
            # "ааа", "123", "??": на таких рядках n-грами не несуть змісту, модель лише вгадала б клас.
            return NluResult(intent="unknown", zone="low", confidence=0.0, is_question=is_question(text),
                             details={"rule": "менше трьох різних літер", "state": context.state})
        intent, confidence, intent_probs = self._top(self.intent, text)
        area, area_p, _ = self._top(self.area, text)
        fragment, fragment_p, _ = self._top(self.fragment, text)

        if intent == "other":
            # Упевнено "поза темою" – для дерева рішень це той самий намір "невідомо".
            intent, zone = "unknown", "low"
        elif confidence >= HIGH:
            zone = "high"
        elif confidence >= LOW:
            zone = "medium"
        else:
            intent, zone = "unknown", "low"

        return NluResult(
            intent=intent,
            zone=zone,
            confidence=round(confidence, 3),
            area=area if area != "-" and area_p >= AREA_MIN else None,
            fragment=fragment if fragment != "-" and fragment_p >= FRAGMENT_MIN else None,
            is_question=is_question(text),
            details={"intent_probs": intent_probs, "area_p": round(area_p, 3), "fragment_p": round(fragment_p, 3),
                     "state": context.state},
        )
