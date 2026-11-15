"""Інтелектуальний компонент С13: класифікатор вільного тексту.

Контролер діалогу працює лише з інтерфейсом Classifier і результатом NluResult, тому справжню модель
(model.py), заглушку для тестів (stub.py) або майбутній LLM-класифікатор можна підміняти без змін у
правилах.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

INTENTS = ("service", "apply", "human", "urgent", "unknown")
INTENT_NAMES = {
    "service": "послуга",
    "apply": "заявка",
    "human": "людина",
    "urgent": "терміново",
    "unknown": "невідомо",
}


@dataclass
class NluResult:
    intent: str  # один з INTENTS
    zone: str  # "high", "medium", "low" – зона впевненості з лаби 3
    confidence: float
    area: str | None = None  # id напряму з content.AREAS
    fragment: str | None = None  # id фрагмента з content.FRAGMENTS
    is_question: bool = False
    status: str = "ok"  # "ok" або "error" (сервіс недоступний, збій моделі)
    details: dict = field(default_factory=dict)  # імовірності для журналу і звіту


@dataclass
class NluContext:
    state: str  # стан, з якого прийшов вільний текст
    area: str | None = None  # напрям, уже обраний у сесії


class Classifier(Protocol):
    def classify(self, text: str, context: NluContext) -> NluResult: ...


class NluUnavailable(Exception):
    """Компонент не відповів: збій, таймаут, недоступний зовнішній сервіс."""
