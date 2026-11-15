"""Заглушки компонента С13 для тестів контролера.

ScriptedClassifier повертає наперед задані результати (stub) і запам'ятовує виклики (mock), щоб тести
перевіряли, що саме контролер передав компоненту. FailingClassifier імітує недоступний сервіс.
Справжня якість розпізнавання цими тестами не перевіряється: для неї є scripts/evaluate_nlu.py.
"""

from __future__ import annotations

from . import NluContext, NluResult, NluUnavailable


class ScriptedClassifier:
    def __init__(self, results: dict[str, NluResult]):
        self.results = results
        self.calls: list[tuple[str, NluContext]] = []

    def classify(self, text: str, context: NluContext) -> NluResult:
        self.calls.append((text, context))
        if text not in self.results:
            return NluResult(intent="unknown", zone="low", confidence=0.0)
        return self.results[text]


class FailingClassifier:
    def __init__(self):
        self.calls: list[tuple[str, NluContext]] = []

    def classify(self, text: str, context: NluContext) -> NluResult:
        self.calls.append((text, context))
        raise NluUnavailable("service unavailable")
