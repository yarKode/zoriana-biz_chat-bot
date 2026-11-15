"""Оцінка класифікатора С13 крос-валідацією і на фразах сценаріїв лаби 2.

    .venv/bin/python scripts/evaluate_nlu.py

Пише artifacts/nlu_metrics.json: розмір і склад датасету, точність і F1 за намірами (5-кратна
стратифікована крос-валідація), частку і точність кожної зони впевненості за поточних меж HIGH і LOW,
таблицю для вибору межі HIGH, якість напряму і фрагмента, результати на фразах сценаріїв.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zoriana_bot.nlu import NluContext  # noqa: E402
from zoriana_bot.nlu.model import HIGH, LOW, IntentClassifier, _pipeline, load_dataset  # noqa: E402

SCENARIO_PHRASES = [
    ("S3", "Здравствуйте, муж подал на развод, нужны алименты на ребёнка", "С1"),
    ("S4", "Потрібна допомога з документами", "С1"),
    ("S7", "Чи ведете ви справи в Харкові?", "С1"),
    ("TC-14", "А скільки коштує?", "С3"),
    ("TC-17", "зараз іде обшук, потрібен адвокат", "С1"),
    ("TC-18", "хочу записатися на консультацію", "С1"),
    ("TC-19", "не хочу говорити з ботом, з'єднайте з адвокатом", "С1"),
]

# Дослідницьке тестування 15.11.2026 (помилка П2): фрази, яких немає в датасеті, з очікуваним наміром і
# напрямом (None – напрям не перевіряється). "before" – результат моделі до виправлення.
# KNOWN_LIMITS – фрази, які після виправлень ще класифікуються не так; тест їх не вимагає, звіт показує.
EXPLORATORY = [
    ("у мене проблема з квартирою", "service", "civil", "urgent, висока 0.62, без напряму"),
    ("мене звільнили з роботи", "service", None, "urgent, висока 0.63"),
    ("питання по спадщині", "service", "civil", "service, висока 0.89, без напряму"),
    ("чоловік пішов з дому, що робити з дітьми", "service", "family", "urgent, середня 0.39"),
    ("сусід побив мою машину", "service", None, "service, середня 0.50"),
    ("потрібно щось робити з боргом", "service", None, "service, середня 0.37"),
    ("отримав повістку до суду", "service", None, "service, висока 0.84"),
    ("податкова прийшла з перевіркою", "service", "state", "service, висока 0.67, без напряму"),
    ("ми з дружиною розходимося", "service", "family", "service, висока 0.70, без напряму"),
    ("колишня не дає бачити сина", "service", "family", "service, висока 0.81, сімейні"),
    ("муж не платит на ребенка", "service", "family", "service, висока 0.88, сімейні"),
    ("маю питання щодо аліментів", "service", "family", "service, висока 0.67, сімейні"),
    # Помилка П3 (підготовка демо): розмиті фрази без наміру.
    ("не знаю, як сформулювати", "unknown", None, "service, висока 0.69"),
    ("хм, навіть не знаю", "unknown", None, "service, висока 0.69"),
    ("щось із папірцями", "unknown", None, "service, висока 0.71"),
    ("ааа", "unknown", None, "service, висока 0.57"),
]
KNOWN_LIMITS = {"щось із папірцями"}


def cv_probs(texts, labels):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(_pipeline(), texts, labels, cv=cv, method="predict_proba")
    classes = np.array(sorted(set(labels)))
    return classes, probs


def main() -> None:
    examples = load_dataset()
    texts = [e.text for e in examples]
    y = np.array([e.intent for e in examples])

    classes, probs = cv_probs(texts, y)
    pred, conf = classes[probs.argmax(1)], probs.max(1)
    report = classification_report(y, pred, output_dict=True, zero_division=0)

    # Зони так, як їх визначає модель: "other" і низька впевненість – "невідомо".
    final = np.where((pred == "other") | (conf < LOW), "other", pred)
    zone = np.where(final == "other", "low", np.where(conf >= HIGH, "high", "medium"))
    zones = {}
    for z in ("high", "medium", "low"):
        mask = zone == z
        zones[z] = {"share": round(float(mask.mean()), 3),
                    "accuracy": round(float((final[mask] == y[mask]).mean()), 3) if mask.any() else None}

    thresholds = []
    for t in (0.35, 0.45, 0.55, 0.65, 0.75):
        mask = (conf >= t) & (pred != "other")
        thresholds.append({"high": t, "share": round(float(mask.mean()), 3),
                           "accuracy": round(float((pred[mask] == y[mask]).mean()), 3)})

    heads = {}
    for attr in ("area", "fragment"):
        labels = np.array([getattr(e, attr) for e in examples])
        cls, pr = cv_probs(texts, labels)
        p = cls[pr.argmax(1)]
        heads[attr] = {"accuracy": round(float(accuracy_score(labels, p)), 3),
                       "macro_f1": round(float(f1_score(labels, p, average="macro")), 3),
                       "classes": dict(Counter(labels.tolist()))}

    model = IntentClassifier.load()
    scenario = []
    for ref, text, state in SCENARIO_PHRASES:
        r = model.classify(text, NluContext(state=state))
        scenario.append({"ref": ref, "text": text, "state": state, "intent": r.intent, "zone": r.zone,
                         "confidence": r.confidence, "area": r.area, "fragment": r.fragment,
                         "is_question": r.is_question})
    exploratory = []
    for text, intent, area, before in EXPLORATORY:
        r = model.classify(text, NluContext(state="С1"))
        ok = r.intent == intent and (area is None or r.area == area)
        exploratory.append({"text": text, "expected_intent": intent, "expected_area": area, "before": before,
                            "after": f"{r.intent}, {r.zone} {r.confidence}, {r.area or 'без напряму'}",
                            "ok": ok, "known_limit": text in KNOWN_LIMITS})

    metrics = {
        "dataset": {"size": len(examples), "intents": dict(Counter(y.tolist())),
                    "ukrainian_share_note": "фрази українською і російською, як у відвідувачів сайту"},
        "model": "TF-IDF (char_wb 2–5 + word 1–2) + LogisticRegression(C=10, class_weight=balanced)",
        "cv": "5-fold stratified, random_state=42",
        "intent": {
            "accuracy": round(float(accuracy_score(y, pred)), 3),
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 3),
            "per_class": {k: {m: round(v[m], 3) for m in ("precision", "recall", "f1-score", "support")}
                          for k, v in report.items() if k in classes},
            "labels": classes.tolist(),
            "confusion": confusion_matrix(y, pred, labels=classes).tolist(),
        },
        "zones": {"HIGH": HIGH, "LOW": LOW, **zones},
        "threshold_table": thresholds,
        "area": heads["area"],
        "fragment": heads["fragment"],
        "scenario_phrases": scenario,
        "exploratory": exploratory,
    }
    out = ROOT / "artifacts" / "nlu_metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("dataset", "zones", "threshold_table", "area", "fragment")},
                     ensure_ascii=False, indent=1))
    print("intent accuracy", metrics["intent"]["accuracy"], "macro F1", metrics["intent"]["macro_f1"])
    for s in scenario:
        print(s)
    for e in exploratory:
        print("OK  " if e["ok"] else "FAIL", e["text"], "|", e["before"], "→", e["after"])


if __name__ == "__main__":
    main()
