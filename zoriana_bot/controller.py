"""Контролер діалогу: реалізує дерево рішень лаби 3 (С1–С17, К1–К5).

Контролер вирішує, коли залишатися в правилах, а коли передати вільний текст класифікатору С13,
і сам обирає наступний стан за результатом класифікатора. Класифікатор тексту відповіді не складає.

Подія від інтерфейсу – один із варіантів:
    {"type": "start"}                 відкриття чату
    {"type": "button", "id": "..."}   натиснута кнопка
    {"type": "text", "text": "..."}   введений текст
    {"type": "idle"}                  10 хв без дій (таймер інтерфейсу)
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .content import AREA_FRAGMENTS, AREAS, FRAGMENTS, PHONE, URGENT_KINDS
from .nlu import INTENT_NAMES, Classifier, NluContext, NluResult
from .tree import APPLICATION, ENDS, FIELDS

MAX_UNKNOWN = 2  # друге "невідомо" веде до адвоката (Г4)
MAX_PHONE_ERRORS = 2  # друга помилка телефону веде до С9

# Стани, через які діалог лише проходить: їх немає в історії для кнопки "Назад".
TRANSIT = {"С12", "С13", "С16", "С17"}


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: str = ""
    history: list[str] = field(default_factory=list)
    area: str | None = None
    area_unknown: bool = False
    name: str | None = None
    phone: str | None = None
    urgent: str | None = None
    question: str | None = None
    phone_errors: int = 0
    unknown_count: int = 0
    resume: str | None = None  # стан, до якого повертають С16 і С17
    pending: NluResult | None = None  # результат класифікатора, який підтверджує С14
    trace: list[tuple[str, str, str]] = field(default_factory=list)  # (з, до, подія) за всю сесію

    def clear_contacts(self) -> None:
        self.name = self.phone = None


@dataclass
class Reply:
    messages: list[str]
    buttons: list[tuple[str, str]]  # (id, підпис)
    state: str
    input: str  # "text", "name", "phone" – що чекає поле введення
    ended: bool
    trace: list[tuple[str, str, str]]  # переходи за цей хід
    nlu: dict | None = None  # результат класифікатора за цей хід (для журналу і демо)


class Inbox:
    """Імітація передачі адвокату: звернення і журнал нерозпізнаних фраз (ціль Ц6).

    Справжнього підключення оператора немає: звернення зберігаються в пам'яті і, якщо задано теку,
    у JSON Lines. Адвокат бачить їх на сторінці /inbox вебчату.
    """

    def __init__(self, folder: Path | None = None):
        self.folder = folder
        self.handoffs: list[dict] = []
        self.unrecognized: list[dict] = []

    def _write(self, name: str, record: dict) -> None:
        if self.folder:
            self.folder.mkdir(parents=True, exist_ok=True)
            with open(self.folder / name, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def handoff(self, record: dict) -> None:
        self.handoffs.append(record)
        self._write("handoffs.jsonl", record)

    def log_unrecognized(self, text: str, state: str) -> None:
        # Лише текст і стан, без імені й телефону (лаба 3, підрозділ 1.3).
        record = {"time": _now(), "text": text, "state": state}
        self.unrecognized.append(record)
        self._write("unrecognized.jsonl", record)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_phone(text: str) -> str | None:
    """10 цифр, що починаються з 0, або +380 і 9 цифр. Повертає номер у форматі +380XXXXXXXXX."""
    digits = re.sub(r"[\s\-()]", "", text)
    if re.fullmatch(r"0\d{9}", digits):
        return "+38" + digits
    if re.fullmatch(r"\+?380\d{9}", digits):
        return "+" + digits.lstrip("+")
    return None


def valid_name(text: str) -> bool:
    return 0 < len(text) <= 60 and bool(re.search(r"[^\W\d_]", text))


MENU = ("menu", "Головне меню")
BACK = ("back", "Назад")


class Controller:
    def __init__(self, nlu: Classifier, inbox: Inbox | None = None):
        self.nlu = nlu
        self.inbox = inbox or Inbox()

    # ------------------------------------------------------------ вхід

    def handle(self, s: Session, event: dict) -> Reply:
        self._out: list[str] = []
        self._turn: list[tuple[str, str, str]] = []
        self._nlu: dict | None = None
        kind = event.get("type")
        button = event.get("id", "")
        text = (event.get("text") or "").strip()

        if kind == "start" or not s.state:
            return self._enter(s, "С1", "початок діалогу")
        if s.state in ENDS:
            # Після кінцевого стану діалог починається знову з меню; текст одразу класифікується.
            self._enter(s, "С1", "новий діалог", record=False)
            if kind == "text" and text:
                return self._classify(s, text)
            return self._reply(s)
        if kind == "idle":
            return self._idle(s)
        if kind == "text":
            if not text:
                return self._reply(s)
            if s.state in FIELDS:
                return self._field(s, text)
            return self._classify(s, text)
        if button == "menu":
            return self._menu(s)
        if button == "back":
            return self._back(s)
        handler = getattr(self, "_on_" + s.state[1:], None)
        reply = handler(s, button) if handler else None
        return reply or self._reply(s)  # невідома кнопка: показати поточний стан ще раз

    # ------------------------------------------------------------ переходи

    def _goto(self, s: Session, to: str, event: str, record: bool = True) -> None:
        if record:
            step = (s.state, to, event)
            s.trace.append(step)
            self._turn.append(step)
        if s.state and s.state not in TRANSIT and s.state not in ENDS and s.state != to:
            s.history.append(s.state)
        if s.history and s.history[-1] == to:
            s.history.pop()  # повернення в стан, з якого вийшли (С16 → С7 тощо)
        s.state = to

    def _enter(self, s: Session, to: str, event: str, *messages: str, record: bool = True) -> Reply:
        self._out.extend(messages)
        self._goto(s, to, event, record)
        if to == "С1":
            s.history.clear()
        return self._reply(s)

    def _reply(self, s: Session) -> Reply:
        prompt, buttons, mode = self._view(s)
        return Reply(
            messages=self._out + ([prompt] if prompt else []),
            buttons=buttons,
            state=s.state,
            input=mode,
            ended=s.state in ENDS,
            trace=self._turn,
            nlu=self._nlu,
        )

    def _view(self, s: Session) -> tuple[str, list[tuple[str, str]], str]:
        """Репліка, кнопки і режим поля введення для поточного стану."""
        area_buttons = [("area:" + k, v[0]) for k, v in AREAS.items()]
        st = s.state
        if st == "С1":
            return ("Добрий день! Допоможу знайти ваш напрям і передати звернення адвокату. Оберіть дію "
                    "або напишіть своє питання.",
                    [("services", "Послуги"), ("apply", "Залишити заявку"), ("lawyer", "Зв’язатися з адвокатом"),
                     ("urgent", "Терміново")], "text")
        if st == "С2":
            return "Оберіть напрям:", area_buttons + [("undecided", "Ще не визначився"), BACK, MENU], "text"
        if st == "С3":
            title, text = AREAS[s.area]
            frag = [("frag:" + f, FRAGMENTS[f][0]) for f in AREA_FRAGMENTS]
            return f"{title}. {text}", frag + [("apply", "Залишити заявку"), BACK, MENU], "text"
        if st == "С4":
            return ("Чи допомогла відповідь?",
                    [("helped", "Так, допомогло"), ("not_helped", "Ні"), ("apply", "Залишити заявку"),
                     ("lawyer", "Зв’язатися з адвокатом"), BACK, MENU], "text")
        if st == "С5":
            return ("На сайті немає відповіді на це питання. Передам його адвокату разом із вашим телефоном.",
                    [("apply", "Залишити заявку"), BACK, MENU], "text")
        if st == "С6":
            return (f"Що сталося? Можете одразу телефонувати: {PHONE}",
                    [("urgent:" + k, v) for k, v in URGENT_KINDS.items()] + [BACK, MENU], "text")
        if st == "С7":
            return "Як до вас звертатися?", [BACK, MENU], "name"
        if st == "С8":
            return "Залиште телефон для дзвінка адвоката.", [BACK, MENU], "phone"
        if st == "С9":
            return (f"Без телефону адвокат не зможе передзвонити. Можете зателефонувати самі: {PHONE}",
                    [("retry_phone", "Ввести номер"), MENU], "text")
        if st == "С10":
            return ("Заявку бачить лише адвокат. Погоджуєтеся на обробку імені й телефону для зворотного дзвінка?",
                    [("agree", "Погоджуюсь"), ("disagree", "Не погоджуюсь"), BACK, MENU], "text")
        if st == "С11":
            return (f"Без згоди я не можу передати ваші дані. Телефон адвоката: {PHONE}",
                    [("back_to_consent", "Назад до згоди"), MENU, ("finish", "Завершити")], "text")
        if st == "С14":
            return self._confirm_text(s.pending), [("yes", "Так"), ("other_area", "Інший напрям"), BACK, MENU], "text"
        if st == "С15":
            return ("Не впевнений, що правильно зрозумів. Оберіть напрям:",
                    area_buttons + [("undecided", "Ще не визначився"), ("lawyer", "Зв’язатися з адвокатом"), BACK,
                                    MENU], "text")
        if st == "С16":
            return ("Заявку ще не надіслано. Повернутися до неї?",
                    [("continue_app", "Продовжити заявку"), MENU], "text")
        if st == "С17":
            return "Ви ще тут? Можемо продовжити з того самого місця.", [("continue", "Продовжити"), MENU], "text"
        # Кінцеві стани: репліку вже додано під час переходу.
        return "", [MENU], "text"

    @staticmethod
    def _confirm_text(r: NluResult | None) -> str:
        if r is None:
            return "Уточніть, будь ласка, ваше питання."
        if r.intent == "service" and r.area:
            return f"Схоже, ваше питання про напрям \"{AREAS[r.area][0]}\". Так?"
        if r.intent == "service" and r.fragment:
            return f"Схоже, ви питаєте: \"{FRAGMENTS[r.fragment][0]}\". Так?"
        return {
            "apply": "Схоже, ви хочете залишити заявку. Так?",
            "human": "Схоже, ви хочете зв’язатися з адвокатом. Так?",
            "urgent": "Схоже, у вас термінова ситуація. Так?",
        }.get(r.intent, "Схоже, ви питаєте про послуги адвоката. Так?")

    # ------------------------------------------------------------ глобальні переходи

    def _menu(self, s: Session) -> Reply:
        if s.state in APPLICATION:
            s.resume = s.state
            return self._enter(s, "С16", "\"Головне меню\" під час заявки")
        if s.state == "С11":
            s.clear_contacts()  # лаба 3: вихід із С11 видаляє ім'я і телефон
        return self._enter(s, "С1", "\"Головне меню\"")

    def _back(self, s: Session) -> Reply:
        if not s.history:
            return self._reply(s)
        previous = s.history.pop()
        step = (s.state, previous, "\"Назад\"")
        s.trace.append(step)
        self._turn.append(step)
        s.state = previous
        if previous == "С8":
            s.phone_errors = 0
        return self._reply(s)

    def _idle(self, s: Session) -> Reply:
        if s.state == "С17":
            s.clear_contacts()
            return self._enter(s, "К5", "ще 10 хв без дій",
                               "Сесію завершено. Незавершену заявку не надіслано. Будемо раді допомогти пізніше.")
        s.resume = s.state
        return self._enter(s, "С17", "10 хв без дій")

    # ------------------------------------------------------------ гібридний компонент С13

    def _classify(self, s: Session, text: str) -> Reply:
        origin = s.state
        self._goto(s, "С13", "вільний текст")
        try:
            r = self.nlu.classify(text, NluContext(state=origin, area=s.area))
        except Exception as error:  # збій моделі або недоступний сервіс
            self._nlu = {"status": "error", "error": str(error)}
            s.unknown_count += 1
            if s.unknown_count >= MAX_UNKNOWN:
                return self._to_lawyer_unknown(s, text)
            return self._enter(s, "С15", "компонент недоступний",
                               "Не вдалося обробити текст: сервіс розпізнавання тимчасово недоступний.")
        self._nlu = asdict(r) | {"intent_name": INTENT_NAMES[r.intent]}

        if r.zone == "low":
            s.unknown_count += 1
            self.inbox.log_unrecognized(text, origin)
            if s.unknown_count >= MAX_UNKNOWN:
                return self._to_lawyer_unknown(s, text)
            return self._enter(s, "С15", "низька впевненість, уперше")
        if r.zone == "medium":
            s.pending = r
            s.question = text
            return self._enter(s, "С14", "середня впевненість")
        s.question = text
        return self._route(s, r, "висока впевненість")

    def _to_lawyer_unknown(self, s: Session, text: str) -> Reply:
        s.question = text
        return self._enter(s, "С8", "\"невідомо\" вдруге",
                           "Передам ваше повідомлення адвокату, вона уточнить питання сама.")

    def _route(self, s: Session, r: NluResult, event: str) -> Reply:
        """Ланцюг перевірок рисунка 1.3 лаби 3: людина/терміново → заявка → фрагмент → напрям → питання."""
        s.unknown_count = 0
        if r.area:
            s.area, s.area_unknown = r.area, False
        if r.intent in ("human", "urgent"):
            if r.intent == "urgent":
                s.urgent = "з тексту повідомлення"
                return self._enter(s, "С8", f"{event}, намір \"терміново\"",
                                   f"Передам звернення адвокату з позначкою \"терміново\". Або телефонуйте зараз: {PHONE}")
            return self._enter(s, "С8", f"{event}, намір \"людина\"", "Передам ваше звернення адвокату.")
        if r.intent == "apply":
            return self._enter(s, "С7", f"{event}, намір \"заявка\"")
        if r.fragment:
            s.question = None
            return self._enter(s, "С4", f"{event}, знайдено фрагмент сайту", FRAGMENTS[r.fragment][1])
        if s.area and r.area:
            s.question = None
            return self._enter(s, "С3", f"{event}, визначено напрям")
        if r.is_question:
            return self._enter(s, "С5", f"{event}, питання без фрагмента")
        s.question = None
        return self._enter(s, "С2", f"{event}, ні питання, ні напряму")

    # ------------------------------------------------------------ поля заявки С7, С8

    def _field(self, s: Session, text: str) -> Reply:
        if s.state == "С7":
            if not valid_name(text):
                self._out.append("Напишіть, будь ласка, ім’я літерами.")
                return self._reply(s)
            s.name = text
            return self._enter(s, "С8", "введено ім’я")
        phone = normalize_phone(text)
        if phone:
            s.phone, s.phone_errors = phone, 0
            return self._enter(s, "С10", "коректний номер")
        s.phone_errors += 1
        if s.phone_errors >= MAX_PHONE_ERRORS:
            return self._enter(s, "С9", "некоректний номер, помилка друга")
        return self._enter(s, "С8", "некоректний номер, помилка перша",
                           "Номер має містити 10 цифр, наприклад 067 123 45 67.")

    # ------------------------------------------------------------ кнопки станів

    def _on_1(self, s: Session, b: str) -> Reply | None:
        if b == "services":
            return self._enter(s, "С2", "\"Послуги\"")
        if b == "apply":
            return self._enter(s, "С7", "\"Залишити заявку\"")
        if b == "lawyer":
            return self._enter(s, "С8", "\"Зв’язатися з адвокатом\"")
        if b == "urgent":
            return self._enter(s, "С6", "\"Терміново\"")
        return None

    def _pick_area(self, s: Session, b: str) -> Reply | None:
        if b.startswith("area:") and b[5:] in AREAS:
            s.area, s.area_unknown = b[5:], False
            s.unknown_count = 0
            return self._enter(s, "С3", "обрано напрям")
        if b == "undecided":
            s.area, s.area_unknown = None, True
            return self._enter(s, "С7", "\"Ще не визначився\"", "Залиште заявку – адвокат уточнить напрям.")
        return None

    def _on_2(self, s: Session, b: str) -> Reply | None:
        return self._pick_area(s, b)

    def _on_3(self, s: Session, b: str) -> Reply | None:
        if b.startswith("frag:") and b[5:] in FRAGMENTS:
            return self._enter(s, "С4", "кнопка фрагмента", FRAGMENTS[b[5:]][1])
        if b == "apply":
            return self._enter(s, "С7", "\"Залишити заявку\"")
        return None

    def _on_4(self, s: Session, b: str) -> Reply | None:
        if b == "helped":
            return self._enter(s, "К1", "\"Так, допомогло\"", "Радий, що допоміг! Якщо з’являться питання – пишіть.")
        if b == "not_helped":
            return self._enter(s, "С7", "\"Ні\"", "Точну відповідь дасть адвокат. Залишимо заявку.")
        if b == "apply":
            return self._enter(s, "С7", "\"Залишити заявку\"")
        if b == "lawyer":
            return self._enter(s, "С8", "\"Зв’язатися з адвокатом\"")
        return None

    def _on_5(self, s: Session, b: str) -> Reply | None:
        if b == "apply":
            return self._enter(s, "С7", "\"Залишити заявку\"", "Текст вашого питання додам до заявки.")
        return None

    def _on_6(self, s: Session, b: str) -> Reply | None:
        if b.startswith("urgent:") and b[7:] in URGENT_KINDS:
            s.urgent = URGENT_KINDS[b[7:]]
            return self._enter(s, "С8", "обрано вид ситуації",
                               f"Залиште телефон – передам звернення адвокату з позначкою \"терміново\". "
                               f"Або телефонуйте зараз: {PHONE}")
        return None

    def _on_9(self, s: Session, b: str) -> Reply | None:
        if b == "retry_phone":
            s.phone_errors = 0
            return self._enter(s, "С8", "\"Ввести номер\"")
        return None

    def _on_10(self, s: Session, b: str) -> Reply | None:
        if b == "agree":
            self._goto(s, "С12", "\"Погоджуюсь\"")
            return self._handoff(s)
        if b == "disagree":
            return self._enter(s, "С11", "\"Не погоджуюсь\"")
        return None

    def _on_11(self, s: Session, b: str) -> Reply | None:
        if b == "back_to_consent":
            return self._enter(s, "С10", "\"Назад до згоди\"")
        if b == "finish":
            s.clear_contacts()
            return self._enter(s, "К4", "\"Завершити\"", f"Звернення не передано. Телефон адвоката: {PHONE}")
        return None

    def _on_14(self, s: Session, b: str) -> Reply | None:
        if b == "yes" and s.pending:
            r, s.pending = s.pending, None
            return self._route(s, r, "намір підтверджено")
        if b == "other_area":
            s.pending = None
            return self._enter(s, "С2", "\"Інший напрям\"")
        return None

    def _on_15(self, s: Session, b: str) -> Reply | None:
        if b == "lawyer":
            return self._enter(s, "С8", "\"Зв’язатися з адвокатом\"")
        return self._pick_area(s, b)

    def _on_16(self, s: Session, b: str) -> Reply | None:
        if b == "continue_app" and s.resume:
            return self._enter(s, s.resume, "\"Продовжити заявку\"")
        return None

    def _on_17(self, s: Session, b: str) -> Reply | None:
        if b == "continue" and s.resume:
            return self._enter(s, s.resume, "\"Продовжити\"")
        return None

    # ------------------------------------------------------------ С12 передача адвокату

    def _handoff(self, s: Session) -> Reply:
        area = AREAS[s.area][0] if s.area else "не визначено"
        record = {
            "time": _now(),
            "session": s.id,
            "name": s.name,
            "phone": s.phone,
            "area": area,
            "urgent": s.urgent,
            "question": s.question,
            "path": [step[1] for step in s.trace],
        }
        self.inbox.handoff(record)
        who = f", {s.name}" if s.name else ""
        if s.urgent:
            return self._enter(s, "К3", "звернення з позначкою \"терміново\"",
                               f"Дякую{who}. Термінове звернення передано адвокату. Телефон для дзвінка: {PHONE}")
        return self._enter(s, "К2", "звичайне звернення",
                           f"Дякую{who}. Заявку з напрямом \"{area}\" передано адвокату. "
                           f"Телефон для дзвінка: {PHONE}")
