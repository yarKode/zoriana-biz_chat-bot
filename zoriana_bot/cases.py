"""Тест-кейси прототипу як дані. Їх проганяють pytest (tests/test_cases.py) і scripts/run_cases.py,
який записує фактичні результати в artifacts/test_results.json для звіту.

nlu = "model"   – справжній класифікатор; очікуваний шлях перевіряється як підпослідовність станів.
nlu = "stub"    – заглушка з наперед заданими результатами; шлях перевіряється точно, а виклики
                  класифікатора (mock) – за expect_calls.
nlu = "failing" – імітація недоступного сервісу.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .controller import Controller, Inbox, Session
from .nlu import NluResult
from .nlu.stub import FailingClassifier, ScriptedClassifier

_CONF = {"high": 0.9, "medium": 0.45, "low": 0.2}


def B(button: str, if_state: str | None = None) -> dict:
    return {"type": "button", "id": button, "if_state": if_state}


def T(text: str) -> dict:
    return {"type": "text", "text": text}


IDLE = {"type": "idle"}


def N(intent: str, zone: str, area: str | None = None, fragment: str | None = None, q: bool = False) -> NluResult:
    return NluResult(intent=intent, zone=zone, confidence=_CONF[zone], area=area, fragment=fragment, is_question=q)


@dataclass
class Case:
    id: str
    title: str
    kind: str  # основний, альтернативний, проблемний, гібридний, інтеграційний
    scenario: str  # посилання на сценарій лаби 2 або "–"
    steps: list[dict]
    expect_path: list[str]
    expect_end: str
    nlu: str = "stub"
    stub: dict[str, NluResult] = field(default_factory=dict)
    expect_handoff: dict | None = None  # поля звернення; {} – звернення не має бути
    expect_calls: list[tuple[str, str, str | None]] | None = None  # (текст, стан, напрям) викликів класифікатора
    expect_unrecognized: int | None = None
    expected_text: str = ""  # очікуваний результат словами, для таблиці звіту


S3 = "Здравствуйте, муж подал на развод, нужны алименты на ребёнка"
S4 = "Потрібна допомога з документами"
S7 = "Чи ведете ви справи в Харкові?"
NAME, PHONE_OK = "Андрій", "067 123 45 67"

CASES: list[Case] = [
    Case("TC-01", "Основний шлях кнопками", "основний", "S1",
         [B("services"), B("area:business"), B("frag:cost"), B("not_helped"), T(NAME), T(PHONE_OK), B("agree")],
         ["С1", "С2", "С3", "С4", "С7", "С8", "С10", "С12", "К2"], "К2",
         expect_handoff={"area": "Господарські спори", "name": NAME, "phone": "+380671234567", "urgent": None},
         expected_text="Опис напряму, фрагмент про вартість, заявка з напрямом \"Господарські спори\" передана (К2)"),
    Case("TC-02", "Термінове звернення", "альтернативний", "S2",
         [B("urgent"), B("urgent:traffic"), T("0981112233"), B("agree")],
         ["С1", "С6", "С8", "С10", "С12", "К3"], "К3",
         expect_handoff={"urgent": "ДТП або позбавлення права керування", "name": None},
         expected_text="Ім’я не запитується, звернення з позначкою \"терміново\" (К3)"),
    Case("TC-03", "Вільний текст російською, справжній класифікатор", "гібридний", "S3",
         [T(S3), B("yes", if_state="С14"), B("frag:online"), B("apply"), T("Ірина"), T("+380671234567"),
          B("agree")],
         ["С1", "С13", "С3", "С4", "С7", "С8", "С10", "С12", "К2"], "К2", nlu="model",
         expect_handoff={"area": "Сімейні справи", "name": "Ірина"},
         expected_text="Класифікатор: послуга, \"Сімейні справи\"; діалог повертається в правила (С3), "
                       "заявка з напрямом"),
    Case("TC-04", "Середня впевненість і підтвердження наміру", "гібридний", "S3, S3а",
         [T(S3), B("yes"), B("frag:online"), B("helped")],
         ["С1", "С13", "С14", "С3", "С4", "К1"], "К1",
         stub={S3: N("service", "medium", area="family")},
         expect_calls=[(S3, "С1", None)],
         expected_text="\"Схоже, ваше питання про напрям \"Сімейні справи\". Так?\" → опис напряму → "
                       "відповіді досить (К1)"),
    Case("TC-05", "Нерозпізнаний запит, справжній класифікатор", "проблемний", "S4",
         [T(S4), B("undecided"), T("Олег"), T("0501234567"), B("agree")],
         ["С1", "С13", "С15", "С7", "С8", "С10", "С12", "К2"], "К2", nlu="model",
         expect_handoff={"area": "не визначено"}, expect_unrecognized=1,
         expected_text="Намір \"невідомо\", одне уточнення кнопками, фраза в журналі нерозпізнаних, заявка "
                       "з напрямом \"не визначено\""),
    Case("TC-06", "Друге \"невідомо\" передає звернення адвокату", "проблемний", "S4",
         [T("бла-бла"), T("ще раз бла"), T(PHONE_OK), B("agree")],
         ["С1", "С13", "С15", "С13", "С8", "С10", "С12", "К2"], "К2",
         expect_handoff={"question": "ще раз бла"}, expect_unrecognized=2,
         expected_text="Після другого \"невідомо\" – одразу телефон, текст повідомлення в зверненні"),
    Case("TC-07", "Некоректний телефон двічі", "проблемний", "S5",
         [B("lawyer"), T("067 12"), T("не хочу давати номер"), B("retry_phone"), T("0671234567"), B("agree")],
         ["С1", "С8", "С8", "С9", "С8", "С10", "С12", "К2"], "К2",
         expect_calls=[],
         expected_text="Перша помилка – підказка формату, друга – С9 з телефоном сайту; текст у полі не йде "
                       "до класифікатора"),
    Case("TC-08", "Назад і вихід у меню із заявки", "альтернативний", "S6",
         [B("services"), B("area:civil"), B("back"), B("area:family"), B("apply"), T("Ірина"), B("menu"),
          B("menu")],
         ["С1", "С2", "С3", "С2", "С3", "С7", "С8", "С16", "С1"], "С1",
         expect_handoff={},
         expected_text="\"Назад\" повертає список напрямів; вихід із заявки перепитує (С16); нічого не надіслано"),
    Case("TC-09", "Повернення до незавершеної заявки", "альтернативний", "S6",
         [B("apply"), T("Ірина"), B("menu"), B("continue_app"), T(PHONE_OK), B("agree")],
         ["С1", "С7", "С8", "С16", "С8", "С10", "С12", "К2"], "К2",
         expect_handoff={"name": "Ірина"},
         expected_text="\"Продовжити заявку\" повертає до поля телефону, ім’я збережено"),
    Case("TC-10", "Питання без відповіді на сайті, справжній класифікатор", "гібридний", "S7",
         [T(S7), B("apply"), T("Марина"), T(PHONE_OK), B("agree")],
         ["С1", "С13", "С5", "С7", "С8", "С10", "С12", "К2"], "К2", nlu="model",
         expect_handoff={"question": S7},
         expected_text="Бот не складає відповідь: \"На сайті немає відповіді…\", текст питання в заявці"),
    Case("TC-11", "Відмова від згоди і повернення до неї", "альтернативний", "S8",
         [B("apply"), T(NAME), T(PHONE_OK), B("menu"), B("continue_app"), B("disagree"), B("back_to_consent"),
          B("disagree"), B("menu")],
         ["С1", "С7", "С8", "С10", "С16", "С10", "С11", "С10", "С11", "С1"], "С1",
         expect_handoff={},
         expected_text="Без згоди дані не передаються; \"Назад до згоди\" і вихід у меню працюють"),
    Case("TC-12", "Відмова від згоди і завершення", "альтернативний", "S8",
         [B("apply"), T(NAME), T(PHONE_OK), B("disagree"), B("finish")],
         ["С1", "С7", "С8", "С10", "С11", "К4"], "К4",
         expect_handoff={},
         expected_text="Звернення не передано, ім’я і телефон видалено, на екрані телефон сайту (К4)"),
    Case("TC-13", "Неактивність", "проблемний", "–",
         [B("services"), IDLE, B("continue"), IDLE, IDLE],
         ["С1", "С2", "С17", "С2", "С17", "К5"], "К5",
         expect_handoff={},
         expected_text="Одне нагадування, \"Продовжити\" повертає до кроку; друга пауза закриває сесію (К5)"),
    Case("TC-14", "Питання в описі напряму, справжній класифікатор", "гібридний", "–",
         [B("services"), B("area:family"), T("А скільки коштує?"), B("helped")],
         ["С1", "С2", "С3", "С13", "С4", "К1"], "К1", nlu="model",
         expected_text="Фрагмент про вартість; обраний напрям зберігається"),
    Case("TC-15", "Контекст, переданий класифікатору", "інтеграційний", "–",
         [B("services"), B("area:family"), T("А скільки коштує?"), B("lawyer"), T(PHONE_OK), B("agree")],
         ["С1", "С2", "С3", "С13", "С4", "С8", "С10", "С12", "К2"], "К2",
         stub={"А скільки коштує?": N("service", "high", fragment="cost", q=True)},
         expect_calls=[("А скільки коштує?", "С3", "family")],
         expect_handoff={"area": "Сімейні справи"},
         expected_text="Класифікатор отримує текст, стан С3 і напрям \"family\"; напрям доходить до звернення"),
    Case("TC-16", "Сервіс розпізнавання недоступний", "проблемний", "–",
         [T("хочу розлучитися"), T("аліменти"), T(PHONE_OK), B("agree")],
         ["С1", "С13", "С15", "С13", "С8", "С10", "С12", "К2"], "К2", nlu="failing",
         expect_handoff={"question": "аліменти"},
         expected_text="Повідомлення про збій і кнопки напрямів; другий збій – передача адвокату, без циклу"),
    Case("TC-17", "Термінова ситуація вільним текстом, справжній класифікатор", "гібридний", "–",
         [T("зараз іде обшук, потрібен адвокат"), T(PHONE_OK), B("agree")],
         ["С1", "С13", "С8", "С10", "С12", "К3"], "К3", nlu="model",
         expect_handoff={"urgent": "з тексту повідомлення"},
         expected_text="Намір \"терміново\": одразу телефон, звернення з позначкою (К3)"),
    Case("TC-18", "Запис вільним текстом, справжній класифікатор", "гібридний", "–",
         [T("хочу записатися на консультацію")],
         ["С1", "С13", "С7"], "С7", nlu="model",
         expected_text="Намір \"заявка\": бот запитує ім’я"),
    Case("TC-19", "Прохання живої людини, справжній класифікатор", "гібридний", "–",
         [T("не хочу говорити з ботом, з'єднайте з адвокатом")],
         ["С1", "С13", "С8"], "С8", nlu="model",
         expected_text="Намір \"людина\": бот запитує телефон для передачі адвокату"),
    Case("TC-20", "Некоректне ім’я", "проблемний", "–",
         [B("apply"), T("123"), T("Олег")],
         ["С1", "С7", "С8"], "С8",
         expected_text="Ім’я без літер не приймається, бот просить ще раз"),
    Case("TC-21", "Загальний запит без напряму", "гібридний", "–",
         [T("потрібен адвокат"), B("area:criminal")],
         ["С1", "С13", "С2", "С3"], "С3",
         stub={"потрібен адвокат": N("service", "high")},
         expected_text="Висока впевненість, але ні напряму, ні питання – список напрямів"),
    Case("TC-22", "Невірно визначений намір", "гібридний", "–",
         [T("питання про розлучення"), B("other_area"), B("area:civil")],
         ["С1", "С13", "С14", "С2", "С3"], "С3",
         stub={"питання про розлучення": N("service", "medium", area="family")},
         expected_text="\"Інший напрям\" повертає до списку напрямів"),
    Case("TC-23", "Уточнення кнопками після \"невідомо\"", "проблемний", "–",
         [T("ммм"), B("area:criminal")],
         ["С1", "С13", "С15", "С3"], "С3",
         stub={"ммм": N("unknown", "low")},
         expected_text="Після першого \"невідомо\" відвідувач обирає напрям кнопкою"),
    Case("TC-24", "\"Ще не визначився\" і вихід із С9 у меню", "альтернативний", "–",
         [B("services"), B("undecided"), T("Олег"), T("abc"), T("xyz"), B("menu")],
         ["С1", "С2", "С7", "С8", "С8", "С9", "С1"], "С1",
         expect_handoff={},
         expected_text="Заявка без напряму; після двох помилок телефону можна вийти в меню"),
    Case("TC-25", "Підтвердження фрагмента сайту", "гібридний", "–",
         [T("а консультація платна"), B("yes")],
         ["С1", "С13", "С14", "С4"], "С4",
         stub={"а консультація платна": N("service", "medium", fragment="cost")},
         expected_text="\"Схоже, ви питаєте: \"Вартість\". Так?\" → фрагмент про вартість"),
    Case("TC-26", "Підтвердження питання без фрагмента", "гібридний", "–",
         [T("а у вихідні працюєте"), B("yes")],
         ["С1", "С13", "С14", "С5"], "С5",
         stub={"а у вихідні працюєте": N("service", "medium", q=True)},
         expected_text="Після \"Так\" – чесне \"На сайті немає відповіді…\" і пропозиція заявки"),
    Case("TC-27", "Підтвердження наміру \"заявка\" і вихід з імені", "гібридний", "–",
         [T("запишіть мене"), B("yes"), B("menu"), B("continue_app"), T("Олег")],
         ["С1", "С13", "С14", "С7", "С16", "С7", "С8"], "С8",
         stub={"запишіть мене": N("apply", "medium")},
         expected_text="Після \"Так\" бот запитує ім’я; вихід у меню перепитує, \"Продовжити заявку\" повертає"),
    Case("TC-28", "Підтвердження наміру \"людина\"", "гібридний", "–",
         [T("хочу з людиною"), B("yes")],
         ["С1", "С13", "С14", "С8"], "С8",
         stub={"хочу з людиною": N("human", "medium")},
         expected_text="Після \"Так\" бот запитує телефон для передачі адвокату"),
    Case("TC-29", "Зв’язок з адвокатом з уточнення", "проблемний", "–",
         [T("ммм"), B("lawyer")],
         ["С1", "С13", "С15", "С8"], "С8",
         stub={"ммм": N("unknown", "low")},
         expected_text="З уточнення кнопками можна одразу перейти до адвоката"),
]


def _classifier(case: Case):
    if case.nlu == "stub":
        return ScriptedClassifier(case.stub)
    if case.nlu == "failing":
        return FailingClassifier()
    from .nlu.model import IntentClassifier

    return IntentClassifier.load()


def run_case(case: Case) -> dict:
    """Проганяє кейс і повертає фактичний результат і статус перевірки."""
    nlu = _classifier(case)
    inbox = Inbox()
    bot = Controller(nlu, inbox)
    session = Session()
    reply = bot.handle(session, {"type": "start"})
    transcript = [{"bot": reply.messages, "buttons": [b[1] for b in reply.buttons], "state": reply.state}]
    inputs = []
    for step in case.steps:
        if step.get("if_state") and reply.state != step["if_state"]:
            continue
        if step["type"] == "button":
            labels = dict(reply.buttons)
            inputs.append(f"[{labels.get(step['id'], step['id'])}]")
        elif step["type"] == "text":
            inputs.append(f"\"{step['text']}\"")
        else:
            inputs.append("(10 хв без дій)")
        reply = bot.handle(session, step)
        transcript.append({"user": inputs[-1], "bot": reply.messages, "buttons": [b[1] for b in reply.buttons],
                           "state": reply.state, "nlu": reply.nlu})

    path = [step[1] for step in session.trace]
    problems = []
    if case.nlu == "model":
        it = iter(path)
        if not all(state in it for state in case.expect_path):
            problems.append("шлях не містить очікуваних станів")
    elif path != case.expect_path:
        problems.append("шлях відрізняється від очікуваного")
    if reply.state != case.expect_end:
        problems.append(f"кінцевий стан {reply.state} замість {case.expect_end}")
    handoff = inbox.handoffs[-1] if inbox.handoffs else None
    if case.expect_handoff == {} and handoff:
        problems.append("звернення передано, хоча не мало")
    if case.expect_handoff:
        if not handoff:
            problems.append("звернення не передано")
        else:
            for key, value in case.expect_handoff.items():
                if handoff.get(key) != value:
                    problems.append(f"у зверненні {key}={handoff.get(key)!r} замість {value!r}")
    if case.expect_calls is not None:
        calls = [(text, ctx.state, ctx.area) for text, ctx in nlu.calls]
        if calls != case.expect_calls:
            problems.append(f"виклики класифікатора {calls} замість {case.expect_calls}")
    if case.expect_unrecognized is not None and len(inbox.unrecognized) != case.expect_unrecognized:
        problems.append(f"у журналі нерозпізнаних {len(inbox.unrecognized)} фраз замість {case.expect_unrecognized}")

    return {
        "id": case.id,
        "title": case.title,
        "kind": case.kind,
        "scenario": case.scenario,
        "nlu": case.nlu,
        "inputs": inputs,
        "expected": case.expected_text,
        "expected_path": case.expect_path,
        "actual_path": path,
        "end": reply.state,
        "handoff": handoff,
        "transcript": transcript,
        "trace": session.trace,
        "problems": problems,
        "status": "пройдено" if not problems else "не пройдено",
    }
