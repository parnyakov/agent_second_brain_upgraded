"""Живая карточка прогресса (agent-infra-backlog item 33, шаг 4).

Пиннит ровно те свойства, которые владелец описал как требования: только
осмысленные события становятся статусами, всё остальное прячется,
дедупликация и троттлинг в четыре секунды, ленивое создание, «продолжаю»
после тишины, удаление в конце — и, отдельно, что сбой карточки не может
стоить доставки ответа.
"""

import asyncio
import json

import pytest

from d_brain.services import progress

# ── заглушки ──────────────────────────────────────────────────────────────


class FakeMessage:
    def __init__(self, message_id: int) -> None:
        self.message_id = message_id


class FakeBot:
    """Считает вызовы Telegram и умеет падать на любом из них."""

    def __init__(self, *, fail_send=False, fail_edit=False, fail_delete=False):
        self.sent: list[tuple[int, str, dict]] = []
        self.edits: list[tuple[int, int, str]] = []
        self.deleted: list[tuple[int, int]] = []
        self.fail_send = fail_send
        self.fail_edit = fail_edit
        self.fail_delete = fail_delete
        self._next_id = 100

    async def send_message(self, chat_id, text, **kwargs):
        if self.fail_send:
            raise RuntimeError("telegram down")
        self.sent.append((chat_id, text, kwargs))
        self._next_id += 1
        return FakeMessage(self._next_id)

    async def edit_message_text(self, text=None, chat_id=None, message_id=None, **kw):
        if self.fail_edit:
            raise RuntimeError("telegram down")
        self.edits.append((chat_id, message_id, text))

    async def delete_message(self, chat_id, message_id):
        if self.fail_delete:
            raise RuntimeError("telegram down")
        self.deleted.append((chat_id, message_id))

    async def send_chat_action(self, chat_id, action):
        pass


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def assistant(*blocks, sidechain=False, model="claude-opus-5"):
    return {
        "type": "assistant",
        "isSidechain": sidechain,
        "message": {"model": model, "content": list(blocks)},
    }


def text_block(text):
    return {"type": "text", "text": text}


def tool_block(name, **inp):
    return {"type": "tool_use", "name": name, "input": inp}


def write(path, *records):
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@pytest.fixture
def transcript(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text("")
    return path


def make_card(bot, transcript, clock, **kwargs):
    return progress.ProgressCard(
        bot, 7, progress.ProgressFeed(transcript), clock=clock, **kwargs
    )


def open_card(card, clock, *, step=3.0):
    """Довести карточку до создания сообщения.

    Два тика, потому что отсчёт ленивого создания идёт от ПЕРВОЙ записи в
    стенограмме, а не от создания объекта: первый тик видит записи и
    засекает начало хода, второй — уже за порогом — рисует карточку."""
    asyncio.run(card.tick())
    clock.advance(step)
    asyncio.run(card.tick())


# ── ярлыки ────────────────────────────────────────────────────────────────


def test_tool_labels_are_human_and_russian():
    assert progress.tool_label("Read", {"file_path": "/a/b/transcript.py"}) == (
        "читаю файл transcript.py"
    )
    assert progress.tool_label("Bash", {"description": "Запускаю тесты"}) == (
        "запускаю команду: Запускаю тесты"
    )
    assert progress.tool_label("WebSearch", {"query": "погода"}) == "ищу в вебе: погода"
    assert progress.tool_label("Task", {}) == "подзадача агенту"
    assert progress.tool_label("Grep", {"pattern": "x"}) == "ищу по файлам"


def test_unknown_and_mcp_tools_get_a_neutral_label():
    label = progress.tool_label("mcp__slack__post", {"text": "x"})
    assert label == "внешний инструмент"
    assert progress.tool_label("BrandNewTool", {}) == "работаю: BrandNewTool"
    # Мусор вместо имени не должен ронять карточку.
    assert progress.tool_label(None, None) == "работаю"


def test_raw_bash_command_is_never_shown():
    """«Длинные сырые команды» прячем: в ярлык попадает только описание."""
    label = progress.tool_label(
        "Bash",
        {
            "command": "curl -s https://secret.example/api | jq '.token' > /tmp/t",
            "description": "Тяну статус",
        },
    )
    assert "curl" not in label and "secret" not in label
    assert label == "запускаю команду: Тяну статус"


def test_first_line_is_truncated_and_stops_at_the_reply_marker():
    long = "я" * 300
    assert len(progress.first_visible_line(long)) == progress.MAX_STATUS_LEN
    assert progress.first_visible_line("Сейчас посмотрю.\nВторая строка") == (
        "Сейчас посмотрю."
    )
    # Всё, что после открывающего маркера, — это сам ответ.
    assert progress.first_visible_line("Готово.\n<<<R:ab12>>>\nВот итог") == "Готово."
    assert progress.first_visible_line("<<<R:ab12>>>\nВот итог") is None


# ── что прячется ──────────────────────────────────────────────────────────


def test_hidden_events_produce_no_status():
    """Результаты инструментов, содержимое файлов, вывод команд, текст
    субагента и служебные записи не становятся статусами."""
    hidden = [
        # tool_result: содержимое прочитанного файла / вывод команды
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "content": "def secret():\n    return 42\n" * 50,
                    }
                ]
            },
        },
        # текст фонового субагента
        assistant(text_block("Я субагент и я что-то нашёл"), sidechain=True),
        # служебная запись самого Claude Code
        assistant(text_block("Context compacted"), model="<synthetic>"),
        # размышления
        assistant({"type": "thinking", "thinking": "надо бы проверить пароль"}),
        {"type": "summary", "summary": "что-то"},
        "не словарь вовсе",
    ]
    for rec in hidden:
        assert progress.statuses_from_record(rec) == [], rec


def test_the_answer_itself_never_reaches_the_card(transcript):
    """Регрессия (слепое ревью, находка 1). Claude Code закрывает запись на
    каждом вызове инструмента, поэтому ответ, начавшийся до вызова и
    продолжившийся после, лежит в ДВУХ записях — и во второй маркера
    ``<<<R:id>>>`` уже нет. Обрезки по маркеру внутри одной записи не
    хватает: продолжение ответа уезжало в карточку."""
    feed = progress.ProgressFeed(transcript)
    write(transcript, assistant(tool_block("Read", file_path="/b.py")))
    assert feed.poll() == "читаю файл b.py"

    # Ответ начался и прервался на вызов инструмента.
    write(transcript, assistant(text_block("Вот итог.\n<<<R:ab12>>>\nНачало ответа")))
    assert feed.poll() == "Вот итог."

    # Вызов инструмента посреди ответа — его ярлык показывать МОЖНО.
    write(transcript, assistant(tool_block("Read", file_path="/c.py")))
    assert feed.poll() == "читаю файл c.py"

    # А продолжение ответа во ВТОРОЙ записи, уже без маркера, — нельзя.
    write(transcript, assistant(text_block("Продолжение ответа: пароль 12345.")))
    assert feed.poll() is None


def test_the_answer_cannot_leak_through_a_second_text_block(transcript):
    """Регрессия (второе слепое ревью). Та же дыра, но ВНУТРИ одной записи:
    открывающий маркер в первом текстовом блоке, продолжение ответа — во
    втором, где маркера уже нет и отрезать нечего. Лечится склейкой
    текстовых блоков записи перед разбором — так же, как это делает
    `_record_text` в transcript.py для настоящего пути доставки."""
    rec = assistant(
        text_block("Готово.\n<<<R:ab1>>>\nНачало ответа"),
        text_block("Продолжение ответа: пароль 12345."),
    )
    assert progress.statuses_from_record(rec) == ["Готово."]


def test_repeated_identical_labels_still_count_as_alive(transcript):
    """Регрессия (слепое ревью, находка 2). Ярлык «ищу по файлам» одинаков
    для любого Grep. Живость, посчитанная по СМЕНЕ статуса, объявляла бы
    молчанием агента, который три минуты подряд грепает, — то есть карточка
    врала бы ровно в том состоянии, ради которого её и делали."""
    clock, bot = Clock(), FakeBot()
    card = make_card(bot, transcript, clock)
    write(transcript, assistant(tool_block("Grep", pattern="a")))
    open_card(card, clock)
    assert "ищу по файлам" in bot.sent[0][1]

    # Три минуты непрерывного грепа: статус не меняется, но записи идут.
    for pattern in "bcdefghijkl":
        write(transcript, assistant(tool_block("Grep", pattern=pattern)))
        clock.advance(20)
        asyncio.run(card.tick())

    assert bot.edits == [], "работающий агент не должен объявляться молчащим"


def test_hidden_events_never_reach_the_card(transcript):
    clock, bot = Clock(), FakeBot()
    card = make_card(bot, transcript, clock)
    write(
        transcript,
        {
            "type": "user",
            "message": {
                "content": [{"type": "tool_result", "content": "СЕКРЕТНЫЙ ВЫВОД"}]
            },
        },
        assistant(text_block("текст субагента"), sidechain=True),
    )
    open_card(card, clock)
    # Карточка создалась (работа идёт), но ни одно скрытое событие в неё не
    # попало — только нейтральный дефолт.
    assert len(bot.sent) == 1
    shown = bot.sent[0][1]
    assert "СЕКРЕТНЫЙ" not in shown and "субагент" not in shown
    assert progress.DEFAULT_STATUS in shown


# ── ленивое создание ──────────────────────────────────────────────────────


def test_no_card_for_a_fast_reply(transcript):
    """Ход короче пары секунд карточки не получает вовсе."""
    clock, bot = Clock(), FakeBot()
    card = make_card(bot, transcript, clock)
    write(transcript, assistant(tool_block("Read", file_path="/a/b.py")))

    clock.advance(0.5)
    asyncio.run(card.tick())
    clock.advance(1.0)  # суммарно 1.5 с — всё ещё меньше MIN_DELAY_SECONDS
    asyncio.run(card.tick())
    asyncio.run(card.close())

    assert bot.sent == [] and bot.edits == [] and bot.deleted == []


def test_lazy_start_is_anchored_on_the_first_record(transcript):
    """Регрессия (слепое ревью, находка 4). Карточка создаётся до отправки
    промпта, а ask() тратит до него пару секунд на плюмбинг пейна. Если
    отсчитывать ленивое создание от создания ОБЪЕКТА, секундный ответ всё
    равно получал бы карточку-мигалку. Отсчёт идёт от первой записи."""
    clock, bot = Clock(), FakeBot()
    card = make_card(bot, transcript, clock)

    # Плюмбинг ask(): объект живёт уже дольше MIN_DELAY, но ход не начался.
    clock.advance(progress.MIN_DELAY_SECONDS + 1)
    asyncio.run(card.tick())
    assert bot.sent == []

    # Ход пошёл — с этого момента и начинается отсчёт.
    write(transcript, assistant(tool_block("Read", file_path="/a.py")))
    asyncio.run(card.tick())
    assert bot.sent == [], "отсчёт должен начаться заново, от первой записи"

    clock.advance(progress.MIN_DELAY_SECONDS)
    asyncio.run(card.tick())
    assert len(bot.sent) == 1


def test_first_message_is_silent(transcript):
    clock, bot = Clock(), FakeBot()
    card = make_card(bot, transcript, clock)
    write(transcript, assistant(tool_block("Read", file_path="/a/b.py")))
    open_card(card, clock)

    assert len(bot.sent) == 1
    assert bot.sent[0][2].get("disable_notification") is True
    assert "читаю файл b.py" in bot.sent[0][1]


# ── дедупликация и троттлинг ──────────────────────────────────────────────


def test_identical_statuses_are_deduplicated(transcript):
    clock, bot = Clock(), FakeBot()
    card = make_card(bot, transcript, clock)
    write(transcript, assistant(tool_block("Read", file_path="/a/b.py")))
    open_card(card, clock)
    assert len(bot.sent) == 1

    # Тот же самый инструмент с тем же файлом ещё дважды, с запасом по
    # времени — сообщение не трогаем вообще.
    for _ in range(2):
        write(transcript, assistant(tool_block("Read", file_path="/a/b.py")))
        clock.advance(10)
        asyncio.run(card.tick())
    assert bot.edits == []


def test_updates_are_throttled_to_one_per_four_seconds(transcript):
    clock, bot = Clock(), FakeBot()
    card = make_card(bot, transcript, clock)
    write(transcript, assistant(tool_block("Read", file_path="/a.py")))
    open_card(card, clock)
    assert len(bot.sent) == 1 and bot.edits == []

    # Три разных статуса в пределах окна троттлинга → ни одной правки.
    for name in ("b.py", "c.py", "d.py"):
        write(transcript, assistant(tool_block("Read", file_path=f"/{name}")))
        clock.advance(1)
        asyncio.run(card.tick())
    assert bot.edits == []

    # Окно прошло — уходит ОДНА правка, и это ПОСЛЕДНИЙ статус: статус
    # заменяется, а не копится.
    clock.advance(progress.THROTTLE_SECONDS)
    asyncio.run(card.tick())
    assert len(bot.edits) == 1
    assert "читаю файл d.py" in bot.edits[0][2]
    assert "b.py" not in bot.edits[0][2] and "c.py" not in bot.edits[0][2]


def test_card_follows_only_this_turn(transcript):
    """Курсор встаёт на конец файла: события прошлого хода не всплывают.

    А раз своих записей у хода ещё нет, до COLD_START_SECONDS карточки не
    будет вовсе — молчание честнее, чем чужой статус."""
    write(transcript, assistant(tool_block("Read", file_path="/старое.py")))
    clock, bot = Clock(), FakeBot()
    card = make_card(bot, transcript, clock)
    clock.advance(5)
    asyncio.run(card.tick())
    assert bot.sent == []

    # Ход молчит дольше порога холодного старта — тогда карточка нужна, но
    # с нейтральным «думаю», а не с событием чужого хода.
    clock.advance(progress.COLD_START_SECONDS)
    asyncio.run(card.tick())
    assert len(bot.sent) == 1
    assert "старое" not in bot.sent[0][1]
    assert progress.DEFAULT_STATUS in bot.sent[0][1]


# ── «продолжаю» после тишины ──────────────────────────────────────────────


def test_quiet_stream_gets_a_keepalive_line(transcript):
    clock, bot = Clock(), FakeBot()
    card = make_card(bot, transcript, clock)
    write(transcript, assistant(tool_block("Bash", description="Гоню тесты")))
    open_card(card, clock)
    assert "Гоню тесты" in bot.sent[0][1]

    # Полторы минуты тишины — ничего не меняется.
    clock.advance(90)
    asyncio.run(card.tick())
    assert bot.edits == []

    # За двумя минутами появляется «продолжаю» со счётчиком минут.
    clock.advance(40)
    asyncio.run(card.tick())
    assert len(bot.edits) == 1
    assert "продолжаю — 2 мин" in bot.edits[0][2]

    # Счётчик растёт, но не дёргает сообщение каждую секунду.
    clock.advance(1)
    asyncio.run(card.tick())
    assert len(bot.edits) == 1
    clock.advance(60)
    asyncio.run(card.tick())
    assert "продолжаю — 3 мин" in bot.edits[-1][2]


def test_quiet_status_wording():
    # Достижимые входы: строка появляется не раньше QUIET_SECONDS.
    assert progress.quiet_status(progress.QUIET_SECONDS) == "продолжаю — 2 мин"
    assert progress.quiet_status(125) == "продолжаю — 2 мин"
    assert progress.quiet_status(605) == "продолжаю — 10 мин"


# ── конец хода ────────────────────────────────────────────────────────────


def test_card_is_deleted_at_the_end(transcript):
    clock, bot = Clock(), FakeBot()
    card = make_card(bot, transcript, clock)
    write(transcript, assistant(tool_block("Read", file_path="/a.py")))
    open_card(card, clock)
    message_id = card.message_id
    assert message_id is not None

    asyncio.run(card.close())
    assert bot.deleted == [(7, message_id)]
    # Повторный close ничего не делает — страховочный вызов в finally.
    asyncio.run(card.close())
    assert len(bot.deleted) == 1


def test_a_late_tick_after_close_cannot_orphan_a_second_card(transcript):
    """Регрессия. Отмена фоновой задачи доставляется только на ближайшей
    точке ожидания, так что один тик может успеть пройти ПОСЛЕ close(). Если
    к этому моменту в стенограмме появилось новое событие, карточка
    создавалась заново — и вторую, осиротевшую, было уже некому удалить."""
    clock, bot = Clock(), FakeBot()
    card = make_card(bot, transcript, clock)
    write(transcript, assistant(tool_block("Read", file_path="/a.py")))
    open_card(card, clock)
    asyncio.run(card.close())

    write(transcript, assistant(tool_block("Read", file_path="/b.py")))
    clock.advance(15)
    asyncio.run(card.tick())

    assert len(bot.sent) == 1, "вторая карточка осиротела бы в чате"
    assert len(bot.deleted) == 1


# ── сбой карточки ─────────────────────────────────────────────────────────


def test_a_card_that_cannot_be_created_is_given_up_on(transcript):
    clock, bot = Clock(), FakeBot(fail_send=True)
    card = make_card(bot, transcript, clock)
    for _ in range(5):
        write(transcript, assistant(tool_block("Read", file_path="/a.py")))
        clock.advance(10)
        asyncio.run(card.tick())
    assert bot.sent == [] and card.message_id is None
    asyncio.run(card.close())  # не бросает
    assert bot.deleted == []


def test_edit_and_delete_failures_never_raise(transcript):
    clock = Clock()
    bot = FakeBot(fail_edit=True, fail_delete=True)
    card = make_card(bot, transcript, clock)
    write(transcript, assistant(tool_block("Read", file_path="/a.py")))
    clock.advance(5)
    asyncio.run(card.tick())
    write(transcript, assistant(tool_block("Read", file_path="/b.py")))
    clock.advance(10)
    asyncio.run(card.tick())  # правка падает — молча
    asyncio.run(card.close())  # удаление падает — молча
    assert bot.edits == [] and bot.deleted == []


def test_a_failed_edit_does_not_mark_the_text_as_shown(transcript):
    """Регрессия (слепое ревью, находка 10). Если правка не прошла, в чате
    висит СТАРЫЙ текст. Считая новый показанным, дедупликация блокировала бы
    исправляющую перерисовку до самой смены статуса."""
    clock = Clock()
    bot = FakeBot(fail_edit=True)
    card = make_card(bot, transcript, clock)
    write(transcript, assistant(tool_block("Read", file_path="/a.py")))
    open_card(card, clock)

    write(transcript, assistant(tool_block("Read", file_path="/b.py")))
    clock.advance(10)
    asyncio.run(card.tick())  # правка падает
    assert bot.edits == []

    # Правки снова проходят — тот же самый статус должен уйти повторно,
    # а не быть проглочен дедупликацией.
    bot.fail_edit = False
    clock.advance(10)
    asyncio.run(card.tick())
    assert len(bot.edits) == 1
    assert "читаю файл b.py" in bot.edits[0][2]


def test_a_broken_transcript_line_does_not_stop_the_feed(transcript):
    feed = progress.ProgressFeed(transcript)
    with transcript.open("a", encoding="utf-8") as f:
        f.write("{это не json\n")
    write(transcript, assistant(tool_block("Read", file_path="/a.py")))
    assert feed.poll() == "читаю файл a.py"


# ── интеграция с обработчиком чата ────────────────────────────────────────


class FakeManager:
    def __init__(self, transcript=None, reply="готово", on_ask=None):
        self.transcript = transcript
        self.reply = reply
        self.on_ask = on_ask
        self.sent: list[tuple[int, str]] = []

    def progress_transcript(self):
        return self.transcript

    async def send_message(self, user_id, prompt):
        self.sent.append((user_id, prompt))
        if self.on_ask is not None:
            await self.on_ask()
        return self.reply

    def is_turn_active(self) -> bool:
        return False


def test_progress_transcript_gates(transcript, tmp_path):
    """``progress_transcript`` — единственный шлагбаум карточки. Каждый из
    четырёх отказов штатный, и ни один не бросает."""
    import types

    from d_brain.services.chat_session import ChatSessionManager

    def manager(session, engine="claude"):
        mgr = ChatSessionManager(tmp_path, session=session)
        mgr._settings = types.SimpleNamespace(chat_engine=engine)
        return mgr

    class Have:
        def current_transcript_path(self):
            return transcript

    class Missing:
        def current_transcript_path(self):
            return tmp_path / "нет-такого.jsonl"

    class NoSession:
        def current_transcript_path(self):
            return None

    class Boom:
        def current_transcript_path(self):
            raise RuntimeError("boom")

    assert manager(Have()).progress_transcript() == transcript
    # Движок Codex не трогаем: его rollout-JSONL другой схемы.
    assert manager(Have(), engine="codex").progress_transcript() is None
    # Несуществующий файл: TranscriptTail встал бы на смещение 0 и проиграл
    # бы ВСЮ историю файла как «события этого хода».
    assert manager(Missing()).progress_transcript() is None
    assert manager(NoSession()).progress_transcript() is None
    assert manager(Boom()).progress_transcript() is None


def test_handler_skips_the_card_when_there_is_no_transcript(monkeypatch):
    """Под Codex и до первого запуска сессии progress_transcript() отдаёт
    None — карточки нет, ответ приходит как обычно."""
    from d_brain.bot.handlers import chat

    mgr = FakeManager(transcript=None)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._process_and_reply(bot, chat_id=7, user_id=1, prompt="привет"))

    assert bot.edits == [] and bot.deleted == []
    assert [text for _, text, _ in bot.sent] == ["готово"]


def test_handler_delivers_the_reply_even_if_the_card_explodes(monkeypatch, transcript):
    """Сбой карточки не имеет права уронить доставку ответа."""
    from d_brain.bot.handlers import chat

    class ExplodingCard:
        message_id = 1
        ticks = 0

        async def tick(self):
            type(self).ticks += 1
            raise RuntimeError("boom")

        async def close(self):
            raise RuntimeError("boom")

    async def slow_turn():
        await asyncio.sleep(0.1)

    mgr = FakeManager(transcript=transcript, on_ask=slow_turn)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    monkeypatch.setattr(progress, "POLL_SECONDS", 0.01)
    # Настоящая фоновая задача, а не None: иначе падающий tick() не
    # вызывается ни разу и «сбой карточки» проверен только наполовину.
    monkeypatch.setattr(
        chat,
        "_start_progress_card",
        lambda bot, chat_id, manager: (
            ExplodingCard(),
            asyncio.ensure_future(chat._progress_loop(ExplodingCard())),
        ),
    )
    bot = FakeBot()
    asyncio.run(chat._process_and_reply(bot, chat_id=7, user_id=1, prompt="привет"))

    assert ExplodingCard.ticks > 0, "падающий tick() должен был реально вызваться"
    assert [text for _, text, _ in bot.sent] == ["готово"]


def test_the_retry_turn_gets_its_own_card(monkeypatch, transcript):
    """Регрессия (слепое ревью, находка 5). Повторный ход после пустого
    ответа — такой же полноценный ход, который может идти минутами."""
    from d_brain.bot.handlers import chat

    started: list[int] = []
    monkeypatch.setattr(
        chat, "_start_progress_card", lambda *a: (started.append(1), (None, None))[1]
    )
    mgr = FakeManager(transcript=transcript, reply="")
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._process_and_reply(bot, chat_id=7, user_id=1, prompt="привет"))

    assert len(mgr.sent) == 2, "должно было быть две попытки"
    assert len(started) == 2, "у повторного хода должна быть своя карточка"


def test_a_hanging_close_cannot_hold_up_the_reply(monkeypatch, transcript):
    """Регрессия (слепое ревью, находка 6). delete_message — полный
    round-trip к Telegram, и он стоит ПЕРЕД отправкой ответа; у aiogram
    дефолтный таймаут запроса около минуты."""
    from d_brain.bot.handlers import chat

    class HangingCard:
        async def tick(self):
            pass

        async def close(self):
            await asyncio.sleep(3600)

    mgr = FakeManager(transcript=transcript)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    monkeypatch.setattr(chat, "_start_progress_card", lambda *a: (HangingCard(), None))
    monkeypatch.setattr(chat, "_CARD_CLOSE_TIMEOUT", 0.05)
    bot = FakeBot()

    async def run():
        await asyncio.wait_for(
            chat._process_and_reply(bot, chat_id=7, user_id=1, prompt="привет"),
            timeout=5,
        )

    asyncio.run(run())
    assert [text for _, text, _ in bot.sent] == ["готово"]


def test_the_card_is_removed_before_an_error_message(monkeypatch, transcript):
    """Регрессия (слепое ревью, находка 9). Инвариант «последнее сообщение в
    чате — ответ, а не карточка» должен держаться и на аварийном пути."""
    from d_brain.bot.handlers import chat

    order: list[str] = []

    class Card:
        async def tick(self):
            pass

        async def close(self):
            order.append("карточка снята")

    class Boom(FakeManager):
        async def send_message(self, user_id, prompt):
            raise RuntimeError("ход развалился")

    monkeypatch.setattr(chat, "_get_manager", lambda: Boom(transcript=transcript))
    monkeypatch.setattr(chat, "_start_progress_card", lambda *a: (Card(), None))

    class RecordingBot(FakeBot):
        async def send_message(self, chat_id, text, **kwargs):
            order.append("сообщение: " + text[:20])
            return await super().send_message(chat_id, text, **kwargs)

    bot = RecordingBot()
    asyncio.run(chat._process_and_reply(bot, chat_id=7, user_id=1, prompt="привет"))

    assert order[0] == "карточка снята", order


def test_stopping_a_card_task_that_never_ran_does_not_leak_cancellation(transcript):
    """Регрессия. ``CancelledError`` наследует ``BaseException``, поэтому
    ``suppress(Exception)`` его не ловит: задача, отменённая до первого
    запуска, роняла ``_stop_progress_card`` — а он вызывается ПЕРЕД отправкой
    ответа, то есть унёс бы с собой ответ."""
    from d_brain.bot.handlers import chat

    async def scenario():
        card = make_card(FakeBot(), transcript, Clock())
        task = asyncio.create_task(chat._progress_loop(card))
        task.cancel()  # отменяем до того, как цикл вообще стартовал
        await chat._stop_progress_card(card, task)

    asyncio.run(scenario())  # не должно бросить


def test_handler_removes_the_card_before_the_reply(monkeypatch, transcript):
    """Карточка снимается ДО ответа, и ответ приходит отдельным сообщением."""
    from d_brain.bot.handlers import chat

    async def slow_turn():
        write(transcript, assistant(tool_block("Read", file_path="/a.py")))
        # Достаточно долго, чтобы фоновая задача успела несколько тиков.
        await asyncio.sleep(0.15)

    mgr = FakeManager(transcript=transcript, on_ask=slow_turn)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    monkeypatch.setattr(progress, "MIN_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(progress, "POLL_SECONDS", 0.01)
    bot = FakeBot()
    asyncio.run(chat._process_and_reply(bot, chat_id=7, user_id=1, prompt="привет"))

    card_messages = [i for i, (_, t, _) in enumerate(bot.sent) if "⏳" in t]
    assert card_messages, "карточка должна была появиться"
    assert bot.deleted, "карточка должна быть удалена"
    # Ответ — последнее сообщение, и он не склеен с карточкой.
    assert bot.sent[-1][1] == "готово"
