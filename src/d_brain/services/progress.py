"""Живая карточка прогресса — одно сообщение в Telegram, которое
редактируется на месте, пока идёт ход.

Зачем. Человек отправил сообщение и видит тишину:
непонятно, живой бот или завис. Индикатор «печатает» этого не решает — он
одинаковый и для двух секунд, и для двадцати минут. Карточка показывает,
что именно происходит прямо сейчас: «читаю файл», «запускаю команду»,
«подзадача агенту».

Откуда события. НЕ с экрана. Тот же JSONL-стенограф сессии, который с
22.09 является источником самого ответа (``transcript.py``, ``ReplyTail``),
уже дописывается по ходу хода — значит, это наш готовый поток событий, без
второго механизма чтения. Карточка открывает СВОЙ :class:`TranscriptTail`
на том же файле: читать один файл двумя курсорами дешевле и безопаснее, чем
разветвлять горячий путь доставки ответа ради вспомогательной функции.

Что из записи становится статусом (и что не становится) — см.
:func:`statuses_from_record`. Коротко: только текст ассистента и вызовы
инструментов; результаты инструментов, содержимое файлов, вывод команд и
сырые командные строки не показываются никогда.

Граница ответственности. Карточка — вспомогательная и заведомо теряемая:
она НЕ проходит через долговечную очередь ``outbox.py`` (её потеря ничего
не стоит, а забивать очередь перерисовками раз в четыре секунды — прямой
вред основному каналу). Из этого следует главное правило модуля: **ни одна
ошибка здесь не имеет права дойти до доставки ответа.** Каждый вызов к
Telegram обёрнут, неудачное создание карточки выключает её насовсем, а
:meth:`ProgressCard.tick` и :meth:`ProgressCard.close` не бросают вообще
ничего.

Движок Codex сюда не заведён: его rollout-JSONL — другая схема, и
``chat_session`` просто не отдаёт для него путь (см. ``progress_transcript``).
"""

from __future__ import annotations

import html
import logging
import os
import re
from pathlib import Path
from typing import Any

from d_brain.services.transcript import TranscriptTail

logger = logging.getLogger(__name__)

# Первая строка текста ассистента обрезается до этого (образец владельца —
# «до 120 символов»). Длинный абзац в статусе превращает карточку в
# нечитаемую стену, которая ещё и перерисовывается каждые четыре секунды.
MAX_STATUS_LEN = 120

# Карточка создаётся лениво: для мгновенного ответа она успеет только
# мигнуть и удалиться, что хуже, чем её отсутствие. Отсчёт идёт от первой
# записи в стенограмме, а не от создания объекта — см. ProgressCard._tick.
MIN_DELAY_SECONDS = 2.0
# Сколько ждать ПЕРВОЙ записи, прежде чем показать карточку всё равно. Так
# бывает, когда модель долго думает перед первым словом (записей ещё нет, а
# человек уже ждёт) и когда стенограмму по какой-то причине не видно вовсе.
# Ход короче этого порога карточки не заслуживает в любом случае.
COLD_START_SECONDS = 10.0
# Не перерисовывать чаще раза в 4 секунды. Telegram ограничивает частоту
# правок одного сообщения, а человек всё равно не читает быстрее.
THROTTLE_SECONDS = 4.0
# Тишина дольше двух минут — строка «продолжаю» со счётчиком. Ровно то, что
# она сообщает: событий нет, процесс жив.
QUIET_SECONDS = 120.0
# Как часто спрашивать стенограмму. Дешёвое инкрементальное чтение хвоста.
POLL_SECONDS = 1.0

# Пока модель ещё ничего не сказала и не позвала инструмент, но работа уже
# идёт дольше MIN_DELAY_SECONDS.
DEFAULT_STATUS = "думаю"

# Маркеры доставки ответа (``<<<R:id>>>`` / ``<<<E:id>>>``). Всё, что после
# открывающего маркера, — это сам ответ; он уйдёт отдельным сообщением и в
# карточке ему делать нечего.
_MARKER_RE = re.compile(r"<<<[RE]:\w+>>>")


# ── ярлыки инструментов ──────────────────────────────────────────────────
# Человеческие, спокойные, по-русски. Ключ — имя инструмента как его пишет
# Claude Code в ``tool_use.name``.
_TOOL_LABELS: dict[str, str] = {
    "Read": "читаю файл",
    "Write": "пишу файл",
    "Edit": "правлю файл",
    "MultiEdit": "правлю файл",
    "NotebookEdit": "правлю блокнот",
    "Bash": "запускаю команду",
    "BashOutput": "смотрю фоновую команду",
    "KillShell": "останавливаю фоновую команду",
    "Glob": "ищу файлы",
    "Grep": "ищу по файлам",
    "WebSearch": "ищу в вебе",
    "WebFetch": "открываю страницу",
    "Task": "подзадача агенту",
    "Agent": "подзадача агенту",
    "TodoWrite": "обновляю план",
    "Skill": "запускаю навык",
    "SlashCommand": "выполняю команду",
    "ExitPlanMode": "показываю план",
    "AskUserQuestion": "уточняю вопрос",
    "Artifact": "собираю страницу",
    "ToolSearch": "подбираю инструмент",
}

# Имена инструментов, к ярлыку которых дописывается имя файла из аргумента.
_FILE_ARG_TOOLS = {
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "MultiEdit": "file_path",
    "NotebookEdit": "notebook_path",
}

# Максимум для «хвоста» ярлыка (имя файла, описание команды). Короче, чем
# MAX_STATUS_LEN: ярлык должен читаться с одного взгляда.
_DETAIL_LEN = 48


def truncate(text: str, limit: int) -> str:
    """Обрезка по длине с многоточием. Пустую строку возвращает как есть."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def tool_label(name: Any, tool_input: Any = None) -> str:
    """Человеческий ярлык для одного вызова инструмента.

    Сырые аргументы сюда НЕ попадают: из ``Bash`` берётся только
    человекочитаемое ``description`` (никогда ``command`` — «длинные сырые
    команды» прячем по условию задачи), из файловых инструментов — только
    базовое имя файла, без пути и без содержимого.

    Незнакомый инструмент (включая любой ``mcp__*``) получает нейтральный
    ярлык, а не падение: набор инструментов меняется от релиза к релизу, и
    карточка не должна об этом знать.
    """
    if not isinstance(name, str) or not name.strip():
        return "работаю"
    name = name.strip()
    args = tool_input if isinstance(tool_input, dict) else {}

    base = _TOOL_LABELS.get(name)
    if base is None:
        if name.startswith("mcp__"):
            return "внешний инструмент"
        return f"работаю: {truncate(name, _DETAIL_LEN)}"

    arg_key = _FILE_ARG_TOOLS.get(name)
    if arg_key:
        raw = args.get(arg_key)
        if isinstance(raw, str) and raw.strip():
            return f"{base} {truncate(os.path.basename(raw.strip()), _DETAIL_LEN)}"
        return base

    if name in ("Bash", "Task", "Agent", "BashOutput"):
        # ``description`` — то, что модель написала ДЛЯ ЧЕЛОВЕКА; именно его
        # и показываем. Его отсутствие — обычное дело, тогда голый ярлык.
        desc = args.get("description")
        if isinstance(desc, str) and desc.strip():
            return f"{base}: {truncate(desc, _DETAIL_LEN)}"
        return base

    if name == "Skill":
        skill = args.get("skill")
        if isinstance(skill, str) and skill.strip():
            return f"{base} {truncate(skill, _DETAIL_LEN)}"
        return base

    if name == "WebSearch":
        query = args.get("query")
        if isinstance(query, str) and query.strip():
            return f"{base}: {truncate(query, _DETAIL_LEN)}"
        return base

    return base


def first_visible_line(text: Any) -> str | None:
    """Первая осмысленная строка текста ассистента, обрезанная до
    :data:`MAX_STATUS_LEN`, или ``None``.

    Всё, начиная с первого маркера доставки, отрезается: за ним идёт сам
    ответ, который придёт отдельным сообщением. Показывать его ещё и в
    карточке — значит показать ответ дважды и в обрезанном виде.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    marker = _MARKER_RE.search(text)
    head = text[: marker.start()] if marker else text
    for line in head.splitlines():
        stripped = line.strip()
        if stripped:
            return truncate(stripped, MAX_STATUS_LEN)
    return None


def _record_blocks(rec: Any) -> list[dict] | str | None:
    """Контент записи ассистента, если она вообще может нести статус."""
    if not isinstance(rec, dict) or rec.get("isSidechain"):
        return None
    if rec.get("type") != "assistant":
        return None
    message = rec.get("message")
    if not isinstance(message, dict) or message.get("model") == "<synthetic>":
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return None


def _has_reply_marker(rec: Any) -> bool:
    """Несёт ли запись открывающий маркер ответа ``<<<R:id>>>``."""
    try:
        content = _record_blocks(rec)
        if content is None:
            return False
        if isinstance(content, str):
            return "<<<R:" in content
        return any(
            b.get("type") == "text" and "<<<R:" in str(b.get("text") or "")
            for b in content
        )
    except Exception:  # noqa: BLE001 — дрейф схемы не стоит ни одного хода
        return False


def statuses_from_record(rec: Any, *, tagged: bool = False) -> list:
    """Статусы, которые несёт одна запись стенограммы, в порядке появления.

    С ``tagged=True`` отдаёт пары ``(статус, это_текст_ассистента)`` — вызову
    нужен этот признак, чтобы после начала ответа глушить текст, но не
    ярлыки инструментов (модель может позвать инструмент посреди ответа).

    Показываем ТОЛЬКО:
      * первую строку текстового блока ассистента;
      * вызовы инструментов, переведённые в ярлыки.

    Не показываем НИЧЕГО из остального, и это список того, что здесь
    сознательно отброшено:
      * ``type != "assistant"`` — записи пользователя несут ``tool_result``:
        содержимое прочитанного файла, вывод команды, ответ веб-поиска.
        Именно то, что требуется прятать;
      * ``isSidechain`` — текст фонового субагента. В карточке главного хода
        он выглядел бы как слова самого ассистента (та же причина, по
        которой его отбрасывает ``ReplyTail``);
      * ``model == "<synthetic>"`` — служебные записи самого Claude Code
        (уведомления о компактации и прочее), не то, что сказала модель;
      * блоки ``thinking``/``tool_result`` внутри контента — не текст и не
        вызов.

    Никогда не бросает: схема JSONL не является опубликованным API, и
    неожиданная форма записи должна стоить ноль статусов, а не исключение.
    """
    out: list[tuple[str, bool]] = []
    try:
        content = _record_blocks(rec)
        if content is None:
            pass
        elif isinstance(content, str):
            line = first_visible_line(content)
            if line:
                out.append((line, True))
        else:
            # Текстовые блоки ОДНОЙ записи склеиваются перед разбором — ровно
            # как это делает `_record_text` в transcript.py для настоящего
            # пути доставки. Поблочный разбор дырявый: если открывающий
            # маркер попал в первый блок, а продолжение ответа — во второй,
            # то во втором маркера уже нет и отрезать нечего, и он уезжал в
            # карточку (второе слепое ревью). Защёлка _answer_started в
            # ProgressFeed закрывает только межзаписную половину этой дыры.
            text_parts: list[str] = []
            for block in content:
                kind = block.get("type")
                if kind == "text":
                    raw = block.get("text")
                    if isinstance(raw, str) and raw:
                        text_parts.append(raw)
                elif kind == "tool_use":
                    out.append(
                        (tool_label(block.get("name"), block.get("input")), False)
                    )
            line = first_visible_line("\n".join(text_parts))
            if line:
                # Перед ярлыками: реплика предшествует вызову, а «последний
                # победил» должен доставаться самому свежему событию записи.
                out.insert(0, (line, True))
    except Exception:  # noqa: BLE001 — дрейф схемы не стоит ни одного хода
        logger.debug("progress: skipping a malformed record", exc_info=True)
        return []
    return out if tagged else [status for status, _ in out]


class ProgressFeed:
    """Поток статусов одного хода поверх стенограммы.

    Привязывается к КОНЦУ файла в момент создания — ровно как ``ReplyTail``,
    и по той же причине: события предыдущего хода не должны выдаваться за
    события текущего.

    :meth:`poll` отдаёт НОВЕЙШИЙ статус, появившийся со времени прошлого
    вызова, и ``None``, если нового нет либо он дословно повторяет прошлый
    отданный. Промежуточные статусы внутри одного опроса схлопываются
    намеренно: статус заменяется, а не копится, и показывать «читаю файл
    a.py», которое уже сменилось на «читаю файл b.py», незачем.
    """

    def __init__(self, path: Path | str) -> None:
        self._tail = TranscriptTail.at_end(path)
        self._last: str | None = None
        #: Пришла ли хоть одна запись в ПОСЛЕДНЕМ опросе — «движок жив»,
        #: отдельно от «есть что показать». Держать эти два вопроса врозь
        #: обязательно: ярлыки вроде «ищу по файлам» одинаковы для любого
        #: Grep, и если считать живость по СМЕНЕ статуса, то агент, который
        #: три минуты подряд грепает, выглядел бы замолчавшим — карточка
        #: врала бы ровно в том состоянии, ради которого её и делали
        #: (слепое ревью, находка 2).
        self.had_records = False
        #: Виден ли уже открывающий маркер ответа. С этого момента текст
        #: ассистента в карточку не пускается вообще — см. :meth:`poll`.
        self._answer_started = False

    def poll(self) -> str | None:
        try:
            records = self._tail.poll_new_records()
        except Exception:  # noqa: BLE001 — карточка не ломает ход
            logger.debug("progress: transcript poll failed", exc_info=True)
            self.had_records = False
            return None
        self.had_records = bool(records)
        latest: str | None = None
        for rec in records:
            for status, is_text in statuses_from_record(rec, tagged=True):
                if is_text and self._answer_started:
                    # Обрезки по маркеру внутри одной записи НЕ ХВАТАЕТ:
                    # Claude Code закрывает запись на каждом вызове
                    # инструмента, поэтому ответ, начавшийся до вызова и
                    # продолжившийся после, лежит в ДВУХ записях — и во
                    # второй никакого маркера уже нет (см. докстринг
                    # ReplyTail в transcript.py). Без этой защёлки первые
                    # 120 символов продолжения ответа уезжали в карточку
                    # (слепое ревью, находка 1).
                    continue
                latest = status
            if _has_reply_marker(rec):
                # Ответ начался — но защёлка ставится ПОСЛЕ разбора этой же
                # записи: текст ДО маркера внутри неё — обычная реплика
                # («сейчас посмотрю»), и first_visible_line уже гарантирует,
                # что за маркер она не заглядывает. Глушится всё, что
                # приходит ПОСЛЕ.
                self._answer_started = True
        if latest is None or latest == self._last:
            return None
        self._last = latest
        return latest


def render(status: str) -> str:
    """HTML карточки для одного статуса. Один значок, без мусора.

    ``quote=False`` намеренно: статус подставляется в ТЕЛО тега, а не в
    атрибут, и апостроф из чужого ``description`` иначе уехал бы в чат как
    ``&#x27;`` — Telegram в HTML-режиме документирует только ``&lt;``,
    ``&gt;`` и ``&amp;``."""
    return f"⏳ <i>{html.escape(status, quote=False)}</i>"


def quiet_status(seconds: float) -> str:
    """Строка «продолжаю» со счётчиком минут для затянувшейся тишины."""
    minutes = max(1, int(seconds // 60))
    return f"продолжаю — {minutes} мин"


class ProgressCard:
    """Одно редактируемое сообщение Telegram на время одного хода.

    Использование: создать до отправки промпта, звать :meth:`tick` раз в
    :data:`POLL_SECONDS` из фоновой задачи, в конце — :meth:`close`.

    ``bot`` — любой объект с ``send_message`` / ``edit_message_text`` /
    ``delete_message`` (в тестах это заглушка). ``clock`` — монотонные
    секунды; вынесен параметром, чтобы троттлинг и двухминутная тишина
    проверялись без настоящего ожидания.
    """

    def __init__(
        self,
        bot: Any,
        chat_id: int,
        feed: ProgressFeed,
        *,
        clock: Any,
        min_delay: float | None = None,
        throttle: float | None = None,
        quiet: float | None = None,
    ) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._feed = feed
        self._clock = clock
        # Значения по умолчанию читаются ЗДЕСЬ, а не в сигнатуре: иначе они
        # застынут на момент импорта и подмена модульной константы (тесты,
        # будущая настройка) ни на что не повлияла бы.
        self._min_delay = MIN_DELAY_SECONDS if min_delay is None else min_delay
        self._throttle = THROTTLE_SECONDS if throttle is None else throttle
        self._quiet = QUIET_SECONDS if quiet is None else quiet

        now = clock()
        self._started = now
        self._last_event = now
        self._last_render = 0.0
        self.message_id: int | None = None
        #: Текст, который сейчас реально висит в сообщении.
        self._shown: str | None = None
        #: Последний пришедший статус, ещё не показанный (ждёт троттлинга).
        self._pending: str | None = None
        #: Карточку выключили насовсем — создать сообщение не удалось.
        self._disabled = False
        #: Стенограмма отдала хоть одну запись, то есть ход реально пошёл.
        self._live = False

    # ── один шаг ──────────────────────────────────────────────────────
    async def tick(self) -> None:
        """Прочитать новые события и, если пора, перерисовать карточку.

        Не бросает ничего и ни при каких обстоятельствах: карточка —
        вспомогательная, а звать её будут из задачи, живущей рядом с
        доставкой ответа.
        """
        try:
            await self._tick()
        except Exception:  # noqa: BLE001 — см. докстринг модуля
            logger.debug("progress: tick failed", exc_info=True)

    async def _tick(self) -> None:
        if self._disabled:
            return
        now = self._clock()
        status = self._feed.poll()
        if self._feed.had_records:
            # Живость считается по ЛЮБОЙ пришедшей записи, а не по смене
            # статуса: см. ProgressFeed.had_records.
            self._last_event = now
            if not self._live:
                # Ход начался только сейчас. Отсчёт ленивого создания идёт
                # отсюда, а не с момента создания объекта: между ними ask()
                # тратит пару секунд на плюмбинг пейна, и якорь на создании
                # означал бы карточку-мигалку даже для секундного ответа
                # (слепое ревью, находка 4).
                self._live = True
                self._started = now
        if status is not None:
            self._pending = status

        if not self._live and (now - self._started) < COLD_START_SECONDS:
            # Записей ещё нет вовсе. Дольше COLD_START_SECONDS так бывает,
            # когда модель долго думает перед первым словом — вот тогда
            # карточка нужна («думаю»), а до того молчание уместнее.
            return
        if self.message_id is None and (now - self._started) < self._min_delay:
            # Ленивое создание: быстрый ответ карточки вообще не увидит.
            return

        text = self._next_text(now)
        if text is None:
            return
        if text == self._shown:
            # Дедупликация: тот же самый статус — сообщение не трогаем.
            self._pending = None
            return
        if self.message_id is not None and (now - self._last_render) < self._throttle:
            # Троттлинг: статус остаётся в _pending и уйдёт следующим тиком.
            return

        self._last_render = now
        # _shown выставляют _create/_edit и ТОЛЬКО при успехе, и ровно по той
        # же причине _pending не гасится заранее: иначе после неудачной
        # правки объект считал бы новый текст показанным, в чате висел бы
        # старый, а повторить было бы уже нечего — карточка замирала бы на
        # устаревшем статусе до конца хода (слепое ревью, находка 10).
        if self.message_id is None:
            sent = await self._create(text)
        else:
            sent = await self._edit(text)
        if sent:
            self._pending = None

    def _next_text(self, now: float) -> str | None:
        """Что должно висеть в карточке прямо сейчас, или ``None``, если
        менять нечего."""
        if self._pending is not None:
            return render(self._pending)
        silence = now - self._last_event
        if silence >= self._quiet:
            return render(quiet_status(silence))
        if self._shown is None:
            # Первая отрисовка: событий ещё нет, но работа уже идёт.
            return render(DEFAULT_STATUS)
        return None

    # ── Telegram ──────────────────────────────────────────────────────
    async def _create(self, text: str) -> bool:
        try:
            msg = await self._bot.send_message(
                self._chat_id, text, disable_notification=True
            )
        except Exception:  # noqa: BLE001
            # Создать не вышло — выключаем насовсем. Повторять каждую
            # секунду при, скажем, заблокированном чате означает лить в лог
            # и в Telegram мусор всё время хода.
            self._disabled = True
            logger.debug("progress: could not create the card", exc_info=True)
            return False
        self.message_id = getattr(msg, "message_id", None)
        if self.message_id is None:
            self._disabled = True
            return False
        self._shown = text
        return True

    async def _edit(self, text: str) -> bool:
        try:
            await self._bot.edit_message_text(
                text=text, chat_id=self._chat_id, message_id=self.message_id
            )
        except Exception:  # noqa: BLE001 — правка не критична, ход идёт дальше
            logger.debug("progress: could not edit the card", exc_info=True)
            return False
        self._shown = text
        return True

    async def close(self) -> None:
        """Убрать карточку. Итоговый ответ приходит отдельным сообщением —
        карточке в истории чата делать нечего.

        Не бросает: вызывается прямо перед отправкой ответа."""
        # ПЕРВЫМ делом и синхронно, до единого await: иначе запоздавший тик
        # фоновой задачи (отмена доставляется только на ближайшей точке
        # ожидания, а при жёсткой отмене всего хода она может не доставиться
        # вовремя вовсе) увидит ``message_id is None`` плюс свежее событие в
        # стенограмме — и СОЗДАСТ вторую карточку, которую уже некому
        # удалить. Воспроизводится; см. тест
        # test_a_late_tick_after_close_cannot_orphan_a_second_card.
        self._disabled = True
        message_id, self.message_id = self.message_id, None
        if message_id is None:
            return
        try:
            await self._bot.delete_message(self._chat_id, message_id)
        except Exception:  # noqa: BLE001
            logger.debug("progress: could not delete the card", exc_info=True)
