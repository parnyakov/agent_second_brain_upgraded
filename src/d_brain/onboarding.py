# ruff: noqa: E501
"""Resumable first-run onboarding for a personal d-brain vault (state v2).

One state engine, two front-ends:

* a non-interactive CLI the agent drives from Telegram
  (``python -m d_brain.onboarding --vault PATH status|next|answer|...``);
* a terminal wizard (``start`` / ``resume``) that uses the same engine and
  saves every answer the moment it is given.

State lives in ``<vault>/.onboarding/state.json`` (owner-only, atomic write).
It holds profile answers but never service tokens: answers that look like
secrets are refused before they reach disk.  Generated vault files carry a
managed block between ``<!-- onboarding:begin -->`` and
``<!-- onboarding:end -->``; re-rendering replaces only that block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

STATE_VERSION = 2
STATE_DIR = ".onboarding"
STATE_FILE = "state.json"
BLOCK_BEGIN = "<!-- onboarding:begin -->"
BLOCK_END = "<!-- onboarding:end -->"
STATUSES = ("pending", "in_progress", "done", "deferred")
IMPORT_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm"}
MAX_IMPORT_BYTES = 5 * 1024 * 1024
MAX_ANSWER_CHARS = 20_000
AFFIRMATIVE = {"да", "yes", "y", "ok", "ок", "подтверждаю", "верно", "всё верно", "все верно", "согласен", "согласна"}
RESUME_HINT = "dbrain onboarding resume (в терминале) или /onboarding (в Telegram)"


class OnboardingError(ValueError):
    """User-facing validation error (Russian message, CLI exit code 2)."""


# ── Section definitions ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Question:
    id: str
    prompt: str
    hint: str = ""
    required: bool = False
    multiline: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "hint": self.hint,
            "required": self.required,
            "multiline": self.multiline,
        }


@dataclass(frozen=True)
class Section:
    id: str
    title: str
    why: str
    questions: tuple[Question, ...]

    def question(self, question_id: str) -> Question:
        for item in self.questions:
            if item.id == question_id:
                return item
        known = ", ".join(q.id for q in self.questions)
        raise OnboardingError(
            f"В разделе «{self.id}» нет вопроса «{question_id}». Варианты: {known}."
        )


SECTION_DEFS: tuple[Section, ...] = (
    Section(
        "profile",
        "Профиль",
        "Базовые факты о вас нужны, чтобы агент не переспрашивал их в каждом разговоре и отвечал в вашем часовом поясе и на вашем языке.",
        (
            Question("name", "Как к вам обращаться?", "Имя или имя и отчество", required=True),
            Question("timezone", "Ваш часовой пояс?", "Например: Europe/Moscow, Asia/Almaty", required=True),
            Question("language", "На каком языке вам удобнее общаться?", "Например: русский"),
            Question("role", "Ваша роль или должность?", "Например: руководитель отдела продаж", required=True),
            Question("about", "Коротко о вас и текущем этапе: чем заняты, что сейчас важно?", "2–5 предложений", multiline=True),
        ),
    ),
    Section(
        "projects",
        "Проекты и люди",
        "Список проектов и ключевых людей даёт агенту карту вашей работы: он будет правильно раскладывать заметки и понимать, о ком и о чём речь.",
        (
            Question("projects", "Какие у вас активные проекты или направления?", "По одному на строку, можно с короткой пометкой", required=True, multiline=True),
            Question("people", "С кем вы чаще всего работаете? Кто за что отвечает?", "По одному человеку на строку: имя — роль", multiline=True),
            Question("responsibilities", "За какие результаты вы отвечаете лично?", "", multiline=True),
        ),
    ),
    Section(
        "goals",
        "Цели",
        "Цели на год, месяц и неделю помогают агенту расставлять приоритеты: он будет напоминать о главном и связывать ежедневные дела с тем, чего вы хотите достичь.",
        (
            Question("vision_3y", "Каким вы видите результат через 3 года?", "Необязательно: можно пропустить и вернуться позже", multiline=True),
            Question("year", "Главные цели на этот год?", "2–4 цели, по одной на строку", required=True, multiline=True),
            Question("month", "Приоритеты текущего месяца?", "2–3 пункта", required=True, multiline=True),
            Question("week", "ОДИН главный результат этой недели?", "Одна фраза: что должно быть сделано к концу недели", required=True),
        ),
    ),
    Section(
        "preferences",
        "Стиль работы",
        "Эти правила определяют, как агент с вами разговаривает и что может делать сам, а что обязан сначала согласовать.",
        (
            Question("style", "Как вам удобнее получать ответы?", "Например: коротко, списком, сначала вывод", required=True, multiline=True),
            Question("autonomy", "Что агент может делать сам, а что только после вашего согласия?", "", required=True, multiline=True),
            Question("avoid", "Чего агенту делать нельзя или как не стоит отвечать?", "", multiline=True),
            Question("agent_name", "Как назвать агента?", "Необязательно: любое имя или пропустить — тогда без имени"),
            Question("agent_gender", "Как агенту говорить о себе?", "мужской («сделал»), женский («сделала») или нейтрально («готово»)"),
        ),
    ),
    Section(
        "services",
        "Сервисы",
        "Составим план подключений: агент будет знать, где лежат ваши документы и календарь. Каждое подключение настраивается отдельно и позже, а пароли и токены в чат не пишутся.",
        (
            Question("used", "Какими сервисами вы пользуетесь в работе?", "Например: Google Drive, Google Calendar, Gmail, Notion", required=True, multiline=True),
            Question("wanted", "Какие из них хотите подключить к агенту в первую очередь?", "", multiline=True),
        ),
    ),
    Section(
        "notes",
        "Заметки",
        "Ваши старые заметки быстро дадут агенту контекст. Мы копируем их отдельно, оригиналы не трогаем, а раскладывать по папкам агент начнёт только после вашего согласия.",
        (
            Question("sources", "Где сейчас хранятся ваши заметки?", "Например: Apple Notes, Google Keep, Notion, папка с файлами. Можно ответить «нет заметок»", multiline=True),
            Question("plan", "Итог разбора импортированных заметок", "Заполняет агент после того, как вы согласовали план разбора", multiline=True),
        ),
    ),
    Section(
        "review",
        "Проверка",
        "Перед началом работы вы проверяете, как агент понял ваш профиль, память и цели. Этот контекст он будет использовать во всех следующих ответах.",
        (
            Question("approved", "Всё верно? Подтверждаете профиль, память и цели?", "Ответьте «да», когда всё проверено", required=True),
            Question("corrections", "Что поправили при проверке?", "", multiline=True),
        ),
    ),
    Section(
        "capabilities",
        "Возможности",
        "Короткая памятка о том, что умеет агент: напоминания и расписания, ежедневная обработка записей, команды и как поправить контекст позже.",
        (
            Question("questions", "Остались вопросы по возможностям?", "", multiline=True),
        ),
    ),
)

SECTIONS: tuple[str, ...] = tuple(section.id for section in SECTION_DEFS)
_BY_ID = {section.id: section for section in SECTION_DEFS}


def get_section(section_id: str) -> Section:
    try:
        return _BY_ID[section_id]
    except KeyError:
        raise OnboardingError(
            f"Неизвестный раздел «{section_id}». Варианты: {', '.join(SECTIONS)}."
        ) from None


# ── Secrets guard ─────────────────────────────────────────────────────────

_SECRET_PATTERNS = (
    re.compile(r"\d{6,}:[A-Za-z0-9_-]{30,}"),
    re.compile(r"github_pat_"),
    re.compile(r"(?<![A-Za-z0-9])ghp_[A-Za-z0-9]"),
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"-----BEGIN"),
)
_KEYLIKE = re.compile(r"[A-Za-z0-9_+=-]{32,}")

SECRET_MESSAGE = (
    "Похоже на токен, ключ или пароль — такое не сохраняется в памяти онбординга. "
    "Токены хранятся только в файле .env на сервере. Если вы отправили настоящий "
    "токен в чат, отзовите его и выпустите новый. Если это не секрет, "
    "перефразируйте ответ без длинного кода."
)


def looks_like_secret(value: str) -> bool:
    if any(pattern.search(value) for pattern in _SECRET_PATTERNS):
        return True
    for token in _KEYLIKE.findall(value):
        if re.search(r"\d", token) and re.search(r"[A-Za-z]", token):
            return True
    return False


# ── Answer normalisation ──────────────────────────────────────────────────

AGENT_GENDERS = {
    "мужской": "мужской",
    "м": "мужской",
    "муж": "мужской",
    "мужской род": "мужской",
    "женский": "женский",
    "ж": "женский",
    "жен": "женский",
    "женский род": "женский",
    "нейтрально": "нейтрально",
    "нейтральный": "нейтрально",
    "нейтральная": "нейтрально",
    "н": "нейтрально",
}
AGENT_GENDER_TEXT = {
    "мужской": "мужской род («сделал», «проверил», «готов»)",
    "женский": "женский род («сделала», «проверила», «готова»)",
    "нейтрально": "нейтрально («готово», «записано», «сделано»)",
}


def normalize_answer(section_id: str, question_id: str, value: str) -> str:
    """Validate and canonicalise answers with a fixed format."""
    if not value:
        return value
    if (section_id, question_id) == ("profile", "timezone"):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise OnboardingError(
                f"Не знаю часовой пояс «{value}». Нужно название из базы часовых поясов, "
                "например: Europe/Moscow, Europe/Kaliningrad, Asia/Yekaterinburg, "
                "Asia/Almaty, Europe/Berlin, UTC."
            ) from None
        return value
    if (section_id, question_id) == ("preferences", "agent_gender"):
        canonical = AGENT_GENDERS.get(value.lower().rstrip("."))
        if canonical is None:
            raise OnboardingError(
                "Ответ на вопрос о роде: «мужской», «женский» или «нейтрально» (или пропустить)."
            )
        return canonical
    return value


# ── State ─────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def state_path(vault: Path) -> Path:
    return vault / STATE_DIR / STATE_FILE


def _fresh_sections() -> dict[str, dict[str, Any]]:
    return {
        name: {"status": "pending", "answers": {}, "updated_at": ""}
        for name in SECTIONS
    }


def new_state() -> dict[str, Any]:
    now = _now()
    return {
        "version": STATE_VERSION,
        "created_at": now,
        "updated_at": now,
        "sections": _fresh_sections(),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


REDACTED = "[удалено: похоже на секрет]"


def redact_answers(raw: Any) -> Any:
    """Replace secret-looking answer strings (v1 and v2 layouts) in place.

    Only answers are scrubbed: the import manifest legitimately contains
    content hashes and export file names with long hex ids.
    """
    if not isinstance(raw, dict):
        return raw
    groups = []
    if isinstance(raw.get("answers"), dict):  # v1
        groups.extend(v for v in raw["answers"].values() if isinstance(v, dict))
    for entry in (raw.get("sections") or {}).values():  # v2
        if isinstance(entry, dict) and isinstance(entry.get("answers"), dict):
            groups.append(entry["answers"])
    for answers in groups:
        for key, value in list(answers.items()):
            if isinstance(value, str) and looks_like_secret(value):
                answers[key] = REDACTED
    return raw


def _write_json(path: Path, data: Any) -> None:
    _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    path.chmod(0o600)


def save_state(vault: Path, state: dict[str, Any]) -> None:
    path = state_path(vault)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    state["updated_at"] = _now()
    _write_json(path, redact_answers(state))


_V1_KEYS = {
    "profile": {"name": "name", "timezone": "timezone", "language": "language", "role": "role", "about": "about"},
    "work": {"projects": "projects", "people": "people", "responsibilities": "responsibilities"},
    "goals": {"vision": "vision_3y", "year": "year", "month": "month", "week": "week"},
    "preferences": {"style": "style", "autonomy": "autonomy", "avoid": "avoid"},
    "services": {"used": "used", "wanted": "wanted"},
    "notes": {"source": "sources"},
}


def migrate_v1(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a v1 state (``completed`` list + ``answers``) to v2."""
    state = new_state()
    state["migrated_from"] = 1
    completed = set(raw.get("completed") or [])
    answers = raw.get("answers") or {}
    updated = str(raw.get("updated_at") or "")
    for old_name, mapping in {**_V1_KEYS, "review": {}}.items():
        new_name = "projects" if old_name == "work" else old_name
        section = state["sections"][new_name]
        old = answers.get(old_name)
        old = old if isinstance(old, dict) else {}
        for old_key, new_key in mapping.items():
            value = old.get(old_key)
            if isinstance(value, str) and value.strip() and not looks_like_secret(value):
                section["answers"][new_key] = value.strip()
        if old_name == "review" and old.get("accepted") is True:
            section["answers"]["approved"] = "да"
        if old.get("deferred"):
            section["status"] = "deferred"
        elif old_name in completed and (old_name != "review" or old.get("accepted") is True):
            section["status"] = "done"
        elif section["answers"]:
            section["status"] = "in_progress"
        if section["status"] != "pending":
            section["updated_at"] = updated
    return state


def load_state(vault: Path) -> dict[str, Any]:
    path = state_path(vault)
    if not path.exists():
        return new_state()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OnboardingError(
            f"Файл состояния {path} повреждён ({exc}). Выполните «restart --yes», старый файл будет сохранён в архиве."
        ) from exc
    if not isinstance(raw, dict):
        raise OnboardingError(f"Файл состояния {path} повреждён.")
    version = raw.get("version", 0)
    if version == 1:
        backup = path.with_name("state.v1.backup.json")
        if backup.exists():
            backup = path.with_name(f"state.v1.backup.{datetime.now():%Y%m%d-%H%M%S}.json")
        _write_json(backup, redact_answers(json.loads(json.dumps(raw))))
        state = migrate_v1(raw)
        save_state(vault, state)
        return state
    if version != STATE_VERSION:
        raise OnboardingError(
            f"Версия состояния {version} не поддерживается (ожидалась {STATE_VERSION})."
        )
    sections = raw.setdefault("sections", {})
    for name in SECTIONS:
        entry = sections.setdefault(name, {})
        if entry.get("status") not in STATUSES:
            entry["status"] = "pending"
        entry.setdefault("answers", {})
        entry.setdefault("updated_at", "")
    return raw


# ── Engine ────────────────────────────────────────────────────────────────


def _is_answered(entry: dict[str, Any], question: Question) -> bool:
    return question.id in entry["answers"]


def missing_required(state: dict[str, Any], section_id: str) -> list[str]:
    entry = state["sections"][section_id]
    return [
        q.id
        for q in get_section(section_id).questions
        if q.required and not str(entry["answers"].get(q.id, "")).strip()
    ]


def first_unanswered(state: dict[str, Any], section_id: str) -> str | None:
    entry = state["sections"][section_id]
    for question in get_section(section_id).questions:
        if not _is_answered(entry, question):
            return question.id
    return None


def next_section(state: dict[str, Any]) -> str | None:
    sections = state["sections"]
    for wanted in ("in_progress", "pending", "deferred"):
        for name in SECTIONS:
            if sections[name]["status"] == wanted:
                return name
    return None


def _touch(state: dict[str, Any], section_id: str, status: str | None = None) -> None:
    entry = state["sections"][section_id]
    if status:
        entry["status"] = status
    entry["updated_at"] = _now()


def status_payload(vault: Path, state: dict[str, Any]) -> dict[str, Any]:
    sections = []
    for index, section in enumerate(SECTION_DEFS, start=1):
        entry = state["sections"][section.id]
        sections.append(
            {
                "id": section.id,
                "index": index,
                "title": section.title,
                "status": entry["status"],
                "answered": sum(1 for q in section.questions if _is_answered(entry, q)),
                "total_questions": len(section.questions),
                "missing_required": missing_required(state, section.id),
                "updated_at": entry["updated_at"],
            }
        )
    done = sum(1 for item in sections if item["status"] == "done")
    upcoming = next_section(state)
    return {
        "version": state["version"],
        "vault": str(vault),
        "done": done,
        "total": len(SECTIONS),
        "finished": upcoming is None,
        "next_section": upcoming,
        "updated_at": state.get("updated_at", ""),
        "sections": sections,
    }


def section_payload(state: dict[str, Any], section_id: str) -> dict[str, Any]:
    section = get_section(section_id)
    entry = state["sections"][section_id]
    questions = []
    for question in section.questions:
        item = question.to_dict()
        item["answer"] = entry["answers"].get(question.id)
        questions.append(item)
    return {
        "id": section.id,
        "index": SECTIONS.index(section.id) + 1,
        "total": len(SECTIONS),
        "title": section.title,
        "why": section.why,
        "status": entry["status"],
        "questions": questions,
        "answers": dict(entry["answers"]),
        "first_unanswered": first_unanswered(state, section_id),
        "missing_required": missing_required(state, section_id),
    }


def next_payload(state: dict[str, Any]) -> dict[str, Any]:
    upcoming = next_section(state)
    return {
        "finished": upcoming is None,
        "section": section_payload(state, upcoming) if upcoming else None,
    }


def save_answer(vault: Path, section_id: str, question_id: str, value: str) -> dict[str, Any]:
    section = get_section(section_id)
    question = section.question(question_id)
    value = value.strip()
    if len(value) > MAX_ANSWER_CHARS:
        raise OnboardingError(
            f"Ответ слишком длинный ({len(value)} символов, максимум {MAX_ANSWER_CHARS}). Сократите его или сохраните текст заметкой."
        )
    if question.required and not value:
        raise OnboardingError(f"Вопрос «{question.prompt}» обязательный — пустой ответ не сохраняется.")
    if looks_like_secret(value):
        raise OnboardingError(SECRET_MESSAGE)
    value = normalize_answer(section_id, question_id, value)
    state = load_state(vault)
    state["sections"][section_id]["answers"][question_id] = value
    _touch(state, section_id, "in_progress")
    save_state(vault, state)
    return state


def _is_affirmative(value: str) -> bool:
    return value.strip().lower().rstrip(".!") in AFFIRMATIVE


def complete_section(vault: Path, section_id: str, *, today: date | None = None) -> tuple[dict[str, Any], list[str]]:
    get_section(section_id)
    state = load_state(vault)
    missing = missing_required(state, section_id)
    if missing:
        prompts = "; ".join(f"{qid}: «{get_section(section_id).question(qid).prompt}»" for qid in missing)
        raise OnboardingError(f"Раздел «{section_id}» нельзя завершить — нет обязательных ответов: {prompts}.")
    prerequisites = {"review": SECTIONS[: SECTIONS.index("review")], "capabilities": ("review",)}
    blocking = [
        name
        for name in prerequisites.get(section_id, ())
        if state["sections"][name]["status"] not in (("done",) if section_id == "capabilities" else ("done", "deferred"))
    ]
    if blocking:
        titles = ", ".join(f"«{get_section(name).title}» ({name})" for name in blocking)
        raise OnboardingError(
            f"Раздел «{section_id}» можно завершить только после разделов: {titles}. "
            "Завершите их или отложите (defer)."
            if section_id == "review"
            else f"Раздел «{section_id}» можно завершить только после проверки: {titles}."
        )
    if section_id == "review" and not _is_affirmative(state["sections"]["review"]["answers"]["approved"]):
        raise OnboardingError(
            "Раздел «review» завершается только после явного подтверждения («да»). Сначала исправьте то, что неверно."
        )
    written = render_section(vault, state, section_id, today=today)
    if section_id == "capabilities":
        state["sections"]["capabilities"]["guide_shown_at"] = _now()
    _touch(state, section_id, "done")
    save_state(vault, state)
    return state, written


def defer_section(vault: Path, section_id: str) -> dict[str, Any]:
    get_section(section_id)
    state = load_state(vault)
    if state["sections"][section_id]["status"] == "done":
        raise OnboardingError(f"Раздел «{section_id}» уже завершён. Чтобы изменить его, используйте reopen.")
    _touch(state, section_id, "deferred")
    save_state(vault, state)
    return state


def reopen_section(vault: Path, section_id: str) -> dict[str, Any]:
    get_section(section_id)
    state = load_state(vault)
    _touch(state, section_id, "in_progress")
    save_state(vault, state)
    return state


def restart(vault: Path) -> Path | None:
    path = state_path(vault)
    archive: Path | None = None
    if path.exists():
        stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
        archive = path.with_name(f"state.{stamp}.json")
        counter = 1
        while archive.exists():
            archive = path.with_name(f"state.{stamp}-{counter}.json")
            counter += 1
        try:
            _write_json(archive, redact_answers(json.loads(path.read_text(encoding="utf-8"))))
            path.unlink()
        except (OSError, json.JSONDecodeError):
            path.replace(archive)  # unreadable: keep the bytes as they are
    save_state(vault, new_state())
    return archive


# ── Rendering ─────────────────────────────────────────────────────────────


def upsert_managed_block(path: Path, body: str, *, heading: str) -> None:
    """Write ``body`` between the onboarding markers, keeping other text."""
    block = f"{BLOCK_BEGIN}\n{body.strip()}\n{BLOCK_END}"
    if not path.exists():
        _write_text(path, f"{heading}\n\n{block}\n")
        return
    current = path.read_text(encoding="utf-8")
    start = current.find(BLOCK_BEGIN)
    end = current.find(BLOCK_END, start + len(BLOCK_BEGIN)) if start != -1 else -1
    if start != -1 and end != -1:
        updated = current[:start] + block + current[end + len(BLOCK_END):]
    else:
        updated = current.rstrip("\n") + "\n\n" + block + "\n"
    _write_text(path, updated)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.onboarding.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _text(answers: dict[str, Any], key: str, empty: str = "_Не указано._") -> str:
    value = str(answers.get(key, "") or "").strip()
    return value or empty


def _answers(state: dict[str, Any], section_id: str) -> dict[str, Any]:
    return state["sections"][section_id]["answers"]


def render_section(vault: Path, state: dict[str, Any], section_id: str, *, today: date | None = None) -> list[str]:
    """Render the vault files of one section; return vault-relative paths."""
    today = today or date.today()
    a = _answers(state, section_id)
    targets: list[tuple[str, str, str]] = []  # (relative path, heading for new file, body)
    if section_id == "profile":
        targets.append((
            "personal/about.md",
            "# Обо мне",
            "## Профиль\n\n"
            f"- Имя: {_text(a, 'name')}\n"
            f"- Часовой пояс: {_text(a, 'timezone')}\n"
            f"- Язык: {_text(a, 'language')}\n"
            f"- Роль: {_text(a, 'role')}\n\n"
            f"## Контекст\n\n{_text(a, 'about')}",
        ))
    elif section_id == "projects":
        targets.append((
            "projects/_index.md",
            "# Проекты",
            f"## Активные проекты\n\n{_text(a, 'projects')}\n\n"
            f"## Люди\n\n{_text(a, 'people')}\n\n"
            f"## Моя ответственность\n\n{_text(a, 'responsibilities')}",
        ))
    elif section_id == "goals":
        later = "_Не задано. Можно добавить позже: «обнови видение на 3 года…»._"
        targets += [
            ("goals/0-vision-3y.md", "# Видение на 3 года", f"## Через 3 года\n\n{_text(a, 'vision_3y', later)}"),
            ("goals/1-yearly.md", "# Цели года", f"## Цели на {today.year} год\n\n{_text(a, 'year')}"),
            ("goals/2-monthly.md", "# Цели месяца", f"## Приоритеты на {today:%m.%Y}\n\n{_text(a, 'month')}"),
            ("goals/3-weekly.md", "# ONE Big Thing недели", f"## Неделя {today:%G-W%V}: главный результат\n\n{_text(a, 'week')}"),
        ]
    elif section_id == "preferences":
        profile = _answers(state, "profile")
        targets.append((
            "MEMORY.md",
            "# Долговременная память",
            "## Профиль\n\n"
            f"- Имя: {_text(profile, 'name')}\n"
            f"- Часовой пояс: {_text(profile, 'timezone')}\n"
            f"- Язык: {_text(profile, 'language')}\n\n"
            f"## Как отвечать\n\n{_text(a, 'style')}\n\n"
            f"## Самостоятельность\n\n{_text(a, 'autonomy')}\n\n"
            f"## Ограничения\n\n{_text(a, 'avoid')}\n\n"
            "## Агент\n\n"
            f"- Имя: {_text(a, 'agent_name', 'не задано')}\n"
            f"- Как говорить о себе: {AGENT_GENDER_TEXT.get(str(a.get('agent_gender') or ''), AGENT_GENDER_TEXT['нейтрально'])}",
        ))
    elif section_id == "services":
        targets.append((
            "personal/connections.md",
            "# Подключения",
            "> Пароли, токены и ключи сюда НЕ записываются — они хранятся только в `.env` на сервере.\n"
            "> Каждый сервис подключается отдельно, с явно согласованными правами доступа.\n\n"
            f"## Использую\n\n{_text(a, 'used')}\n\n"
            f"## Хочу подключить\n\n{_text(a, 'wanted')}",
        ))
    elif section_id == "notes":
        targets.append((
            "inbox/README.md",
            "# Входящие",
            "## Импорт заметок\n\n"
            "- Копии импортированных заметок лежат в `inbox/raw-import/<дата>/` — оригиналы у вас не изменяются и не удаляются.\n"
            "- Файлы в `raw-import/` агент не удаляет: это резервная копия.\n"
            "- Сначала агент читает копии и предлагает план разбора в `inbox/import-plan.md` (файл → куда → почему).\n"
            "- Заметки раскладываются по папкам только после вашего согласия.\n\n"
            f"## Где хранились заметки\n\n{_text(a, 'sources')}\n\n"
            f"## Итог разбора\n\n{_text(a, 'plan', '_Разбор ещё не проводился._')}",
        ))
    elif section_id == "review":
        lines = [
            "# Проверка онбординга",
            "",
            f"- Подтверждено: {_now()}",
            f"- Ответ: {_text(a, 'approved')}",
            "",
            "## Исправления при проверке",
            "",
            _text(a, "corrections", "_Нет._"),
            "",
            "## Состояние разделов",
            "",
        ]
        for section in SECTION_DEFS:
            status = state["sections"][section.id]["status"]
            lines.append(f"- {section.title}: {'done' if section.id == 'review' else status}")
        _write_text(vault / STATE_DIR / "review.md", "\n".join(lines) + "\n")
        return [f"{STATE_DIR}/review.md"]
    written = []
    for relative, heading, body in targets:
        upsert_managed_block(vault / relative, body, heading=heading)
        written.append(relative)
    return written


# ── Notes import ──────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_notes(vault: Path, source: Path, *, today: date | None = None) -> dict[str, Any]:
    """Copy supported note files into ``inbox/raw-import/<date>/``.

    Idempotent by content hash (manifest in state under ``notes``); never
    modifies the source; skips symlinks and files larger than 5 MB.
    """
    today = today or date.today()
    source = source.expanduser()
    if not source.is_absolute():
        source = Path.cwd() / source
    if source.is_symlink():
        raise OnboardingError(f"Путь {source} — символическая ссылка; укажите настоящую папку или файл.")
    if not source.exists():
        raise OnboardingError(f"Путь не найден: {source}")
    raw_root = (vault / "inbox" / "raw-import").resolve()
    if source.resolve() == raw_root or raw_root in source.resolve().parents:
        raise OnboardingError("Нельзя импортировать из inbox/raw-import самого vault.")

    candidates: list[tuple[Path, Path]] = []  # (absolute, relative)
    skipped: list[dict[str, str]] = []
    if source.is_file():
        candidates.append((source, Path(source.name)))
    else:
        for root, dirs, files in os.walk(source, followlinks=False):
            root_path = Path(root)
            for name in sorted(dirs):
                if (root_path / name).is_symlink():
                    skipped.append({"path": str((root_path / name).relative_to(source)), "reason": "symlink"})
            dirs[:] = sorted(d for d in dirs if not (root_path / d).is_symlink())
            for name in sorted(files):
                item = root_path / name
                candidates.append((item, item.relative_to(source)))

    state = load_state(vault)
    notes = state["sections"]["notes"]
    manifest: dict[str, Any] = notes.setdefault("manifest", {})
    dest_root = vault / "inbox" / "raw-import" / today.isoformat()
    copied = duplicates = unsupported = 0
    files_copied: list[str] = []
    for item, relative in candidates:
        if item.is_symlink():
            skipped.append({"path": str(relative), "reason": "symlink"})
            continue
        if not item.is_file():
            continue
        if item.suffix.lower() not in IMPORT_EXTENSIONS:
            unsupported += 1
            continue
        if item.stat().st_size > MAX_IMPORT_BYTES:
            skipped.append({"path": str(relative), "reason": "too_large"})
            continue
        digest = _sha256(item)
        known = manifest.get(digest)
        if known and (vault / known["dest"]).exists():
            duplicates += 1
            continue
        destination = dest_root / relative
        counter = 1
        while destination.exists():
            destination = destination.with_name(f"{Path(relative).stem}-{counter}{Path(relative).suffix}")
            counter += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
        dest_rel = destination.relative_to(vault).as_posix()
        manifest[digest] = {"source": str(item), "dest": dest_rel, "imported_at": _now()}
        files_copied.append(dest_rel)
        copied += 1
    report = {
        "source": str(source),
        "destination": dest_root.relative_to(vault).as_posix(),
        "copied": copied,
        "duplicates": duplicates,
        "unsupported": unsupported,
        "skipped": skipped,
        "files": files_copied,
    }
    notes["last_import"] = {k: v for k, v in report.items() if k != "files"} | {"at": _now()}
    if notes["status"] in ("pending", "deferred"):
        notes["status"] = "in_progress"
    notes["updated_at"] = _now()
    save_state(vault, state)
    return report


# ── Terminal wizard ───────────────────────────────────────────────────────

InputFn = Callable[[str], str]
_STATUS_RU = {"pending": "не начат", "in_progress": "в работе", "done": "готов", "deferred": "отложен"}


def _read_answer(question: Question, input_fn: InputFn) -> str:
    hint = f" ({question.hint})" if question.hint else ""
    optional = "" if question.required else " [Enter — пропустить]"
    if not question.multiline:
        return input_fn(f"{question.prompt}{hint}{optional}: ").strip()
    print(f"{question.prompt}{hint}{optional}")
    print("  (несколько строк; закончите строкой с одной точкой «.»)")
    lines: list[str] = []
    while True:
        line = input_fn("").rstrip()
        if line == ".":
            break
        if not lines and not line and not question.required:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _print_map(state: dict[str, Any]) -> None:
    print("\nКарта онбординга:")
    for index, section in enumerate(SECTION_DEFS, start=1):
        status = state["sections"][section.id]["status"]
        print(f"  {index}. {section.title} — {_STATUS_RU[status]}")
    print("Можно остановиться в любой момент (Ctrl+C): каждый ответ уже сохранён.")


def run_wizard(vault: Path, *, input_fn: InputFn = input) -> int:
    state = load_state(vault)
    _print_map(state)
    try:
        while True:
            section_id = next_section(state)
            if section_id is None:
                print("\nОнбординг завершён. Напишите боту в Telegram — агент уже знает ваш контекст.")
                return 0
            section = get_section(section_id)
            print(f"\n=== {SECTIONS.index(section_id) + 1}/{len(SECTIONS)}. {section.title} ===")
            print(f"Зачем: {section.why}")
            if section_id == "capabilities":
                guide = vault / "GUIDE.md"
                print(f"Памятка о возможностях: {guide if guide.exists() else 'GUIDE.md в папке vault'}")
            if section_id == "review":
                print("Проверьте файлы: personal/about.md, projects/_index.md, MEMORY.md, goals/.")
            while (question_id := first_unanswered(state, section_id)) is not None:
                question = section.question(question_id)
                value = _read_answer(question, input_fn)
                if question.required and not value:
                    print("Это обязательный вопрос.")
                    continue
                try:
                    state = save_answer(vault, section_id, question_id, value)
                except OnboardingError as exc:
                    print(exc)
                    continue
                if section_id == "notes" and question_id == "sources":
                    _wizard_import(vault, input_fn)
                    state = load_state(vault)
            try:
                state, written = complete_section(vault, section_id)
            except OnboardingError as exc:
                print(exc)
                if section_id == "review":
                    state["sections"]["review"]["answers"].pop("approved", None)
                    save_state(vault, state)
                print(f"Исправьте ответы и продолжите: {RESUME_HINT}")
                return 1
            for relative in written:
                print(f"  записано: {relative}")
    except (EOFError, KeyboardInterrupt):
        print(f"\nПрогресс сохранён. Продолжить: {RESUME_HINT}")
        return 130


def _wizard_import(vault: Path, input_fn: InputFn) -> None:
    raw = input_fn("Путь к папке или файлу с заметками для импорта [Enter — пропустить]: ").strip()
    if not raw:
        return
    try:
        report = import_notes(vault, Path(raw))
    except OnboardingError as exc:
        print(exc)
        return
    print(_format_import(report))


def _format_import(report: dict[str, Any]) -> str:
    lines = [
        f"Импорт в {report['destination']}: скопировано {report['copied']}, "
        f"уже было {report['duplicates']}, неподходящий формат {report['unsupported']}.",
    ]
    reasons = {"symlink": "символическая ссылка", "too_large": "больше 5 МБ"}
    for item in report["skipped"]:
        lines.append(f"  пропущено: {item['path']} ({reasons.get(item['reason'], item['reason'])})")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(2, f"Ошибка в команде: {message}\n")


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="вывод в JSON")
    parser = _Parser(description="Возобновляемый онбординг d-brain")
    parser.add_argument("--vault", type=Path, default=Path("vault"))
    sub = parser.add_subparsers(dest="command", parser_class=_Parser)
    sub.add_parser("start", help="мастер в терминале: начать или продолжить")
    sub.add_parser("resume", help="мастер в терминале: продолжить")
    sub.add_parser("status", parents=[common], help="прогресс по разделам")
    sub.add_parser("next", parents=[common], help="следующий раздел и его вопросы")
    answer = sub.add_parser("answer", parents=[common], help="сохранить один ответ")
    answer.add_argument("section")
    answer.add_argument("question_id")
    source = answer.add_mutually_exclusive_group(required=True)
    source.add_argument("--value")
    source.add_argument("--stdin", action="store_true")
    for name, help_text in (
        ("complete", "завершить раздел и записать файлы"),
        ("defer", "отложить раздел"),
        ("reopen", "вернуться к разделу"),
        ("edit", "то же, что reopen"),
    ):
        item = sub.add_parser(name, parents=[common], help=help_text)
        item.add_argument("section")
    notes = sub.add_parser("import-notes", parents=[common], help="скопировать заметки в inbox/raw-import")
    notes.add_argument("path", type=Path)
    again = sub.add_parser("restart", parents=[common], help="архивировать состояние и начать заново")
    again.add_argument("--yes", action="store_true")
    return parser


def _emit(args: argparse.Namespace, payload: dict[str, Any], text: str) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(text)


def _status_text(payload: dict[str, Any]) -> str:
    lines = [f"Онбординг: готово {payload['done']}/{payload['total']}"]
    for item in payload["sections"]:
        lines.append(f"  {item['index']}. {item['title']} ({item['id']}) — {_STATUS_RU[item['status']]}")
    upcoming = payload["next_section"]
    lines.append(f"Следующий раздел: {upcoming or 'онбординг завершён'}")
    lines.append(f"Последнее сохранение: {payload['updated_at'] or 'ещё не было'}")
    return "\n".join(lines)


def _next_text(payload: dict[str, Any]) -> str:
    section = payload["section"]
    if section is None:
        return "Онбординг завершён."
    lines = [f"{section['index']}/{section['total']}. {section['title']} ({section['id']}) — {_STATUS_RU[section['status']]}", f"Зачем: {section['why']}"]
    for question in section["questions"]:
        mark = "✓" if question["answer"] is not None else ("*" if question["required"] else "-")
        lines.append(f"  {mark} {question['id']}: {question['prompt']}")
    lines.append(f"Первый вопрос без ответа: {section['first_unanswered'] or 'нет'}")
    return "\n".join(lines)


def _dispatch(args: argparse.Namespace, vault: Path) -> int:
    command = args.command or "resume"
    if command in ("start", "resume"):
        return run_wizard(vault)
    if command == "status":
        payload = status_payload(vault, load_state(vault))
        _emit(args, payload, _status_text(payload))
        return 0
    if command == "next":
        payload = next_payload(load_state(vault))
        _emit(args, payload, _next_text(payload))
        return 0
    if command == "answer":
        value = sys.stdin.read() if args.stdin else args.value
        state = save_answer(vault, args.section, args.question_id, value)
        payload = {
            "ok": True,
            "section": args.section,
            "question_id": args.question_id,
            "status": state["sections"][args.section]["status"],
            "first_unanswered": first_unanswered(state, args.section),
            "missing_required": missing_required(state, args.section),
        }
        _emit(args, payload, f"Сохранено: {args.section}.{args.question_id}")
        return 0
    if command == "complete":
        state, written = complete_section(vault, args.section)
        payload = {"ok": True, "section": args.section, "status": "done", "written": written, "next_section": next_section(state)}
        _emit(args, payload, f"Раздел {args.section} завершён. Записано: {', '.join(written) or 'нет файлов'}")
        return 0
    if command == "defer":
        state = defer_section(vault, args.section)
        payload = {"ok": True, "section": args.section, "status": "deferred", "next_section": next_section(state)}
        _emit(args, payload, f"Раздел {args.section} отложен. Продолжить: {RESUME_HINT}")
        return 0
    if command in ("reopen", "edit"):
        reopen_section(vault, args.section)
        payload = {"ok": True, "section": args.section, "status": "in_progress"}
        _emit(args, payload, f"Раздел {args.section} открыт заново; остальные разделы не изменены.")
        return 0
    if command == "import-notes":
        report = import_notes(vault, args.path)
        _emit(args, {"ok": True, **report}, _format_import(report))
        return 0
    if command == "restart":
        if not args.yes:
            raise OnboardingError("restart удаляет текущий прогресс из работы — повторите с флагом --yes.")
        archive = restart(vault)
        payload = {"ok": True, "archived": str(archive) if archive else None}
        _emit(args, payload, f"Начинаем заново. Старое состояние: {archive or 'не было'}. Дальше: dbrain onboarding start")
        return 0
    raise OnboardingError(f"Неизвестная команда {command}.")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    vault = args.vault.expanduser().resolve()
    try:
        if not vault.is_dir():
            raise OnboardingError(f"Папка vault не найдена: {vault}")
        return _dispatch(args, vault)
    except OnboardingError as exc:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
