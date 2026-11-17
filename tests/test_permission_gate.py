"""Permission dialogs relayed to the person (2026-11-02..05).

The gate turns a dialog on a session's pane into ONE Telegram question with
«Да» / «Нет», keeps a record on disk while it waits (the chat path parks
messages meanwhile), and answers the CLI only with the person's button —
for the current dialog, never a stale one, never by timeout.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from d_brain.services import permission_gate
from d_brain.services.tmux_parse import read_permission_dialog

FIXTURES = Path(__file__).parent / "fixtures"
RM_GLOB = read_permission_dialog(
    (FIXTURES / "pane_permission_dangerous_rm.txt").read_text(encoding="utf-8")
)
RM_VAR = read_permission_dialog(
    (FIXTURES / "pane_permission_unframed.txt").read_text(encoding="utf-8")
)
ASK_RULE = read_permission_dialog(
    (FIXTURES / "pane_permission_ask_rule.txt").read_text(encoding="utf-8")
)
OWNER = 111


class FakeSession:
    def __init__(self, runtime_dir: Path, dialog=None):
        self.runtime_dir = runtime_dir
        self.dialog = dialog
        self.answers: list[tuple[str, bool]] = []
        self.result = "done"

    def permission_dialog(self):
        return self.dialog

    def answer_permission(self, fingerprint: str, allow: bool) -> str:
        self.answers.append((fingerprint, allow))
        if self.result == "done":
            self.dialog = None
        return self.result


class FakeBot:
    def __init__(self):
        self.sent: list[tuple[int, str, dict]] = []
        self.edits: list[tuple[str, dict]] = []
        self.fail = False

    async def send_message(self, chat_id, text, **kw):
        if self.fail:
            raise RuntimeError("network")
        self.sent.append((chat_id, text, kw))
        return SimpleNamespace(message_id=500 + len(self.sent))

    async def edit_message_text(self, text, **kw):
        self.edits.append((text, kw))


@pytest.fixture
def clock():
    return {"now": 1000.0}


def make(tmp_path, clock, dialog=RM_GLOB, **kw):
    session = FakeSession(tmp_path, dialog)
    gate = permission_gate.PermissionGate(
        {"main": session},
        chat_id=OWNER,
        allowed_ids=[OWNER],
        clock=lambda: clock["now"],
        **kw,
    )
    return gate, session, FakeBot()


def step(gate, bot):
    asyncio.run(gate.step(bot))


def press(gate, nonce, allow, user=OWNER, slot="main", html=""):
    return asyncio.run(gate.answer(slot, nonce, allow, user_id=user, current_html=html))


def test_new_dialog_is_one_question_with_yes_and_no(tmp_path, clock):
    gate, session, bot = make(tmp_path, clock)
    step(gate, bot)
    step(gate, bot)  # still the same dialog: no second message
    assert len(bot.sent) == 1
    chat_id, text, kw = bot.sent[0]
    assert chat_id == OWNER
    rows = kw["reply_markup"].inline_keyboard
    assert [b.text for b in rows[0]] == ["Да", "Нет"]
    rec = permission_gate.load(tmp_path)
    assert rows[0][0].callback_data == f"perm:main:{rec.nonce}:y"
    assert rows[0][1].callback_data == f"perm:main:{rec.nonce}:n"
    assert permission_gate.pending(tmp_path)
    assert session.answers == []  # nothing is ever answered by itself


def test_question_is_plain_russian_with_the_command(tmp_path, clock):
    gate, _s, bot = make(tmp_path, clock)
    step(gate, bot)
    text = bot.sent[0][1]
    assert text.startswith("🔐 <b>Фоновая задача просит подтвердить действие</b>")
    assert (
        "Удалить файлы <code>*</code> в папке <code>/srv/demo/vault/sheets</code>."
        in text
    )
    assert "<code>cd /srv/demo/vault &amp;&amp; mkdir -p sheets" in text
    assert "фоновый агент «general-purpose»" in text
    assert "новые сообщения ждут в очереди" in text
    assert text.endswith("Разрешить?")


def test_variable_path_is_explained(tmp_path, clock):
    text = permission_gate.describe(RM_VAR, "main")
    assert "Удалить файлы <code>*</code> в папке <code>$d</code>." in text
    assert "Путь задан переменной" in text


def test_approval_rule_and_cron_wording(tmp_path, clock):
    no_agent = permission_gate.describe(
        SimpleNamespace(**{**vars(ASK_RULE), "agent": None}), "cron"
    )
    assert no_agent.startswith("🔐 <b>Задача по расписанию просит")
    assert "нужно твоё подтверждение" in no_agent


def test_long_command_is_cut(tmp_path, clock):
    long = SimpleNamespace(**{**vars(ASK_RULE), "command": "x" * 2000})
    text = permission_gate.describe(long, "main")
    assert "x" * permission_gate.COMMAND_LIMIT + " …</code>" in text
    assert "x" * (permission_gate.COMMAND_LIMIT + 1) not in text


def test_yes_answers_the_dialog_and_says_so(tmp_path, clock):
    gate, session, bot = make(tmp_path, clock)
    step(gate, bot)
    rec = permission_gate.load(tmp_path)
    toast, text = press(gate, rec.nonce, True)
    assert session.answers == [(RM_GLOB.fingerprint, True)]
    assert toast == "Разрешил."
    assert text.endswith("✅ <b>Разрешил.</b>")
    assert "Разрешить?" not in text
    assert not permission_gate.pending(tmp_path)  # the queue resumes


def test_no_answers_the_dialog_and_says_so(tmp_path, clock):
    gate, session, bot = make(tmp_path, clock)
    step(gate, bot)
    toast, text = press(gate, permission_gate.load(tmp_path).nonce, False)
    assert session.answers == [(RM_GLOB.fingerprint, False)]
    assert toast == "Отклонил."
    assert "🚫 <b>Отклонил.</b>" in text


def test_stale_button_does_nothing_and_says_it_is_outdated(tmp_path, clock):
    gate, session, bot = make(tmp_path, clock)
    step(gate, bot)
    old = permission_gate.load(tmp_path).nonce
    # The agent's next request: a different dialog, a new question.
    session.dialog = ASK_RULE
    step(gate, bot)
    assert len(bot.sent) == 2
    assert "Окно закрылось без ответа" in bot.edits[0][0]
    toast, text = press(gate, old, True, html="вопрос\n\nРазрешить?")
    assert toast == "Запрос устарел."
    assert (
        text
        == "вопрос\n\n⌛ <i>Запрос устарел: окно уже закрыто. Ничего не нажимал.</i>"
    )
    assert session.answers == []
    assert permission_gate.pending(tmp_path)  # the new one still waits


def test_button_of_a_stranger_is_refused(tmp_path, clock):
    gate, session, bot = make(tmp_path, clock)
    step(gate, bot)
    toast, text = press(gate, permission_gate.load(tmp_path).nonce, True, user=999)
    assert text is None and "не для тебя" in toast
    assert session.answers == []


def test_dialog_changed_under_the_button_is_not_answered(tmp_path, clock):
    gate, session, bot = make(tmp_path, clock)
    step(gate, bot)
    session.result = "changed"
    toast, text = press(gate, permission_gate.load(tmp_path).nonce, True)
    assert toast == "Запрос устарел." and "Ничего не нажимал" in text


def test_key_that_did_not_take_keeps_the_question_open(tmp_path, clock):
    gate, session, bot = make(tmp_path, clock)
    step(gate, bot)
    session.result = "stuck"
    toast, text = press(gate, permission_gate.load(tmp_path).nonce, True)
    assert text is None and "Попробуй ещё раз" in toast
    assert permission_gate.pending(tmp_path)


def test_dialog_closed_elsewhere_drops_the_record(tmp_path, clock):
    gate, session, bot = make(tmp_path, clock)
    step(gate, bot)
    session.dialog = None  # answered in the terminal, or the agent stopped
    step(gate, bot)
    assert not permission_gate.pending(tmp_path)
    text, kw = bot.edits[0]
    assert "Окно закрылось без ответа отсюда" in text
    assert kw["reply_markup"] is None and kw["message_id"] == 501


def test_one_reminder_and_never_an_answer_on_timeout(tmp_path, clock):
    gate, session, bot = make(tmp_path, clock, remind_after=1800.0)
    step(gate, bot)
    clock["now"] += 1799
    step(gate, bot)
    assert len(bot.sent) == 1
    clock["now"] += 1
    step(gate, bot)
    clock["now"] += 7200
    step(gate, bot)
    assert len(bot.sent) == 2  # exactly one reminder
    _chat, text, kw = bot.sent[1]
    assert "Всё ещё жду" in text and kw["reply_to_message_id"] == 501
    assert session.answers == []
    assert permission_gate.pending(tmp_path)  # still the person's call


def test_question_is_retried_when_telegram_was_down(tmp_path, clock):
    gate, session, bot = make(tmp_path, clock)
    bot.fail = True
    step(gate, bot)
    # The record is on disk anyway: nothing is typed into the dialog.
    assert permission_gate.pending(tmp_path)
    bot.fail = False
    step(gate, bot)
    assert len(bot.sent) == 1
    assert permission_gate.load(tmp_path).message_id == 501


def test_record_survives_a_restart(tmp_path, clock):
    gate, session, bot = make(tmp_path, clock)
    step(gate, bot)
    nonce = permission_gate.load(tmp_path).nonce
    again, _s, bot2 = make(tmp_path, clock)
    again._sessions["main"] = session
    step(again, bot2)
    assert bot2.sent == []  # same dialog, same question
    assert press(again, nonce, True)[0] == "Разрешил."


class AbsentableSession(FakeSession):
    """A session whose pane may not exist (the on-demand duty session)."""

    def __init__(self, runtime_dir: Path, dialog=None, present=True):
        super().__init__(runtime_dir, dialog)
        self.present = present
        self.probes = 0

    def exists(self) -> bool:
        return self.present

    def permission_dialog(self):
        self.probes += 1
        return self.dialog


def test_absent_session_is_not_probed(tmp_path, clock):
    """Duty enabled but never needed: its pane does not exist, so the gate
    must not look into it every pass (that flooded the logs of instances
    without a duty session with tmux "can't find session")."""
    main_dir, duty_dir = tmp_path / "main", tmp_path / "duty"
    main_dir.mkdir()
    duty_dir.mkdir()
    main = AbsentableSession(main_dir)
    duty = AbsentableSession(duty_dir, dialog=RM_GLOB, present=False)
    gate = permission_gate.PermissionGate(
        {"main": main, "duty": duty},
        chat_id=OWNER,
        allowed_ids=[OWNER],
        clock=lambda: clock["now"],
    )
    bot = FakeBot()
    step(gate, bot)
    step(gate, bot)
    assert duty.probes == 0 and main.probes == 2
    assert bot.sent == [] and not permission_gate.pending(duty_dir)
    duty.present = True  # created on demand later: watched from then on
    step(gate, bot)
    assert duty.probes == 1 and len(bot.sent) == 1


def test_session_that_vanished_closes_its_question(tmp_path, clock):
    session = AbsentableSession(tmp_path, dialog=RM_GLOB)
    gate = permission_gate.PermissionGate(
        {"duty": session},
        chat_id=OWNER,
        allowed_ids=[OWNER],
        clock=lambda: clock["now"],
    )
    bot = FakeBot()
    step(gate, bot)
    assert permission_gate.pending(tmp_path)
    session.present = False
    step(gate, bot)
    assert not permission_gate.pending(tmp_path)
    assert "Окно закрылось без ответа отсюда" in bot.edits[0][0]


def test_sessions_without_dialogs_are_not_watched(tmp_path):
    gate = permission_gate.PermissionGate(
        {"main": SimpleNamespace(runtime_dir=tmp_path), "cron": None}, chat_id=OWNER
    )
    assert gate._sessions == {}


@pytest.mark.parametrize(
    "data,expected",
    [
        ("perm:main:ab12cd34:y", ("main", "ab12cd34", True)),
        ("perm:cron:ab12cd34:n", ("cron", "ab12cd34", False)),
        ("perm:main:ab12cd34:maybe", None),
        ("perm:main::y", None),
        ("other:main:ab:y", None),
    ],
)
def test_parse_callback(data, expected):
    assert permission_gate.parse_callback(data) == expected


def test_pending_tolerates_anything_that_is_not_a_path():
    assert permission_gate.pending(None) is False
    assert permission_gate.pending(object()) is False


# ── the button handler ────────────────────────────────────────────────


class FakeMessage:
    def __init__(self, html_text: str):
        self.html_text = html_text
        self.edited: list[tuple[str, dict]] = []

    async def edit_text(self, text, **kw):
        self.edited.append((text, kw))


class FakeCallback:
    def __init__(self, data: str, user: int, message: FakeMessage):
        self.data = data
        self.from_user = SimpleNamespace(id=user)
        self.message = message
        self.toasts: list[str] = []

    async def answer(self, text: str = "", **kw):
        self.toasts.append(text)


def test_handler_presses_and_edits_the_question(tmp_path, clock):
    from d_brain.bot.handlers.permission import handle_permission_answer

    gate, session, bot = make(tmp_path, clock)
    step(gate, bot)
    permission_gate.configure(gate)
    try:
        nonce = permission_gate.load(tmp_path).nonce
        msg = FakeMessage(bot.sent[0][1])
        cb = FakeCallback(f"perm:main:{nonce}:n", OWNER, msg)
        asyncio.run(handle_permission_answer(cb))
    finally:
        permission_gate.reset()
    assert session.answers == [(RM_GLOB.fingerprint, False)]
    assert cb.toasts == ["Отклонил."]
    text, kw = msg.edited[0]
    assert "🚫 <b>Отклонил.</b>" in text and kw["reply_markup"] is None


def test_handler_without_a_gate_says_so(tmp_path):
    from d_brain.bot.handlers.permission import handle_permission_answer

    permission_gate.reset()
    cb = FakeCallback("perm:main:ab12cd34:y", OWNER, FakeMessage("q"))
    asyncio.run(handle_permission_answer(cb))
    assert cb.toasts and "не сработала" in cb.toasts[0]
    assert cb.message.edited == []
