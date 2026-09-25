"""The out-of-band delivery watch, run for real.

This script is the one alerting path that shares nothing with the bot, so
it is also the one that has to be tested by EXECUTING it rather than by
importing anything. Everything it touches is faked around it: a stub
``systemctl``, a stub ``pgrep``, its own runtime dir, and an env file that
does not exist — so ``notify.sh``/``backup-notify.sh`` find no token, send
nothing and exit 0.

What the script reports is then read off its own contract, which the unit
already relies on (``SuccessExitStatus=0 1``):

    exit 0  — healthy, nothing sent
    exit 1  — a fault was found and reported; the message is on stderr

Every case below is a pair: the false alarm is gone, and the real fault it
was built for still arrives.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "delivery-watch.sh"

_SYSTEMCTL_STUB = """#!/bin/bash
# `systemctl --user show -p <Property> --value <unit>`
prop=""; unit=""
for a in "$@"; do
    case "$a" in
        -p) ;;
        --user|--system|show|--value) ;;
        ActiveState|StateChangeTimestampMonotonic) prop="$a" ;;
        *) unit="$a" ;;
    esac
done
case "$unit:$prop" in
    *watchdog*:ActiveState) echo "${STUB_WATCHDOG_STATE:-active}" ;;
    *:ActiveState)          echo "${STUB_BOT_STATE:-active}" ;;
    *watchdog*:StateChangeTimestampMonotonic) echo "${STUB_WATCHDOG_CHANGED_US:-0}" ;;
    *:StateChangeTimestampMonotonic)          echo "${STUB_BOT_CHANGED_US:-0}" ;;
esac
exit 0
"""

_PGREP_STUB = "#!/bin/bash\necho \"${STUB_BOT_COUNT:-1}\"\nexit 0\n"

# `curl` stands in for both notify channels' only side effect, so nothing
# leaves the machine. STUB_CURL_OK=0 is the case that matters most: dead
# letters pile up precisely when Telegram sends are failing, and the script
# must not then mark the fault as reported.
_CURL_STUB = """#!/bin/bash
if [ "${STUB_CURL_OK:-1}" = "1" ]; then exit 0; fi
exit 7
"""


@pytest.fixture
def watch(tmp_path):
    """Returns ``run(**env)`` → ``(exit_code, stderr)`` for the real script."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "systemctl").write_text(_SYSTEMCTL_STUB)
    (bindir / "pgrep").write_text(_PGREP_STUB)
    (bindir / "curl").write_text(_CURL_STUB)
    for name in ("systemctl", "pgrep", "curl"):
        os.chmod(bindir / name, 0o755)

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    # A fresh watchdog heartbeat, or F3 fires in every single case.
    (runtime / "STATUS.md").write_text("state: healthy\n")

    def run(**env_extra: str) -> tuple[int, str]:
        env = {
            "PATH": f"{bindir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(tmp_path),
            "DBRAIN_RUNTIME_DIR": str(runtime),
            # No credentials anywhere: notify.sh and backup-notify.sh both
            # bail out before curl, so nothing can leave the machine.
            "DBRAIN_ENV_FILE": str(tmp_path / "nope.env"),
            "DBRAIN_BACKUP_NOTIFY_ENV": str(tmp_path / "nope-backup.env"),
            # notify.sh reaches the STUBBED curl above, never the network.
            "TELEGRAM_BOT_TOKEN": "stub",
            "ALLOWED_USER_IDS": "1",
            # Its 30-minute debounce is keyed on a stamp in the runtime dir,
            # and these tests run the script several times in a row.
            "DBRAIN_NOTIFY_COOLDOWN": "0",
        }
        env.update(env_extra)
        proc = subprocess.run(
            ["bash", str(SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return proc.returncode, proc.stderr

    run.runtime = runtime  # type: ignore[attr-defined]
    return run


def _write_health(runtime: Path, *, streak: int, last_ts: float) -> None:
    (runtime / "ask-health.json").write_text(
        json.dumps(
            {
                "fail_streak": streak,
                "last_status": "busy",
                "last_ts": last_ts,
                "streak_started_ts": last_ts - 120,
            }
        )
    )


def _write_receipt(runtime: Path, at: float) -> None:
    d = runtime / "outbox"
    d.mkdir(parents=True, exist_ok=True)
    (d / "receipts.json").write_text(json.dumps({"ids": [f"{int(at * 1e9):019d}"]}))


def _put(runtime: Path, relative: str, *, age_seconds: float = 0.0) -> Path:
    path = runtime / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": "x"}))
    if age_seconds:
        old = time.time() - age_seconds
        os.utime(path, (old, old))
    return path


# ── baseline ─────────────────────────────────────────────────────────


def test_a_healthy_install_says_nothing(watch):
    code, err = watch()
    assert code == 0
    assert "канал доставки" not in err


# ── F4: the streak, and the receipt that contradicts it ──────────────


def test_a_streak_still_alerts_although_the_brush_offs_were_delivered(watch):
    """THE REAL CASE, and the reason there is no receipt gate here.

    Every failed turn delivers an apology, whose outbox receipt is stamped
    AFTER that turn's ledger row. A gate on "something reached Telegram
    since the last failure" is therefore satisfied by every streak this
    check exists for — it was written, and removed in blind review. The
    false-alarm morning is fixed in the LEDGER instead: a turn the duty
    session covered now scores `ok`, so a streak forms only when the person
    really got nothing but brush-offs."""
    runtime = watch.runtime
    now = time.time()
    _write_health(runtime, streak=5, last_ts=now - 200)
    _write_receipt(runtime, now - 199)  # the brush-off for that very turn

    code, err = watch()
    assert code == 1
    assert "канал доставки под вопросом" in err
    assert "ответы подряд не доставлены" in err


def test_a_ledger_closed_by_a_delivered_answer_says_nothing(watch):
    """NO FALSE ALARM: an answer that reached the person by ANY route resets
    the streak where the ledger is written, and this check honours that
    rather than second-guessing it."""
    _write_health(watch.runtime, streak=0, last_ts=time.time() - 10)
    code, err = watch()
    assert code == 0
    assert "не доставлен" not in err


def test_the_streak_alert_carries_no_number(watch):
    """notify.sh debounces on a cksum of the message, so the count inside
    the text minted a new message every five minutes and alerted forever.
    It belongs in the journal, not in the alert."""
    runtime = watch.runtime
    _write_health(runtime, streak=7, last_ts=time.time() - 100)

    code, err = watch()
    assert code == 1
    fault_line = next(line for line in err.splitlines() if "не доставлены" in line)
    assert not any(ch.isdigit() for ch in fault_line)
    # ...and the exact value is still recoverable from the journal.
    assert "fail_streak 7" in err


# ── F1/F2: a graceful stop is not a fault; a stuck one is ────────────


def test_a_unit_in_a_graceful_stop_is_not_reported_as_down(watch):
    """FALSE ALARM: since the graceful stop was introduced, a restart can
    hold `deactivating` for minutes — longer than this script's own timer.
    Reporting «бот не активен» about a bot doing exactly what it was told
    is the cry-wolf this script's header forbids."""
    monotonic_us = int(float(Path("/proc/uptime").read_text().split()[0]) * 1e6)
    code, err = watch(
        STUB_BOT_STATE="deactivating",
        STUB_BOT_CHANGED_US=str(max(0, monotonic_us - 30_000_000)),  # 30s ago
    )
    assert code == 0
    assert "не активен" not in err


def test_a_unit_stuck_in_a_transition_is_still_reported(watch):
    """THE REAL CASE, and the one the first cut of that fix deleted:
    «бот не поднялся после перезапуска» looks exactly like `activating`, so
    accepting that state unconditionally would have silenced it forever."""
    monotonic_us = int(float(Path("/proc/uptime").read_text().split()[0]) * 1e6)
    code, err = watch(
        STUB_BOT_STATE="activating",
        STUB_BOT_CHANGED_US=str(max(0, monotonic_us - 5_000_000)),
        DBRAIN_TRANSITION_GRACE="1",
    )
    assert code == 1
    assert "не активен" in err


def test_a_dead_unit_is_still_reported(watch):
    code, err = watch(STUB_BOT_STATE="failed")
    assert code == 1
    assert "не активен" in err


# ── F6: the proven loss, which must ALWAYS get through ───────────────


def test_a_reply_in_the_dead_queue_is_reported(watch):
    """The guarantee behind every gate added above: a message this install
    can PROVE it failed to deliver still reaches the owner, out of band, on
    its own evidence — even with a perfectly healthy unit, a clean ledger
    and a fresh receipt."""
    runtime = watch.runtime
    _write_receipt(runtime, time.time())
    _put(runtime, "outbox/dead/0000000000000000001.json")

    code, err = watch()
    assert code == 1
    assert "мёртвой очереди" in err


def test_an_accepted_message_retired_unanswered_is_reported(watch):
    _put(watch.runtime, "inbox/stale/0000000000000000007.json")
    code, err = watch()
    assert code == 1
    assert "ответ так и не ушёл" in err


def test_an_accepted_message_left_stranded_is_reported(watch):
    """«принято, но ответа не было» in its plainest form."""
    _put(watch.runtime, "inbox/0000000000000000007.json", age_seconds=3 * 3600)
    code, err = watch()
    assert code == 1
    assert "зависло без ответа" in err


@pytest.mark.parametrize("age_minutes", [0, 50])
def test_a_long_but_legitimate_turn_is_not_a_loss(watch, age_minutes):
    """FALSE ALARM: the accepted entry stays on disk for the WHOLE handler
    chain, and chat.py runs a second full turn when the first comes back
    empty — so ~50 minutes on disk is ordinary, not a strand."""
    _put(
        watch.runtime,
        "inbox/0000000000000000008.json",
        age_seconds=age_minutes * 60,
    )
    code, err = watch()
    assert code == 0
    assert "зависло" not in err


def test_a_loss_already_reported_does_not_alert_forever(watch):
    """The latch. A pile of dead letters the owner has already been told
    about must not re-alert every half hour — but a NEW loss on top of it
    must."""
    runtime = watch.runtime
    _put(runtime, "outbox/dead/0000000000000000001.json")
    assert watch()[0] == 1

    assert watch()[0] == 0  # same pile, silence

    _put(runtime, "outbox/dead/0000000000000000002.json")
    code, err = watch()
    assert code == 1
    assert "мёртвой очереди" in err


def test_a_different_loss_at_the_same_count_still_alerts(watch):
    """The latch is keyed on WHICH messages were lost, not how many. One
    loss cleared and a different one appearing between two runs leaves the
    count unchanged — a counter would never report the second message."""
    runtime = watch.runtime
    first = _put(runtime, "outbox/dead/0000000000000000001.json")
    assert watch()[0] == 1

    first.unlink()
    _put(runtime, "outbox/dead/0000000000000000002.json")
    code, err = watch()
    assert code == 1
    assert "мёртвой очереди" in err


def test_a_loss_is_not_marked_reported_when_the_alert_could_not_be_sent(watch):
    """The failure modes are correlated: entries land in outbox/dead/ exactly
    when Telegram sends are failing, which is when both notify channels fail
    too. A watermark advanced there would consume the one report of the one
    fault this script guarantees — so it advances only once something left."""
    runtime = watch.runtime
    _put(runtime, "outbox/dead/0000000000000000001.json")

    code, err = watch(STUB_CURL_OK="0")
    assert code == 1
    assert "мёртвой очереди" in err

    # Nothing got through, so the loss is still owed a report.
    code, err = watch(STUB_CURL_OK="0")
    assert code == 1
    assert "мёртвой очереди" in err

    # ...and once one does get through, it goes quiet.
    assert watch()[0] == 1
    assert watch()[0] == 0


def test_a_cleared_pile_can_alert_again(watch):
    """The watermark moves down too: a loss that was cleaned up and then
    happens again is a new incident, not an old one."""
    runtime = watch.runtime
    first = _put(runtime, "outbox/dead/0000000000000000001.json")
    assert watch()[0] == 1
    first.unlink()
    assert watch()[0] == 0

    _put(runtime, "outbox/dead/0000000000000000003.json")
    assert watch()[0] == 1


def test_the_loss_alert_carries_no_number(watch):
    """Same debounce rule as F4 — the identities go to the journal."""
    runtime = watch.runtime
    for i in range(1, 4):
        _put(runtime, f"outbox/dead/000000000000000000{i}.json")
    code, err = watch()
    assert code == 1
    fault_line = next(line for line in err.splitlines() if "мёртвой очереди" in line)
    assert not any(ch.isdigit() for ch in fault_line)
    assert "outbox-dead/0000000000000000003.json" in err
