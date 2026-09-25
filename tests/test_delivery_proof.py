"""The evidence layer: what the durable queues PROVE.

Every test here comes in a pair — one that a false alarm is gone, one that
the real case it was built for still shows up.
"""

from __future__ import annotations

import json
from pathlib import Path

from d_brain.services import chat_queue, delivery_proof, inbox, outbox

# ── the mirrored constants must not drift ────────────────────────────


def test_the_mirrored_directory_names_still_match_their_sources():
    """delivery_proof duplicates these names instead of importing them (so
    the watchdog can answer "is anything lost" without the transport
    stack). A silent rename in any queue would make every check below read
    an empty directory and report "nothing lost" forever — the worst
    possible failure for this module. Pin them."""
    assert delivery_proof.OUTBOX_DIRNAME == outbox.DIRNAME
    assert delivery_proof.OUTBOX_DEAD_DIRNAME == outbox.DEAD_DIRNAME
    assert delivery_proof.RECEIPTS_FILENAME == outbox.RECEIPTS_FILENAME
    assert delivery_proof.RECEIPTS_FILENAME == inbox.RECEIPTS_FILENAME
    assert delivery_proof.INBOX_DIRNAME == inbox.DIRNAME
    assert delivery_proof.INBOX_STALE_DIRNAME == inbox.STALE_DIRNAME
    assert delivery_proof.CHAT_QUEUE_DIRNAME == chat_queue.DIRNAME
    assert delivery_proof.CHAT_QUEUE_STALE_DIRNAME == chat_queue.STALE_DIRNAME


# ── the receipt veto that was REMOVED ────────────────────────────────


def test_this_module_offers_no_timestamp_proof_of_delivery():
    """A regression guard on a design decision, not on code.

    "A receipt newer than the last failed turn means the channel worked" was
    implemented here and removed in blind review: the apology the user gets
    for a failed turn goes out through the outbox too, so its receipt always
    postdates that turn's ledger row. Every streak the delivery backstop
    exists for satisfies the gate, which disarms the backstop entirely while
    looking like a careful check.

    Anyone reaching for it again should read the module docstring first."""
    assert not hasattr(delivery_proof, "last_receipt_ts")
    assert not hasattr(delivery_proof, "delivered_since")


# ── proof of loss ────────────────────────────────────────────────────


def test_a_healthy_install_proves_no_loss(tmp_path):
    """NO FALSE ALARM: an idle install, and one in the middle of an ordinary
    turn, must both read as "nothing lost"."""
    assert delivery_proof.losses(tmp_path, now=1000.0).any is False

    box = inbox.Inbox(tmp_path, clock_fn=lambda: 1000.0)
    box.accept({"update_id": 5, "message": {"chat": {"id": 1}, "text": "привет"}})
    # A turn in flight — accepted a moment ago, being answered right now.
    assert delivery_proof.losses(tmp_path, now=1010.0).any is False


def test_a_reply_in_the_dead_queue_is_a_proven_loss(tmp_path):
    """THE REAL CASE the alert must always survive for: the answer existed
    and could not be delivered."""
    box = outbox.Outbox(tmp_path)
    item = box.enqueue(7, "ответ")
    box.bury(item, reason="refused", now=1000.0)

    loss = delivery_proof.losses(tmp_path, now=1000.0)
    assert loss.dead_replies == 1
    assert loss.any is True
    assert any("мёртвой очереди" in r for r in loss.reasons())


def test_an_accepted_message_retired_unanswered_is_a_proven_loss(tmp_path):
    box = inbox.Inbox(tmp_path, clock_fn=lambda: 1000.0)
    entry = box.accept({"update_id": 9, "message": {"chat": {"id": 1}, "text": "?"}})
    box.retire(entry, reason="too old to replay")

    loss = delivery_proof.losses(tmp_path, now=1000.0)
    assert loss.retired_messages == 1
    assert loss.any is True


def test_an_accepted_message_nobody_answers_is_a_proven_loss(tmp_path):
    """Accepted, and untouched long past any legitimate turn."""
    import os

    box = inbox.Inbox(tmp_path, clock_fn=lambda: 1000.0)
    box.accept({"update_id": 11, "message": {"chat": {"id": 1}, "text": "?"}})
    entry = next((tmp_path / delivery_proof.INBOX_DIRNAME).glob("*.json"))
    touched = entry.stat().st_mtime

    fresh = delivery_proof.losses(tmp_path, now=touched + 60)
    assert fresh.any is False  # still being worked on
    stuck = delivery_proof.losses(
        tmp_path, now=touched + delivery_proof.DEFAULT_STUCK_AFTER + 1
    )
    assert stuck.stranded_messages == 1
    assert stuck.any is True
    assert os.path.exists(entry)


def test_a_long_but_legitimate_turn_is_not_a_stranded_message(tmp_path):
    """NO FALSE ALARM (blind review 5). The inbox entry is removed only when
    the whole handler chain returns, and chat.py runs a SECOND full turn when
    the first comes back empty — so one message can legitimately hold its
    entry for two `chat_turn_timeout`s plus a busy-wait. The threshold has to
    clear that, or every long turn cries wolf."""
    from d_brain.config import Settings

    turn_timeout = Settings.model_fields["chat_turn_timeout"].default
    two_turns = 2 * turn_timeout
    assert delivery_proof.DEFAULT_STUCK_AFTER > two_turns

    box = inbox.Inbox(tmp_path, clock_fn=lambda: 1000.0)
    box.accept({"update_id": 12, "message": {"chat": {"id": 1}, "text": "?"}})
    entry = next((tmp_path / delivery_proof.INBOX_DIRNAME).glob("*.json"))
    at = entry.stat().st_mtime
    assert delivery_proof.losses(tmp_path, now=at + two_turns).any is False


def test_a_replay_attempt_clears_the_strand(tmp_path):
    """An entry a boot replay has picked up is being worked on, not
    abandoned — and `delivery-watch.sh` can only see mtime, so both readers
    have to judge it the same way (blind review 8)."""
    import os

    box = inbox.Inbox(tmp_path, clock_fn=lambda: 1000.0)
    entry = box.accept({"update_id": 13, "message": {"chat": {"id": 1}, "text": "?"}})
    path = next((tmp_path / delivery_proof.INBOX_DIRNAME).glob("*.json"))
    old = 1000.0
    os.utime(path, (old, old))
    assert delivery_proof.losses(
        tmp_path, now=old + delivery_proof.DEFAULT_STUCK_AFTER + 1
    ).stranded_messages == 1

    box.record_attempt(entry)  # the replay touches it
    assert delivery_proof.losses(
        tmp_path, now=old + delivery_proof.DEFAULT_STUCK_AFTER + 1
    ).any is False


def test_an_unreadable_accepted_entry_counts_and_is_not_moved(tmp_path):
    """A read-only check must not quarantine what it is describing — but an
    entry nobody has touched in hours is a loss whether it parses or not."""
    import os

    d = tmp_path / delivery_proof.INBOX_DIRNAME
    d.mkdir(parents=True)
    broken = d / "0000000000000000042.json"
    broken.write_text("{ truncated")
    os.utime(broken, (1000.0, 1000.0))

    loss = delivery_proof.losses(
        tmp_path, now=1000.0 + delivery_proof.DEFAULT_STUCK_AFTER + 1
    )
    assert loss.stranded_messages == 1
    assert broken.exists()


def test_an_entry_that_vanishes_mid_check_is_not_a_loss(tmp_path):
    """NO FALSE ALARM (blind review 7): a turn finishing between the glob
    and the stat is an ordinary race on a busy install — and the OPPOSITE of
    a loss. Counting it made the alert nondeterministic under load."""
    import os

    d = tmp_path / delivery_proof.INBOX_DIRNAME
    d.mkdir(parents=True)
    (d / "0000000000000000043.json").write_text("{}")

    real_stat = os.stat_result.__class__  # noqa: F841 - readability only
    original = Path.stat

    def vanishing(self, *a, **kw):
        if self.name == "0000000000000000043.json":
            raise FileNotFoundError(self)
        return original(self, *a, **kw)

    Path.stat = vanishing  # type: ignore[method-assign]
    try:
        assert delivery_proof.losses(tmp_path, now=1e12).any is False
    finally:
        Path.stat = original  # type: ignore[method-assign]


# ── the "already reported" latch ─────────────────────────────────────


def test_a_new_loss_is_new_even_when_the_count_did_not_move(tmp_path):
    """Blind review 4: one loss cleared and a different one appearing in the
    same interval leaves the count unchanged — a counter-based latch would
    never report the second message."""
    box = outbox.Outbox(tmp_path)
    first = box.enqueue(7, "a")
    box.bury(first, reason="refused", now=1000.0)
    before = delivery_proof.losses(tmp_path, now=1000.0)

    (tmp_path / "outbox" / "dead" / f"{first.id}.json").unlink()
    second = box.enqueue(7, "b")
    box.bury(second, reason="refused", now=1000.0)
    after = delivery_proof.losses(tmp_path, now=1000.0)

    assert after.total == before.total == 1
    assert after.new_since(before.ids)


def test_a_loss_already_reported_is_not_new(tmp_path):
    box = outbox.Outbox(tmp_path)
    box.bury(box.enqueue(7, "a"), reason="refused", now=1000.0)
    loss = delivery_proof.losses(tmp_path, now=1000.0)

    delivery_proof.write_reported(tmp_path, loss.ids)
    assert delivery_proof.read_reported(tmp_path) == loss.ids
    assert loss.new_since(delivery_proof.read_reported(tmp_path)) == ()


def test_a_missing_latch_reports_everything(tmp_path):
    """Absence of a latch must mean "nothing told yet", never "all told"."""
    assert delivery_proof.read_reported(tmp_path) == ()
    box = outbox.Outbox(tmp_path)
    box.bury(box.enqueue(7, "a"), reason="refused", now=1000.0)
    loss = delivery_proof.losses(tmp_path, now=1000.0)
    assert loss.new_since(delivery_proof.read_reported(tmp_path)) == loss.ids


def test_the_reasons_never_carry_a_number(tmp_path):
    """notify.sh debounces on a checksum of the text, so a count inside it
    mints a new message every time it moves — which is how a false-alarm
    morning alert arrived every five minutes."""
    box = outbox.Outbox(tmp_path)
    for _ in range(4):
        box.bury(box.enqueue(7, "x"), reason="refused", now=1000.0)
    reasons = delivery_proof.losses(tmp_path, now=1000.0).reasons()
    assert reasons
    assert not any(ch.isdigit() for r in reasons for ch in r)


def test_a_chat_queue_job_retired_unanswered_is_a_proven_loss(tmp_path):
    queue = chat_queue.ChatQueue(tmp_path, clock_fn=lambda: 1000.0)
    job, _waiting = queue.enqueue(chat_id=7, user_id=7, message_id=1, prompt="?")
    queue.retire(job, reason="too old")

    loss = delivery_proof.losses(tmp_path, now=1000.0)
    assert loss.retired_jobs == 1
    assert loss.any is True


def test_losses_never_raise_on_a_half_built_runtime_dir(tmp_path):
    """Alerting paths run in exactly the conditions that break things."""
    (tmp_path / delivery_proof.OUTBOX_DIRNAME).mkdir()
    (tmp_path / delivery_proof.INBOX_DIRNAME).write_text("not a directory")
    loss = delivery_proof.losses(tmp_path, now=1000.0)
    assert loss.any is False
    assert json.dumps(loss.reasons()) == "[]"
