"""The turn ledger: which CLI turn answered which request (2026-10-29).

Replaces the echo binding (the model copying ``<<<R:rid>>>``/``<<<E:rid>>>``)
with facts the CLI itself reports through hooks. ``scripts/turn_hook.py``
appends one line per event to ``<runtime_dir>/turn-ledger.jsonl``:

* ``submit`` — a UserPromptSubmit: ``prompt_id``, ``session_id`` and the
  ``rid``/``kind`` of the bot's machine tag (:func:`turn_tag`) if the prompt
  carries one. A message typed into a running turn fires another submit
  with the SAME ``prompt_id`` (verified live on 2.1.312): it joins that turn.
* ``stop`` — the turn's final assistant message (``text``) and the count of
  background tasks still running. The final message IS the reply.
* ``stop_failure`` — the turn died on an API error.
* ``session_start`` — informational (``/clear``, compaction, restart).

The bot adds ``interrupt`` lines itself (an Escape fires no Stop).

Delivery is decided by CLAIMS (``turn-ledger.claims.jsonl``, check-and-append
under ``turn-ledger.lock``): exactly one party — the ``ask()`` waiting on the
turn, or the unsolicited-reply pump — takes a turn's reply, deduped by
``prompt_id``. The same file is the delivery log the metric script reads.

Nothing here touches the pane or the ask-lock.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

LEDGER_FILE = "turn-ledger.jsonl"
CLAIMS_FILE = "turn-ledger.claims.jsonl"
LOCK_FILE = "turn-ledger.lock"
CLAIMS_MAX = 4 * 1024 * 1024

# A reply that starts with this is "nothing to report": a background
# notification turn ends quietly instead of pinging the owner.
SENTINEL = "[SILENT]"
# What a turn that ended without any text turns into. Per instance
# (Settings.empty_reply_notice): on => the owner gets this line, a visible
# "something went quiet" signal; off (default) => the turn is treated like a
# bare [SILENT]: recorded (claim state "empty"), nothing is sent.
EMPTY_REPLY = "✅ Ход завершён без текстового ответа."

TAG_KINDS = ("ask", "steer", "maint")
# Keep in sync with scripts/turn_hook.py (pinned by a test).
TAG_RE = re.compile(
    r'<dbrain-turn rid="([A-Za-z0-9_-]{1,64})" kind="(ask|steer|maint)"\s*/>'
)
# Line-only legacy markers a model may still write out of habit (its own
# history is full of them). Stripped from a delivered reply, never required.
_LEGACY_MARKER_LINE = re.compile(
    r"(?m)^[ \t⏺>*-]*<<<[RE]:[A-Za-z0-9_-]+>>>[ \t]*\r?\n?"
)
_LEGACY_MARKER_INLINE = re.compile(r"<<<[RE]:[A-Za-z0-9_-]+>>>")

# Terminal claim states: once one of these is recorded for a prompt_id,
# nobody delivers that turn again.
TERMINAL = (
    "claimed",
    "delivered",
    "sentinel",
    "empty",
    "failed",
    "interrupted",
    "skipped",
)


def turn_tag(rid: str, kind: str) -> str:
    if kind not in TAG_KINDS:
        raise ValueError(f"unknown turn tag kind: {kind!r}")
    return f'<dbrain-turn rid="{rid}" kind="{kind}"/>'


def tagged(prompt: str, rid: str, kind: str) -> str:
    """The prompt with the bot's machine tag appended on its own line."""
    return f"{prompt}\n\n{turn_tag(rid, kind)}"


def parse_tag(prompt: str) -> tuple[str, str] | None:
    found = TAG_RE.findall(prompt or "")
    return found[-1] if found else None


def clean_reply(text: str | None) -> str:
    """The deliverable reply: legacy marker lines and stray tags removed."""
    if not text:
        return ""
    out = _LEGACY_MARKER_LINE.sub("", text)
    out = _LEGACY_MARKER_INLINE.sub("", out)
    out = TAG_RE.sub("", out)
    return out.strip()


def is_silent(text: str | None) -> bool:
    return bool(text) and text.lstrip().startswith(SENTINEL)


def strip_sentinel(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith(SENTINEL):
        return stripped[len(SENTINEL) :].strip()
    return text


# ── reading the ledger ───────────────────────────────────────────────────


class LedgerReader:
    """Incremental reader over ``turn-ledger.jsonl`` that survives the hook's
    size rotation (``.jsonl`` → ``.jsonl.1``). Only whole lines are consumed."""

    def __init__(self, runtime_dir: Path | str) -> None:
        self.path = Path(runtime_dir) / LEDGER_FILE
        self._ino: int | None = None
        self._offset = 0

    @classmethod
    def at_end(cls, runtime_dir: Path | str) -> LedgerReader:
        reader = cls(runtime_dir)
        try:
            st = reader.path.stat()
            reader._ino, reader._offset = st.st_ino, st.st_size
        except OSError:
            pass
        return reader

    @staticmethod
    def _read_from(path: Path, offset: int) -> tuple[list[dict], int]:
        try:
            with path.open("rb") as f:
                f.seek(offset)
                chunk = f.read()
        except OSError:
            return [], 0
        out: list[dict] = []
        consumed = 0
        start = 0
        while True:
            nl = chunk.find(b"\n", start)
            if nl == -1:
                break
            line = chunk[start:nl].strip()
            consumed = start = nl + 1
            if not line:
                continue
            try:
                rec = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(rec, dict):
                out.append(rec)
        return out, consumed

    def read_new(self) -> list[dict]:
        """Every complete record appended since the last call. Never raises."""
        try:
            st = self.path.stat()
        except OSError:
            return []
        records: list[dict] = []
        if self._ino is not None and st.st_ino != self._ino:
            # Rotated: finish the old generation first, if it is still there.
            rotated = self.path.with_name(self.path.name + ".1")
            try:
                if rotated.stat().st_ino == self._ino:
                    old, _ = self._read_from(rotated, self._offset)
                    records.extend(old)
            except OSError:
                pass
            self._offset = 0
        self._ino = st.st_ino
        new, consumed = self._read_from(self.path, self._offset)
        self._offset += consumed
        records.extend(new)
        return records


def read_rotated(runtime_dir: Path | str) -> list[dict]:
    """The rotated generation (``.jsonl.1``) only."""
    path = Path(runtime_dir) / LEDGER_FILE
    return LedgerReader._read_from(path.with_name(path.name + ".1"), 0)[0]


def read_all(runtime_dir: Path | str) -> list[dict]:
    """The whole ledger, rotated generation first (for tools and the pump's
    first pass)."""
    path = Path(runtime_dir) / LEDGER_FILE
    old, _ = LedgerReader._read_from(path.with_name(path.name + ".1"), 0)
    new, _ = LedgerReader._read_from(path, 0)
    return old + new


def append_bot_event(runtime_dir: Path | str, ev: str, **fields) -> None:
    """A line written by the bot itself (``interrupt``). Best effort."""
    try:
        path = Path(runtime_dir) / LEDGER_FILE
        rec = {"ev": ev, "ts": time.time(), "by": "bot", **fields}
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, (json.dumps(rec, ensure_ascii=False) + "\n").encode())
        finally:
            os.close(fd)
    except OSError as exc:
        logger.warning("turn-ledger: could not append %s: %s", ev, exc)


# ── folding events into turns ────────────────────────────────────────────


@dataclass
class Turn:
    prompt_id: str
    first_ts: float
    session_id: str = ""
    origin: str = "user"
    tags: list[tuple[str, str]] = field(default_factory=list)
    submits: int = 0
    stop: dict | None = None
    stops: int = 0
    failure: dict | None = None
    interrupted_ts: float | None = None

    @property
    def rids(self) -> list[str]:
        return [rid for rid, _ in self.tags]

    @property
    def bound_rid(self) -> str | None:
        """The rid of the ``ask()`` this turn belongs to, if any."""
        for rid, kind in self.tags:
            if kind in ("ask", "maint"):
                return rid
        return None

    @property
    def is_maint(self) -> bool:
        return any(kind == "maint" for _, kind in self.tags)

    @property
    def kinds(self) -> list[str]:
        return [kind for _, kind in self.tags]

    @property
    def has_request(self) -> bool:
        """Carries something a person asked (an ask or a steered message)."""
        return any(k in ("ask", "steer") for k in self.kinds) or (
            self.origin == "user" and not self.tags
        )

    @property
    def end_ts(self) -> float | None:
        if self.stop is not None:
            return float(self.stop.get("ts") or 0.0)
        if self.failure is not None:
            return float(self.failure.get("ts") or 0.0)
        return self.interrupted_ts

    @property
    def closed(self) -> bool:
        return self.end_ts is not None


class TurnBook:
    """Turns by ``prompt_id``, folded incrementally from ledger records.

    Records without a ``prompt_id`` (a CLI that predates the field) degrade
    to positional binding: a submit joins the open turn if there is one,
    otherwise starts an anonymous turn; a stop closes the newest open turn."""

    def __init__(self) -> None:
        self.turns: dict[str, Turn] = {}
        self._order: list[str] = []
        self._anon = 0

    def _open_key(self) -> str | None:
        for key in reversed(self._order):
            if not self.turns[key].closed:
                return key
        return None

    def add(self, rec: dict) -> Turn | None:
        ev = rec.get("ev")
        pid = str(rec.get("prompt_id") or "")
        ts = float(rec.get("ts") or 0.0)
        if ev == "submit":
            if not pid:
                pid = self._open_key() or ""
                if not pid:
                    self._anon += 1
                    pid = f"anon-{ts:.6f}-{self._anon}"
            turn = self.turns.get(pid)
            if turn is None:
                turn = Turn(
                    prompt_id=pid,
                    first_ts=ts,
                    session_id=str(rec.get("session_id") or ""),
                    origin=str(rec.get("origin") or "user"),
                )
                self.turns[pid] = turn
                self._order.append(pid)
            turn.submits += 1
            if turn.interrupted_ts is not None and turn.stop is None:
                # Input after an Escape that the CLI filed under the same
                # prompt_id: the turn runs again, its Stop is still coming.
                turn.interrupted_ts = None
            rid, kind = rec.get("rid"), rec.get("kind")
            if rid and kind in TAG_KINDS:
                turn.tags.append((str(rid), str(kind)))
            return turn
        if ev in ("stop", "stop_failure", "interrupt"):
            if not pid:
                pid = self._open_key() or ""
            turn = self.turns.get(pid)
            if turn is None and ev == "stop" and pid:
                # A Stop whose submit this reader never saw (the submit hook
                # failed, or it predates the reader). Its final message is
                # still a reply: keep it as a turn of unknown origin rather
                # than drop it.
                turn = Turn(
                    prompt_id=pid,
                    first_ts=ts,
                    session_id=str(rec.get("session_id") or ""),
                    origin="unknown",
                )
                self.turns[pid] = turn
                self._order.append(pid)
            if turn is None:
                return None
            if ev == "stop":
                turn.stops += 1
                turn.stop = rec  # the LAST stop of a prompt_id is its final word
            elif ev == "stop_failure":
                turn.failure = rec
            elif turn.interrupted_ts is None and turn.stop is None:
                turn.interrupted_ts = ts
            return turn
        return None

    def extend(self, records: Iterable[dict]) -> None:
        for rec in records:
            try:
                self.add(rec)
            except (TypeError, ValueError):
                continue

    def by_rid(self, rid: str) -> Turn | None:
        for key in reversed(self._order):
            if rid in self.turns[key].rids:
                return self.turns[key]
        return None

    def ordered(self) -> list[Turn]:
        return [self.turns[k] for k in self._order]

    def prune(self, before_ts: float) -> None:
        """Forget turns that ended (or, never closed, began) before
        ``before_ts`` — keeps a long-running reader's memory bounded."""
        keep: list[str] = []
        for key in self._order:
            turn = self.turns[key]
            end = turn.end_ts
            if (end if end is not None else turn.first_ts) < before_ts:
                del self.turns[key]
            else:
                keep.append(key)
        self._order = keep

    def last_open(self) -> Turn | None:
        key = self._open_key()
        return self.turns[key] if key else None


def stop_text(turn: Turn, work_dir: Path | None = None) -> str | None:
    """The turn's final message: from the Stop payload, else (older CLI) from
    the transcript the Stop pointed at."""
    if turn.stop is None:
        return None
    text = turn.stop.get("text")
    if isinstance(text, str):
        return text
    path = turn.stop.get("transcript_path")
    if path:
        from d_brain.services.transcript import turn_end

        end = turn_end(path, prompt_id=turn.prompt_id)
        if end is not None:
            return end.text
    return None


# ── claims: who delivered what (also the metric log) ─────────────────────


class Claims:
    def __init__(self, runtime_dir: Path | str) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.path = self.runtime_dir / CLAIMS_FILE
        self._lock = self.runtime_dir / LOCK_FILE

    @contextmanager
    def _locked(self):
        fd = os.open(self._lock, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def records(self) -> list[dict]:
        old, _ = LedgerReader._read_from(self.path.with_name(self.path.name + ".1"), 0)
        new, _ = LedgerReader._read_from(self.path, 0)
        return old + new

    def states(self) -> dict[str, dict]:
        """Last claim record per prompt_id."""
        out: dict[str, dict] = {}
        for rec in self.records():
            pid = rec.get("prompt_id")
            if pid and rec.get("state") != "ask_outcome":
                out[str(pid)] = rec
        return out

    def _append(self, rec: dict) -> None:
        try:
            if self.path.stat().st_size > CLAIMS_MAX:
                os.replace(self.path, self.path.with_name(self.path.name + ".1"))
        except OSError:
            pass
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, (json.dumps(rec, ensure_ascii=False) + "\n").encode())
        finally:
            os.close(fd)

    def claim(
        self, prompt_id: str, *, by: str, state: str = "claimed", **fields
    ) -> bool:
        """Take this turn's reply. False if somebody already did."""
        with self._locked():
            prev = self.states().get(prompt_id)
            if prev is not None and prev.get("state") in TERMINAL:
                return False
            self._append(
                {
                    "ts": time.time(),
                    "prompt_id": prompt_id,
                    "by": by,
                    "state": state,
                    **fields,
                }
            )
            return True

    def record(self, **fields) -> None:
        """A follow-up or metric-only line (no dedup decision). Best effort."""
        try:
            with self._locked():
                self._append({"ts": time.time(), **fields})
        except OSError as exc:
            logger.warning("turn-ledger: could not record %s: %s", fields, exc)
