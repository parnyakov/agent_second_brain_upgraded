"""Delivery metric over the reply ledger: the share of NOT delivered replies.

Inputs are the two files the reply-ledger path already writes in a session's
runtime dir — ``turn-ledger.jsonl`` (what the CLI did, from its hooks) and
``turn-ledger.claims.jsonl`` (who delivered what, with latency). Nothing new
is logged for this; the metric is a read-only fold, so it can be recomputed
for any window and compared before/after a change.

Per CLI turn:

* ``bound``    — carries an ``ask``/``maint`` tag: a request with an ask()
  waiting on it;
* ``steer``    — no ask, but a steered message joined it;
* ``notify``   — CLI-initiated (``<task-notification>``), no tag;
* ``untagged`` — a prompt with no bot tag (typed in ``dbrain attach``, a
  nudge, a pre-ledger prompt).

Outcome: ``delivered`` (by the ask or the pump), ``sentinel``, ``empty``,
``failed`` (StopFailure, or the ask returned an error status),
``interrupted``, ``pending`` (still open / within grace), ``lost`` (ended,
nobody delivered it, past grace — or never ended at all).

A turn NEEDS a reply when it carries a request (bound/steer), or when it is
an unbound turn whose final message is real output (not sentinel, not
empty). The headline rate is ``not delivered / needs reply``, where "not
delivered" is ``lost`` + ``failed``. ``interrupted`` turns are reported but
not counted (a /stop is the person's own choice) — except a turn the bot
itself interrupted on a stall, which is ``failed``.

"Delivered" means handed to the durable outbox: by the pump directly, or by
``ask()`` returning the reply to the chat layer, which queues it there. A
reply that later dies in ``outbox/dead/`` is reported by delivery_proof, not
here.

Asks that never reached the CLI (no submit hook for their rid) count as not
delivered unless the request went elsewhere by design: ``busy``/
``busy_active`` (parked in the chat queue or answered by the duty session)
and ``reset``.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from d_brain.services import turn_ledger

LOST_AFTER = 120.0  # an ended turn nobody delivered for this long is lost
NEVER_ENDED_AFTER = 3 * 3600.0  # an open turn this old never ended
# ask() statuses whose request is answered by another path by design.
_ROUTED_ELSEWHERE = ("ok", "busy", "busy_active", "reset")


@dataclass
class TurnOutcome:
    prompt_id: str
    kind: str
    outcome: str
    first_ts: float
    latency: float | None = None
    by: str | None = None
    needs_reply: bool = True


@dataclass
class Report:
    since: float
    until: float
    turns: list[TurnOutcome] = field(default_factory=list)
    asks_never_seen: int = 0

    def _needing(self) -> list[TurnOutcome]:
        return [t for t in self.turns if t.needs_reply and t.outcome != "interrupted"]

    @property
    def needs_reply(self) -> int:
        return len(self._needing()) + self.asks_never_seen

    @property
    def not_delivered(self) -> int:
        bad = sum(1 for t in self._needing() if t.outcome in ("lost", "failed"))
        return bad + self.asks_never_seen

    @property
    def rate(self) -> float | None:
        return self.not_delivered / self.needs_reply if self.needs_reply else None

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for t in self.turns:
            out.setdefault(t.kind, {}).setdefault(t.outcome, 0)
            out[t.kind][t.outcome] += 1
        return out

    def latencies(self) -> dict[str, float]:
        lat = sorted(
            t.latency
            for t in self.turns
            if t.outcome == "delivered" and t.latency is not None
        )
        if not lat:
            return {}
        p90 = lat[min(len(lat) - 1, int(round(0.9 * (len(lat) - 1))))]
        return {"p50": statistics.median(lat), "p90": p90, "max": lat[-1]}

    def as_dict(self) -> dict:
        return {
            "since": self.since,
            "until": self.until,
            "needs_reply": self.needs_reply,
            "not_delivered": self.not_delivered,
            "rate": self.rate,
            "asks_never_seen": self.asks_never_seen,
            "counts": self.counts(),
            "latency": self.latencies(),
        }


def _kind(turn: turn_ledger.Turn) -> str:
    if turn.bound_rid:
        return "bound"
    if "steer" in turn.kinds:
        return "steer"
    if turn.origin == "task-notification":
        return "notify"
    return "untagged"


def compute(
    runtime_dir: Path | str,
    *,
    since: float,
    until: float | None = None,
    now: float | None = None,
) -> Report:
    now = time.time() if now is None else now
    until = now if until is None else until
    book = turn_ledger.TurnBook()
    book.extend(turn_ledger.read_all(runtime_dir))
    claim_recs = turn_ledger.Claims(runtime_dir).records()
    final: dict[str, dict] = {}
    delivered: dict[str, dict] = {}
    ask_status: dict[str, dict] = {}  # rid -> last ask_outcome
    for rec in claim_recs:
        if rec.get("state") == "ask_outcome":
            if rec.get("rid"):
                ask_status[str(rec["rid"])] = rec
            continue
        pid = rec.get("prompt_id")
        if not pid:
            continue
        final[str(pid)] = rec
        if rec.get("state") == "delivered":
            delivered[str(pid)] = rec

    report = Report(since=since, until=until)
    seen_rids: set[str] = set()
    for turn in book.ordered():
        seen_rids.update(turn.rids)
        if not (since <= turn.first_ts < until):
            continue
        kind = _kind(turn)
        rec = final.get(turn.prompt_id)
        state = rec.get("state") if rec else None
        out = TurnOutcome(turn.prompt_id, kind, "lost", turn.first_ts)
        if turn.prompt_id in delivered:
            d = delivered[turn.prompt_id]
            out.outcome, out.by = "delivered", d.get("by")
            out.latency = d.get("latency")
        elif (
            turn.bound_rid and ask_status.get(turn.bound_rid, {}).get("status") == "ok"
        ):
            # Marker path (ledger off): the ask returned the reply itself.
            out.outcome, out.by = "delivered", "ask"
        elif state in ("sentinel", "empty"):
            out.outcome = state
        elif state == "interrupted" or (
            turn.interrupted_ts is not None and turn.stop is None
        ):
            stalled = "stalled" in str(
                ask_status.get(turn.bound_rid or "", {}).get("detail") or ""
            )
            out.outcome = "failed" if stalled else "interrupted"
        elif state == "failed" or turn.failure is not None:
            out.outcome = "failed"
        elif (
            turn.bound_rid
            and turn.stop is None
            and (
                ask_status.get(turn.bound_rid, {}).get("status")
                not in (None, "timeout")
            )
        ):
            out.outcome = "failed"  # the ask answered with an error status
        elif turn.closed:
            end = turn.end_ts or turn.first_ts
            out.outcome = "pending" if now - end < LOST_AFTER else "lost"
        else:
            out.outcome = (
                "pending" if now - turn.first_ts < NEVER_ENDED_AFTER else "lost"
            )
        if kind in ("notify", "untagged") and out.outcome in ("sentinel", "empty"):
            out.needs_reply = False
        if kind == "notify" and out.outcome in ("pending", "interrupted"):
            out.needs_reply = False
        if out.outcome == "pending":
            out.needs_reply = False
        report.turns.append(out)

    for rid, rec in ask_status.items():
        ts = float(rec.get("ts") or 0.0)
        status = rec.get("status")
        if status in _ROUTED_ELSEWHERE:
            continue
        if rid not in seen_rids and since <= ts < until:
            # The ask's prompt never reached the CLI (no submit hook at all).
            report.asks_never_seen += 1
    return report


def render(report: Report) -> str:
    def fmt(ts: float) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))

    lines = [f"Окно: {fmt(report.since)} — {fmt(report.until)}"]
    rate = report.rate
    lines.append(
        "Не доставлено: "
        + (
            f"{report.not_delivered} из {report.needs_reply} ({rate:.1%})"
            if rate is not None
            else "нет ходов, требующих ответа"
        )
    )
    if report.asks_never_seen:
        lines.append(f"  из них запрос не дошёл до CLI: {report.asks_never_seen}")
    for kind, outcomes in sorted(report.counts().items()):
        parts = ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items()))
        lines.append(f"  {kind}: {parts}")
    lat = report.latencies()
    if lat:
        lines.append(
            "Задержка доставки, с: p50={p50:.1f} p90={p90:.1f} max={max:.1f}".format(
                **lat
            )
        )
    return "\n".join(lines)


def parse_window(text: str) -> float:
    """'24h', '7d', '90m' → seconds."""
    text = text.strip().lower()
    units = {"m": 60.0, "h": 3600.0, "d": 86400.0}
    if text and text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)
