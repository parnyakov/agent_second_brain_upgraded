#!/usr/bin/env python3
"""Live smoke test for CodexExecDriver — real `codex exec`, no mocks.

Manual, standalone, and DELIBERATELY NOT WIRED INTO THE BOT. It constructs a
CodexExecDriver directly against a throwaway runtime dir and a working root
you pass in; it never reads Settings, never touches the live engine selection
(`DBRAIN_CHAT_ENGINE` / `DBRAIN_CRON_ENGINE`), and never touches the running
services. Flipping anything live is a separate, explicit decision (see
agent-infra-backlog item 26).

This is the "one real, honest check" the compressed phase-2 scope asks for:
unit tests prove the parsing and the state machine, this proves the parsing
matches what the real CLI actually emits today.

    uv run scripts/codex-smoke.py                       # full run
    uv run scripts/codex-smoke.py --work-dir /some/dir  # pick the working root
    uv run scripts/codex-smoke.py --keep                # keep the runtime dir

What it exercises, in order:
  1. a fresh-thread ask()  → AskResult(status="ok"), thread_id persisted
  2. a resumed ask()       → same thread_id, continuity (the model recalls
                             turn 1), which is what `codex exec resume` buys
  3. send_control("/clear")→ the thread is dropped
  4. a post-clear ask()    → a NEW thread_id (Codex's own "fresh context")
  5. read-only probes      → is_healthy / is_turn_active / capture_text /
                             last_reply_for_resend / pop_orphan_replies

Sandbox: `workspace-write` (the default), scoped to --work-dir. That is write
protection, not read isolation — `--sandbox workspace-write` does not restrict
reads (phase-0 spike, step 3). The owner removed the privacy half of that finding
on 2026-09-05 and kept the write-protection half, which is why the default is
kept as-is here.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d_brain.services.codex_driver import CodexExecDriver  # noqa: E402

PERSONA = Path(__file__).resolve().parents[1] / "deploy" / "codex-agents.md"

RED, GREEN, DIM, OFF = "\033[31m", "\033[32m", "\033[2m", "\033[0m"
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = f"{GREEN}PASS{OFF}" if ok else f"{RED}FAIL{OFF}"
    print(f"  [{mark}] {label}" + (f"  {DIM}{detail}{OFF}" if detail else ""))
    if not ok:
        failures.append(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="working root for the codex thread (default: a temp dir)",
    )
    ap.add_argument("--keep", action="store_true", help="keep the runtime dir")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    runtime_dir = Path(tempfile.mkdtemp(prefix="codex-smoke-rt-"))
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="codex-smoke-wd-"))
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"codex smoke test\n  work_dir    {work_dir}\n  runtime_dir {runtime_dir}\n")

    drv = CodexExecDriver(
        session_name="codex_smoke",
        work_dir=work_dir,
        runtime_dir=runtime_dir,
        instructions_file=PERSONA if PERSONA.exists() else None,
    )

    print("0. pre-flight")
    check("is_healthy()", drv.is_healthy())
    check("is_turn_active() is False before anything", drv.is_turn_active() is False)
    check(
        "last_reply_for_resend() == ('unavailable', None)",
        drv.last_reply_for_resend() == ("unavailable", None),
    )

    print("\n1. fresh thread")
    r1 = drv.ask(
        "Ответь ровно одним словом: КОДЕКС. Больше ничего не пиши.",
        timeout=args.timeout,
        request_id="smoke-1",
    )
    print(f"   {DIM}AskResult(status={r1.status!r}, reply={r1.reply!r}){OFF}")
    check("status == 'ok'", r1.status == "ok", f"got {r1.status!r} / {r1.detail!r}")
    check("reply is non-empty", bool(r1.reply))
    thread1 = (
        (runtime_dir / "thread_id").read_text().strip()
        if (runtime_dir / "thread_id").exists()
        else None
    )
    check("thread_id persisted", bool(thread1), thread1 or "")

    print("\n2. resume the same thread (continuity)")
    r2 = drv.ask(
        "Какое слово ты только что сказал? Ответь одним словом.",
        timeout=args.timeout,
        request_id="smoke-2",
    )
    print(f"   {DIM}AskResult(status={r2.status!r}, reply={r2.reply!r}){OFF}")
    thread2 = (runtime_dir / "thread_id").read_text().strip()
    check("status == 'ok'", r2.status == "ok", f"got {r2.status!r} / {r2.detail!r}")
    check("thread_id unchanged", thread2 == thread1, f"{thread1} -> {thread2}")
    check(
        "the model remembers turn 1",
        "кодекс" in (r2.reply or "").lower(),
        repr(r2.reply),
    )

    print("\n3. send_control('/clear')")
    drv.send_control("/clear")
    check("thread_id dropped", not (runtime_dir / "thread_id").exists())

    print("\n4. a new thread after /clear")
    r3 = drv.ask("Скажи ровно: ОК", timeout=args.timeout, request_id="smoke-3")
    thread3 = (runtime_dir / "thread_id").read_text().strip()
    print(f"   {DIM}AskResult(status={r3.status!r}, reply={r3.reply!r}){OFF}")
    check("status == 'ok'", r3.status == "ok", f"got {r3.status!r} / {r3.detail!r}")
    check("a genuinely new thread_id", thread3 != thread1, f"{thread1} -> {thread3}")

    print("\n5. read-only probes after the turns")
    status, body = drv.last_reply_for_resend()
    check("last_reply_for_resend() == ('ready', <last reply>)", status == "ready")
    check("resend body matches the last reply", body == r3.reply, repr(body))
    check("pop_orphan_replies() == []", drv.pop_orphan_replies() == [])
    check("is_turn_active() is False again", drv.is_turn_active() is False)
    check("is_pane_turn_active() is False", drv.is_pane_turn_active() is False)
    cap = drv.capture_text()
    check("capture_text() shows the turn journal", "status=ok" in cap)
    print(f"{DIM}--- capture_text() ---\n{cap}{DIM}----------------------{OFF}")

    if not args.keep:
        shutil.rmtree(runtime_dir, ignore_errors=True)
    else:
        print(f"\nruntime dir kept: {runtime_dir}")

    print()
    if failures:
        print(f"{RED}{len(failures)} check(s) failed:{OFF} " + ", ".join(failures))
        return 1
    print(f"{GREEN}all checks passed{OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
