#!/usr/bin/env python3
"""Conservative daily cleanup of abandoned interactive sessions and caches."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import html
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from uuid import UUID
from zoneinfo import ZoneInfo

WEEK = 7 * 86400
ARCHIVE_DAILY_LIMIT = 1024 * 1024 * 1024
STATE = Path.home() / ".local/state/server-cleanup"
CONFIG = Path.home() / ".config/server-cleanup"
PROTECTED = CONFIG / "protected.json"
# The report is timestamped in the owner's timezone: the installer writes TZ
# to .env and systemd passes it in. An unknown or missing zone falls back to
# the machine's own, so a bad value never stops the cleanup.
def _report_zone():
    name = os.environ.get("TZ") or os.environ.get("DBRAIN_TIMEZONE")
    if name:
        try:
            return ZoneInfo(name)
        except Exception:
            pass
    return datetime.now().astimezone().tzinfo


TZ = _report_zone()


CORE_PREFIXES = ("dbrain_",)
IDLE_CODEX = "Ask Codex to do anything"


def command(*args: str, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def tmux_sessions() -> list[dict]:
    result = command(
        "tmux", "list-sessions", "-F",
        "#{session_name}\t#{session_created}\t#{session_activity}\t#{session_attached}\t#{session_windows}",
    )
    if result.returncode != 0:
        if "no server running" in result.stderr or "No such file" in result.stderr:
            return []
        raise RuntimeError(f"tmux list-sessions failed: {result.stderr.strip()}")
    sessions = []
    for line in result.stdout.splitlines():
        name, created, activity, attached, windows = line.split("\t")
        sessions.append(dict(name=name, created=int(created), activity=int(activity),
                             attached=int(attached), windows=int(windows)))
    return sessions


def protected_sessions() -> dict[str, int]:
    try:
        data = json.loads(PROTECTED.read_text())
        return {str(k): int(v) for k, v in data.items()}
    except FileNotFoundError:
        return {}


def save_protected(data: dict[str, int]) -> None:
    CONFIG.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = PROTECTED.with_suffix(".tmp")
    with tmp.open("w") as out:
        json.dump(data, out, indent=2, sort_keys=True)
        out.write("\n")
    tmp.chmod(0o600)
    tmp.replace(PROTECTED)


def session_name_from_pane() -> str:
    pane = os.environ.get("TMUX_PANE")
    if not pane:
        raise RuntimeError("Use --session NAME outside a tmux pane")
    result = command("tmux", "display-message", "-p", "-t", pane, "#{session_name}")
    if result.returncode != 0:
        raise RuntimeError("Cannot identify the current tmux session")
    return result.stdout.strip()


def manage_protection(action: str, name: str | None) -> None:
    name = name or session_name_from_pane()
    live = {s["name"]: s for s in tmux_sessions()}
    data = protected_sessions()
    if action == "protect":
        if name not in live:
            raise RuntimeError(f"Session not found: {name}")
        data[name] = live[name]["created"]
        save_protected(data)
        print(f"Protected until explicitly unprotected: {name}")
    else:
        data.pop(name, None)
        save_protected(data)
        print(f"Unprotected: {name}")


def idle_prompt(name: str) -> bool:
    panes = command("tmux", "list-panes", "-t", name, "-F",
                    "#{pane_dead}\t#{pane_current_command}")
    if panes.returncode != 0 or len(panes.stdout.splitlines()) != 1:
        return False
    dead, current = panes.stdout.strip().split("\t", 1)
    if dead != "0" or current not in {"node", "codex", "claude"}:
        return False
    capture = command("tmux", "capture-pane", "-p", "-t", name, "-S", "-60")
    if capture.returncode != 0:
        return False
    tail = capture.stdout.splitlines()[-30:]
    recent = "\n".join(tail)
    # Background agents can keep running even when the main prompt is visible.
    if re.search(r"\bWorking\s*\(|background terminal[s]? running|\bbusy working\b", recent, re.I):
        return False
    if IDLE_CODEX in recent:
        return True
    # Claude Code's empty input prompt. A typed but unanswered request is not idle.
    for line in reversed(tail):
        stripped = line.strip()
        if stripped.startswith("❯"):
            return stripped[1:].strip() == ""
    return False


def reason_to_skip(s: dict, protected: dict[str, int], now: float) -> str | None:
    if s["name"].startswith(CORE_PREFIXES):
        return "core bot session"
    if protected.get(s["name"]) == s["created"]:
        return "protected by user"
    if s["attached"]:
        return "attached"
    if s["windows"] != 1:
        return "multiple windows"
    if now - max(s["created"], s["activity"]) < WEEK:
        return "recent activity"
    if not idle_prompt(s["name"]):
        return "idle state not verified"
    return None


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found")


def swap_free() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("SwapFree:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("SwapFree not found")


def open_inodes() -> set[tuple[int, int]]:
    opened: set[tuple[int, int]] = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (proc / "status").stat().st_uid != os.getuid():
                continue
            for fd in (proc / "fd").iterdir():
                try:
                    st = fd.stat()
                    opened.add((st.st_dev, st.st_ino))
                except (OSError, PermissionError):
                    pass
        except (OSError, PermissionError):
            pass
    return opened


def process_cwds() -> list[Path]:
    result = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (proc / "status").stat().st_uid == os.getuid():
                result.append((proc / "cwd").resolve(strict=True))
        except (OSError, PermissionError):
            pass
    return result


def prune_old_pip_cache(now: float, dry_run: bool) -> tuple[int, int]:
    root = Path.home() / ".cache/pip"
    if not root.is_dir():
        return 0, 0
    opened = open_inodes()
    count = size = 0
    for directory, _, files in os.walk(root, followlinks=False):
        for filename in files:
            path = Path(directory) / filename
            try:
                st = path.lstat()
                if (not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid()
                        or now - st.st_mtime < WEEK
                        or (st.st_dev, st.st_ino) in opened):
                    continue
                count += 1
                size += st.st_size
                if not dry_run:
                    path.unlink()
            except (OSError, PermissionError):
                continue
    if not dry_run:
        for directory, _, _ in os.walk(root, topdown=False, followlinks=False):
            if Path(directory) != root:
                try:
                    Path(directory).rmdir()
                except OSError:
                    pass
    return count, size


def active_claude_ids() -> set[str]:
    """Keep scratchpads associated with any running Claude process."""
    ids: set[str] = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            raw = (proc / "cmdline").read_bytes().decode(errors="ignore")
            if "claude" not in raw.lower():
                continue
            ids.update(re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", raw))
        except (OSError, PermissionError):
            continue
    return ids


def tree_usage_and_latest(root: Path, opened: set[tuple[int, int]]) -> tuple[int, float, bool]:
    """Measure allocated bytes and newest modification without following links."""
    total = 0
    latest = root.lstat().st_mtime
    busy = False
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            try:
                st = (Path(directory) / name).lstat()
                latest = max(latest, st.st_mtime)
                total += st.st_blocks * 512
                busy |= ((st.st_dev, st.st_ino) in opened
                         or stat.S_ISSOCK(st.st_mode) or stat.S_ISFIFO(st.st_mode))
            except (OSError, PermissionError):
                pass
    return total, latest, busy


def prune_claude_scratchpads(now: float, dry_run: bool) -> tuple[int, int]:
    root = Path(f"/tmp/claude-{os.getuid()}")
    if not root.is_dir() or root.is_symlink() or root.stat().st_uid != os.getuid():
        return 0, 0
    active = active_claude_ids()
    opened = open_inodes()
    cwds = process_cwds()
    count = size = 0
    for project in root.iterdir():
        if not project.is_dir() or project.is_symlink():
            continue
        for session in project.iterdir():
            try:
                UUID(session.name)
            except ValueError:
                continue
            scratch = session / "scratchpad"
            if (session.name in active or not scratch.is_dir() or scratch.is_symlink()
                    or scratch.stat().st_uid != os.getuid()):
                continue
            bytes_used, latest, busy = tree_usage_and_latest(scratch, opened)
            if (busy or any(scratch == cwd or scratch in cwd.parents for cwd in cwds)
                    or now - latest < WEEK):
                continue
            count += 1
            size += bytes_used
            if not dry_run:
                shutil.rmtree(scratch)
    return count, size


def archive_old_subagent_logs(now: float, dry_run: bool) -> tuple[int, int, int]:
    """Compress completed subagent transcripts; retain recoverable gzip files."""
    root = Path.home() / ".claude/projects"
    if not root.is_dir():
        return 0, 0, 0
    active = active_claude_ids()
    opened = open_inodes()
    candidates = []
    for path in root.glob("*/[0-9a-f]*-*/subagents/*.jsonl"):
        try:
            st = path.lstat()
            parent_id = path.parent.parent.name
            UUID(parent_id)
            if (not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid()
                    or parent_id in active or now - st.st_mtime < WEEK
                    or st.st_size < 1_000_000 or (st.st_dev, st.st_ino) in opened
                    or path.with_suffix(".jsonl.gz").exists()):
                continue
            candidates.append((st.st_mtime, path, st))
        except (OSError, ValueError):
            continue
    count = source_bytes = archive_bytes = 0
    for _, path, original in sorted(candidates):
        if source_bytes + original.st_size > ARCHIVE_DAILY_LIMIT and count:
            break
        target = path.with_suffix(".jsonl.gz")
        if dry_run:
            count += 1
            source_bytes += original.st_size
            continue
        temp = path.with_suffix(".jsonl.gz.tmp")
        try:
            with path.open("rb") as src, temp.open("wb") as raw:
                os.fchmod(raw.fileno(), 0o600)
                with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1) as packed:
                    shutil.copyfileobj(src, packed, length=1024 * 1024)
                raw.flush()
                os.fsync(raw.fileno())
            # Reading to EOF verifies the compressed stream and its CRC.
            with gzip.open(temp, "rb") as check:
                while check.read(1024 * 1024):
                    pass
            current = path.lstat()
            if (current.st_ino != original.st_ino or current.st_size != original.st_size
                    or current.st_mtime_ns != original.st_mtime_ns):
                temp.unlink()
                continue
            packed_size = temp.stat().st_size
            temp.replace(target)
            path.unlink()
            count += 1
            source_bytes += original.st_size
            archive_bytes += packed_size
        except (OSError, EOFError, gzip.BadGzipFile):
            temp.unlink(missing_ok=True)
            continue
    return count, source_bytes, archive_bytes


def prune_finished_job_tmp(now: float, dry_run: bool) -> tuple[int, int]:
    """Remove only temporary files of jobs that ended over a week ago."""
    root = Path.home() / ".claude/jobs"
    if not root.is_dir():
        return 0, 0
    active_short = {session_id[:8] for session_id in active_claude_ids()}
    opened = open_inodes()
    cwds = process_cwds()
    count = size = 0
    for job in root.iterdir():
        if not job.is_dir() or not re.fullmatch(r"[0-9a-f]{8}", job.name):
            continue
        temp = job / "tmp"
        if job.name in active_short or not temp.is_dir() or temp.is_symlink():
            continue
        try:
            state = json.loads((job / "state.json").read_text())
            if state.get("state") not in {"done", "stopped", "failed"}:
                continue
            updated = datetime.fromisoformat(state["updatedAt"].replace("Z", "+00:00")).timestamp()
            bytes_used, latest, busy = tree_usage_and_latest(temp, opened)
            if (now - max(updated, latest) < WEEK or busy
                    or any(temp == cwd or temp in cwd.parents for cwd in cwds)):
                continue
            count += 1
            size += bytes_used
            if not dry_run:
                shutil.rmtree(temp)
        except (OSError, ValueError, KeyError):
            continue
    return count, size


def mib(value: int) -> str:
    return f"{value / 1048576:.0f} МиБ"


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    result = command("du", "-sx", "-B1", str(path), timeout=45)
    if result.returncode:
        return 0
    return int(result.stdout.split()[0])


def disk_breakdown() -> dict[str, int]:
    home = Path.home()
    vault = home / "projects/agent-second-brain/vault"
    paths = {
        "Временные файлы Claude": Path(f"/tmp/claude-{os.getuid()}"),
        "История и задания Claude": home / ".claude",
        "Кэш npm": home / ".npm/_cacache",
        "Кэш uv": home / ".cache/uv",
        "Рабочие деревья": vault / ".claude/worktrees",
        "Результаты проектов": vault / "output",
        "Системный журнал": Path("/var/log/journal"),
    }
    return {name: directory_bytes(path) for name, path in paths.items()}


def uv_cache_busy() -> bool:
    """Long-running `uv run` bot services hold the shared cache lock."""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if (proc / "comm").read_text().strip() != "uv":
                continue
            if b"uv\x00run\x00" in (proc / "cmdline").read_bytes():
                return True
        except (OSError, PermissionError):
            continue
    return False


def vacuum_old_journal(dry_run: bool) -> str:
    if dry_run:
        return "would remove archived entries older than 7 days"
    result = command("sudo", "-n", "journalctl", "--rotate", "--vacuum-time=7d",
                     "--no-pager", timeout=60)
    if result.returncode:
        return "ошибка: журналы не очищены"
    freed = re.findall(r"freed ([0-9.]+[KMG]?B)", result.stdout + result.stderr)
    return "очищены записи старше 7 дней" + (f" (до {freed[0]})" if freed else "")


def report_text(report: dict) -> str:
    closed = ", ".join(report["closed"]) or "нет"
    lines = ["🧹 <b>Ежедневная уборка сервера</b>",
             f"Сессии закрыты: {html.escape(closed)}.",
             f"Кэш pip старше 7 дней: {report['pip_files']} файлов, {mib(report['pip_bytes'])}.",
             f"Старые временные папки Claude: {report['scratchpads']} шт., {mib(report['scratchpad_bytes'])}.",
             f"Временные файлы завершённых заданий: {report['job_tmp']} шт., {mib(report['job_tmp_bytes'])}.",
             f"Архивировано логов агентов: {report['logs_archived']} шт., {mib(report['logs_original_bytes'])} → {mib(report['logs_archive_bytes'])}."]
    if report["uv_result"]:
        lines.append(f"Кэш uv: {html.escape(report['uv_result'])}.")
    lines.append(f"Системный журнал: {html.escape(report['journal_result'])}.")
    lines.extend([
        f"Доступная RAM: {mib(report['memory_before'])} → {mib(report['memory_after'])}.",
        f"Свободный swap: {mib(report['swap_before'])} → {mib(report['swap_after'])}.",
        f"Диск: занято {report['disk_used_percent']:.1f}%, свободно {report['disk_after'] / 1073741824:.1f} ГиБ.",
        f"Оставлено сессий: {report['sessions_left']} (защищённых или активных: {report['protected_or_active']}).",
    ])
    if report.get("disk_delta") is not None:
        lines.append(f"Изменение занятого места с прошлого отчёта: {report['disk_delta'] / 1073741824:+.1f} ГиБ.")
    top = sorted(report["disk_breakdown"].items(), key=lambda item: item[1], reverse=True)[:4]
    lines.append("Крупнейшие категории: " + "; ".join(
        f"{html.escape(name)} {size / 1073741824:.1f} ГиБ" for name, size in top
    ) + ".")
    if report["errors"]:
        lines.append("Ошибки: " + html.escape("; ".join(report["errors"])[:800]))
    return "\n".join(lines)


def send_telegram(message: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    raw_ids = os.environ.get("ALLOWED_USER_IDS", "")
    if not token or not raw_ids:
        raise RuntimeError("Telegram credentials or recipient are missing")
    try:
        user_ids = json.loads(raw_ids)
    except json.JSONDecodeError:
        user_ids = [part.strip() for part in raw_ids.split(",") if part.strip()]
    if not user_ids:
        raise RuntimeError("No Telegram recipient configured")
    body = urllib.parse.urlencode(dict(chat_id=int(user_ids[0]), text=message,
                                       parse_mode="HTML")).encode()
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage", data=body, method="POST"
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.load(response)
    if not result.get("ok"):
        raise RuntimeError("Telegram rejected the report")


def run_cleanup(dry_run: bool) -> dict:
    now = time.time()
    before = tmux_sessions()
    protected = protected_sessions()
    report = dict(at=datetime.now(TZ).isoformat(), dry_run=dry_run, closed=[], skipped={},
                  pip_files=0, pip_bytes=0, scratchpads=0, scratchpad_bytes=0,
                  job_tmp=0, job_tmp_bytes=0,
                  logs_archived=0, logs_original_bytes=0, logs_archive_bytes=0,
                  uv_result="", journal_result="", errors=[], memory_before=mem_available(),
                  swap_before=swap_free(), disk_before=shutil.disk_usage("/").free)
    for session in before:
        name = session["name"]
        reason = reason_to_skip(session, protected, now)
        if reason:
            report["skipped"][name] = reason
            continue
        if dry_run:
            report["closed"].append(name)
            continue
        # Recheck immediately before closing: another client may have attached.
        fresh = next((x for x in tmux_sessions() if x["name"] == name), None)
        if not fresh or reason_to_skip(fresh, protected_sessions(), time.time()):
            report["skipped"][name] = "state changed before close"
            continue
        result = command("tmux", "kill-session", "-t", name)
        if result.returncode == 0:
            report["closed"].append(name)
        else:
            report["errors"].append(f"tmux {name}: {result.stderr.strip()[:100]}")
    report["pip_files"], report["pip_bytes"] = prune_old_pip_cache(now, dry_run)
    try:
        report["scratchpads"], report["scratchpad_bytes"] = prune_claude_scratchpads(now, dry_run)
    except OSError as exc:
        report["errors"].append(f"Claude scratchpads: {exc.strerror or type(exc).__name__}")
    try:
        report["job_tmp"], report["job_tmp_bytes"] = prune_finished_job_tmp(now, dry_run)
    except OSError as exc:
        report["errors"].append(f"Claude job temp: {exc.strerror or type(exc).__name__}")
    try:
        (report["logs_archived"], report["logs_original_bytes"],
         report["logs_archive_bytes"]) = archive_old_subagent_logs(now, dry_run)
    except OSError as exc:
        report["errors"].append(f"Claude log archive: {exc.strerror or type(exc).__name__}")
    if not dry_run:
        uv = shutil.which("uv")
        if uv:
            if uv_cache_busy():
                report["uv_result"] = "пропущен — его используют запущенные сервисы"
            else:
                try:
                    result = command(uv, "cache", "prune", timeout=45)
                    if result.returncode == 0:
                        report["uv_result"] = "удалены неиспользуемые записи"
                    else:
                        report["errors"].append("uv cache prune failed")
                except subprocess.TimeoutExpired:
                    report["errors"].append("uv cache prune timed out")
    try:
        report["journal_result"] = vacuum_old_journal(dry_run)
        if report["journal_result"].startswith("ошибка"):
            report["errors"].append("journalctl vacuum failed")
    except subprocess.TimeoutExpired:
        report["journal_result"] = "не завершён за 60 секунд"
        report["errors"].append("journalctl vacuum timed out")
    report["memory_after"] = mem_available()
    report["swap_after"] = swap_free()
    disk = shutil.disk_usage("/")
    report["disk_after"] = disk.free
    # Match `df`'s Use% denominator (reserved filesystem blocks are excluded).
    report["disk_used_percent"] = 100 * disk.used / (disk.used + disk.free)
    report["disk_used"] = disk.used
    report["disk_breakdown"] = disk_breakdown()
    try:
        prior = json.loads((STATE / "latest.json").read_text())
        report["disk_delta"] = disk.used - int(prior["disk_used"])
    except (FileNotFoundError, ValueError, KeyError):
        report["disk_delta"] = None
    left = tmux_sessions()
    report["sessions_left"] = len(left)
    report["protected_or_active"] = sum(
        1 for s in left if report["skipped"].get(s["name"]) in
        {"core bot session", "protected by user", "attached", "recent activity"}
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "protect", "unprotect", "list", "test-report"])
    parser.add_argument("--session")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.action in {"protect", "unprotect"}:
        manage_protection(args.action, args.session)
        return 0
    if args.action == "list":
        print(json.dumps(protected_sessions(), indent=2))
        return 0
    if args.action == "test-report":
        send_telegram("🧹 Проверка связи: ежедневная уборка сервера настроена. Первый запуск сегодня вечером.")
        return 0
    STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (STATE / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Cleanup already running", file=sys.stderr)
            return 0
        report = run_cleanup(args.dry_run)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.dry_run:
            return 0
        out = STATE / f"report-{datetime.now(TZ):%Y-%m-%d}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        out.chmod(0o600)
        latest = STATE / "latest.json"
        latest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        latest.chmod(0o600)
        send_telegram(report_text(report))
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"server cleanup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
