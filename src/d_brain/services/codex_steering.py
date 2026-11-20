"""Private per-turn Codex app server shared by exec and Telegram steering."""

import json
import logging
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from websockets.exceptions import WebSocketException
from websockets.sync.client import unix_connect

logger = logging.getLogger(__name__)


@contextmanager
def chat_server(binary: str, cwd: Path, sandbox: str, model: str | None):
    # Keep the Unix socket inside an owner-only directory, including for
    # instances which share /tmp. No control endpoint is exposed on TCP.
    with tempfile.TemporaryDirectory(prefix="dbrain-turn-") as directory:
        socket = Path(directory) / "rpc"
        argv = [
            binary,
            "app-server",
            "--listen",
            f"unix://{socket}",
            "-c",
            f"sandbox_mode={sandbox}",
            "-c",
            "approval_policy=never",
        ]
        if model:
            argv += ["-c", f"model={json.dumps(model)}"]
        # The systemd units keep child processes alive on a bot crash.
        # Couple this per-turn server to its owner on Linux as well.
        setpriv = shutil.which("setpriv")
        if setpriv:
            argv = [setpriv, "--pdeathsig", "TERM", "--", *argv]
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 10
            while not socket.exists():
                if proc.poll() is not None or time.monotonic() >= deadline:
                    raise OSError("Codex chat server did not open its socket")
                time.sleep(0.025)
            yield socket
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


@contextmanager
def connection(socket):
    with unix_connect(
        str(socket),
        uri="ws://localhost/",
        compression=None,
        max_size=None,
        open_timeout=5,
        close_timeout=1,
    ) as ws:

        def request(i, method, params):
            ws.send(json.dumps({"id": i, "method": method, "params": params}))
            deadline = time.monotonic() + 15
            while True:
                msg = json.loads(ws.recv(timeout=max(0, deadline - time.monotonic())))
                if msg.get("id") == i:
                    if "error" in msg:
                        raise OSError(
                            f"Codex rejected {method}: {msg['error'].get('code')}"
                        )
                    return msg["result"]

        request(1, "initialize", {"clientInfo": {"name": "dbrain", "version": "1"}})
        ws.send(json.dumps({"method": "initialized"}))
        yield request


def prepare_resume(socket, thread, cwd, sandbox, model):
    # Remote exec resumes without overrides. Load the thread with today's
    # explicit settings first, including threads last used in read-only mode.
    try:
        with connection(socket) as request:
            result = request(
                2,
                "thread/resume",
                {
                    "threadId": thread,
                    "cwd": str(cwd),
                    "sandbox": sandbox,
                    "approvalPolicy": "never",
                    "model": model,
                    "excludeTurns": True,
                },
            )
            if result["thread"]["id"] != thread:
                raise OSError("Codex resumed a different thread")
    except (WebSocketException, ValueError, KeyError) as exc:
        raise OSError("Invalid response while resuming Codex chat") from exc


def steer(socket: Path, thread: str, text: str) -> bool:
    """Only acknowledge input accepted by the exact running turn."""
    try:
        with connection(socket) as request:
            result = request(
                2,
                "thread/turns/list",
                {
                    "threadId": thread,
                    "limit": 1,
                    "itemsView": "notLoaded",
                },
            )
            turns = result.get("data", [])
            if not turns or turns[0]["status"] != "inProgress":
                return False
            turn_id = turns[0]["id"]
            response = request(
                3,
                "turn/steer",
                {
                    "threadId": thread,
                    "expectedTurnId": turn_id,
                    "input": [{"type": "text", "text": text}],
                },
            )
            return response.get("turnId") == turn_id
    except Exception as exc:
        logger.warning("Codex steer unavailable (%s)", type(exc).__name__)
        return False
