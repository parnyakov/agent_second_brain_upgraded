"""Minimal sd_notify client (no external deps).

Lets the bot/watchdog talk to systemd's Type=notify + WatchdogSec without
pulling in a library. All functions are no-ops when not run under systemd
(NOTIFY_SOCKET unset), so the code runs fine in dev and tests.
"""

import os
import socket


def notify(state: str) -> bool:
    """Send a state line to systemd (e.g. 'READY=1', 'WATCHDOG=1').

    Returns False if not running under systemd or on socket error.
    """
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    # Abstract namespace sockets start with '@'.
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(addr)
            sock.sendall(state.encode())
        return True
    except OSError:
        return False


def watchdog_interval(default: float = 60.0) -> float:
    """Ping interval = half of WATCHDOG_USEC (systemd's deadline), or default."""
    usec = os.environ.get("WATCHDOG_USEC")
    if not usec:
        return default
    try:
        return max(1.0, (int(usec) / 1_000_000) / 2)
    except ValueError:
        return default
