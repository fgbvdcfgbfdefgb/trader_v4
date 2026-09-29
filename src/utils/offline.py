"""
Hard offline guard.

The brief is "must run offline".  Rather than *hoping* no library phones
home (transformers and huggingface_hub both try by default), we block
outbound sockets at the interpreter level and set the standard offline
environment variables before anything else is imported.

Loopback and AF_UNIX stay open so SQLite, CUDA IPC and local IPC still work.
"""
from __future__ import annotations

import os
import socket

_LOCAL_PREFIXES = ("127.", "::1", "localhost", "0.0.0.0", "")
_installed = False


def _is_local(addr) -> bool:
    if isinstance(addr, (bytes, str)):
        return True                      # AF_UNIX path
    try:
        host = addr[0]
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(host, str):
        return False
    return host.startswith(_LOCAL_PREFIXES) or host == "::"


class OfflineViolation(RuntimeError):
    pass


def set_env_offline() -> None:
    for k, v in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "NO_PROXY": "*",
    }.items():
        os.environ.setdefault(k, v)


def enforce(strict: bool = True) -> None:
    """Block non-loopback connects. Idempotent and safe to call per-process."""
    global _installed
    set_env_offline()
    if _installed or not strict:
        return
    _installed = True

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guarded_connect(self, address):
        if not _is_local(address):
            raise OfflineViolation(
                f"outbound network blocked (tried {address!r}). "
                "This trainer is required to run fully offline; every input "
                "must already be in the repo. Pass --allow-network to lift.")
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        if not _is_local(address):
            raise OfflineViolation(f"outbound network blocked ({address!r})")
        return real_connect_ex(self, address)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
