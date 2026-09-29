"""POSIX ``fcntl`` compatibility shim for Windows.

``src/agents/data_manager.py`` uses ``fcntl.flock(fd, LOCK_EX/LOCK_UN)`` to make the
read-modify-write of the append-only per-persona JSONL files atomic (see
``_append_jsonl`` and ``mark_generation_rejected``). ``fcntl`` is POSIX-only and
does not exist on Windows, which makes the whole project fail to import there.

All concurrency in Agentopia happens through **threads inside a single process**
(``ThreadPoolExecutor`` in ``src/world/world.py`` and ``src/agents/role_agent.py``),
and every run writes into its own ``data/<world>_<runid>/`` directory. So the mutual
exclusion ``flock`` provides can be reproduced exactly with one ``threading.Lock``
per underlying file, with no cross-process locking required.

Only the API surface ``data_manager.py`` actually uses is implemented: ``flock``,
``LOCK_EX``, ``LOCK_SH``, ``LOCK_UN``, ``LOCK_NB``.

Lock/unlock pairing matches POSIX ``flock`` semantics for this usage: the caller
locks once and unlocks in a ``finally`` block, never nesting on the same file.
"""

from __future__ import annotations

import errno
import os
import threading

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8

# keyed by (st_dev, st_ino) so every open() of the same file shares one lock
_locks: dict = {}
_locks_guard = threading.Lock()


def _file_key(fd: int):
    """Identify the underlying file, not the file descriptor.

    Each ``open()`` returns a new fd, so the fd number cannot be the key.
    ``os.fstat`` exposes ``st_dev``/``st_ino`` on Windows as well (volume serial
    number + file index), which uniquely identifies the file.
    """
    try:
        st = os.fstat(fd)
        return (st.st_dev, st.st_ino)
    except OSError:
        # Fall back to the fd itself; still safe, just less sharing.
        return fd


def _lock_for(fd: int) -> threading.Lock:
    key = _file_key(fd)
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = _locks[key] = threading.Lock()
        return lock


def flock(fd: int, operation: int) -> None:
    """Emulate ``fcntl.flock`` using a per-file ``threading.Lock``.

    ``LOCK_EX``/``LOCK_SH`` acquire, ``LOCK_UN`` releases, and ``LOCK_NB`` turns a
    contended acquire into ``OSError(EWOULDBLOCK)`` instead of blocking.
    """
    lock = _lock_for(fd)

    if operation & LOCK_UN:
        try:
            lock.release()
        except RuntimeError:
            # Not held — e.g. released twice, or the fd was closed first.
            pass
        return

    if operation & LOCK_NB:
        if not lock.acquire(blocking=False):
            raise OSError(errno.EWOULDBLOCK, "Resource temporarily unavailable")
        return

    lock.acquire()
