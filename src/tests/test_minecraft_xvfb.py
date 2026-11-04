"""The private Xvfb of the Minecraft pipe backend: stale display locks are reclaimed, live ones are skipped,
and the server is stopped with SIGTERM so that it removes its own lock (a SIGKILL leaves it behind)."""
from __future__ import annotations

import os
import subprocess

import pytest

from sb_pomdp import minecraft_env as M


class _FakeXvfb:
    def __init__(self, args, **kwargs):
        self.args = args
        self.terminated = self.killed = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if not (self.terminated or self.killed):
            raise subprocess.TimeoutExpired(self.args, timeout)
        return 0


def _backend_shell():
    backend = M._PipeBackend.__new__(M._PipeBackend)
    backend.xvfb = None
    backend.log = None
    return backend


@pytest.fixture
def x_dirs(tmp_path, monkeypatch):
    lock_dir, socket_dir = tmp_path / "locks", tmp_path / "sockets"
    lock_dir.mkdir()
    socket_dir.mkdir()
    monkeypatch.setattr(M, "_X_LOCK_DIR", lock_dir)
    monkeypatch.setattr(M, "_X_SOCKET_DIR", socket_dir)
    monkeypatch.setattr(M, "_XVFB_ATTEMPTS", 400)
    monkeypatch.setattr(__import__("time"), "sleep", lambda s: None)
    # liveness through /proc is Linux-only; the tests define it as "this process"
    monkeypatch.setattr(M, "_pid_alive", lambda pid: pid == os.getpid())
    return lock_dir, socket_dir


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="pid liveness is read from /proc")
def test_pid_alive_reads_proc():
    assert M._pid_alive(os.getpid())
    assert not M._pid_alive(999999999)
    assert not M._pid_alive(0)


def test_stale_display_means_a_dead_pid(x_dirs):
    lock_dir, _ = x_dirs
    (lock_dir / ".X300-lock").write_text(f"{os.getpid():10d}\n")  # a live process: not stale
    (lock_dir / ".X301-lock").write_text("  999999999\n")  # no such process: stale
    (lock_dir / ".X302-lock").write_text("garbage\n")  # unreadable: treated as live (never removed)
    assert not M._PipeBackend._stale_display(300)
    assert M._PipeBackend._stale_display(301)
    assert not M._PipeBackend._stale_display(302)
    assert not M._PipeBackend._stale_display(303)  # no lock at all


def test_start_reclaims_stale_locks_and_skips_live_ones(x_dirs, monkeypatch):
    lock_dir, socket_dir = x_dirs
    # every display is locked: the even numbers by dead processes, the odd ones by this process
    for number in range(M._X_DISPLAY_MIN, M._X_DISPLAY_MAX):
        pid = 999999999 if number % 2 == 0 else os.getpid()
        (lock_dir / f".X{number}-lock").write_text(f"{pid:10d}\n")
        (socket_dir / f"X{number}").write_text("")
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kw: started.append(_FakeXvfb(args)) or started[-1])
    backend = _backend_shell()
    display = backend._start_xvfb(seed=7)
    number = int(display[1:])
    assert number % 2 == 0, "a display locked by a live process must not be reused"
    assert not (lock_dir / f".X{number}-lock").exists() and not (socket_dir / f"X{number}").exists()
    assert backend.xvfb is started[-1] and started[-1].args[1] == display
    # the live locks are untouched
    assert all((lock_dir / f".X{n}-lock").exists() for n in range(M._X_DISPLAY_MIN + 1, M._X_DISPLAY_MAX, 2))


def test_start_fails_when_every_display_is_live(x_dirs, monkeypatch):
    lock_dir, _ = x_dirs
    for number in range(M._X_DISPLAY_MIN, M._X_DISPLAY_MAX):
        (lock_dir / f".X{number}-lock").write_text(f"{os.getpid():10d}\n")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Xvfb must not be started"))
    with pytest.raises(RuntimeError):
        _backend_shell()._start_xvfb(seed=1)


def test_stop_terminates_before_killing():
    backend = _backend_shell()
    backend.xvfb = proc = _FakeXvfb(["Xvfb", ":5"])
    backend._stop_xvfb()
    assert proc.terminated and not proc.killed and backend.xvfb is None
    backend._stop_xvfb()  # idempotent
