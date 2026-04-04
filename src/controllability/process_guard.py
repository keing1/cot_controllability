"""Process guard: PID-file-based orphan cleanup for multiprocessing scripts.

When a parent process is killed with SIGKILL, its spawned worker processes
survive as orphans.  ProcessGuard writes a PID file on startup, and kills
any PIDs recorded in that file from a previous run before proceeding.

Usage:
    guard = ProcessGuard("results/logs/my_script.pids")
    guard.kill_previous()          # kill orphans from last run
    ...
    pool = ProcessPoolExecutor(...)
    guard.save_pids(pool)          # record current worker PIDs
    ...
    guard.cleanup()                # remove PID file on normal exit
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


class ProcessGuard:
    def __init__(self, pidfile: str | Path) -> None:
        self.pidfile = Path(pidfile)

    def kill_previous(self) -> int:
        """Kill workers recorded in the PID file from a previous run.

        Returns the number of processes killed.
        """
        if not self.pidfile.exists():
            return 0
        try:
            data = json.loads(self.pidfile.read_text())
        except (json.JSONDecodeError, OSError):
            self.pidfile.unlink(missing_ok=True)
            return 0
        killed = 0
        for pid in data.get("pids", []):
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except ProcessLookupError:
                pass
        self.pidfile.unlink(missing_ok=True)
        if killed:
            time.sleep(1)  # let OS reap killed processes
        return killed

    def save_pids(self, pool: ProcessPoolExecutor) -> None:
        """Record the PIDs of all pool workers (+ parent) to the PID file."""
        pids = [p.pid for p in pool._processes.values() if p.pid is not None]
        pids.append(os.getpid())
        self.pidfile.parent.mkdir(parents=True, exist_ok=True)
        self.pidfile.write_text(json.dumps({"pids": pids}))

    def cleanup(self) -> None:
        """Remove the PID file (call on normal exit)."""
        self.pidfile.unlink(missing_ok=True)

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers that kill the process group.

        Also registers an atexit handler to clean up the PID file.
        """
        os.setpgrp()

        def _handler(signum, _frame):
            self.cleanup()
            os.killpg(os.getpgrp(), signal.SIGKILL)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        atexit.register(self.cleanup)
