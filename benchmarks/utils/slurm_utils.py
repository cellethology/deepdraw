"""Utilities for interacting with Slurm workloads."""

import getpass
import os
import shutil
import subprocess
import sys
import time

_SQUEUE_WAIT_ENV = "AL_DISABLE_SQUEUE_WAIT"
_SQUEUE_INTERVAL_ENV = "AL_SQUEUE_POLL_SECONDS"
_DEFAULT_POLL_SECONDS = 60.0


def wait_for_slurm_jobs(
    dataset_name: str,
    squeue_wait_env: str = _SQUEUE_WAIT_ENV,
    squeue_interval_env: str = _SQUEUE_INTERVAL_ENV,
    default_poll_seconds: float = _DEFAULT_POLL_SECONDS,
) -> None:
    """Poll `squeue` until no active jobs remain for the current user."""
    if os.environ.get(squeue_wait_env):
        return
    if not shutil.which("squeue"):
        print(
            "[INFO] `squeue` not available; skipping queue wait before next dataset.",
            file=sys.stderr,
        )
        return
    user = (
        os.environ.get("USER")
        or os.environ.get("LOGNAME")
        or os.environ.get("SLURM_JOB_USER")
        or getpass.getuser()
    )
    if not user:
        print(
            "[WARN] Could not determine user for `squeue`; skipping queue wait.",
            file=sys.stderr,
        )
        return
    try:
        poll_seconds = float(os.environ.get(squeue_interval_env, default_poll_seconds))
    except ValueError:
        poll_seconds = default_poll_seconds

    print(
        f"[INFO] Waiting for Slurm jobs to finish before starting the next dataset "
        f"(current: {dataset_name})."
    )
    while True:
        try:
            result = subprocess.run(
                ["squeue", "-h", "-u", user],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"[WARN] Failed to query `squeue` ({exc.stderr.strip()}), "
                "skipping queue wait.",
                file=sys.stderr,
            )
            return
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            print("[INFO] Slurm queue is empty; continuing to next dataset.")
            return
        print(
            f"[INFO] {len(lines)} Slurm job(s) still active. "
            f"Polling again in {poll_seconds:.0f} seconds..."
        )
        time.sleep(max(1.0, poll_seconds))
