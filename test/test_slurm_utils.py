"""Unit tests for Slurm queue helpers."""

from __future__ import annotations

import subprocess

from job_sub.utils import slurm_utils


def test_wait_for_slurm_jobs_respects_disable_env(monkeypatch) -> None:
    monkeypatch.setenv("AL_DISABLE_SQUEUE_WAIT", "1")
    monkeypatch.setattr(
        slurm_utils.shutil,
        "which",
        lambda _name: (_ for _ in ()).throw(AssertionError("should not query squeue")),
    )

    slurm_utils.wait_for_slurm_jobs("dataset_a")


def test_wait_for_slurm_jobs_skips_when_squeue_missing(monkeypatch, capsys) -> None:
    monkeypatch.delenv("AL_DISABLE_SQUEUE_WAIT", raising=False)
    monkeypatch.setattr(slurm_utils.shutil, "which", lambda _name: None)

    slurm_utils.wait_for_slurm_jobs("dataset_a")

    captured = capsys.readouterr()
    assert "squeue` not available" in captured.err


def test_wait_for_slurm_jobs_skips_when_user_missing(monkeypatch, capsys) -> None:
    monkeypatch.delenv("AL_DISABLE_SQUEUE_WAIT", raising=False)
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)
    monkeypatch.delenv("SLURM_JOB_USER", raising=False)
    monkeypatch.setattr(slurm_utils.shutil, "which", lambda _name: "/usr/bin/squeue")
    monkeypatch.setattr(slurm_utils.getpass, "getuser", lambda: "")

    slurm_utils.wait_for_slurm_jobs("dataset_a")

    captured = capsys.readouterr()
    assert "Could not determine user" in captured.err


def test_wait_for_slurm_jobs_polls_until_empty(monkeypatch, capsys) -> None:
    monkeypatch.delenv("AL_DISABLE_SQUEUE_WAIT", raising=False)
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setenv("AL_SQUEUE_POLL_SECONDS", "bad-value")
    monkeypatch.setattr(slurm_utils.shutil, "which", lambda _name: "/usr/bin/squeue")
    calls: list[list[str]] = []
    sleeps: list[float] = []
    results = [
        subprocess.CompletedProcess(["squeue"], 0, stdout="job1\njob2\n", stderr=""),
        subprocess.CompletedProcess(["squeue"], 0, stdout="\n", stderr=""),
    ]

    def fake_run(cmd, check, capture_output, text):
        calls.append(cmd)
        assert check is True
        assert capture_output is True
        assert text is True
        return results.pop(0)

    monkeypatch.setattr(slurm_utils.subprocess, "run", fake_run)
    monkeypatch.setattr(
        slurm_utils.time, "sleep", lambda seconds: sleeps.append(seconds)
    )

    slurm_utils.wait_for_slurm_jobs("dataset_a", default_poll_seconds=2.5)

    assert calls == [["squeue", "-h", "-u", "alice"], ["squeue", "-h", "-u", "alice"]]
    assert sleeps == [2.5]
    captured = capsys.readouterr()
    assert "2 Slurm job(s) still active" in captured.out
    assert "Slurm queue is empty" in captured.out


def test_wait_for_slurm_jobs_skips_on_squeue_error(monkeypatch, capsys) -> None:
    monkeypatch.delenv("AL_DISABLE_SQUEUE_WAIT", raising=False)
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setattr(slurm_utils.shutil, "which", lambda _name: "/usr/bin/squeue")

    def fake_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["squeue"],
            stderr="scheduler unavailable\n",
        )

    monkeypatch.setattr(slurm_utils.subprocess, "run", fake_run)

    slurm_utils.wait_for_slurm_jobs("dataset_a")

    captured = capsys.readouterr()
    assert "Failed to query `squeue`" in captured.err
    assert "scheduler unavailable" in captured.err
