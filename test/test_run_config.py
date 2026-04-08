"""Unit tests for job_sub.run_config orchestration helpers."""

from __future__ import annotations

import importlib
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest


@pytest.fixture
def load_run_config(monkeypatch):
    """Load job_sub.run_config with patched dataset loading side effects."""

    def _load(datasets: list[SimpleNamespace] | None = None):
        import job_sub.utils.config_utils as config_utils

        fake_datasets = datasets or [
            SimpleNamespace(
                name="dataset_a",
                metadata_path="meta_a.csv",
                embedding_dir="/tmp/emb_a",
                subset_ids_path=None,
            )
        ]

        monkeypatch.setattr(
            config_utils,
            "load_datasets_or_raise",
            lambda argv, config_path: (fake_datasets, "datasets/test.yaml"),
        )
        monkeypatch.setattr(
            config_utils,
            "seed_env_from_datasets",
            lambda *args, **kwargs: None,
        )

        fake_entry = types.ModuleType("run_active_learning")
        fake_entry.run_one_experiment = lambda cfg: None
        monkeypatch.setitem(sys.modules, "run_active_learning", fake_entry)

        monkeypatch.delitem(sys.modules, "job_sub.run_config", raising=False)
        return importlib.import_module("job_sub.run_config")

    return _load


def test_parse_boolish(load_run_config) -> None:
    module = load_run_config()
    assert module._parse_boolish(True, default=False)
    assert module._parse_boolish("yes", default=False)
    assert module._parse_boolish("1", default=False)
    assert not module._parse_boolish("off", default=True)
    assert not module._parse_boolish("0", default=True)
    assert module._parse_boolish("not-a-bool", default=True)


def test_single_array_enabled_prefers_override(tmp_path, load_run_config, monkeypatch):
    module = load_run_config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("single_array_across_datasets: false\n")
    monkeypatch.setattr(module, "_CONFIG_PATH", config_path)

    assert module._single_array_across_datasets_enabled(
        ["single_array_across_datasets=true"]
    )
    assert not module._single_array_across_datasets_enabled(
        ["single_array_across_datasets=invalid"]
    )


def test_single_array_enabled_reads_config_when_no_override(
    tmp_path, load_run_config, monkeypatch
) -> None:
    module = load_run_config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("single_array_across_datasets: true\n")
    monkeypatch.setattr(module, "_CONFIG_PATH", config_path)
    assert module._single_array_across_datasets_enabled([])

    config_path.write_text("single_array_across_datasets: false\n")
    assert not module._single_array_across_datasets_enabled([])

    monkeypatch.setattr(module, "_CONFIG_PATH", tmp_path / "missing.yaml")
    assert not module._single_array_across_datasets_enabled([])


def test_run_single_array_sweep_adds_dataset_index_and_clears_dataset_envs(
    load_run_config, monkeypatch
) -> None:
    datasets = [
        SimpleNamespace(
            name="ds0",
            metadata_path="m0.csv",
            embedding_dir="/tmp/e0",
            subset_ids_path="/tmp/s0.txt",
        ),
        SimpleNamespace(
            name="ds1",
            metadata_path="m1.csv",
            embedding_dir="/tmp/e1",
            subset_ids_path=None,
        ),
        SimpleNamespace(
            name="ds2",
            metadata_path="m2.csv",
            embedding_dir="/tmp/e2",
            subset_ids_path=None,
        ),
    ]
    module = load_run_config(datasets)
    captured: dict[str, object] = {}

    def fake_run_dataset_sweep(cmd, env, dataset_name):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["dataset_name"] = dataset_name

    monkeypatch.setattr(module, "_run_dataset_sweep", fake_run_dataset_sweep)
    monkeypatch.setenv(module._DATASET_ENV, "stale_dataset")
    monkeypatch.setenv(module._METADATA_ENV, "stale_metadata")
    monkeypatch.setenv(module._EMBED_DIR_ENV, "stale_embed")
    monkeypatch.setenv(module._SUBSET_ENV, "stale_subset")
    monkeypatch.setenv(module._DATASET_INDEX_ENV, "99")

    module._run_single_array_sweep(["foo=bar"])

    cmd = captured["cmd"]
    env = captured["env"]
    assert captured["dataset_name"] == "ALL_DATASETS"
    assert "dataset_index=0,1,2" in cmd
    assert env[module._HYDRA_CHILD_ENV] == "1"
    assert module._DATASET_ENV not in env
    assert module._METADATA_ENV not in env
    assert module._EMBED_DIR_ENV not in env
    assert module._SUBSET_ENV not in env
    assert module._DATASET_INDEX_ENV not in env


def test_run_single_array_sweep_keeps_user_dataset_index(load_run_config, monkeypatch):
    module = load_run_config()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        module,
        "_run_dataset_sweep",
        lambda cmd, env, dataset_name: captured.update(
            {"cmd": cmd, "env": env, "dataset_name": dataset_name}
        ),
    )

    module._run_single_array_sweep(["foo=bar", "dataset_index=7"])

    cmd = captured["cmd"]
    assert "dataset_index=7" in cmd
    assert sum(1 for part in cmd if part.startswith("dataset_index=")) == 1


def test_main_runs_per_dataset_mode(load_run_config, monkeypatch) -> None:
    datasets = [
        SimpleNamespace(
            name="ds0",
            metadata_path="m0.csv",
            embedding_dir="/tmp/e0",
            subset_ids_path="/tmp/s0.txt",
        ),
        SimpleNamespace(
            name="ds1",
            metadata_path="m1.csv",
            embedding_dir="/tmp/e1",
            subset_ids_path=None,
        ),
    ]
    module = load_run_config(datasets)
    monkeypatch.setattr(module, "collect_user_overrides", lambda argv: ["foo=bar"])
    monkeypatch.setattr(
        module, "_single_array_across_datasets_enabled", lambda overrides: False
    )

    calls: list[tuple[list[str], dict, str]] = []
    waits: list[str] = []

    def fake_run_dataset_sweep(cmd, env, dataset_name):
        calls.append((cmd, env, dataset_name))

    monkeypatch.setattr(module, "_run_dataset_sweep", fake_run_dataset_sweep)
    monkeypatch.setattr(module, "wait_for_slurm_jobs", lambda name: waits.append(name))
    monkeypatch.setenv(module._SUBSET_ENV, "stale_subset")

    module.main()

    assert len(calls) == 2
    cmd0, env0, name0 = calls[0]
    cmd1, env1, name1 = calls[1]
    assert name0 == "ds0"
    assert name1 == "ds1"
    assert cmd0[:3] == [sys.executable, str(module._SCRIPT_PATH), "-m"]
    assert cmd1[:3] == [sys.executable, str(module._SCRIPT_PATH), "-m"]
    assert env0[module._HYDRA_CHILD_ENV] == "1"
    assert env0[module._DATASET_INDEX_ENV] == "0"
    assert env0[module._DATASET_ENV] == "ds0"
    assert env0[module._SUBSET_ENV] == "/tmp/s0.txt"
    assert env1[module._DATASET_INDEX_ENV] == "1"
    assert env1[module._DATASET_ENV] == "ds1"
    assert module._SUBSET_ENV not in env1
    assert waits == ["ds0", "ds1"]


def test_main_runs_single_array_mode(load_run_config, monkeypatch) -> None:
    module = load_run_config()
    monkeypatch.setattr(module, "collect_user_overrides", lambda argv: ["foo=bar"])
    monkeypatch.setattr(
        module, "_single_array_across_datasets_enabled", lambda overrides: True
    )
    called: list[list[str]] = []
    monkeypatch.setattr(
        module, "_run_single_array_sweep", lambda overrides: called.append(overrides)
    )

    module.main()
    assert called == [["foo=bar"]]


def test_run_dataset_sweep_handles_uncompleted_job_error(
    load_run_config, monkeypatch
) -> None:
    module = load_run_config()

    def raise_uncompleted(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["python", "-m"],
            stderr="submitit.core.utils.UncompletedJobError: unfinished",
        )

    monkeypatch.setattr(module.subprocess, "run", raise_uncompleted)
    module._run_dataset_sweep(["python", "-m"], {}, "dataset_a")


def test_run_dataset_sweep_reraises_other_errors(load_run_config, monkeypatch) -> None:
    module = load_run_config()

    def raise_other(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=2,
            cmd=["python", "-m"],
            stderr="boom",
        )

    monkeypatch.setattr(module.subprocess, "run", raise_other)
    with pytest.raises(subprocess.CalledProcessError):
        module._run_dataset_sweep(["python", "-m"], {}, "dataset_a")
