"""Unit tests for seed job expansion helpers."""

from __future__ import annotations

from omegaconf import OmegaConf

from job_sub.utils import seed_jobs


def test_generate_seed_values_uses_start_and_count() -> None:
    cfg = OmegaConf.create({"num_seeds_per_job": 3, "seed_start": 5})
    assert seed_jobs.generate_seed_values(cfg) == [5, 6, 7]


def test_generate_seed_values_rejects_invalid_count() -> None:
    cfg = OmegaConf.create({"num_seeds_per_job": 0, "seed_start": 5})
    try:
        seed_jobs.generate_seed_values(cfg)
    except ValueError as exc:
        assert "num_seeds_per_job must be >= 1" in str(exc)
    else:  # pragma: no cover - explicit failure path
        raise AssertionError("Expected ValueError for invalid seed count")


def test_materialize_seed_cfgs_sets_per_seed_output_and_overrides(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "num_seeds_per_job": 2,
            "seed_start": 10,
            "al_settings": {"output_dir": "/tmp/base_output", "seed": 0},
        }
    )
    monkeypatch.setattr(seed_jobs, "_extract_hydra_overrides", lambda: ["foo=bar"])

    seed_cfgs = seed_jobs.materialize_seed_cfgs(cfg)

    assert len(seed_cfgs) == 2
    assert seed_cfgs[0]["al_settings"]["seed"] == 10
    assert seed_cfgs[0]["al_settings"]["output_dir"] == "/tmp/base_output/seed_10"
    assert seed_cfgs[1]["al_settings"]["seed"] == 11
    assert seed_cfgs[1]["al_settings"]["output_dir"] == "/tmp/base_output/seed_11"
    assert seed_cfgs[0]["hydra_overrides"] == ["foo=bar"]
    assert seed_cfgs[1]["hydra_overrides"] == ["foo=bar"]


def test_max_seed_workers_respects_parallel_and_slurm(monkeypatch) -> None:
    cfg = OmegaConf.create({"parallelize_seeds": True})
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    monkeypatch.setattr(seed_jobs.os, "cpu_count", lambda: 8)

    assert seed_jobs.max_seed_workers(cfg, num_tasks=10) == 3


def test_max_seed_workers_returns_one_when_parallel_disabled(monkeypatch) -> None:
    cfg = OmegaConf.create({"parallelize_seeds": False})
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setattr(seed_jobs.os, "cpu_count", lambda: 12)

    assert seed_jobs.max_seed_workers(cfg, num_tasks=10) == 1


def test_run_seed_jobs_serial_executes_all(monkeypatch) -> None:
    cfg = OmegaConf.create({})
    raw_cfgs = [{"id": 1}, {"id": 2}]
    monkeypatch.setattr(seed_jobs, "materialize_seed_cfgs", lambda _cfg: raw_cfgs)
    monkeypatch.setattr(seed_jobs, "max_seed_workers", lambda _cfg, _n: 1)

    calls: list[dict] = []
    monkeypatch.setattr(seed_jobs, "run_seed_experiment", lambda raw: calls.append(raw))

    seed_jobs.run_seed_jobs(cfg)
    assert calls == raw_cfgs


def test_run_seed_jobs_parallel_executes_all(monkeypatch) -> None:
    cfg = OmegaConf.create({})
    raw_cfgs = [{"id": 1}, {"id": 2}, {"id": 3}]
    monkeypatch.setattr(seed_jobs, "materialize_seed_cfgs", lambda _cfg: raw_cfgs)
    monkeypatch.setattr(seed_jobs, "max_seed_workers", lambda _cfg, _n: 2)

    submitted: list[dict] = []

    class _DoneFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, raw_cfg):
            submitted.append(raw_cfg)
            return _DoneFuture(fn(raw_cfg))

    monkeypatch.setattr(seed_jobs, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(seed_jobs, "as_completed", lambda futures: futures)
    monkeypatch.setattr(seed_jobs, "run_seed_experiment", lambda raw: raw["id"])

    seed_jobs.run_seed_jobs(cfg)
    assert submitted == raw_cfgs


def test_extract_hydra_overrides_when_hydra_initialized(monkeypatch) -> None:
    class _HydraConfigOK:
        @staticmethod
        def initialized():
            return True

        @staticmethod
        def get():
            return type(
                "_Cfg",
                (),
                {"overrides": type("_Overrides", (), {"task": ["foo=bar", 7]})()},
            )()

    monkeypatch.setattr(seed_jobs, "HydraConfig", _HydraConfigOK)
    assert seed_jobs._extract_hydra_overrides() == ["foo=bar", "7"]


def test_extract_hydra_overrides_returns_empty_on_hydra_error(monkeypatch) -> None:
    class _HydraConfigErr:
        @staticmethod
        def initialized():
            return True

        @staticmethod
        def get():
            raise RuntimeError("hydra unavailable")

    monkeypatch.setattr(seed_jobs, "HydraConfig", _HydraConfigErr)
    assert seed_jobs._extract_hydra_overrides() == []
