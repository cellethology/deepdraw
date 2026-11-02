"""Tests for per-seed worker helpers."""

from __future__ import annotations

from omegaconf import OmegaConf

from benchmarks.utils import seed_runner


def test_run_seed_experiment_with_snapshot_writes_resolved_config(
    tmp_path, monkeypatch
) -> None:
    output_dir = tmp_path / "seed_3"
    raw_cfg = {
        "value": 7,
        "derived": "${value}",
        "hydra_overrides": ["value=7"],
        "al_settings": {"output_dir": str(output_dir)},
    }
    captured = []
    monkeypatch.setattr(seed_runner, "run_one_experiment", captured.append)

    seed_runner.run_seed_experiment_with_snapshot(raw_cfg)

    snapshot = OmegaConf.load(output_dir / ".hydra/config.yaml")
    overrides = OmegaConf.load(output_dir / ".hydra/overrides.yaml")
    assert snapshot.derived == 7
    assert list(overrides) == ["value=7"]
    assert captured[0].derived == 7
