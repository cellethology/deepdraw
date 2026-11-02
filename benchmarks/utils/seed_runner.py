"""Helpers for running per-seed experiments in parallel worker processes."""

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from benchmarks.run_active_learning import run_one_experiment
from benchmarks.utils.config_utils import ensure_resolvers


def run_seed_experiment(raw_cfg: dict[str, Any]) -> None:
    """Recreate DictConfig and run experiment (used by multiprocessing workers)."""
    ensure_resolvers()
    cfg = OmegaConf.create(raw_cfg)
    OmegaConf.resolve(cfg)
    run_one_experiment(cfg)


def run_seed_experiment_with_snapshot(raw_cfg: dict[str, Any]) -> None:
    """Persist a Hydra-compatible config snapshot, then run one seed."""
    ensure_resolvers()
    cfg = OmegaConf.create(raw_cfg)
    OmegaConf.resolve(cfg)

    output_dir = Path(str(cfg.al_settings.output_dir))
    hydra_dir = output_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=hydra_dir / "config.yaml")

    overrides = OmegaConf.select(cfg, "hydra_overrides", default=[])
    OmegaConf.save(
        config=OmegaConf.create(list(overrides)), f=hydra_dir / "overrides.yaml"
    )
    run_one_experiment(cfg)
