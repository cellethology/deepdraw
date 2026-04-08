"""Helpers for expanding a Hydra config into per-seed job configs."""

import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from job_sub.utils.seed_runner import run_seed_experiment


def generate_seed_values(cfg) -> list[int]:
    """Return sequential seeds to run inside a single Hydra job."""
    num_seeds = int(OmegaConf.select(cfg, "num_seeds_per_job", default=1))
    if num_seeds < 1:
        raise ValueError("num_seeds_per_job must be >= 1")
    seed_start = int(OmegaConf.select(cfg, "seed_start", default=0))
    return list(range(seed_start, seed_start + num_seeds))


def materialize_seed_cfgs(cfg) -> list[dict[str, Any]]:
    """Return serialized configs for each seed to support multiprocessing."""
    base_output_dir = Path(str(cfg.al_settings.output_dir))
    base_cfg = OmegaConf.to_container(cfg, resolve=False)
    overrides = _extract_hydra_overrides()
    seed_cfgs: list[dict[str, Any]] = []
    for seed in generate_seed_values(cfg):
        seed_output_dir = base_output_dir / f"seed_{seed}"
        seed_cfg = copy.deepcopy(base_cfg)
        al_settings = seed_cfg.setdefault("al_settings", {})
        al_settings["seed"] = seed
        al_settings["output_dir"] = str(seed_output_dir)
        seed_cfg["hydra_overrides"] = list(overrides)
        seed_cfgs.append(seed_cfg)
    return seed_cfgs


def max_seed_workers(cfg, num_tasks: int) -> int:
    """Decide max workers based on config flag and available CPUs."""
    cpu_count = os.cpu_count() or 1
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get(
        "SLURM_CPUS_ON_NODE"
    )
    available = cpu_count
    if slurm_cpus:
        try:
            available = max(1, int(str(slurm_cpus).split("(")[0]))
        except ValueError:
            available = cpu_count
    print(
        f"[seed_jobs] cpu_count={cpu_count} slurm_cpus={slurm_cpus} "
        f"allocated_cpus={available}"
    )
    if not OmegaConf.select(cfg, "parallelize_seeds", default=True):
        return 1
    workers = max(1, min(available, num_tasks))
    print(f"[seed_jobs] seed_workers={workers} num_tasks={num_tasks}")
    return workers


def run_seed_jobs(cfg) -> None:
    """Run all seed configurations, parallelizing when enabled."""
    seed_cfgs = materialize_seed_cfgs(cfg)
    workers = max_seed_workers(cfg, len(seed_cfgs))
    if workers == 1 or len(seed_cfgs) == 1:
        for raw_cfg in seed_cfgs:
            run_seed_experiment(raw_cfg)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_seed_experiment, raw_cfg) for raw_cfg in seed_cfgs
        ]
        for future in as_completed(futures):
            future.result()


def _extract_hydra_overrides() -> list[str]:
    """Return task-level overrides from Hydra runtime if available."""
    if HydraConfig.initialized():
        try:
            return [str(item) for item in HydraConfig.get().overrides.task]
        except Exception:  # pragma: no cover - defensive
            return []
    return []
