"""Tests for configurable Hydra output paths in benchmarks/conf/config.yaml."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra._internal.callbacks import Callbacks
from hydra.core.global_hydra import GlobalHydra
from hydra.core.plugins import Plugins
from hydra.types import HydraContext
from omegaconf import OmegaConf

from benchmarks.utils.config_utils import register_launcher_settings

_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "conf" / "config.yaml"
)


def _register_now_resolver_for_tests() -> None:
    """Provide deterministic timestamps so config interpolation can be asserted."""
    values = {
        "%Y-%m-%d": "2099-12-31",
        "%H-%M-%S": "23-59-58",
    }
    OmegaConf.register_new_resolver(
        "now",
        lambda fmt: values.get(str(fmt), "UNKNOWN"),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "hydra",
        lambda key: "/tmp/hydra_runtime_output" if key == "runtime.output_dir" else "",
        replace=True,
    )


def test_hydra_sweep_dir_defaults_to_benchmarks_multirun() -> None:
    _register_now_resolver_for_tests()
    cfg = OmegaConf.load(_CONFIG_PATH)

    assert cfg.results_root_dir == "results"
    resolved_sweep_dir = OmegaConf.select(cfg, "hydra.sweep.dir")
    expected_dir = f"{cfg.results_root_dir}/2099-12-31/23-59-58"
    assert resolved_sweep_dir == expected_dir
    assert OmegaConf.select(cfg, "hydra.run.dir") == expected_dir
    assert (
        OmegaConf.select(cfg, "al_settings.output_dir") == "/tmp/hydra_runtime_output"
    )


def test_hydra_sweep_dir_uses_configured_results_root_dir() -> None:
    _register_now_resolver_for_tests()
    cfg = OmegaConf.load(_CONFIG_PATH)
    cfg.results_root_dir = "/tmp/custom_al_runs"

    resolved_sweep_dir = OmegaConf.select(cfg, "hydra.sweep.dir")
    assert resolved_sweep_dir == "/tmp/custom_al_runs/2099-12-31/23-59-58"
    assert OmegaConf.select(cfg, "hydra.run.dir") == resolved_sweep_dir


@pytest.mark.parametrize(
    "launcher", [None, "submitit_local", "submitit_slurm", "basic"]
)
def test_launcher_settings_compose_without_cross_platform_options(launcher):
    if launcher in {"submitit_local", "submitit_slurm"}:
        pytest.importorskip("hydra_plugins.hydra_submitit_launcher")
    register_launcher_settings(_CONFIG_PATH.parent)
    with initialize_config_dir(config_dir=str(_CONFIG_PATH.parent), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"hydra/launcher={launcher}"] if launcher else [],
            return_hydra_config=True,
        )
        settings = cfg.hydra.launcher
        instance = Plugins.instance().instantiate_launcher(
            hydra_context=HydraContext(
                config_loader=GlobalHydra.instance().config_loader(),
                callbacks=Callbacks(),
            ),
            task_function=lambda config: None,
            config=cfg,
        )
        if launcher == "submitit_slurm":
            assert type(instance).__name__ == "SlurmLauncher"
            assert settings.partition is None
            assert settings.mem_per_cpu == "30GB"
            assert settings.qos is None
        else:
            assert all(
                key not in settings for key in ("partition", "qos", "mem_per_cpu")
            )
        if launcher == "submitit_local":
            assert type(instance).__name__ == "LocalLauncher"
            assert settings.cpus_per_task == 1
            assert settings.timeout_min == 1440
        elif launcher in {None, "basic"}:
            assert type(instance).__name__ == "BasicLauncher"
