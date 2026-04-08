"""Tests for configurable Hydra output paths in job_sub/conf/config.yaml."""

from pathlib import Path

from omegaconf import OmegaConf

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "job_sub" / "conf" / "config.yaml"


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


def test_hydra_sweep_dir_defaults_to_job_sub_multirun() -> None:
    _register_now_resolver_for_tests()
    cfg = OmegaConf.load(_CONFIG_PATH)

    resolved_sweep_dir = OmegaConf.select(cfg, "hydra.sweep.dir")
    expected_dir = f"{cfg.results_root_dir}/2099-12-31/23-59-58"
    assert resolved_sweep_dir == expected_dir
    assert (
        OmegaConf.select(cfg, "al_settings.output_dir") == "/tmp/hydra_runtime_output"
    )


def test_hydra_sweep_dir_uses_configured_results_root_dir() -> None:
    _register_now_resolver_for_tests()
    cfg = OmegaConf.load(_CONFIG_PATH)
    cfg.results_root_dir = "/tmp/custom_al_runs"

    resolved_sweep_dir = OmegaConf.select(cfg, "hydra.sweep.dir")
    assert resolved_sweep_dir == "/tmp/custom_al_runs/2099-12-31/23-59-58"
