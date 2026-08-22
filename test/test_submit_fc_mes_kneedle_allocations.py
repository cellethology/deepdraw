from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from job_sub import submit_fc_mes_kneedle_allocations as submitter


def test_corrected_fc_inputs_are_pinned() -> None:
    assert submitter.DATASETS_FILE == "datasets/166k_datasets.yaml"
    assert submitter.EMBEDDING_MODEL == "166k_alphagenome_1bp_embeddingkneedle"
    assert submitter.SCORE_MODE == "induced_over_basal"
    assert submitter.DEFAULT_EVALUATED_ROUNDS == 30


def test_complete_summary_requires_corrected_configuration(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    payload = {
        "completed_rounds": 30,
        "query_strategy": "MES",
        "embedding_model": submitter.EMBEDDING_MODEL,
        "dataset_name": "example",
        "seed": 4,
        "measurement_simulation": {
            "enabled": True,
            "score_mode": submitter.SCORE_MODE,
            "expression_columns": list(submitter.EXPRESSION_COLUMNS),
            "replicates_per_construct": 2,
            "noise_sigma_log10_expression": 0.1,
        },
    }
    summary_path.write_text(json.dumps(payload))

    assert submitter.complete_summary(
        summary_path,
        dataset_name="example",
        seed=4,
        replicates=2,
        noise_sigma=0.1,
        evaluated_rounds=30,
    )

    payload["embedding_model"] = "166k_alphagenome_1bp_embedding_default"
    summary_path.write_text(json.dumps(payload))
    assert not submitter.complete_summary(
        summary_path,
        dataset_name="example",
        seed=4,
        replicates=2,
        noise_sigma=0.1,
        evaluated_rounds=30,
    )


def test_active_manifest_keys_maps_only_queued_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "submission_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "allocation": "12x2",
                "dataset_name": "dataset_a",
                "job_ids": ["100_0", "100_1"],
                "seeds": [0, 1],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(submitter.shutil, "which", lambda _: "/fake/squeue")
    monkeypatch.setattr(
        submitter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="100_1\n"),
    )

    assert submitter.active_manifest_keys(tmp_path) == {("12x2", "dataset_a", 1)}


def test_run_seed_builds_30_round_fc_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output_dir = tmp_path / "12x2" / "example" / "seed_7"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(submitter.subprocess, "run", fake_run)
    monkeypatch.setattr(submitter, "complete_summary", lambda *args, **kwargs: True)
    runner = submitter.RunSeed(
        output_root=str(tmp_path),
        allocation="12x2",
        dataset_index=3,
        dataset_name="example",
        constructs=12,
        replicates=2,
        noise_sigma=0.1,
        evaluated_rounds=30,
    )

    runner(7)

    command = captured["command"]
    assert "datasets_file=datasets/166k_datasets.yaml" in command
    assert "embedding_model=166k_alphagenome_1bp_embeddingkneedle" in command
    assert "al_settings.max_rounds=29" in command
    assert "measurement_simulation.score_mode=induced_over_basal" in command
    assert "measurement_simulation.noise_sigma_log10_expression=0.1" in command
