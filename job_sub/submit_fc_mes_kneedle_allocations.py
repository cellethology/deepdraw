from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import submitit
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
# Keep the venv entry-point path. Resolving its symlink can bypass the venv.
PYTHON = Path(sys.executable).absolute()
DATASETS_FILE = "datasets/166k_datasets.yaml"
DATASETS_PATH = REPO / "job_sub" / DATASETS_FILE
SLURM_BIN = Path("/soft/slurm/slurm-25.05.2_installation/bin")

EMBEDDING_MODEL = "166k_alphagenome_1bp_embeddingkneedle"
LABEL_KEY = "Fold Change (Induced/Basal)"
SCORE_MODE = "induced_over_basal"
EXPRESSION_COLUMNS = (
    "Basal Exp (GFP, au)",
    "Induced Exp (GFP, au)",
)
DEFAULT_NOISE_SIGMA_LOG10_EXPRESSION = 0.10
DEFAULT_EVALUATED_ROUNDS = 30
CONFIRMATION_Z = 1.96
NUM_SEEDS = 30

ALLOCATIONS = {
    "24x1": (24, 1),
    "12x2": (12, 2),
    "8x3": (8, 3),
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    metadata_path: Path
    embedding_dir: Path
    subset_ids_path: Path


@dataclass(frozen=True)
class RunSeed:
    output_root: str
    allocation: str
    dataset_index: int
    dataset_name: str
    constructs: int
    replicates: int
    noise_sigma: float
    evaluated_rounds: int

    def __call__(self, seed: int) -> str:
        output_dir = (
            Path(self.output_root)
            / self.allocation
            / self.dataset_name
            / f"seed_{seed}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["AL_DATASET_INDEX"] = str(self.dataset_index)
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["HYDRA_FULL_ERROR"] = "1"

        expression_columns = json.dumps(list(EXPRESSION_COLUMNS), separators=(",", ":"))
        command = [
            str(PYTHON),
            "job_sub/run_config.py",
            f"datasets_file={DATASETS_FILE}",
            "single_array_across_datasets=false",
            f"num_seeds_per_job={NUM_SEEDS}",
            "seed_start=0",
            f"embedding_model={EMBEDDING_MODEL}",
            "initial_selection_strategy=probcover_euclidean",
            "query_strategy=botorch_mes",
            "predictor=botorch_gp",
            f"al_settings.seed={seed}",
            f"al_settings.starting_batch_size={self.constructs}",
            f"al_settings.batch_size={self.constructs}",
            f"al_settings.max_rounds={self.evaluated_rounds - 1}",
            f"al_settings.label_key={json.dumps(LABEL_KEY)}",
            "measurement_simulation.enabled=true",
            f"measurement_simulation.replicates_per_construct={self.replicates}",
            (f"measurement_simulation.noise_sigma_log10_expression={self.noise_sigma}"),
            f"measurement_simulation.score_mode={SCORE_MODE}",
            f"measurement_simulation.expression_columns={expression_columns}",
            f"measurement_simulation.confirmation_z={CONFIRMATION_Z}",
            f"hydra.run.dir={json.dumps(str(output_dir))}",
        ]
        subprocess.run(command, cwd=REPO, env=env, check=True)

        summary_path = output_dir / "summary.json"
        if not complete_summary(
            summary_path,
            dataset_name=self.dataset_name,
            seed=seed,
            replicates=self.replicates,
            noise_sigma=self.noise_sigma,
            evaluated_rounds=self.evaluated_rounds,
        ):
            raise RuntimeError(f"Missing or invalid expected output: {summary_path}")
        return str(summary_path)


def load_datasets() -> list[DatasetConfig]:
    config = OmegaConf.load(DATASETS_PATH)
    return [
        DatasetConfig(
            name=str(dataset.name),
            metadata_path=Path(str(dataset.metadata_path)),
            embedding_dir=Path(str(dataset.embedding_dir)),
            subset_ids_path=Path(str(dataset.subset_ids_path)),
        )
        for dataset in config.datasets
    ]


def validate_inputs(datasets: list[DatasetConfig]) -> None:
    if len(datasets) != 39:
        raise ValueError(f"Expected 39 FC datasets, found {len(datasets)}.")

    missing: list[Path] = []
    metadata_paths: set[Path] = set()
    for dataset in datasets:
        metadata_paths.add(dataset.metadata_path)
        for path in (
            dataset.metadata_path,
            dataset.embedding_dir / f"{EMBEDDING_MODEL}.npz",
            dataset.subset_ids_path,
        ):
            if not path.is_file():
                missing.append(path)
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise FileNotFoundError(
            f"Missing corrected FC sweep inputs:\n{preview}{suffix}"
        )

    required_columns = {*EXPRESSION_COLUMNS, LABEL_KEY}
    for metadata_path in metadata_paths:
        columns = set(pd.read_csv(metadata_path, nrows=0).columns)
        absent = sorted(required_columns - columns)
        if absent:
            raise KeyError(f"{metadata_path} is missing required columns: {absent}")


def parse_int_spec(spec: str, upper_bound: int, label: str) -> list[int]:
    if spec == "all":
        return list(range(upper_bound))
    values = sorted({int(value) for value in spec.split(",")})
    if not values or values[0] < 0 or values[-1] >= upper_bound:
        raise ValueError(f"Invalid {label}: {spec}")
    return values


def complete_summary(
    summary_path: Path,
    *,
    dataset_name: str,
    seed: int,
    replicates: int,
    noise_sigma: float,
    evaluated_rounds: int,
) -> bool:
    if not summary_path.is_file():
        return False
    try:
        with summary_path.open() as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    simulation = payload.get("measurement_simulation") or {}
    return (
        int(payload.get("completed_rounds", -1)) == evaluated_rounds
        and str(payload.get("query_strategy", "")).upper() == "MES"
        and str(payload.get("embedding_model", "")) == EMBEDDING_MODEL
        and str(payload.get("dataset_name", "")) == dataset_name
        and int(payload.get("seed", -1)) == seed
        and bool(simulation.get("enabled"))
        and str(simulation.get("score_mode", "")) == SCORE_MODE
        and tuple(simulation.get("expression_columns") or ()) == EXPRESSION_COLUMNS
        and int(simulation.get("replicates_per_construct", -1)) == replicates
        and abs(float(simulation.get("noise_sigma_log10_expression", -1)) - noise_sigma)
        < 1e-12
    )


def active_manifest_keys(output_root: Path) -> set[tuple[str, str, int]]:
    manifest_path = output_root / "submission_manifest.jsonl"
    if not manifest_path.is_file():
        return set()

    job_to_key: dict[str, tuple[str, str, int]] = {}
    with manifest_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                job_ids = record["job_ids"]
                seeds = record["seeds"]
                if len(job_ids) != len(seeds):
                    raise ValueError("job_ids and seeds have different lengths")
                for job_id, seed in zip(job_ids, seeds, strict=True):
                    job_to_key[str(job_id)] = (
                        str(record["allocation"]),
                        str(record["dataset_name"]),
                        int(seed),
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid manifest record on line {line_number}: {exc}"
                ) from exc

    if not job_to_key:
        return set()

    squeue = shutil.which("squeue")
    if squeue is None:
        raise RuntimeError(
            "Cannot safely check the existing manifest because squeue was not found."
        )
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or getpass.getuser()
    result = subprocess.run(
        [squeue, "-r", "-h", "-u", user, "-o", "%i"],
        check=True,
        capture_output=True,
        text=True,
    )
    active_job_ids = {
        line.strip() for line in result.stdout.splitlines() if line.strip()
    }
    return {job_to_key[job_id] for job_id in active_job_ids if job_id in job_to_key}


def build_executor(
    *,
    output_root: Path,
    allocation: str,
    task_count: int,
    job_prefix: str,
    partitions: str,
    qos: str,
    mem_per_cpu: str,
    timeout_min: int,
):
    folder = output_root / allocation / "slurm_logs" / "%j"
    executor = submitit.AutoExecutor(folder=folder, cluster="slurm")
    executor.update_parameters(
        timeout_min=timeout_min,
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=1,
        slurm_partition=partitions,
        slurm_qos=qos,
        slurm_mem_per_cpu=mem_per_cpu,
        slurm_array_parallelism=task_count,
        slurm_job_name=f"{job_prefix}_{allocation}",
        slurm_signal_delay_s=120,
    )
    return executor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit corrected MES FC allocation sweeps on kneedle embeddings."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--allocations",
        default="24x1,12x2,8x3",
        help="Comma-separated allocation names.",
    )
    parser.add_argument(
        "--dataset-indices", default="all", help="Comma-separated indices or 'all'."
    )
    parser.add_argument(
        "--seeds", default="all", help="Comma-separated seeds or 'all'."
    )
    parser.add_argument(
        "--noise-sigma", type=float, default=DEFAULT_NOISE_SIGMA_LOG10_EXPRESSION
    )
    parser.add_argument(
        "--evaluated-rounds", type=int, default=DEFAULT_EVALUATED_ROUNDS
    )
    parser.add_argument("--partitions", default="intel-sc3,amd-ep2,amd-ep5")
    parser.add_argument("--qos", default="huge")
    parser.add_argument("--mem-per-cpu", default="30GB")
    parser.add_argument("--timeout-min", type=int, default=2880)
    parser.add_argument("--job-prefix", default="dd_fc_k01")
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["PATH"] = f"{SLURM_BIN}:{os.environ.get('PATH', '')}"
    if args.noise_sigma < 0:
        raise ValueError("--noise-sigma must be nonnegative.")
    if args.evaluated_rounds < 1:
        raise ValueError("--evaluated-rounds must be positive.")

    output_root = args.output_root.expanduser().resolve()
    datasets = load_datasets()
    validate_inputs(datasets)
    dataset_indices = parse_int_spec(
        args.dataset_indices, len(datasets), "dataset indices"
    )
    seeds = parse_int_spec(args.seeds, NUM_SEEDS, "seeds")
    allocations = [value.strip() for value in args.allocations.split(",")]
    unknown = set(allocations).difference(ALLOCATIONS)
    if unknown:
        raise ValueError(f"Unknown allocations: {sorted(unknown)}")

    active_keys = active_manifest_keys(output_root)
    plans = []
    complete_count = 0
    active_count = 0
    for allocation in allocations:
        constructs, replicates = ALLOCATIONS[allocation]
        for dataset_index in dataset_indices:
            dataset_name = datasets[dataset_index].name
            missing = []
            for seed in seeds:
                summary_path = (
                    output_root
                    / allocation
                    / dataset_name
                    / f"seed_{seed}"
                    / "summary.json"
                )
                if complete_summary(
                    summary_path,
                    dataset_name=dataset_name,
                    seed=seed,
                    replicates=replicates,
                    noise_sigma=args.noise_sigma,
                    evaluated_rounds=args.evaluated_rounds,
                ):
                    complete_count += 1
                elif (allocation, dataset_name, seed) in active_keys:
                    active_count += 1
                else:
                    missing.append(seed)
            if missing:
                plans.append(
                    (
                        allocation,
                        constructs,
                        replicates,
                        dataset_index,
                        dataset_name,
                        missing,
                    )
                )

    total_tasks = sum(len(plan[-1]) for plan in plans)
    print(f"repo={REPO}")
    print(f"python={PYTHON}")
    print(f"output_root={output_root}")
    print(f"datasets_file={DATASETS_FILE}")
    print(f"embedding_model={EMBEDDING_MODEL}")
    print(f"datasets={len(dataset_indices)}")
    print(f"noise_sigma_log10_expression={args.noise_sigma}")
    print(f"score_mode={SCORE_MODE}")
    print(f"evaluated_rounds={args.evaluated_rounds}")
    print(f"array_parents={len(plans)}")
    print(f"tasks_to_submit={total_tasks}")
    print(f"completed_tasks_skipped={complete_count}")
    print(f"active_tasks_skipped={active_count}")
    print(f"partitions={args.partitions}")
    print(f"qos={args.qos}")
    print(f"submit={args.submit}")
    if not args.submit or not plans:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "submission_manifest.jsonl"
    with manifest_path.open("a") as manifest:
        for allocation, constructs, replicates, index, dataset, missing in plans:
            runner = RunSeed(
                output_root=str(output_root),
                allocation=allocation,
                dataset_index=index,
                dataset_name=dataset,
                constructs=constructs,
                replicates=replicates,
                noise_sigma=args.noise_sigma,
                evaluated_rounds=args.evaluated_rounds,
            )
            executor = build_executor(
                output_root=output_root,
                allocation=allocation,
                task_count=len(missing),
                job_prefix=args.job_prefix,
                partitions=args.partitions,
                qos=args.qos,
                mem_per_cpu=args.mem_per_cpu,
                timeout_min=args.timeout_min,
            )
            jobs = executor.map_array(runner, missing)
            record = {
                "submitted_at": datetime.now().isoformat(timespec="seconds"),
                "array_parent_id": str(jobs[0].job_id).split("_")[0],
                "allocation": allocation,
                "dataset_index": index,
                "dataset_name": dataset,
                "seeds": missing,
                "job_ids": [str(job.job_id) for job in jobs],
                "array_parallelism": len(missing),
                "datasets_file": DATASETS_FILE,
                "embedding_model": EMBEDDING_MODEL,
                "noise_sigma_log10_expression": args.noise_sigma,
                "evaluated_rounds": args.evaluated_rounds,
            }
            manifest.write(json.dumps(record, sort_keys=True) + "\n")
            manifest.flush()
            print(
                f"submitted parent={record['array_parent_id']} "
                f"allocation={allocation} dataset={dataset} tasks={len(missing)}"
            )


if __name__ == "__main__":
    main()
