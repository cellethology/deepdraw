# Retrospective Benchmarks And Cluster Runs

This directory contains the original retrospective active learning benchmark workflow. It is intended for evaluating Deepdraw configurations on datasets where all labels are already known, not for running a new experimental campaign from an unlabeled design pool.

For the user-facing experimental workflow, see the repository-level [README.md](../README.md).

## Entry Point

The main entry point is [run_config.py](run_config.py). It reads dataset definitions from YAML files in [datasets/](datasets/) and runs experiments either locally with `submitit_local` or on a Slurm cluster with `submitit_slurm`.

```bash
python job_sub/run_config.py hydra/launcher=submitit_local
```

## Configuration

The benchmark runner uses [Hydra](https://hydra.cc/) configuration files under [conf/](conf/).

Key config groups:

- predictors: [conf/predictor/](conf/predictor/)
- query strategies: [conf/query_strategy/](conf/query_strategy/)
- initial selection strategies: [conf/initial_selection_strategy/](conf/initial_selection_strategy/)
- feature and target transforms: [conf/transforms/](conf/transforms/)

Main config: [conf/config.yaml](conf/config.yaml)

Example dataset YAML:

```yaml
datasets:
  - name: my_dataset
    metadata_path: /path/to/metadata.csv
    embedding_dir: /path/to/embeddings_dir
    subset_ids_path: /path/to/subset_ids.txt  # optional
```

Example active learning settings:

```yaml
al_settings:
  batch_size: 12
  starting_batch_size: 12
  max_rounds: 29
  label_key: "Fold Change (Induced/Basal)"
  output_dir: ${hydra:runtime.output_dir}
  seed: 0
```

## Running A Single Configuration

Run locally:

```bash
python job_sub/run_config.py \
  hydra/launcher=submitit_local
```

Run locally with explicit components:

```bash
python job_sub/run_config.py \
  hydra/launcher=submitit_local \
  query_strategy=botorch_qlog_nei \
  predictor=botorch_gp \
  initial_selection_strategy=probcover_euclidean \
  embedding_model=1m_alphagenome_1bp_embeddings_kneedle \
  al_settings.batch_size=12 \
  al_settings.max_rounds=29
```

Run on Slurm:

```bash
python job_sub/run_config.py \
  hydra/launcher=submitit_slurm \
  query_strategy=topk \
  predictor=rf_regressor \
  initial_selection_strategy=probcover_euclidean \
  al_settings.batch_size=12
```

## Running Parameter Sweeps

Use Hydra multirun mode:

```bash
python job_sub/run_config.py --multirun \
  hydra/launcher=submitit_local \
  query_strategy=random,topk,botorch_qlog_nei \
  predictor=ridge_regressor,rf_regressor,botorch_gp
```

Sweep embeddings and strategies:

```bash
python job_sub/run_config.py --multirun \
  hydra/launcher=submitit_local \
  embedding_model=evo2_meanpool_block28_pca8,enformer_evo2_concat \
  query_strategy=botorch_qlog_nei,botorch_ucb \
  initial_selection_strategy=core_set,probcover_euclidean
```

Submit a Slurm sweep:

```bash
python job_sub/run_config.py --multirun \
  hydra/launcher=submitit_slurm \
  embedding_model=enformer_evo2_concat,evo2_meanpool_block28_pca8 \
  query_strategy=botorch_qlog_nei,botorch_ucb,botorch_mes \
  predictor=botorch_gp,gaussian_regressor \
  initial_selection_strategy=core_set,probcover_euclidean
```

## Slurm Settings

Cluster settings live under `hydra.launcher` in [conf/config.yaml](conf/config.yaml):

```yaml
hydra:
  launcher:
    timeout_min: 720
    partition: intel-sc3,wzt_20250411
    cpus_per_task: 1
    qos: huge
    mem_per_cpu: 30GB
    submitit_folder: slurm_logs
```

The wrapper can fan out across datasets and parameter combinations. `single_array_across_datasets=true` submits one multirun that sweeps `dataset_index` across all configured datasets in a single Slurm array.

## Aggregating Results

Aggregate summaries:

```bash
python job_sub/aggregate_summaries.py \
  --base-dir results/my_experiment \
  --output results/aggregated_summary.csv
```

The benchmark runner writes per-run files such as:

```text
results/
├── results.csv
├── summary.json
└── error.txt  # only when a run fails
```

## Data Format

Retrospective benchmarks use paired metadata and embedding files.

Embedding NPZ:

```python
{
    "ids": np.ndarray,
    "embeddings": np.ndarray,
}
```

Metadata CSV:

```csv
id,Fold Change (Induced/Basal),other_features
0,5.23,value1
1,3.87,value2
```

The benchmark data loader uses embedding `ids` as row indices into the metadata CSV.
