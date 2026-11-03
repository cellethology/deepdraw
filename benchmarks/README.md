# Retrospective benchmarks

These optional tools evaluate Deepdraw on datasets whose labels are already known. For new experimental campaigns, use the [main workflow](../README.md).

Run the commands below from a source checkout's repository root. Install the benchmark dependencies first:

```bash
python -m pip install -e ".[analysis]"
```

For Slurm, use `python -m pip install -e ".[analysis,cluster]"`.

## Prepare the data

Create a dataset YAML using paths for your machine:

```yaml
datasets:
  - name: my_dataset
    metadata_path: /absolute/path/metadata.csv
    embedding_dir: /absolute/path/embeddings
    subset_ids_path: /absolute/path/subset_ids.txt  # optional; omit for the full pool
```

Each embedding file must contain a 2D `embeddings` array and integer `ids` giving zero-based row positions in the metadata CSV. The CSV must contain the score column you want to maximize. This benchmark loader uses row indices; the experimental CLI also supports string design IDs.

The optional subset file contains one integer ID per line. Use an absolute path for `datasets_file`; local definitions in `benchmarks/datasets/` are ignored by Git and excluded from release packages.

## Run one configuration

For an embedding file named `my_embeddings.npz` and a score column named `score`:

```bash
python -m benchmarks.run_config \
  datasets_file=/absolute/path/datasets.yaml \
  embedding_model=my_embeddings \
  al_settings.label_key=score \
  query_strategy=botorch_mes
```

`embedding_model` is the NPZ filename without its extension. A single run uses the first dataset in the YAML; set `dataset_index` to choose another. Results are written under `results/` unless `results_root_dir` is overridden.

Settings are defined in [conf/config.yaml](conf/config.yaml). Shared algorithm presets are in [deepdraw/conf](../deepdraw/conf/); Hydra uses their filenames, such as `predictor=botorch_gp` and `query_strategy=botorch_mes`.

The default batch size is 12. `al_settings.max_rounds` controls query batches after initialization; `al_settings.top_fraction` controls the top fraction used for benchmark metrics. Feature and target transforms default to `standardize` and `log_standardize`.

## Compare configurations

This local sweep compares three query strategies across all datasets in the YAML, using one seed:

```bash
python -m benchmarks.run_config --multirun \
  datasets_file=/absolute/path/datasets.yaml \
  embedding_model=my_embeddings \
  al_settings.label_key=score \
  predictor=botorch_gp \
  initial_selection_strategy=probcover \
  query_strategy=random,topk,botorch_mes \
  al_settings.seed=0 \
  hydra.sweep.dir=results/comparison
```

Omit `al_settings.seed=0` to use the seed range configured by `num_seeds_per_job` and `seed_start`. With the default `seed_mode=array`, each seed is a separate Hydra job. `seed_mode=inline` runs the configured seeds within each job.

## Run on Slurm

Choose your cluster's partition and resource settings:

```bash
python -m benchmarks.run_config --multirun \
  datasets_file=/absolute/path/datasets.yaml \
  embedding_model=my_embeddings \
  al_settings.label_key=score \
  predictor=botorch_gp \
  initial_selection_strategy=probcover \
  query_strategy=botorch_mes \
  al_settings.seed=0 \
  hydra/launcher=submitit_slurm \
  hydra.launcher.partition=your_partition
```

Local runs use Hydra's `basic` launcher. Optional SubmitIt settings are in [conf/launcher_settings](conf/launcher_settings/). Set `hydra.launcher.cpus_per_task`, memory, timeout, and QoS for your cluster as needed. `single_array_across_datasets=true` groups all datasets in one sweep; with Slurm this uses one array.

## Noise experiments and stopping

[noise_simulation.py](noise_simulation.py) adds simulated measurements and replicate-based uncertainty to retrospective runs. Its settings are under `measurement_simulation` in the main benchmark config.

`confirmed_top_stopping` optionally stops a noise benchmark after a true top circuit passes simulated-measurement confirmation and the minimum batch count is reached. It requires the pool's known true scores. Both noise simulation and confirmed-top stopping are disabled by default.

The general threshold and EI rules are available through `al_settings.stopping_rule`, with `al_settings.phenotype_threshold` or `al_settings.epsilon`. Choose one stopping mechanism per run.

## Inspect and combine results

Each experiment directory contains `results.csv` and `summary.json`. Noise simulations also write `simulated_measurements.csv` and `training_observations.csv`.

For the comparison sweep above:

```bash
python -m benchmarks.aggregate_summaries \
  --sweep-dir results/comparison

python -m benchmarks.report_failed_jobs \
  --sweep-dir results/comparison
```

Aggregation writes `combined_summaries.by_round.csv` under each dataset directory. Use `--overwrite` to recompute an existing aggregation.

The entry point is [run_config.py](run_config.py). [run_active_learning.py](run_active_learning.py) implements the benchmark loop used by that entry point.
