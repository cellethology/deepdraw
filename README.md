# Gene Circuit Design - Active Learning

[![CI](https://github.com/cellethology/gene_circuit_design/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/cellethology/gene_circuit_design/actions/workflows/pre-commit.yml)
[![Coverage](https://img.shields.io/codecov/c/github/cellethology/gene_circuit_design?logo=codecov)](https://codecov.io/gh/cellethology/gene_circuit_design)

Active learning framework for efficiently discovering high-expression DNA sequences using machine learning and pre-computed sequence embeddings (SEI, EVO2, Enformer, etc.).

## Quick Start

### Installation

```bash
# Clone the repository
git clone git@github.com:cellethology/gene_circuit_design.git
cd gene_circuit_design

# Install dependencies
uv sync --python 3.10

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Using Deepdraw With a New Design Pool

Deepdraw can now be run as a real experimental active-learning loop, not only as
a retrospective benchmark. Start with:

- a CSV design pool containing one row per candidate sequence
- an embeddings NPZ for the same rows, with arrays named `embeddings` and `ids`
  (or `sample_ids`)
- no expression labels yet

Create the first batch to measure:

```bash
deepdraw init \
  --pool-csv designs.csv \
  --embeddings embeddings.npz \
  --sequence-column sequence \
  --output-dir runs/my_deepdraw_run \
  --starting-batch-size 24 \
  --batch-size 24
```

This writes:

- `runs/my_deepdraw_run/round_000_to_measure.csv`
- `runs/my_deepdraw_run/latest_recommendations.csv`
- `runs/my_deepdraw_run/deepdraw_state.json`
- `runs/my_deepdraw_run/selection_history.csv`

After measuring that first batch, add the measured values to a CSV that keeps
either `deepdraw_id` or `deepdraw_pool_index` from the recommendation file, then
ask Deepdraw for the next batch:

```bash
deepdraw suggest \
  --run-dir runs/my_deepdraw_run \
  --measurements measurements.csv \
  --label-column expression
```

Repeat `deepdraw suggest` as the measurement table grows. The CLI reuses the
same model, query strategy, batch size, seed, and transform settings recorded in
`deepdraw_state.json`.

The retrospective benchmark entry point is still [job_sub/run_config.py](job_sub/run_config.py). It reads dataset definitions from [job_sub/datasets/datasets.yaml](job_sub/datasets/datasets.yaml) and runs experiments either locally (`submitit_local`) or on a Slurm cluster (`submitit_slurm`).

## How It Works

1. **Load Data**: Pre-computed embeddings (NPZ) + a design pool CSV
2. **Initial Selection**: Select starting samples using strategy (Random, K-means, CoreSet, ProbCover) without requiring labels
3. **Active Learning Loop**:
   - Train a predictor on labeled samples
   - Use query strategy to select next batch of informative samples
   - Add selected samples to labeled pool
   - Repeat for N rounds
4. **Track Recommendations**: Save each recommended batch and the full selection history

For retrospective experiments, Deepdraw also computes metrics such as Spearman
correlation and top-discovery rates because all labels are already known.

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management. All settings are defined in YAML files.

### 1. Setup Datasets Configuration

Create or edit [job_sub/datasets/datasets.yaml](job_sub/datasets/datasets.yaml):

```yaml
datasets:
  - name: my_dataset
    metadata_path: /path/to/metadata.csv
    embedding_dir: /path/to/embeddings_dir
    subset_ids_path: /path/to/subset_ids.txt  # optional
```

### 2. Main Configuration

The main config is in [job_sub/conf/config.yaml](job_sub/conf/config.yaml):

```yaml
defaults:
  - predictor: botorch_gp
  - query_strategy: botorch_log_ei
  - initial_selection_strategy: core_set
  - transforms@feature_transforms: standardize
  - transforms@target_transforms: log_standardize
  - override hydra/launcher: submitit_slurm  # or submitit_local

# Data paths (set via environment or datasets.yaml)
datasets_file: datasets/datasets.yaml
dataset_name: ${env:AL_DATASET_NAME}
metadata_path: ${env:AL_METADATA_PATH}
embedding_dir: ${env:AL_EMBEDDING_ROOT}
embedding_model: enformer_evo2_concat
embedding_path: ${embedding_dir}/${embedding_model}.npz

# Active learning settings
al_settings:
  batch_size: 12
  starting_batch_size: 12
  max_rounds: 9
  label_key: "Fold Change (Induced/Basal)"
  output_dir: ${hydra:runtime.output_dir}
  seed: 0
```

### Available Components

**Predictors** ([job_sub/conf/predictor/](job_sub/conf/predictor/)):
- `linear_regressor`: Linear regression
- `ridge_regressor`: Ridge regression
- `bayes_ridge`: Bayesian ridge regression
- `kn_regressor`: K-nearest neighbors
- `rf_regressor`: Random forest
- `gradboost_regressor`: Gradient boosting
- `histgradboost_regressor`: Histogram-based gradient boosting
- `mlp_regressor`: Multi-layer perceptron neural network
- `gaussian_regressor`: Gaussian process (sklearn)
- `botorch_gp`: Gaussian process (BoTorch)

**Query Strategies** ([job_sub/conf/query_strategy/](job_sub/conf/query_strategy/)):
- `random`: Random sampling baseline
- `topk`: Select top-k by predicted value
- `toploglikelihood`: Select by log-likelihood
- `predstdhybrid`: Hybrid prediction-uncertainty strategy
- `botorch_log_ei`: Log Expected Improvement (Bayesian optimization)
- `botorch_log_pi`: Log Probability of Improvement
- `botorch_ucb`: Upper Confidence Bound
- `botorch_mes`: Max-value Entropy Search
- `botorch_qlog_nei`: Noisy Expected Improvement
- `botorch_ts`: Thompson Sampling

**Initial Selection** ([job_sub/conf/initial_selection_strategy/](job_sub/conf/initial_selection_strategy/)):
- `random`: Random sampling
- `kmeans`: K-means clustering with centroid selection
- `core_set`: Greedy k-center coverage
- `density_core_set`: Density-weighted k-center coverage
- `probcover`: Probabilistic coverage-based selection

**Transforms** ([job_sub/conf/transforms/](job_sub/conf/transforms/)):
- Feature: `standardize`, `normalize`, `pca`
- Target: `log_standardize`, `standardize`, `quantile`

## Project Structure

```
├── core/                      # Core active learning modules
│   ├── experiment.py          # Main ActiveLearningExperiment orchestrator
│   ├── data_loader.py         # Data loading (NPZ embeddings + CSV metadata)
│   ├── query_strategies.py   # Query strategy implementations
│   ├── predictor_trainer.py  # ML model training and prediction
│   ├── initial_selection_strategies.py  # Initial pool selection
│   ├── metrics_calculator.py # Performance metrics computation
│   └── round_tracker.py      # Results tracking across rounds
├── job_sub/                   # Job submission and utilities
│   ├── conf/                  # Hydra configuration files
│   │   ├── config.yaml        # Main config (launcher, defaults)
│   │   ├── predictor/         # Predictor configs
│   │   ├── query_strategy/    # Query strategy configs
│   │   ├── initial_selection_strategy/  # Initial selection configs
│   │   └── transforms/        # Transform configs
│   ├── datasets/              # Dataset definitions
│   │   └── datasets.yaml      # Define datasets with paths
│   ├── run_config.py         # Main entry point (local/SLURM)
│   └── aggregate_summaries.py # Aggregate results from multiple runs
├── configs/                   # Legacy experiment configuration files
├── plotting/                  # Visualization scripts
├── test/                      # Unit and integration tests
└── run_active_learning.py    # Core experiment runner (called by run_config.py)
```

## Usage Examples

### Running a Single Configuration

Run one experiment with specific settings:

```bash
# Run locally with default settings
python job_sub/run_config.py \
  hydra/launcher=submitit_local

# Run locally with custom configuration
python job_sub/run_config.py \
  hydra/launcher=submitit_local \
  query_strategy=botorch_log_ei \
  predictor=botorch_gp \
  initial_selection_strategy=core_set \
  embedding_model=enformer_evo2_concat \
  al_settings.batch_size=12 \
  al_settings.max_rounds=9

# Run on Slurm cluster with specific configuration
python job_sub/run_config.py \
  hydra/launcher=submitit_slurm \
  query_strategy=topk \
  predictor=rf_regressor \
  initial_selection_strategy=probcover \
  al_settings.batch_size=8
```

### Running Parameter Sweeps

Run multiple experiments with different parameter combinations using `--multirun`:

```bash
# Sweep locally over multiple query strategies and predictors
python job_sub/run_config.py --multirun \
  hydra/launcher=submitit_local \
  query_strategy=random,topk,botorch_log_ei \
  predictor=linear_regressor,rf_regressor,botorch_gp

# Sweep locally over embeddings and strategies
python job_sub/run_config.py --multirun \
  hydra/launcher=submitit_local \
  embedding_model=evo2_meanpool_block28_pca8,enformer_evo2_concat \
  query_strategy=botorch_log_ei,botorch_ucb \
  initial_selection_strategy=core_set,probcover

# Sweep on Slurm cluster (submits multiple jobs)
python job_sub/run_config.py --multirun \
  hydra/launcher=submitit_slurm \
  embedding_model=enformer_evo2_concat,evo2_meanpool_block28_pca8 \
  query_strategy=botorch_log_ei,botorch_ucb,botorch_mes \
  predictor=botorch_gp,gaussian_regressor \
  initial_selection_strategy=core_set,probcover

# The --multirun flag automatically:
# 1. Creates all parameter combinations
# 2. Submits separate jobs for each combination
# 3. Organizes results by parameters
```

### Aggregate and Analyze Results

```bash
# Aggregate summaries from multiple runs
python job_sub/aggregate_summaries.py \
  --base-dir results/my_experiment \
  --output results/aggregated_summary.csv

# Generate plots
python plotting/averaged_performance_analysis.py \
  --results-base-path results/my_experiment \
  --output-dir plots/my_experiment \
  --metric max_extreme_value_auc
```

## Data Format

### Embeddings (NPZ format)
```python
# embeddings.npz structure:
{
  'ids': [0, 1, 2, ...],         # Shape: (N,), row ids from the pool CSV
  'embeddings': np.array([...])  # Shape: (N, embedding_dim)
}
```

`deepdraw init` also accepts `sample_ids` instead of `ids`. If your pool has a
stable identifier column, pass `--id-column`; otherwise Deepdraw uses row indices.

### Metadata (CSV format)
```csv
id,expression,other_features
seq_001,5.23,value1
seq_002,3.87,value2
```

The `id` column must match `sample_ids` in the NPZ file.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov-report=html

# Run specific test file
pytest test/test_experiment.py
```

## Slurm Cluster Configuration

When using `hydra/launcher=submitit_slurm`, configure cluster settings in [job_sub/conf/config.yaml](job_sub/conf/config.yaml):

```yaml
hydra:
  launcher:
    timeout_min: 160           # Job timeout in minutes
    partition: gpu,cpu         # Comma-separated list of partitions
    cpus_per_task: 4           # CPUs per job
    mem_per_cpu: 16GB          # Memory per CPU
    qos: normal                # Quality of service
    submitit_folder: slurm_logs  # Where to save job logs
```

The script automatically loops through all datasets in `datasets.yaml`, submits separate Slurm jobs for each parameter combination, and monitors job completion.

## Output

Results are saved to the configured `output_dir`:

```
results/{experiment_name}/
├── results.csv                # Round-by-round metrics
├── summary.json               # Aggregated summary metrics
└── selected_variants.csv      # Track of all selected samples
```

## Contributing

This project uses pre-commit hooks for code quality:

```bash
# Run hooks manually
pre-commit run --all-files

# Specific hooks
pre-commit run ruff --all-files
pre-commit run pytest
```

## License

[Add license information]

## Citation

[Add citation information]
