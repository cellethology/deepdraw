# Deepdraw

[![CI](https://github.com/cellethology/deepdraw/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/cellethology/deepdraw/actions/workflows/pre-commit.yml)
[![Coverage](https://img.shields.io/codecov/c/github/cellethology/deepdraw?logo=codecov)](https://codecov.io/gh/cellethology/deepdraw)

Deepdraw selects batches of genetic circuits to measure using sequence embeddings and accumulated experimental results. Its default search combines ProbCover initial selection, Gaussian-process prediction, and maximum-value entropy search (MES).

The active-learning loop runs on CPU. AlphaGenome embedding generation is optional. For retrospective evaluation on fully measured datasets, see [benchmarks](https://github.com/cellethology/deepdraw/blob/main/benchmarks/README.md).

## Install

Use Python 3.11 for the full installation. On Linux or macOS, create and activate an environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install Deepdraw with AlphaGenome embedding generation:

```bash
python -m pip install "deepdraw-opt[alphagenome]" \
  "https://github.com/google-deepmind/alphagenome_research/archive/dad09dd11a480fb8c16ae703a606f55a6e3e968b.zip"
```

Include both arguments; the second installs Google's research library.

To skip AlphaGenome and use existing embeddings, install only Deepdraw (Python 3.10 or later):

```bash
python -m pip install deepdraw-opt
```

The package name is `deepdraw-opt`; the command and Python import are `deepdraw`. Check the installation with `deepdraw --help`. Your own data can be used without a repository checkout.

## Try the dummy example

The example includes 60 designs, synthetic embeddings, and measurements matching the default first batch. It requires no model download.

Download the repository or clone it, then run the example from its root:

```bash
git clone https://github.com/cellethology/deepdraw.git
cd deepdraw
```

```bash
deepdraw init \
  --pool-csv examples/deepdraw_dummy/design_pool.csv \
  --embeddings examples/deepdraw_dummy/embeddings.npz \
  --sequence-column sequence \
  --id-column variant_id \
  --stopping-rule threshold \
  --phenotype-threshold 19

deepdraw suggest \
  --run-dir deepdraw_run \
  --measurements examples/deepdraw_dummy/measurements.csv \
  --label-column Expression
```

These commands write `deepdraw_run/round_000_to_measure.csv` and `round_001_to_measure.csv`. See the [example README](https://github.com/cellethology/deepdraw/blob/main/examples/deepdraw_dummy/README.md) to simulate another measurement update and exercise stopping.

## Use your own data

### 1. Prepare the design pool and embeddings

Create `designs.csv` with one row per candidate, a unique ID, and its DNA sequence:

```csv
variant_id,sequence
variant_001,ATGCGTACGTTAGCGA
variant_002,ATGCGTACGATAGCAA
variant_003,ATGCGTACGCTAGCTA
```

Provide an NPZ containing:

| Array | Contents |
| --- | --- |
| `embeddings` | A numeric array of shape `(number_of_designs, number_of_features)`. |
| `ids` | Design IDs matching the CSV's ID column, one per embedding row. |

`sample_ids` is also accepted as the ID-array name. Zero-based pool row indices are supported; an NPZ without IDs must follow the CSV's row order exactly. Keep the pool and embeddings at their original paths while continuing a run.

You can use embeddings from another model or generate them with the [AlphaGenome workflow below](#generate-alphagenome-embeddings).

### 2. Select the first batch

```bash
deepdraw init \
  --pool-csv designs.csv \
  --embeddings embeddings.npz \
  --sequence-column sequence \
  --id-column variant_id \
  --output-dir runs/my_project \
  --stopping-rule threshold \
  --phenotype-threshold 10
```

Choose a phenotype cutoff appropriate for your experiment. Deepdraw maximizes the supplied score.

The run directory contains:

```text
runs/my_project/
├── deepdraw_state.json
├── round_000_to_measure.csv
├── latest_recommendations.csv
└── selection_history.csv
```

Measure the designs in `round_000_to_measure.csv`. Recommendation files retain the original pool columns; `selection_history.csv` also records `deepdraw_round`.

### 3. Add measurements and request the next batch

Create a cumulative `measurements.csv` containing design IDs and one measured score per design:

```csv
variant_id,Expression
variant_001,1.42
variant_002,3.87
variant_003,5.11
```

Use the IDs actually recommended for your run. Include every previously recommended design; additional measured designs from the pool can also be included.

```bash
deepdraw suggest \
  --run-dir runs/my_project \
  --measurements measurements.csv \
  --label-column Expression
```

Deepdraw trains on the measured designs and writes `round_001_to_measure.csv`. Measure that batch, append its scores to the same cumulative file, and repeat `suggest` for later rounds. Measured designs are excluded from new batches.

When a stopping rule fires, the command reports the reason and writes no new batch. Use a new `--output-dir` when starting another run.

## Search settings and stopping

The CLI selects the recommended search settings automatically. Algorithm flags are optional overrides:

| Flag | Default | Alternatives |
| --- | --- | --- |
| `--starting-batch-size`, `--batch-size` | `12` | Any positive integer; capped by available designs. |
| `--seed` | `0` | An integer from `0` to `4294967295`. |
| `--initial-selection-strategy` | `probcover` | `core_set`, `density_core_set`, `kmeans`, `probcover_cosine`, `typiclust`, `random`. |
| `--predictor` | `gp` | `linear_gp`, `gradboost_regressor`, `rf_ensemble`. |
| `--query-strategy` | `mes` | `topk`, `predstdhybrid`, `ucb`, `pes`, `gibbon`, `log_pi`, `log_ei`, `qlog_nei`, `qlog_ei`, `random`. |
| `--feature-transforms` | `standardize` | `none`. |
| `--target-transforms` | `log_standardize` | `none`. |
| `--stopping-rule` | `none` | `threshold`, `ei`. |

`probcover` uses Euclidean distance; `probcover_cosine` uses cosine distance.

Gradient boosting supports `topk`, `predstdhybrid`, and `random`. The random-forest ensemble also supports `ucb`, `log_pi`, `log_ei`, `qlog_nei`, and `qlog_ei`. The entropy queries `mes`, `pes`, and `gibbon` require `gp` or `linear_gp`.

The default target transform applies `log(1 + score)` followed by standardization. Use `--target-transforms none` for scores that cannot be log-transformed.

Fixed-threshold stopping, used above, stops when a measured score meets or exceeds `--phenotype-threshold`. For an adaptive EI example using the bundled data:

```bash
deepdraw init \
  --pool-csv examples/deepdraw_dummy/design_pool.csv \
  --embeddings examples/deepdraw_dummy/embeddings.npz \
  --sequence-column sequence \
  --id-column variant_id \
  --output-dir runs/dummy_ei \
  --stopping-rule ei \
  --epsilon 0.01

deepdraw suggest \
  --run-dir runs/dummy_ei \
  --measurements examples/deepdraw_dummy/measurements.csv \
  --label-column Expression
```

After each measured batch, `suggest` stops if maximum expected improvement over the unmeasured pool is at most epsilon. With the default target transform, epsilon is in `log(1 + score)` units before standardization. The value above is for the dummy example; choose it for your objective. EI stopping reuses a compatible fitted GP or fits a separate GP when another predictor is selected.

Omitting the stopping options disables threshold and EI stopping. All runs finish when no unmeasured designs remain. Use `--log-level WARNING` for quieter output.

## Generate AlphaGenome embeddings

Use the full installation above. If you already have embeddings, skip this section.

Accept access to `google/alphagenome-all-folds` on Hugging Face, then authenticate:

```bash
hf auth login
```

An `HF_TOKEN` environment variable is also supported. For an NVIDIA GPU on Linux, install CUDA support:

```bash
python -m pip install --upgrade "jax[cuda12]>=0.6.0"
```

CPU execution is also supported; use CPU on macOS. Start with a batch size of 1.

```bash
deepdraw embed-alphagenome \
  --pool-csv designs.csv \
  --sequence-column sequence \
  --id-column variant_id \
  --output embeddings_alphagenome_raw.npz \
  --model-version all_folds \
  --resolution 1 \
  --species human \
  --pooling mean \
  --batch-size 1

deepdraw pca \
  --input embeddings_alphagenome_raw.npz \
  --output embeddings.npz \
  --selection-method kneedle
```

Use the resulting `embeddings.npz` with `deepdraw init`.

## Repository layout and development

| Directory | Contents |
| --- | --- |
| [deepdraw/](https://github.com/cellethology/deepdraw/tree/main/deepdraw) | CLI, experimental workflow, embedding tools, and shared presets. |
| [core/](https://github.com/cellethology/deepdraw/blob/main/core/README.md) | Models, selection strategies, training, and stopping rules. |
| [examples/deepdraw_dummy/](https://github.com/cellethology/deepdraw/blob/main/examples/deepdraw_dummy/README.md) | Runnable example with synthetic data. |
| [benchmarks/](https://github.com/cellethology/deepdraw/blob/main/benchmarks/README.md) | Retrospective evaluation, noise simulation, and Hydra/Slurm tools. |
| [envs/alphagenome/](https://github.com/cellethology/deepdraw/tree/main/envs/alphagenome) | Optional uv environment for development. |
| [test/](https://github.com/cellethology/deepdraw/tree/main/test) | Unit and workflow tests. |

From a source checkout, install development dependencies and run the tests:

```bash
python -m pip install -e ".[analysis]" --group dev
pytest
```

Command-specific help:

```bash
deepdraw init --help
deepdraw suggest --help
deepdraw pca --help
deepdraw embed-alphagenome --help
```
