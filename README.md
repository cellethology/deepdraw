# Deepdraw: genetic circuit design with genomic foundation model

[![CI](https://github.com/cellethology/deepdraw/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/cellethology/deepdraw/actions/workflows/pre-commit.yml)
[![Coverage](https://img.shields.io/codecov/c/github/cellethology/deepdraw?logo=codecov)](https://codecov.io/gh/cellethology/deepdraw)

Deepdraw is an active learning algorithm for genetic circuit design. It uses genomic foundation model (GFM) embeddings to make accurate predictions from very few experimental observations. At each iteration, Deepdraw integrates measurements from previous rounds with sequence-level circuit embeddings and proposes informative candidate designs in practical batches of 12, a 100-fold reduction relative to prior active learning approaches for circuit design.

This README is for users who want to apply Deepdraw to their own design pool. Retrospective benchmarking, Hydra sweeps, and Slurm array jobs are documented separately in [job_sub/README.md](job_sub/README.md).

## Quick Start

### 1. Install

Prerequisites: Git, Python, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/cellethology/deepdraw.git
cd deepdraw

uv sync --python 3.10
uv run deepdraw --help
```

Plain `uv sync` installs the user-facing Deepdraw runtime. If you are developing the package and want test, lint, notebook, and plotting tools, include the dev group:

```bash
uv sync --python 3.10 --group dev
```

Optional retrospective analysis and cluster-job tooling can be added only when needed:

```bash
uv sync --extra analysis
uv sync --extra cluster
```

If you are testing the current development branch before it is merged:

```bash
git checkout codex/deepdraw-user-workflow
```

### 2. Run The Dummy Example

The fastest way to verify the workflow is to run the bundled dummy example. It includes a 60-sequence design pool, a matching embeddings file, and fake first-round measurements.

```bash
uv run deepdraw init \
  --pool-csv examples/deepdraw_dummy/design_pool.csv \
  --embeddings examples/deepdraw_dummy/embeddings.npz \
  --sequence-column sequence \
  --id-column variant_id
```

Deepdraw writes the first batch to:

```text
deepdraw_run/round_000_to_measure.csv
```

Now simulate receiving measurements from the first experimental round. In a real project, this should be a cumulative file, for example `measurements.csv`, that you keep appending to after each round. The dummy example provides its starter version as `measurements.csv`.

```bash
uv run deepdraw suggest \
  --run-dir deepdraw_run \
  --measurements examples/deepdraw_dummy/measurements.csv \
  --label-column Expression
```

Deepdraw trains on all rows in the measurement file and writes the next batch to:

```text
deepdraw_run/round_001_to_measure.csv
```

The dummy example uses the same defaults as a real run: first batch size 12, later batch size 12, seed 0, ProbCover-euclidean initial selection, BoTorch GP prediction, and MES acquisition. See [Useful Flags](#useful-flags) for ways to change output location, batch size, model, acquisition strategy, and preprocessing.

## Use Deepdraw On Your Own Project

Deepdraw expects you to start with an unlabeled design pool. You do not need any experimental measurements for the first round.

### 1. Prepare A Design Pool

Create a CSV with one row per candidate design. Include a sequence column and, preferably, a stable design ID column.

```csv
variant_id,sequence
variant_001,ATGCGTACGTTAGCGA
variant_002,ATGCGTACGATAGCAA
variant_003,ATGCGTACGCTAGCTA
```

The stable ID column is recommended because it makes measurement files easier to merge across rounds.

### 2. Generate GFM Embeddings

Generate one embedding vector per design using your chosen genomic foundation model. Deepdraw currently expects embeddings to be provided as an NPZ file; embedding generation itself is outside the `deepdraw` CLI.

Required NPZ structure:

```python
{
    "embeddings": np.ndarray,  # shape: (num_designs, embedding_dim)
    "ids": np.ndarray,         # variant IDs or row indices aligned to the pool CSV
}
```

`sample_ids` is also accepted instead of `ids`. If you pass `--id-column variant_id`, the NPZ IDs should match that CSV column. If you do not pass an ID column, use row indices `0, 1, 2, ...`.

### 3. Select The First Batch

Run `deepdraw init` to choose the first experimental batch from embeddings only.

```bash
uv run deepdraw init \
  --pool-csv designs.csv \
  --embeddings embeddings.npz \
  --sequence-column sequence \
  --id-column variant_id \
  --output-dir runs/my_deepdraw_run \
  --starting-batch-size 12 \
  --batch-size 12
```

Outputs:

```text
runs/my_deepdraw_run/
├── deepdraw_state.json
├── latest_recommendations.csv
├── round_000_to_measure.csv
└── selection_history.csv
```

Send `round_000_to_measure.csv` to the wet lab.
Recommendation CSVs keep the original design-pool columns; `selection_history.csv`
adds only `deepdraw_round` so you can see which batch each design came from.

### 4. Add Measurements

After the first experiment, create one cumulative measurements CSV, such as `measurements.csv`. The easiest approach is to copy `round_000_to_measure.csv` and add a measured label column.

```csv
variant_id,sequence,Expression
variant_001,ATGCGTACGTTAGCGA,1.42
variant_005,ATGCGTACGAAAGCGA,3.87
variant_009,ATGCGTACGCAAGTTA,5.11
```

Keep the stable ID column, such as `variant_id`, in every measurement update. Deepdraw uses that column to map measurements back to the original design pool.

You can include extra measured designs that were not recommended by Deepdraw, as long as they are present in the original design pool. `deepdraw suggest` trains on every measured design in `measurements.csv`, then excludes all measured designs from the next recommendation batch.

Keep updating this same file over time. After you measure `round_001_to_measure.csv`, append those new rows to `measurements.csv`; after `round_002_to_measure.csv`, append those rows too. Each `deepdraw suggest` call expects the measurements file to contain every previously recommended design with a measured label.

### 5. Select The Next Batch

Run `deepdraw suggest` with the cumulative measurement table:

```bash
uv run deepdraw suggest \
  --run-dir runs/my_deepdraw_run \
  --measurements measurements.csv \
  --label-column Expression
```

This writes:

```text
runs/my_deepdraw_run/round_001_to_measure.csv
```

Measure that batch, append the new labels to the same `measurements.csv`, and run `deepdraw suggest` again:

```bash
uv run deepdraw suggest \
  --run-dir runs/my_deepdraw_run \
  --measurements measurements.csv \
  --label-column Expression
```

The next output will be:

```text
runs/my_deepdraw_run/round_002_to_measure.csv
```

The loop is:

```text
design pool + embeddings
        |
        v
deepdraw init
        |
        v
measure round_000
        |
        v
deepdraw suggest
        |
        v
measure round_001
        |
        v
append round_001 labels to measurements.csv
        |
        v
deepdraw suggest
        |
        v
repeat
```

## Recommended Defaults

For a first real campaign, use the defaults unless you have a reason to compare strategies:

```bash
uv run deepdraw init \
  --pool-csv designs.csv \
  --embeddings embeddings.npz \
  --sequence-column sequence \
  --id-column variant_id \
  --output-dir runs/my_deepdraw_run \
  --starting-batch-size 12 \
  --batch-size 12
```

The default workflow uses:

- initial selection: `probcover_euclidean`
- predictor: `gp`
- query strategy: `mes`
- feature transforms: `standardize`
- target transforms: `log_standardize`

For very small smoke tests, you can use faster settings: `random`, `ridge_regressor`, `topk`, and no transforms.

## Useful Flags

The examples above use production defaults. You can override pieces of the workflow when you need a different experimental setup or a faster local smoke test.

Change the number of designs per round:

```bash
uv run deepdraw init \
  --pool-csv designs.csv \
  --embeddings embeddings.npz \
  --sequence-column sequence \
  --id-column variant_id \
  --output-dir runs/my_deepdraw_run \
  --starting-batch-size 24 \
  --batch-size 12
```

Choose where outputs are written:

```bash
uv run deepdraw init \
  --pool-csv designs.csv \
  --embeddings embeddings.npz \
  --sequence-column sequence \
  --id-column variant_id \
  --output-dir runs/my_deepdraw_run
```

If the run directory already exists, `deepdraw init` stops rather than overwriting `deepdraw_state.json`. Add `--force` only when you intentionally want to overwrite an existing run directory.

Use a faster local smoke-test configuration by swapping in lighter model and transform settings:

```bash
uv run deepdraw init \
  --pool-csv examples/deepdraw_dummy/design_pool.csv \
  --embeddings examples/deepdraw_dummy/embeddings.npz \
  --sequence-column sequence \
  --id-column variant_id \
  --output-dir /tmp/deepdraw_dummy_fast_run \
  --starting-batch-size 4 \
  --batch-size 3 \
  --seed 11 \
  --initial-selection-strategy random \
  --predictor ridge_regressor \
  --query-strategy topk \
  --feature-transforms none \
  --target-transforms none \
  --force
```

Common override flags:

- `--starting-batch-size`: number of designs in the first batch.
- `--batch-size`: number of designs in each later batch.
- `--seed`: random seed for reproducible initial selection and stochastic model components.
- `--initial-selection-strategy`: first-round strategy, such as `probcover_euclidean`, `core_set`, or `random`.
- `--predictor`: model used after measurements arrive, such as `gp` or `ridge_regressor`. Existing `botorch_*` names are also accepted.
- `--query-strategy`: acquisition strategy for later rounds, such as `mes`, `qlog_nei`, or `topk`. Existing `botorch_*` names are also accepted.
- `--feature-transforms`: feature preprocessing config, such as `standardize` or `none`.
- `--target-transforms`: label preprocessing config, such as `log_standardize` or `none`.
- `--log-level`: progress output verbosity; default is `INFO`, use `WARNING` for quieter runs.

## CLI Reference

Create a run and select the first batch:

```bash
uv run deepdraw init --help
```

Train on measured labels and select the next batch:

```bash
uv run deepdraw suggest --help
```

Required/common arguments:

- `--pool-csv`: CSV containing candidate designs.
- `--embeddings`: NPZ containing GFM embeddings aligned to the design pool.
- `--sequence-column`: sequence column in the design pool CSV.
- `--id-column`: optional stable design ID column in the pool CSV.
- `--output-dir`: run directory where Deepdraw writes state and recommendations. Defaults to `deepdraw_run`.
- `--measurements`: cumulative CSV containing measured labels for all previous Deepdraw recommendations.
- `--label-column`: measured target column, such as `Expression` or `Fold Change`.

## Repository Layout

```text
├── deepdraw/                  # User-facing Deepdraw CLI and workflow
├── examples/deepdraw_dummy/   # Tiny runnable example
├── core/                      # Active learning models, strategies, and trainers
├── job_sub/                   # Retrospective benchmark and Slurm/Hydra tooling
├── test/                      # Unit and workflow tests
└── utils/                     # Supporting utilities
```

## Testing

```bash
uv run pytest test/test_deepdraw_workflow.py
uv run pytest
```

## Citation

Citation information will be added with the manuscript/release.
