# Deepdraw dummy example

This example provides 60 designs, synthetic embeddings, and measurements for the default first batch. Follow the [installation steps](https://github.com/cellethology/deepdraw/blob/main/README.md#install), then run these commands in a working directory. No AlphaGenome download is needed.

## Select and measure two batches

```bash
# Copy the example data.
deepdraw example

# Select the first batch.
deepdraw init \
  --pool-csv deepdraw_dummy/design_pool.csv \
  --embeddings deepdraw_dummy/embeddings.npz \
  --sequence-column sequence \
  --id-column variant_id \
  --stopping-rule threshold \
  --phenotype-threshold 19

# Add measurements and select the next batch.
deepdraw suggest \
  --run-dir deepdraw_run \
  --measurements deepdraw_dummy/measurements.csv \
  --label-column Expression
```

`init` writes `deepdraw_run/round_000_to_measure.csv`. The supplied measurements match that batch; `suggest` trains the default GP and writes `round_001_to_measure.csv` using MES.

To simulate measuring the second batch, create a cumulative measurement file with the same synthetic score formula used for the bundled data:

```bash
# Simulate the next measurements.
python - <<'PYTHON'
import pandas as pd

previous = pd.read_csv("deepdraw_dummy/measurements.csv")
batch = pd.read_csv("deepdraw_run/round_001_to_measure.csv")
batch["Expression"] = [
    round(1 + 0.31 * int(identifier.rsplit("_", 1)[1]), 2)
    for identifier in batch["variant_id"]
]
pd.concat([previous, batch[previous.columns]], ignore_index=True).to_csv(
    "deepdraw_run/measurements.csv", index=False
)
PYTHON

# Update the run and check stopping.
deepdraw suggest \
  --run-dir deepdraw_run \
  --measurements deepdraw_run/measurements.csv \
  --label-column Expression
```

The run stops when a measured score reaches 19. For real experiments, keep one cumulative measurements file and append newly measured designs before each `suggest` call.

The bundled measurements correspond to the default initial strategy, batch size, and seed. If you change those settings, supply measurements matching the newly selected batch.

## Regenerate the inputs

```bash
# Regenerate the example inputs.
python deepdraw_dummy/make_dummy_inputs.py
```
