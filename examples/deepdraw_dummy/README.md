# Deepdraw dummy example

This example provides 60 designs, synthetic embeddings, and measurements for the default first batch. Follow the [installation steps](../../README.md#install), then run these commands from the repository root. No AlphaGenome download is needed.

## Select and measure two batches

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

The first command writes `deepdraw_run/round_000_to_measure.csv`. The supplied measurements match that batch; the second command trains the default GP and writes `round_001_to_measure.csv` using MES.

To simulate measuring the second batch, create a cumulative measurement file with the same synthetic score formula used for the bundled data:

```bash
python - <<'PYTHON'
import pandas as pd

previous = pd.read_csv("examples/deepdraw_dummy/measurements.csv")
batch = pd.read_csv("deepdraw_run/round_001_to_measure.csv")
batch["Expression"] = [
    round(1 + 0.31 * int(identifier.rsplit("_", 1)[1]), 2)
    for identifier in batch["variant_id"]
]
pd.concat([previous, batch[previous.columns]], ignore_index=True).to_csv(
    "deepdraw_run/measurements.csv", index=False
)
PYTHON

deepdraw suggest \
  --run-dir deepdraw_run \
  --measurements deepdraw_run/measurements.csv \
  --label-column Expression
```

The run stops when a measured score reaches 19. For real experiments, keep one cumulative measurements file and append newly measured designs before each `suggest` call.

The bundled measurements correspond to the default initial strategy, batch size, and seed. If you change those settings, supply measurements matching the newly selected batch.

## Regenerate the inputs

```bash
python examples/deepdraw_dummy/make_dummy_inputs.py
```
