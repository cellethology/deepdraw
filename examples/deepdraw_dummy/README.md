# Deepdraw Dummy Workflow

This directory contains a tiny design pool and matching embeddings for checking
the user-facing Deepdraw loop without any real experimental data.

Run the first batch selection:

```bash
deepdraw init \
  --pool-csv examples/deepdraw_dummy/design_pool.csv \
  --embeddings examples/deepdraw_dummy/embeddings.npz \
  --sequence-column sequence \
  --id-column variant_id \
  --output-dir /tmp/deepdraw_dummy_run \
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

Then simulate the first measurement update:

```bash
deepdraw suggest \
  --run-dir /tmp/deepdraw_dummy_run \
  --measurements examples/deepdraw_dummy/measurements_round0.csv \
  --label-column Expression
```

The bundled `measurements_round0.csv` matches the deterministic initial random
selection from the command above (`--seed 11`, `--starting-batch-size 4`).
Regenerate the files with:

```bash
python examples/deepdraw_dummy/make_dummy_inputs.py
```
