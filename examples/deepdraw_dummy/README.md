# Deepdraw Dummy Workflow

This directory contains a tiny 24-design pool and matching embeddings for checking
the user-facing Deepdraw loop without any real experimental data.

Run the first batch selection:

```bash
uv run deepdraw init \
  --pool-csv examples/deepdraw_dummy/design_pool.csv \
  --embeddings examples/deepdraw_dummy/embeddings.npz \
  --sequence-column sequence \
  --id-column variant_id
```

Then simulate the first measurement update:

```bash
uv run deepdraw suggest \
  --run-dir deepdraw_run \
  --measurements examples/deepdraw_dummy/measurements_round0.csv \
  --label-column Expression
```

The command above uses the same defaults as a real run, and `measurements_round0.csv` matches its first-round recommendations. For a faster smoke test, use a separate output directory and add:

```bash
--output-dir /tmp/deepdraw_dummy_fast_run \
--starting-batch-size 4 \
--batch-size 3 \
--seed 11 \
--initial-selection-strategy random \
--predictor ridge_regressor \
--query-strategy topk \
--feature-transforms none \
--target-transforms none
```

The bundled `measurements_round0.csv` is for the default command above.
Regenerate the files with:

```bash
uv run python examples/deepdraw_dummy/make_dummy_inputs.py
```
