# Comparison methods

Run Merzbacher, Kobiela, Friedman, or Yin from the repository root using the downloaded Zenodo data.

```bash
conda env create -f benchmarks/comparison_methods/environment.yml
conda activate deepdraw-benchmarks

python -m benchmarks.comparison_methods.run friedman \
  --data-dir /path/to/zenodo_for_deepdraw/data \
  --task IS --space AD_part1_indices
```

Replace `friedman` with `merzbacher`, `kobiela`, or `yin`; the remaining arguments are the same. Each method loads its inputs automatically from `--data-dir`.

All four support `--task IS`, `OR`, or `AND` with a corresponding `--space`. Selection and summary CSVs are saved in `benchmark_output/`; use `--output` to choose another directory.

Yin uses the [bootstrap CNN predictor](https://github.com/christopheryin/iterative_synthetic_enhancer_design/blob/64ec5329f573a437bf229283b5bc2b453472b5b0/R1-MPRA_design/model_training/train_boot_predictors.ipynb), adapted to select measured circuits from a fixed pool; de novo sequence generation is omitted. Friedman and Yin load the deposited candidate FASTA files automatically; use `--sequences-fasta` to supply another file with matching numeric circuit IDs.
