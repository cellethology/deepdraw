# Core modules

These modules provide Deepdraw's active-learning components. For installation and experimental use, start with the [main README](../README.md).

- [experiment.py](experiment.py): retrospective active-learning loop.
- [data_loader.py](data_loader.py): aligned measurement and embedding data for retrospective runs.
- [initial_selection_strategies.py](initial_selection_strategies.py) and [query_strategies.py](query_strategies.py): initial and subsequent batch selection.
- [predictor_trainer.py](predictor_trainer.py): model training and feature/target transforms.
- [botorch_models.py](botorch_models.py) and [stable_rbf_kernel.py](stable_rbf_kernel.py): BoTorch predictors and the RBF kernel.
- [uncertainty.py](uncertainty.py): predictive uncertainty helpers.
- [stopping.py](stopping.py): measured-threshold and expected-improvement stopping used by both the experimental and benchmark workflows.
- [metrics_calculator.py](metrics_calculator.py) and [round_tracker.py](round_tracker.py): retrospective round metrics and run summaries.

Optional noise simulation and confirmed-top benchmark stopping are in [benchmarks/noise_simulation.py](../benchmarks/noise_simulation.py).
