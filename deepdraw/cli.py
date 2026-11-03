"""Command line interface for the production Deepdraw workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from deepdraw.embeddings.alphagenome import embed_design_pool, embed_fasta
from deepdraw.embeddings.pca import reduce_embeddings_pca
from deepdraw.examples import write_dummy_example
from deepdraw.workflow import initialize_run, suggest_next_batch

_LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR")


def _add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default=argparse.SUPPRESS,
        choices=_LOG_LEVEL_CHOICES,
        help="Progress output verbosity.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepdraw",
        description="Run Deepdraw active learning on an experimental design pool.",
    )
    _add_log_level_argument(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    example_parser = subparsers.add_parser(
        "example", help="Copy the bundled dummy example into a new directory."
    )
    _add_log_level_argument(example_parser)
    example_parser.add_argument(
        "--output-dir", default=Path("deepdraw_dummy"), type=Path
    )

    init_parser = subparsers.add_parser(
        "init",
        help="Create a run and choose the first unlabeled batch to measure.",
        description=(
            "Uses the recommended Deepdraw settings automatically. "
            "Algorithm flags are optional overrides."
        ),
    )
    _add_log_level_argument(init_parser)
    init_parser.add_argument("--pool-csv", required=True, type=Path)
    init_parser.add_argument("--embeddings", required=True, type=Path)
    init_parser.add_argument(
        "--output-dir",
        default=Path("deepdraw_run"),
        type=Path,
        help="Run directory for state and recommendations. Defaults to deepdraw_run.",
    )
    init_parser.add_argument("--sequence-column")
    init_parser.add_argument("--id-column")
    init_parser.add_argument("--starting-batch-size", type=int, default=12)
    init_parser.add_argument("--batch-size", type=int, default=12)
    init_parser.add_argument("--seed", type=int, default=0)
    init_parser.add_argument(
        "--stopping-rule", choices=("none", "ei", "threshold"), default="none"
    )
    init_parser.add_argument(
        "--epsilon",
        type=float,
        help="EI cutoff in log(1 + phenotype) units with the default target transform.",
    )
    init_parser.add_argument(
        "--phenotype-threshold",
        type=float,
        help="Stop when a measured phenotype meets or exceeds this value.",
    )
    init_parser.add_argument(
        "--initial-selection-strategy",
        default="probcover",
        help=(
            "Initial strategy (default: probcover). Optional choices: "
            "core_set, density_core_set, kmeans, probcover, "
            "probcover_cosine, typiclust, or random."
        ),
    )
    init_parser.add_argument(
        "--predictor",
        default="gp",
        help=(
            "Predictor (default: gp). Optional choices: gp, linear_gp, "
            "gradboost_regressor, or rf_ensemble. "
            "Existing botorch_* names are accepted."
        ),
    )
    init_parser.add_argument(
        "--query-strategy",
        default="mes",
        help=(
            "Query strategy (default: mes). Optional choices: "
            "topk, predstdhybrid, ucb, pes, mes, gibbon, "
            "log_pi, log_ei, qlog_nei, qlog_ei, or random. "
            "Existing botorch_* names are accepted."
        ),
    )
    init_parser.add_argument(
        "--feature-transforms",
        default="standardize",
        choices=["standardize", "none"],
        help="Feature preprocessing (default: standardize).",
    )
    init_parser.add_argument(
        "--target-transforms",
        default="log_standardize",
        choices=["log_standardize", "none"],
        help="Target preprocessing (default: log_standardize).",
    )
    init_parser.add_argument("--force", action="store_true")

    suggest_parser = subparsers.add_parser(
        "suggest",
        help="Train on measured labels and choose the next batch.",
    )
    _add_log_level_argument(suggest_parser)
    suggest_parser.add_argument("--run-dir", required=True, type=Path)
    suggest_parser.add_argument("--measurements", required=True, type=Path)
    suggest_parser.add_argument("--label-column")
    suggest_parser.add_argument("--measurement-id-column")

    embed_parser = subparsers.add_parser(
        "embed-alphagenome",
        help="Generate AlphaGenome embeddings for a design pool or FASTA file.",
    )
    _add_log_level_argument(embed_parser)
    input_group = embed_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pool-csv", type=Path)
    input_group.add_argument("--fasta", type=Path)
    embed_parser.add_argument("--output", required=True, type=Path)
    embed_parser.add_argument("--sequence-column")
    embed_parser.add_argument("--id-column")
    embed_parser.add_argument("--model-version", default="all_folds")
    embed_parser.add_argument("--batch-size", type=int, default=1)
    embed_parser.add_argument("--pooling", choices=["mean", "none"], default="mean")
    embed_parser.add_argument(
        "--resolution",
        type=int,
        choices=[1, 128],
        default=1,
        help="Embedding resolution in base pairs (default: 1).",
    )
    embed_parser.add_argument(
        "--species",
        choices=["human", "mouse"],
        default="human",
    )
    embed_parser.add_argument(
        "--no-pad-to-multiple",
        action="store_true",
        help="Disable AlphaGenome input padding to multiples of 2048 bp.",
    )
    embed_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable DNA sequence validation before embedding.",
    )
    embed_parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "tpu"],
        help="Force an AlphaGenome runtime device. Defaults to GPU/TPU if available.",
    )

    pca_parser = subparsers.add_parser(
        "pca",
        help="Reduce an embedding NPZ with PCA and kneedle component selection.",
    )
    _add_log_level_argument(pca_parser)
    pca_parser.add_argument("--input", dest="input_file", required=True, type=Path)
    pca_parser.add_argument("--output", required=True, type=Path)
    pca_parser.add_argument(
        "--n-components",
        type=int,
        help=(
            "Number of PCA components to fit. By default this also keeps exactly "
            "that many PCs unless --select-components is passed."
        ),
    )
    pca_parser.set_defaults(exact_n_components=None)
    pca_parser.add_argument(
        "--exact-n-components",
        dest="exact_n_components",
        action="store_true",
        help="Keep exactly --n-components.",
    )
    pca_parser.add_argument(
        "--select-components",
        dest="exact_n_components",
        action="store_false",
        help="Use --selection-method after fitting --n-components as an upper cap.",
    )
    pca_parser.add_argument("--target-variance", type=float, default=0.95)
    pca_parser.add_argument(
        "--selection-method",
        choices=["target-variance", "elbow", "kneedle", "l-method"],
        default="kneedle",
    )
    pca_parser.add_argument(
        "--power-of-two",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Round target-variance selection to a power of two.",
    )
    pca_parser.add_argument(
        "--use-mean-pooling",
        action="store_true",
        help="Mean-pool 3D or ragged embeddings over sequence positions before PCA.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    log_level = getattr(args, "log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(message)s",
    )

    try:
        _run_command(args, parser)
    except (OSError, RuntimeError, ValueError) as exc:
        if log_level == "DEBUG":
            raise
        parser.exit(1, f"Error: {exc}\n")


def _run_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command == "example":
        output_dir = write_dummy_example(args.output_dir)
        print(f"Wrote Deepdraw example: {output_dir}")
        return

    if args.command == "init":
        state = initialize_run(
            pool_csv=args.pool_csv,
            embeddings_path=args.embeddings,
            output_dir=args.output_dir,
            sequence_column=args.sequence_column,
            id_column=args.id_column,
            starting_batch_size=args.starting_batch_size,
            batch_size=args.batch_size,
            seed=args.seed,
            predictor_name=args.predictor,
            query_strategy_name=args.query_strategy,
            initial_selection_strategy_name=args.initial_selection_strategy,
            feature_transforms_name=args.feature_transforms,
            target_transforms_name=args.target_transforms,
            stopping_rule=args.stopping_rule,
            epsilon=args.epsilon,
            phenotype_threshold=args.phenotype_threshold,
            force=args.force,
        )
        print(f"Initialized Deepdraw run: {state.output_dir}")
        print(f"Measure: {state.output_path / 'round_000_to_measure.csv'}")
        return

    if args.command == "suggest":
        state = suggest_next_batch(
            run_dir=args.run_dir,
            measurements_csv=args.measurements,
            label_column=args.label_column,
            measurement_id_column=args.measurement_id_column,
        )
        if state.stopped:
            print(f"Stopped Deepdraw run: {state.stopping_result['reason']}")
            return
        latest_round = state.rounds[-1]["round"]
        round_path = state.output_path / f"round_{latest_round:03d}_to_measure.csv"
        print(f"Wrote Deepdraw round {latest_round}: {round_path}")
        return

    if args.command == "embed-alphagenome":
        if args.pool_csv is not None:
            embed_design_pool(
                pool_csv=args.pool_csv,
                output_path=args.output,
                sequence_column=args.sequence_column,
                id_column=args.id_column,
                model_version=args.model_version,
                batch_size=args.batch_size,
                pooling=args.pooling,
                resolution=args.resolution,
                species=args.species,
                pad_to_multiple=not args.no_pad_to_multiple,
                validate=not args.no_validate,
                device=args.device,
            )
        else:
            embed_fasta(
                fasta_path=args.fasta,
                output_path=args.output,
                model_version=args.model_version,
                batch_size=args.batch_size,
                pooling=args.pooling,
                resolution=args.resolution,
                species=args.species,
                pad_to_multiple=not args.no_pad_to_multiple,
                validate=not args.no_validate,
                device=args.device,
            )
        print(f"Wrote AlphaGenome embeddings: {args.output}")
        return

    if args.command == "pca":
        summary = reduce_embeddings_pca(
            input_file=args.input_file,
            output_file=args.output,
            n_components=args.n_components,
            target_variance=args.target_variance,
            use_mean_pooling=args.use_mean_pooling,
            exact_n_components=args.exact_n_components,
            power_of_two=args.power_of_two,
            selection_method=args.selection_method.replace("-", "_"),
        )
        print(
            f"Wrote PCA embeddings: {args.output} "
            f"({summary.n_components} PCs, "
            f"{summary.cumulative_explained_variance:.2%} variance)"
        )
        return

    parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
