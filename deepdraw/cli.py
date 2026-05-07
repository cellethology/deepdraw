"""Command line interface for the production Deepdraw workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

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

    init_parser = subparsers.add_parser(
        "init",
        help="Create a run and choose the first unlabeled batch to measure.",
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
        "--initial-selection-strategy",
        default="probcover_euclidean",
        help="Name under job_sub/conf/initial_selection_strategy without .yaml.",
    )
    init_parser.add_argument(
        "--predictor",
        default="botorch_gp",
        help="Name under job_sub/conf/predictor without .yaml.",
    )
    init_parser.add_argument(
        "--query-strategy",
        default="botorch_mes",
        help="Name under job_sub/conf/query_strategy without .yaml.",
    )
    init_parser.add_argument(
        "--feature-transforms",
        default="standardize",
        help="Name under job_sub/conf/transforms without .yaml.",
    )
    init_parser.add_argument(
        "--target-transforms",
        default="log_standardize",
        help="Name under job_sub/conf/transforms without .yaml.",
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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, getattr(args, "log_level", "INFO")),
        format="%(message)s",
    )

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
        latest_round = state.rounds[-1]["round"]
        round_path = state.output_path / f"round_{latest_round:03d}_to_measure.csv"
        print(f"Wrote Deepdraw round {latest_round}: {round_path}")
        return

    parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
