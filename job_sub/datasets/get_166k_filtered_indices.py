"""
Utility script to generate subset id files for the 166k dataset.

Reads the metadata CSV, filters rows where "No. of Barcodes" >= threshold,
and writes the matching row indices (sample IDs) to a newline-delimited file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_METADATA_PATH = Path(
    "/home/wangzitongLab/wangzitong/gene_circuit_design/data_new/"
    "Rai_2024_166k/166k_Library_CLASSIC_Data.csv"
)
DEFAULT_OUTPUT_PATH = Path("166k_filtered_indices.txt")
DEFAULT_COLUMN = "No. of Barcodes"
DEFAULT_MIN_VALUE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter the 166k metadata file and emit sample IDs (row indices) "
            "where the barcode count exceeds a threshold."
        )
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=f"Path to the metadata CSV. Defaults to {DEFAULT_METADATA_PATH}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to write the subset ids. Defaults to {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--column-name",
        default=DEFAULT_COLUMN,
        help=f"Column to threshold. Defaults to '{DEFAULT_COLUMN}'.",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=DEFAULT_MIN_VALUE,
        help=(
            f"Minimum value required to keep a row. Defaults to {DEFAULT_MIN_VALUE} "
            "(>= threshold)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metadata_csv)
    if args.column_name not in df.columns:
        raise ValueError(
            f"Column '{args.column_name}' not found in {args.metadata_csv} "
            f"(available: {list(df.columns)})"
        )

    mask = df[args.column_name] >= args.min_value
    selected_indices = df.index[mask].to_list()

    if not selected_indices:
        raise RuntimeError(
            "No rows satisfied the threshold. "
            "Please check the column name and minimum value."
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(str(idx) for idx in selected_indices))

    print(
        f"Wrote {len(selected_indices)} sample IDs (>= {args.min_value} "
        f"{args.column_name}) to {args.output_path}"
    )


if __name__ == "__main__":
    main()
