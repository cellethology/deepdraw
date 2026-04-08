"""
Aggregate results.csv files from Hydra sweeps into by-round CSVs.

Run this script after SubmitIt jobs finish to generate
combined_tracks.csv for every dataset directory under a
specific sweep timestamp directory.

Example:
    python job_sub/aggregate_tracks.py --sweep-dir job_sub/multirun/2026-01-11/19-14-02
"""

import argparse

from pathlib import Path

import pandas as pd
from tqdm import tqdm





def aggregate_tracks(
    sweep_dir: Path,
    
) -> None:
    search_dir = sweep_dir
    result_name = "results.csv"

    result_files = sorted(search_dir.rglob(result_name))#rglob can recursively search for all matched files under search_dir and its subdirectories, and return a generator of Path objects.
    summary_folders=set(path.parents[2] for path in result_files)
    
    for path_dir in tqdm(summary_folders, desc="Reading results"):
        dfs =[]
        files_paths = sorted(path_dir.rglob(result_name))
        for file_path in files_paths:
            selection= pd.read_csv(file_path)
            selection['seed_name'] = file_path.parents[0].name
            selection['setting_name'] = file_path.parents[1].name
            selection['dataset_name'] = file_path.parents[2].name
            dfs.append(selection)
        if not dfs:
            raise ValueError(f"No {result_name} files found in {path_dir} or its subdirectories.")
        dfs = pd.concat(dfs, ignore_index=True)
        dfs.to_csv(path_dir / "combined_tracks.csv", index=False)

        


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help=(
            "Path to a sweep date directory (e.g., job_sub/multirun/2025-12-30) "
            "or a specific sweep timestamp directory "
            "(e.g., job_sub/multirun/2025-12-30/15-30-03)."
        ),
    )


    return parser.parse_args()





def main() -> None:
    args = parse_args()

    aggregate_tracks(
        sweep_dir=args.sweep_dir,
    )



if __name__ == "__main__":
    main()
