"""Build a supervised-learning dataset from WFO ranking outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from etf_rotation_experiments.ml_ranking import data_prep, feature_engineering

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    PROJECT_ROOT.parent
    / "results/run_20251113_194451/ranking_oos_sharpe_compound_top1000.parquet",
]
DEFAULT_OUTPUT = PROJECT_ROOT / "ml_ranking/data/training_dataset.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Ranking parquet files to include in the dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output dataset path",
    )
    parser.add_argument(
        "--min-sample-count",
        type=int,
        default=60,
        help="Minimum required oos_compound_sample_count for a combo",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="oos_compound_sharpe",
        help="Label column to append to the dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_config = data_prep.LoadConfig(
        paths=[path.resolve() for path in args.inputs],
        min_oos_sample_count=args.min_sample_count,
    )
    training_frame = data_prep.prepare_training_frame(load_config)
    features = feature_engineering.build_feature_matrix(training_frame)
    dataset = feature_engineering.attach_standard_label(
        features, training_frame, label_column=args.label_column
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    summary = {
        "rows": len(dataset),
        "columns": list(dataset.columns),
        "output": output_path.as_posix(),
    }
    print(pd.Series(summary))


if __name__ == "__main__":
    main()
