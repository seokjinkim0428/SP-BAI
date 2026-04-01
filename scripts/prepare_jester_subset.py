#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sp_bai.experiments.real_data import DEFAULT_ARM_IDS, prepare_jester_subset


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the complete-rating Jester subset used by the SP-BAI real-data experiments."
    )
    parser.add_argument(
        "--ratings",
        default="jester_ratings.csv",
        help="Path to the raw Jester ratings CSV.",
    )
    parser.add_argument(
        "--output",
        default="jester_subset_50699_8.csv",
        help="Path to the output subset CSV.",
    )
    parser.add_argument(
        "--arm-ids",
        nargs="*",
        type=int,
        default=list(DEFAULT_ARM_IDS),
        help="Joke IDs to keep. The script retains users who rated all selected jokes.",
    )
    args = parser.parse_args()

    output_path = prepare_jester_subset(
        ratings_file=args.ratings,
        output_file=args.output,
        arm_ids=tuple(args.arm_ids),
        verbose=True,
    )
    print(f"Subset ready: {output_path}")


if __name__ == "__main__":
    main()
