"""Paired test of reserving a feasible promise-keeping candidate."""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.config import add_game_args, game_kwargs, DEFAULT_GAMES
from src.evaluation.keep_coverage import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--seed", type=int, default=40000)
    parser.add_argument("--out", default="data/keep_coverage.json")
    add_game_args(parser, budget=400)
    args = parser.parse_args()
    if args.games < 2:
        parser.error("at least two paired games are required")
    report = run(list(range(args.seed, args.seed + args.games)),
                 game_kwargs(args), args.budget)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(report, stream, indent=2)
    print("Broken-obligation categories:", report["break_categories"])
    for name, row in report["metrics"].items():
        print(f"{name}: baseline {row['baseline_mean']}, reserved "
              f"{row['reserved_mean']}, paired difference 95% CI "
              f"{row['reserved_minus_baseline_ci95']}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
