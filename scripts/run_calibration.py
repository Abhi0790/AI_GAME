"""Evaluate P(keeps) on games controlled by a forecast-independent policy."""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.config import GameConfig, add_board_args, board_kwargs
from src.evaluation.fixed_policy import play_fixed, summarise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/fixed_policy_calibration.json")
    add_board_args(parser)
    args = parser.parse_args()
    if args.games < 2:
        parser.error("at least two games are needed for a held-out base rate")

    seeds = list(range(args.seed, args.seed + args.games))
    games = [play_fixed(GameConfig(seed=s, **board_kwargs(args))).calibration
             for s in seeds]
    result = summarise(games, seeds)
    result["config"] = GameConfig(seed=args.seed, **board_kwargs(args)).to_dict()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(result, stream, indent=2)
    print(f"{result['games']} games, {result['predictions']} graded predictions")
    print(f"Brier: model {result['model_brier']}, held-out base rate "
          f"{result['baseline_brier']}")
    print(f"Paired game-level difference (model - baseline), 95% CI: "
          f"{result['paired_difference_ci95']}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
