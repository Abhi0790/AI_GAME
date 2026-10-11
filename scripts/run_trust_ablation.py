"""Paired games: learned trust forecast versus a held-out base rate."""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.config import add_game_args, game_kwargs, DEFAULT_GAMES
from src.evaluation.trust_ablation import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help="paired test games; a multiple of 12 balances seats and personas")
    parser.add_argument("--train-games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=10000)
    parser.add_argument("--out", default="data/trust_ablation.json")
    add_game_args(parser, budget=400)
    args = parser.parse_args()
    if args.games < 2 or args.train_games < 2:
        parser.error("at least two training and two paired test games are required")

    report = run(
        list(range(args.train_seed, args.train_seed + args.train_games)),
        list(range(args.seed, args.seed + args.games)),
        game_kwargs(args), node_budget=args.budget)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    opportunities_path = str(Path(args.out).with_suffix(".opportunities.jsonl.gz"))
    with gzip.open(opportunities_path, "wt") as stream:
        for pair in report["games"]:
            for arm in ("learned", "constant"):
                game = pair[arm]
                for row in game.pop("opportunities"):
                    stream.write(json.dumps({
                        "seed": game["seed"], "seat": game["seat"],
                        "persona": game["persona"], "arm": arm, **row,
                    }) + "\n")
    report["opportunities_file"] = opportunities_path
    with open(args.out, "w") as stream:
        json.dump(report, stream, indent=2)
    print(f"Held-out constant P(keeps): {report['constant_keep_forecast']:.3f} "
          f"from {report['training_predictions']} predictions")
    for name, row in report["metrics"].items():
        print(f"{name}: learned {row['learned_mean']}, constant "
              f"{row['constant_mean']}, paired difference 95% CI "
              f"{row['learned_minus_constant_ci95']}")
    print(f"Saved {args.out}")
    print(f"Saved {opportunities_path}")


if __name__ == "__main__":
    main()
