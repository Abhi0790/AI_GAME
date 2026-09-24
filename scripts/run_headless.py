"""One seeded game, printed so the run can be reproduced from its own log.

    python scripts/run_headless.py --seed 42 --knob LAMBDA_INCENTIVE=0.6
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

from src.common.config import GameConfig, add_board_args, board_kwargs
from src.harness import play, knob_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search", default="expectiminimax",
                        choices=["expectiminimax", "mcts"])
    parser.add_argument("--node-budget", type=int, default=1500)
    parser.add_argument("--knob", action="append", default=[],
                        metavar="NAME=VALUE", help="override a tuning constant")
    add_board_args(parser)
    args = parser.parse_args()

    knobs = dict(k.split("=", 1) for k in args.knob)
    cfg = GameConfig(seed=args.seed, search=args.search,
                     node_budget=args.node_budget,
                     knobs={k: float(v) for k, v in knobs.items()},
                     **board_kwargs(args))

    # Everything needed to replay this run, before it runs.
    print("config: " + json.dumps(cfg.to_dict(), sort_keys=True, default=str))
    print("knobs : " + json.dumps({**knob_values(), **cfg.knobs}, sort_keys=True))

    play(cfg, verbose=True)


if __name__ == "__main__":
    main()
