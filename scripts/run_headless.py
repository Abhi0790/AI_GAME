"""One seeded game, printed so the run can be reproduced from its own log.

    python scripts/run_headless.py --seed 42 --knob LAMBDA_INCENTIVE=0.6
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

from src.common.config import GameConfig, add_game_args, game_kwargs
from src.harness import play, knob_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    add_game_args(parser)
    args = parser.parse_args()
    cfg = GameConfig(seed=args.seed, **game_kwargs(args))

    # Everything needed to replay this run, before it runs.
    print("config: " + json.dumps(cfg.to_dict(), sort_keys=True, default=str))
    print("knobs : " + json.dumps({**knob_values(), **cfg.knobs}, sort_keys=True))

    play(cfg, verbose=True)


if __name__ == "__main__":
    main()
