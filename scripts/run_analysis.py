"""Run N games and print/save evaluation metrics summary."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from src.common.config import GameConfig, add_board_args, board_kwargs
from src.harness import game_setup, play_many
from src.engine.replay import save_replay
from src.evaluation.metrics import MIN_GAMES, final_scores
from src.evaluation.analysis import (
    print_game_summary, print_tournament_summary, export_game_data,
)


def main():
    parser = argparse.ArgumentParser(description="Run games and analyse results")
    parser.add_argument("--games", type=int, default=MIN_GAMES,
                        help=f"Number of games (below {MIN_GAMES} is not reportable)")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--export", type=str, default=None,
                        help="Directory to export JSON data")
    parser.add_argument("--save-replays", action="store_true",
                        help="Save replay files")
    add_board_args(parser)
    args = parser.parse_args()

    all_results = []
    all_histories = []
    all_personas = []

    cfgs = [GameConfig(seed=args.seed + i, **board_kwargs(args)) for i in range(args.games)]
    for i, (cfg, runner) in enumerate(zip(cfgs, play_many(cfgs))):
        seed = cfg.seed
        print(f"\n{'='*40}")
        print(f"  Game {i+1}/{args.games}  (seed={seed})")
        print(f"{'='*40}")

        # Personas rotate one seat per seed, so the per-persona tables below
        # are not measuring the colour.
        personas = runner.personas()
        all_personas.append(personas)

        # The metrics read the seat list off the board.
        with game_setup(cfg):
            all_results.append(final_scores(runner.state))
            all_histories.append(runner.history)

            print_game_summary(runner.history, runner.state,
                               personas, runner.calibration, runner.reversals)

            if args.save_replays:
                path = save_replay(
                    runner.history, runner.state,
                    directory="replays",
                    filename=f"game_{seed}.json",
                    board=cfg.make_board().to_dict(),
                )
                print(f"Replay saved: {path}")

        if args.export:
            export_game_data(
                runner.history, runner.state,
                filepath=os.path.join(args.export, f"game_{seed}.json"),
            )

    if args.games > 1:
        with game_setup(GameConfig(**board_kwargs(args))):
            print_tournament_summary(all_results, all_histories, all_personas)


if __name__ == "__main__":
    main()
