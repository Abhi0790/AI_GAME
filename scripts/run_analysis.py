"""Run N games and print/save evaluation metrics summary."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random

from src.common.schemas import Player
from src.agents.agent import Agent
from src.engine.runner import GameRunner
from src.engine.replay import save_replay
from src.evaluation.metrics import final_scores
from src.evaluation.analysis import (
    print_game_summary, print_tournament_summary, export_game_data,
)


def main():
    parser = argparse.ArgumentParser(description="Run games and analyse results")
    parser.add_argument("--games", type=int, default=5, help="Number of games")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--export", type=str, default=None,
                        help="Directory to export JSON data")
    parser.add_argument("--save-replays", action="store_true",
                        help="Save replay files")
    args = parser.parse_args()

    all_results = []
    all_histories = []

    for i in range(args.games):
        seed = args.seed + i
        random.seed(seed)
        print(f"\n{'='*40}")
        print(f"  Game {i+1}/{args.games}  (seed={seed})")
        print(f"{'='*40}")

        agents = [
            Agent(Player.RED, "Opportunist"),
            Agent(Player.BLUE, "Honest"),
            Agent(Player.GREEN, "Paranoid"),
            Agent(Player.GOLD, "Vengeful"),
        ]
        runner = GameRunner(agents)
        runner.run()
        personas = runner.personas()

        scores = final_scores(runner.state)
        all_results.append(scores)
        all_histories.append(runner.history)

        print_game_summary(runner.history, runner.state,
                           runner.personas(), runner.calibration)

        if args.save_replays:
            path = save_replay(
                runner.history, runner.state,
                directory="replays",
                filename=f"game_{seed}.json",
            )
            print(f"Replay saved: {path}")

        if args.export:
            export_game_data(
                runner.history, runner.state,
                filepath=os.path.join(args.export, f"game_{seed}.json"),
            )

    if args.games > 1:
        print_tournament_summary(all_results, all_histories, personas)


if __name__ == "__main__":
    main()
