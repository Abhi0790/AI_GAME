"""Multi-game win rates, by persona, with intervals.

    python scripts/run_tournament.py --games 20

Personas rotate one seat per game (the harness does it), so every number is a
statement about the persona and not about the colour it sat in.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from src.common.config import GameConfig, add_board_args, board_kwargs
from src.harness import play_many
from src.evaluation.metrics import (
    MIN_GAMES, betrayal_rate_per_persona, preference_reversals,
    fmt_rate, fmt_mean,
)


def run_tournament(games: int = MIN_GAMES, board: dict | None = None):
    board = board or {}
    persona_wins = {}
    by_persona = {}
    briers = []
    reversals = []

    for runner in play_many(GameConfig(seed=i, **board) for i in range(games)):
        personas = runner.personas()
        for name, rate in betrayal_rate_per_persona(runner.history, personas).items():
            by_persona.setdefault(name, []).append(rate)
        score = runner.brier_score()
        if score is not None:
            briers.append(score)
        reversals += runner.reversals

        centers = runner.center_counts()
        max_c = max(centers.values())
        winners = [p for p, c in centers.items() if c == max_c]
        for w in winners:
            persona_wins[personas[w]] = persona_wins.get(personas[w], 0) + 1 / len(winners)

    print(f"\nTournament Results ({games} games, personas rotated one seat per game)")
    if games < MIN_GAMES:
        print(f"  {games} games is below MIN_GAMES={MIN_GAMES}: nothing below "
              f"is reportable.")

    print("\nWin share by persona — the number the betrayal table never reports:")
    for name, w in sorted(persona_wins.items(), key=lambda r: -r[1]):
        # Shared wins make this fractional, so round for the interval and say
        # so rather than pretending the count is exact.
        print(f"  {name:12s}: {fmt_rate(round(w), games)}  {'#' * int(w / games * 40)}")

    print("\nBetrayal rate by persona, pooled:")
    for name, values in sorted(by_persona.items()):
        print(f"  {name:12s}: {fmt_mean(values, '{:.1%}')}")

    rev = preference_reversals(reversals)
    if rev["n"]:
        print(f"\nPreference reversals: {fmt_rate(rev['reversed'], rev['n'])} of "
              f"priced deals were signed and then, in the same turn, worth more "
              f"broken than kept")
        for kind, (n, b, _r) in sorted(rev["by_kind"].items()):
            print(f"  {kind:12s}: {fmt_rate(b, n)}")

    if briers:
        print(f"\nP(keeps) Brier across the tournament: "
              f"{fmt_mean(briers)}  (0.25 = a coin flip)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=MIN_GAMES)
    add_board_args(ap)
    _args = ap.parse_args()
    run_tournament(_args.games, board_kwargs(_args))
