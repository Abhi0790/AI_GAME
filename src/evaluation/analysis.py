"""
Analysis utilities — human-readable summaries and JSON export.
"""

import json
import os
from typing import List, Dict, Any

from src.common.schemas import Player, GameState
from src.evaluation.metrics import (
    supply_center_timeline,
    betrayal_events,
    betrayal_rate_per_player,
    betrayal_rate_per_persona,
    accusation_stats,
    adjudications_per_turn,
    turns_to_coalition,
    alliance_durations,
    final_scores,
    win_rates,
    brier_score,
    reliability_bins,
)


def print_game_summary(history: list, final_state: GameState,
                       personas: Dict[Player, str] = None,
                       calibration: list = None):
    """Print a formatted single-game summary to stdout.

    *personas* turns seat colours into the thing the report actually compares.
    "Red betrays more than Blue" says nothing on its own.
    """
    print("\n" + "=" * 60)
    print("  GAME SUMMARY")
    print("=" * 60)

    # Final scores
    scores = final_scores(final_state)
    print("\nFinal Supply Center Counts:")
    for p in Player:
        bar = "█" * scores[p]
        print(f"  {p.value:6s}: {scores[p]}  {bar}")

    winner = max(scores, key=scores.get)
    print(f"\n  Winner: {winner.value}")

    # Betrayals
    betrayals = betrayal_events(history)
    print(f"\nTotal Betrayals: {len(betrayals)}")
    for b in betrayals:
        print(f"  Turn {b['turn']}: {b['commitment_type']} between "
              f"{b['players']} broken by {b['broken_by']}")

    # Betrayal rates
    rates = betrayal_rate_per_player(history)
    print("\nBetrayal Rates:")
    for p in Player:
        label = f"{p.value} ({personas[p]})" if personas else p.value
        print(f"  {label:24s}: {rates[p]:.1%}")

    if personas:
        print("\nBetrayal Rate by Persona:")
        for name, rate in sorted(betrayal_rate_per_persona(history, personas).items()):
            print(f"  {name:12s}: {rate:.1%}")

        acc = accusation_stats(history, personas)
        if acc["total"]:
            print(f"\nAccusations: {acc['total']} made, {acc['confirmed']} confirmed by "
                  f"the engine, {acc['refuted']} refuted ({acc['known_lies']} were lies)")
            for name, row in sorted(acc["by_persona"].items()):
                print(f"  {name:12s}: {row['made']} made, {row['refuted']} refuted")

    nodes = adjudications_per_turn(history)
    if any(nodes):
        print(f"\nSearch cost: {sum(nodes)} adjudications, "
              f"{sum(nodes) / max(1, len(nodes)):.0f} per turn")
    coalition = turns_to_coalition(history)
    print(f"First standing alliance: " +
          (f"turn {coalition}" if coalition else "never formed"))

    if calibration:
        score = brier_score(calibration)
        if score is not None:
            graded = len([c for c in calibration if c.observed is not None])
            print(f"\nP(keeps) calibration over {graded} graded promises: "
                  f"Brier {score:.3f}  (0.25 = a coin flip)")
            for b in reliability_bins(calibration, bins=5):
                bar = "#" * round(b["observed_rate"] * 20)
                print(f"  predicted {b['bin_lower']:.1f}-{b['bin_upper']:.1f} "
                      f"(n={b['count']:>3}) -> kept {b['observed_rate']:.0%} {bar}")

    # Alliance durations
    alliances = alliance_durations(history)
    if alliances:
        avg_dur = sum(a["duration"] for a in alliances) / len(alliances)
        kept = sum(1 for a in alliances if a["status"] == "kept")
        print(f"\nAlliances: {len(alliances)} total, "
              f"{kept} kept, avg duration {avg_dur:.1f} turns")

    # SC timeline (compact)
    timeline = supply_center_timeline(history)
    print("\nSupply Centers Over Time:")
    header = "  Turn  " + "  ".join(f"{p.value:>5}" for p in Player)
    print(header)
    turns = max(len(v) for v in timeline.values()) if timeline else 0
    for t in range(turns):
        row = f"  {t + 1:4d}  "
        for p in Player:
            val = timeline[p][t] if t < len(timeline[p]) else 0
            row += f"  {val:>5}"
        print(row)

    print("=" * 60)


def print_tournament_summary(
    game_results: List[Dict[Player, int]],
    all_histories: List[list],
    personas: Dict[Player, str] = None,
):
    """Print cross-game tournament statistics."""
    print("\n" + "=" * 60)
    print("  TOURNAMENT SUMMARY")
    print("=" * 60)

    n = len(game_results)
    print(f"\nGames Played: {n}")

    wr = win_rates(game_results)
    print("\nWin Rates:")
    for p in Player:
        bar = "█" * int(wr[p] * 20)
        print(f"  {p.value:6s}: {wr[p]:.1%}  {bar}")

    # Aggregate betrayal rates
    all_rates: Dict[Player, List[float]] = {p: [] for p in Player}
    for history in all_histories:
        rates = betrayal_rate_per_player(history)
        for p in Player:
            all_rates[p].append(rates[p])

    print("\nAverage Betrayal Rates:")
    for p in Player:
        avg = sum(all_rates[p]) / max(1, len(all_rates[p]))
        label = f"{p.value} ({personas[p]})" if personas else p.value
        print(f"  {label:24s}: {avg:.1%}")

    if personas:
        pooled: Dict[str, List[float]] = {}
        for history in all_histories:
            for name, rate in betrayal_rate_per_persona(history, personas).items():
                pooled.setdefault(name, []).append(rate)
        print("\nBy Persona:")
        for name, values in sorted(pooled.items()):
            avg = sum(values) / len(values)
            print(f"  {name:12s}: {avg:.1%}  {'#' * round(avg * 30)}")

    print("=" * 60)


def export_game_data(
    history: list,
    final_state: GameState,
    filepath: str,
):
    """Export game data as a JSON file."""
    data = {
        "final_scores": {p.value: s for p, s in final_scores(final_state).items()},
        "betrayals": betrayal_events(history),
        "betrayal_rates": {p.value: r for p, r in betrayal_rate_per_player(history).items()},
        "alliances": alliance_durations(history),
        "sc_timeline": {p.value: v for p, v in supply_center_timeline(history).items()},
    }
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported game data to {filepath}")
