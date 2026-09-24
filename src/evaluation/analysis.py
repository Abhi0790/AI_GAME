"""
Analysis utilities — human-readable summaries and JSON export.
"""

import json
import os
from typing import List, Dict, Any

from src.engine.board import players
from src.common.schemas import Player, GameState, leaders
from src.evaluation.metrics import (
    deal_mix,
    pair_trust_divergence,
    supply_center_timeline,
    betrayal_events,
    betrayal_rate_per_player,
    obligations,
    preference_reversals,
    adjudications_per_turn,
    turns_to_coalition,
    alliance_durations,
    final_scores,
    win_rates,
    brier_score,
    reliability_bins,
    MIN_GAMES,
    fmt_rate,
    fmt_mean,
)


def _betrayal_counts(history: list, personas: Dict[Player, str] = None
                     ) -> Dict[str, List[int]]:
    """{persona: [broken, obligated]} — the counts behind the rate, which is
    what an interval needs. Counted per (turn, partner) by `obligations`, the
    same unit `betrayal_rate_per_persona` uses."""
    counts: Dict[str, List[int]] = {}
    if not personas:
        return counts
    for step in history:
        for (p, _q), broke in obligations(step).items():
            row = counts.setdefault(personas.get(p, "Unknown"), [0, 0])
            row[1] += 1
            row[0] += broke
    return counts


def print_game_summary(history: list, final_state: GameState,
                       personas: Dict[Player, str] = None,
                       calibration: list = None, reversals: list = None):
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
    for p in players():
        bar = "█" * scores[p]
        print(f"  {p.value:6s}: {scores[p]}  {bar}")

    ahead = leaders(scores)
    print("\n  Winner: " + (ahead[0].value if len(ahead) == 1
                            else "shared — " + ", ".join(sorted(p.value for p in ahead))))

    # Betrayals
    betrayals = betrayal_events(history)
    print(f"\nTotal Betrayals: {len(betrayals)}")
    for b in betrayals:
        print(f"  Turn {b['turn']}: {b['commitment_type']} between "
              f"{b['players']} broken by {b['broken_by']}")

    # Betrayal rates. Keyed by Player for the seat colour, and by persona for
    # everything the report actually compares — the persona table is the one
    # to read, the colour one is a sanity check.
    counts = _betrayal_counts(history, personas)
    print("\nBetrayal Rates by seat:")
    rates = betrayal_rate_per_player(history)
    for p in players():
        label = f"{p.value} ({personas[p]})" if personas else p.value
        print(f"  {label:24s}: {rates[p]:.1%}")

    if personas:
        print("\nBetrayal Rate by Persona:")
        for name in sorted(counts):
            broke, total = counts[name]
            print(f"  {name:12s}: {fmt_rate(broke, total)}")

    mix = deal_mix(history)
    if mix["made"]:
        print("\nPromises made, and how often they were broken:")
        for kind in sorted(mix["made"]):
            print(f"  {kind:12s}: "
                  f"{fmt_rate(mix['broken'].get(kind, 0), mix['made'][kind])}")

    pair = pair_trust_divergence(history)
    if pair["n"]:
        print(f"\n\"Keeps promises to me\" vs \"keeps promises\": "
              f"mean gap {pair['mean_where_diverged']:.3f} where they diverged, "
              f"{pair['share_over_0.1']:.0%} of beliefs apart by more than 0.1")

    nodes = adjudications_per_turn(history)
    if any(nodes):
        print(f"\nSearch cost: {sum(nodes)} adjudications, "
              f"{sum(nodes) / max(1, len(nodes)):.0f} per turn")
    coalition = turns_to_coalition(history)
    print(f"First standing alliance: " +
          (f"turn {coalition}" if coalition else "never formed"))

    rev = preference_reversals(reversals or [], personas)
    if rev["n"]:
        print(f"\nPreference reversals: {rev['reversed']}/{rev['n']} = "
              f"{rev['rate']:.0%} of deals were signed and then, in the same turn, "
              f"worth more broken than kept (mean edge {rev['mean_gap']:+.2f})")
        for kind, (n, b, _r) in sorted(rev["by_kind"].items()):
            print(f"  {kind:12s}: {fmt_rate(b, n)}")
        for k, (n, b, _r) in sorted(rev["by_persona"].items()):
            print(f"  {k:12s}: {fmt_rate(b, n)}")

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
    alliances = alliance_durations(history, horizon=len(history))
    if alliances:
        avg_dur = sum(a["duration"] for a in alliances) / len(alliances)
        kept = sum(1 for a in alliances if a["status"] == "kept")
        censored = sum(1 for a in alliances if a["status"] == "censored")
        print(f"\nAlliances: {len(alliances)} total, {kept} ran to term, "
              f"{censored} still live at the horizon, "
              f"avg observed duration {avg_dur:.1f} turns")
        if censored:
            print(f"  (the {censored} censored ones are a lower bound, not a result)")

    # SC timeline (compact)
    timeline = supply_center_timeline(history, final_state)
    print("\nSupply Centers Over Time:")
    header = "  Turn  " + "  ".join(f"{p.value:>5}" for p in players())
    print(header)
    turns = max(len(v) for v in timeline.values()) if timeline else 0
    for t in range(turns):
        row = f"  {t + 1:4d}  "
        for p in players():
            val = timeline[p][t] if t < len(timeline[p]) else 0
            row += f"  {val:>5}"
        print(row)

    print("=" * 60)


def print_tournament_summary(
    game_results: List[Dict[Player, int]],
    all_histories: List[list],
    all_personas: List[Dict[Player, str]] = None,
):
    """Print cross-game tournament statistics.

    *all_personas* is one mapping per game, because the harness rotates the
    personas one seat per seed. Pooling them under a single game's mapping —
    which is what this used to do — attributes every rate to the wrong
    persona in three games out of four.
    """
    print("\n" + "=" * 60)
    print("  TOURNAMENT SUMMARY")
    print("=" * 60)

    n = len(game_results)
    print(f"\nGames Played: {n}" +
          (f"   [below MIN_GAMES={MIN_GAMES}: nothing here is reportable]"
           if n < MIN_GAMES else ""))

    wr = win_rates(game_results)
    print("\nWin Rates by seat (should be flat — the personas rotate):")
    for p in players():
        print(f"  {p.value:6s}: {fmt_rate(round(wr[p] * n), n)}")

    print("\nAverage Betrayal Rates by seat:")
    for p in players():
        values = [betrayal_rate_per_player(h)[p] for h in all_histories]
        print(f"  {p.value:6s}: {fmt_mean(values, '{:.1%}')}")

    if all_personas:
        pooled: Dict[str, List[int]] = {}
        wins: Dict[str, float] = {}
        for history, personas, scores in zip(all_histories, all_personas,
                                             game_results):
            for name, (broke, total) in _betrayal_counts(history, personas).items():
                row = pooled.setdefault(name, [0, 0])
                row[0] += broke
                row[1] += total
            best = max(scores.values())
            champs = [p for p, c in scores.items() if c == best]
            for c in champs:
                wins[personas[c]] = wins.get(personas[c], 0) + 1 / len(champs)

        print("\nBetrayal Rate by Persona (pooled over every commitment):")
        for name, (broke, total) in sorted(pooled.items()):
            print(f"  {name:12s}: {fmt_rate(broke, total)}")

        print("\nWin share by Persona:")
        for name, w in sorted(wins.items(), key=lambda r: -r[1]):
            print(f"  {name:12s}: {fmt_rate(round(w), n)}")

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
        "alliances": alliance_durations(history, horizon=len(history)),
        "sc_timeline": {p.value: v
                        for p, v in supply_center_timeline(
                            history, final_state).items()},
    }
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported game data to {filepath}")
