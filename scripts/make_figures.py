"""Run the experiments behind the six report figures and write the PNGs.

    python scripts/make_figures.py --games 8 --out figures/

Each figure needs a different experiment, so this script is mostly the
experiment harness; the drawing lives in src/evaluation/figures.py.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os
import random
from collections import defaultdict

from src.common.schemas import Player
from src.agents.agent import Agent
from src.agents.planner.planner import PlannerConfig
from src.engine.runner import GameRunner
from src.evaluation import figures
from src.evaluation.metrics import (
    betrayal_rate_by_turn, vcoop_at_break, turns_to_coalition, final_scores,
    reliability_bins, brier_score,
)
from src.evaluation.sweep import baseline_brier

SEATS = [(Player.RED, "Opportunist"), (Player.BLUE, "Honest"),
         (Player.GREEN, "Paranoid"), (Player.GOLD, "Vengeful")]


def play(seed, broadcast=True, **planner_kwargs):
    random.seed(seed)
    agents = [Agent(p, persona, PlannerConfig(**planner_kwargs)) for p, persona in SEATS]
    runner = GameRunner(agents, broadcast_enabled=broadcast)
    runner.run(verbose=False)
    return runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = list(range(1, args.games + 1))
    summary = {}

    # ── 1, 2, 5: one corpus of ordinary games ───────────────────────────
    print(f"Playing {len(seeds)} games at the default settings...")
    runners = [play(s) for s in seeds]
    personas = runners[0].personas()

    pooled_rates = defaultdict(lambda: defaultdict(list))
    vcoop_rows, calibration = [], []
    for r in runners:
        for persona, series in betrayal_rate_by_turn(r.history, personas).items():
            for turn, value in enumerate(series):
                if value is not None:
                    pooled_rates[persona][turn].append(value)
        vcoop_rows += vcoop_at_break(r.history, personas)
        calibration += r.calibration

    max_turn = max((t for p in pooled_rates.values() for t in p), default=0) + 1
    series = {
        persona: [
            sum(turns[t]) / len(turns[t]) if turns.get(t) else None
            for t in range(max_turn)
        ]
        for persona, turns in pooled_rates.items()
    }

    figures.betrayal_rate_vs_turn(series).savefig(
        f"{args.out}/1_betrayal_rate_vs_turn.png", dpi=160)
    figures.vcoop_at_break(vcoop_rows).savefig(
        f"{args.out}/2_vcoop_at_break.png", dpi=160)

    bins = reliability_bins(calibration, bins=8)
    brier = brier_score(calibration)
    figures.reliability_diagram(bins, brier, baseline_brier(calibration)).savefig(
        f"{args.out}/5_reliability_diagram.png", dpi=160)
    summary["brier"] = brier
    summary["baseline_brier"] = baseline_brier(calibration)
    summary["predictions"] = len([c for c in calibration if c.observed is not None])

    # ── 3: rating by persona x search horizon ───────────────────────────
    print("Sweeping the search horizon...")
    by_horizon = {}
    for depth in (1, 2):
        totals = defaultdict(list)
        for s in seeds:
            r = play(s, depth=depth)
            scores = final_scores(r.state)
            for player, centres in scores.items():
                totals[personas[player]].append(centres)
        by_horizon[depth] = {p: sum(v) / len(v) for p, v in totals.items()}
    figures.rating_by_persona_and_horizon(by_horizon).savefig(
        f"{args.out}/3_rating_by_horizon.png", dpi=160)
    summary["rating_by_horizon"] = by_horizon

    # ── 4: turns to coalition, broadcast on vs off ──────────────────────
    print("Running the broadcast on/off control...")
    on = [turns_to_coalition(r.history) for r in runners]
    off = [turns_to_coalition(play(s, broadcast=False).history) for s in seeds]
    figures.turns_to_coalition(on, off).savefig(
        f"{args.out}/4_turns_to_coalition.png", dpi=160)
    summary["turns_to_coalition"] = {"broadcast_on": on, "broadcast_off": off}

    # ── 6: win rate vs budget, expectiminimax vs MCTS ───────────────────
    print("Comparing expectiminimax and MCTS at equal budgets...")
    rows = []
    for search in ("expectiminimax", "mcts"):
        for budget in (400, 900, 1800):
            wins, spent = 0, []
            for s in seeds:
                random.seed(s)
                agents = []
                for i, (player, persona) in enumerate(SEATS):
                    # Seat one agent on the variant under test; the other
                    # three stay on the default search, so the win rate is a
                    # comparison and not a self-play tautology.
                    cfg = PlannerConfig(
                        search=search if i == 0 else "expectiminimax",
                        node_budget=budget)
                    agents.append(Agent(player, persona, cfg))
                runner = GameRunner(agents)
                runner.run(verbose=False)
                scores = final_scores(runner.state)
                if max(scores, key=lambda p: scores[p]) == SEATS[0][0]:
                    wins += 1
                spent += [runner.history[t].nodes[SEATS[0][0]]
                          for t in range(len(runner.history))]
            rows.append({
                "search": search,
                "adjudications": sum(spent) / max(1, len(spent)),
                "win_rate": wins / len(seeds),
                "games": len(seeds),
            })
            print(f"  {search:<16} budget {budget:>5} -> "
                  f"{rows[-1]['adjudications']:.0f} adjudications/turn, "
                  f"win rate {rows[-1]['win_rate']:.0%}")
    figures.win_rate_vs_budget(rows).savefig(
        f"{args.out}/6_win_rate_vs_budget.png", dpi=160)
    summary["search_comparison"] = rows

    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote six figures and summary.json to {args.out}/")


if __name__ == "__main__":
    main()
