"""Run the experiments behind the six report figures and write the PNGs.

    python scripts/make_figures.py --games 20 --games-per-cell 20 --out figures/

Each figure needs a different experiment, so this script is mostly the
experiment harness; the drawing lives in src/evaluation/figures.py.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os
from collections import defaultdict

from src.common.config import GameConfig, PERSONA_ORDER
from src.common.schemas import leaders
from src.harness import play_many
from src.evaluation import figures
from src.evaluation.metrics import (
    MIN_GAMES, betrayal_rate_by_turn, vcoop_at_break, turns_to_coalition,
    final_scores, reliability_bins, brier_score, ci95,
)
from src.evaluation.sweep import baseline_brier


def _centres_by_persona(cfg, runner):
    personas = runner.personas()
    return [(personas[p], c) for p, c in final_scores(runner.state).items()]


def _variant_result(cfg, runner):
    """(win share, budgeted and total adjudications per turn) for the variant seat."""
    seat = next(iter(cfg.seat_planner))
    ahead = leaders(final_scores(runner.state))
    return (1.0 / len(ahead) if seat in ahead else 0.0,
            [step.nodes[seat] for step in runner.history],
            [step.total_nodes[seat] for step in runner.history])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=MIN_GAMES)
    ap.add_argument("--games-per-cell", type=int, default=MIN_GAMES,
                    help="games behind each search x budget cell of figure 6")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = list(range(1, args.games + 1))
    summary = {}

    # ── 1, 2, 5: one corpus of ordinary games ───────────────────────────
    print(f"Playing {len(seeds)} games at the default settings...")
    runners = play_many(GameConfig(seed=s) for s in seeds)
    # Seating rotates with the seed, so each runner has its own mapping —
    # pooling them under runners[0]'s would relabel three games in four.
    pooled_rates = defaultdict(lambda: defaultdict(list))
    vcoop_rows, calibration = [], []
    for r in runners:
        personas = r.personas()
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
    per_depth = {
        1: play_many((GameConfig(seed=s, depth=1) for s in seeds), _centres_by_persona),
        # The default corpus above is already depth 2.
        2: [_centres_by_persona(None, r) for r in runners],
    }
    for depth, games in per_depth.items():
        totals = defaultdict(list)
        for rows in games:
            for persona, centres in rows:
                totals[persona].append(centres)
        by_horizon[depth] = {p: sum(v) / len(v) for p, v in totals.items()}
    figures.rating_by_persona_and_horizon(by_horizon).savefig(
        f"{args.out}/3_rating_by_horizon.png", dpi=160)
    summary["rating_by_horizon"] = by_horizon

    # ── 4: turns to coalition ───────────────────────────────────────────
    coalition = [turns_to_coalition(r.history) for r in runners]
    figures.turns_to_coalition(coalition).savefig(
        f"{args.out}/4_turns_to_coalition.png", dpi=160)
    summary["turns_to_coalition"] = coalition

    # ── 6: win rate vs budget, expectiminimax vs MCTS ───────────────────
    cell_seeds = list(range(1, args.games_per_cell + 1))
    print(f"Comparing expectiminimax and MCTS at equal budgets "
          f"({len(cell_seeds)} games per cell)...")
    if len(cell_seeds) < MIN_GAMES:
        print(f"  warning: {len(cell_seeds)} < MIN_GAMES={MIN_GAMES}; figure 6 "
              f"will be drawn but annotated as below threshold.")

    def variant_seat(s):
        """The seat holding this seed's persona, so all four personas are tested."""
        seating = GameConfig(seed=s).seats()
        want = PERSONA_ORDER[s % len(PERSONA_ORDER)]
        return next((p for p, name in seating.items() if name == want),
                    list(seating)[s % len(seating)])

    cells = [(search, budget) for search in ("expectiminimax", "mcts")
             for budget in (400, 900, 1800)]
    # One seat on the variant at budget B; the others stay on the default planner.
    results = play_many((GameConfig(seed=s, seat_planner={
        variant_seat(s): {"search": search, "node_budget": budget}})
        for search, budget in cells for s in cell_seeds), _variant_result)
    rows = []
    for i, (search, budget) in enumerate(cells):
        games = results[i * len(cell_seeds):(i + 1) * len(cell_seeds)]
        wins = sum(w for w, _, _ in games)
        spent = [n for _, per_turn, _ in games for n in per_turn]
        total = [n for _, _, per_turn in games for n in per_turn]
        rate, lo, hi = ci95(wins, len(cell_seeds))
        rows.append({
            "search": search,
            "budget": budget,
            "adjudications": sum(spent) / max(1, len(spent)),
            # Negotiation pricing and prediction logging as well; identical code in every cell.
            "total_adjudications": sum(total) / max(1, len(total)),
            "win_rate": rate,
            "win_rate_lo": lo,
            "win_rate_hi": hi,
            "games": len(cell_seeds),
        })
        print(f"  {search:<16} budget {budget:>5} -> "
              f"{rows[-1]['adjudications']:.0f} adjudications/turn "
              f"({rows[-1]['total_adjudications']:.0f} with negotiation), "
              f"win rate {rate:.0%} [{lo:.0%}, {hi:.0%}] "
              f"(n={len(cell_seeds)})")
    figures.win_rate_vs_budget(rows).savefig(
        f"{args.out}/6_win_rate_vs_budget.png", dpi=160)
    summary["search_comparison"] = rows
    summary["min_games"] = MIN_GAMES

    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote six figures and summary.json to {args.out}/")


if __name__ == "__main__":
    main()
