"""Paired evaluation of promise-keeping candidate coverage in the planner."""

from typing import Optional

from src.common.config import GameConfig
from src.common.schemas import obligated_parties
from src.evaluation.metrics import final_scores
from src.evaluation.trust_ablation import paired_interval, variant_seat
from src.harness import play_many


CATEGORIES = ("forced", "generator_miss", "prune_miss", "budget_miss",
              "scored_break", "unclassified")


def _break_category(coverage: dict) -> str:
    if not coverage:
        return "unclassified"
    if not coverage.get("keepable"):
        return "forced"
    if coverage.get("evaluated_keep"):
        return "scored_break"
    if not coverage.get("generated_keep"):
        return "generator_miss"
    if not coverage.get("pruned_keep"):
        return "prune_miss"
    return "budget_miss"


def _game_result(cfg: GameConfig, runner) -> dict:
    seat = variant_seat(cfg.seed, cfg.make_board().players)
    scores = final_scores(runner.state)
    ahead = runner.leaders()
    categories = {name: 0 for name in CATEGORIES}
    due_total = broken_total = missed = keepable = scored_break_turns = 0
    for step in runner.history:
        due = [o for o in step.outcomes
               if seat in obligated_parties(o.commitment, step.state.turn)]
        broken = sum(seat in o.broken_by for o in due)
        due_total += len(due)
        broken_total += broken
        coverage = step.traces[seat].coverage
        if coverage.get("keepable"):
            keepable += 1
            if not coverage.get("evaluated_keep"):
                missed += 1
        if broken:
            category = _break_category(coverage)
            categories[category] += broken
            scored_break_turns += category == "scored_break"
    turns = len(runner.history)
    return {
        "seed": cfg.seed,
        "seat": seat.value,
        "persona": runner.personas()[seat],
        "arm": ("reserved" if cfg.seat_planner[seat]["reserve_keep_candidate"]
                else "baseline"),
        "win_share": 1 / len(ahead) if seat in ahead else 0.0,
        "final_centers": scores[seat],
        "due_obligations": due_total,
        "broken_obligations": broken_total,
        "betrayal_rate": broken_total / due_total if due_total else None,
        "keepable_turns": keepable,
        "missed_keep_turns": missed,
        "scored_break_turns": scored_break_turns,
        "break_categories": categories,
        "planner_nodes_per_turn": sum(step.nodes[seat] for step in runner.history) / turns,
    }


def _compare(baseline: list[dict], reserved: list[dict]) -> dict:
    if [r["seed"] for r in baseline] != [r["seed"] for r in reserved]:
        raise ValueError("arms must use matching seeds")
    out = {}
    for metric in ("win_share", "final_centers", "due_obligations",
                   "broken_obligations",
                   "betrayal_rate", "missed_keep_turns", "scored_break_turns",
                   "planner_nodes_per_turn"):
        pairs = [(a[metric], b[metric]) for a, b in zip(baseline, reserved)
                 if a[metric] is not None and b[metric] is not None]
        out[metric] = {
            "baseline_mean": sum(a for a, _ in pairs) / len(pairs) if pairs else None,
            "reserved_mean": sum(b for _, b in pairs) / len(pairs) if pairs else None,
            "reserved_minus_baseline_ci95": paired_interval([b - a for a, b in pairs]),
            "paired_games": len(pairs),
        }
    return out


def _category_totals(games: list[dict]) -> dict:
    return {name: sum(game["break_categories"][name] for game in games)
            for name in CATEGORIES}


def run(seeds: list[int], board_settings: dict, node_budget: int,
        workers: Optional[int] = None) -> dict:
    if not seeds:
        raise ValueError("at least one seed is required")
    shared = {**board_settings, "node_budget": node_budget,
              "event_keyed_rng": True}
    seats = GameConfig(**shared).make_board().players

    def configs(reserve: bool):
        for seed in seeds:
            seat = variant_seat(seed, seats)
            yield GameConfig(seed=seed, seat_planner={seat: {
                "trace_coverage": True, "reserve_keep_candidate": reserve}},
                **shared)

    baseline = play_many(configs(False), _game_result, workers=workers)
    reserved = play_many(configs(True), _game_result, workers=workers)
    return {
        "design": "one rotating seat; baseline versus reserved keeping candidate; paired seeds and event-keyed RNG",
        "seeds": seeds,
        "config": GameConfig(seed=seeds[0], **shared).to_dict(),
        "metrics": _compare(baseline, reserved),
        "break_categories": {
            "baseline": _category_totals(baseline),
            "reserved": _category_totals(reserved),
        },
        "games": [{"baseline": a, "reserved": b}
                  for a, b in zip(baseline, reserved)],
    }


def compare_budgets(low: dict, high: dict) -> dict:
    """Paired change in the safeguard's effect between planner budgets."""
    if low["seeds"] != high["seeds"]:
        raise ValueError("budget runs must use the same seeds")
    low_config, high_config = low["config"].copy(), high["config"].copy()
    low_config.pop("node_budget", None)
    high_config.pop("node_budget", None)
    if low_config != high_config:
        raise ValueError("budget runs must match except for node budget")
    out = {}
    for metric in ("win_share", "final_centers", "due_obligations",
                   "broken_obligations", "betrayal_rate", "missed_keep_turns"):
        pairs = []
        for lo, hi in zip(low["games"], high["games"]):
            values = [pair[arm][metric] for pair in (lo, hi)
                      for arm in ("baseline", "reserved")]
            if all(value is not None for value in values):
                pairs.append((values[1] - values[0], values[3] - values[2]))
        out[metric] = {
            "low_reserved_minus_baseline": (
                sum(a for a, _ in pairs) / len(pairs) if pairs else None),
            "high_reserved_minus_baseline": (
                sum(b for _, b in pairs) / len(pairs) if pairs else None),
            "high_minus_low_ci95": paired_interval([b - a for a, b in pairs]),
            "paired_games": len(pairs),
        }
    return out
