"""Paired test of a learned P(keeps) against a held-out constant forecast."""

import random
from typing import Optional

from src.common.config import GameConfig, PERSONA_ORDER
from src.common.schemas import MessageType, Player, leaders
from src.evaluation.metrics import brier_score, final_scores, obligations
from src.harness import play_many


def variant_seat(seed: int, seats: list[Player]) -> Player:
    """Balance both seat colour and persona across a 24-seed default cycle."""
    period = len(PERSONA_ORDER)
    return seats[(seed // period + seed % period) % len(seats)]


def _training_result(cfg: GameConfig, runner):
    seat = variant_seat(cfg.seed, cfg.make_board().players)
    graded = [p for p in runner.calibration
              if p.observer == seat and p.observed is not None]
    return sum(p.observed is True for p in graded), len(graded)


def _game_result(cfg: GameConfig, runner):
    seat = variant_seat(cfg.seed, cfg.make_board().players)
    scores = final_scores(runner.state)
    ahead = leaders(scores)
    duties = [broke for step in runner.history
              for (player, _partner), broke in obligations(step).items()
              if player == seat]
    deals = {c.id for step in runner.history for c in step.commitments
             if seat in c.players}
    forecasts = [p for p in runner.calibration if p.observer == seat]
    signed = [p.signed_gain for p in runner.reversals if p.player == seat]
    turns = len(runner.history)
    opportunities = _opportunities(runner, seat)
    decisions = {kind: sum(row["decision"] == kind for row in opportunities)
                 for kind in ("Accept", "Counter", "Reject")}
    return {
        "seed": cfg.seed,
        "seat": seat.value,
        "persona": runner.personas()[seat],
        "arm": "constant" if cfg.seat_keep_forecast else "learned",
        "win_share": 1 / len(ahead) if seat in ahead else 0.0,
        "final_centers": scores[seat],
        "deals": len(deals),
        "betrayals": sum(duties),
        "obligations": len(duties),
        "betrayal_rate": sum(duties) / len(duties) if duties else None,
        "mean_signed_gain": sum(signed) / len(signed) if signed else None,
        "brier": brier_score(forecasts),
        "forecast_rows": sum(p.observed is not None for p in forecasts),
        "planner_nodes_per_turn": sum(step.nodes[seat] for step in runner.history) / turns,
        "total_nodes_per_turn": sum(step.total_nodes[seat] for step in runner.history) / turns,
        "offers_received": len(opportunities),
        "priced_offers": sum(row["forecast"] is not None for row in opportunities),
        "early_rejections": sum(row["early_reject"] for row in opportunities),
        "accepted_offers": decisions["Accept"],
        "countered_offers": decisions["Counter"],
        "rejected_offers": decisions["Reject"],
        "opportunities": opportunities,
    }


def _opportunities(runner, seat: Player) -> list[dict]:
    """Record every offer answered by the selected seat and its later verdicts."""
    outcomes = {}
    for step in runner.history:
        for outcome in step.outcomes:
            outcomes.setdefault(outcome.commitment.id, []).append({
                "turn": step.state.turn,
                "kept": outcome.kept,
                "broken_by": [p.value for p in outcome.broken_by],
            })
    rows = []
    for step in runner.history:
        offers = {m.id: m for m in step.messages
                  if m.message_type in (MessageType.PROPOSE, MessageType.COUNTER)}
        signed = {c.id for c in step.commitments}
        for reply in step.messages:
            if reply.sender != seat or reply.reference_id not in offers:
                continue
            if reply.message_type not in (MessageType.ACCEPT, MessageType.COUNTER,
                                          MessageType.REJECT):
                continue
            offer = offers[reply.reference_id]
            if offer.sender == seat:
                continue
            why = reply.rationale or {}
            signed_id = (reply.id if reply.message_type == MessageType.COUNTER
                         else offer.id)
            was_signed = signed_id in signed
            rows.append({
                "turn": step.state.turn,
                "offer_id": offer.id,
                "proposer": offer.sender.value,
                "type": offer.commitment_type.value if offer.commitment_type else None,
                "offer_kind": offer.message_type.value,
                "turns": offer.turns,
                "private": offer.private,
                "decision": reply.message_type.value,
                "forecast": why.get("p_keep"),
                "value_without": why.get("v_none"),
                "value_kept": why.get("v_kept"),
                "value_broken": why.get("v_broken"),
                "expected_margin": (why["ev"] - why["v_none"]
                                    if "ev" in why and "v_none" in why else None),
                "early_reject": reply.message_type == MessageType.REJECT and not why,
                "signed": was_signed,
                "outcomes": outcomes.get(signed_id, []) if was_signed else [],
            })
    return rows


def paired_interval(values: list[float], samples: int = 5000) -> list[Optional[float]]:
    """Mean and percentile bootstrap CI, resampling paired games as clusters."""
    if not values:
        return [None, None, None]
    mean = sum(values) / len(values)
    if len(values) == 1:
        return [mean, None, None]
    rng = random.Random(0)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(samples))
    return [mean, draws[int(0.025 * (samples - 1))],
            draws[int(0.975 * (samples - 1))]]


def summarize_opportunities(games: list[dict]) -> dict:
    rows = [row for game in games for row in game["opportunities"]]
    priced = [row for row in rows if row["forecast"] is not None]
    graded = [outcome for row in rows if row["signed"]
              for outcome in row["outcomes"]]
    return {
        "received": len(rows),
        "priced": len(priced),
        "mean_forecast_when_priced": (
            sum(row["forecast"] for row in priced) / len(priced) if priced else None),
        "early_rejected": sum(row["early_reject"] for row in rows),
        "accepted": sum(row["decision"] == "Accept" for row in rows),
        "countered": sum(row["decision"] == "Counter" for row in rows),
        "rejected": sum(row["decision"] == "Reject" for row in rows),
        "signed": sum(row["signed"] for row in rows),
        "graded_deal_turns": len(graded),
        "deal_keep_rate": (sum(outcome["kept"] for outcome in graded) / len(graded)
                           if graded else None),
    }


def compare(learned: list[dict], constant: list[dict]) -> dict:
    if [r["seed"] for r in learned] != [r["seed"] for r in constant]:
        raise ValueError("arms must contain the same seeds in the same order")
    metrics = ("win_share", "final_centers", "deals", "offers_received",
               "priced_offers", "early_rejections", "accepted_offers",
               "countered_offers", "rejected_offers", "betrayal_rate",
               "mean_signed_gain", "brier", "planner_nodes_per_turn",
               "total_nodes_per_turn")
    out = {}
    for name in metrics:
        pairs = [(a[name], b[name]) for a, b in zip(learned, constant)
                 if a[name] is not None and b[name] is not None]
        deltas = [a - b for a, b in pairs]
        out[name] = {
            "learned_mean": sum(a for a, _ in pairs) / len(pairs) if pairs else None,
            "constant_mean": sum(b for _, b in pairs) / len(pairs) if pairs else None,
            "learned_minus_constant_ci95": paired_interval(deltas),
            "paired_games": len(pairs),
        }
    return out


def compare_budgets(low: dict, high: dict) -> dict:
    """Paired high-minus-low change in the learned-versus-control effect."""
    if low.get("training_seeds") != high.get("training_seeds"):
        raise ValueError("budget runs must use the same training seeds")
    low_config, high_config = low.get("config", {}).copy(), high.get("config", {}).copy()
    low_config.pop("node_budget", None)
    high_config.pop("node_budget", None)
    if low_config != high_config:
        raise ValueError("budget runs must match except for node budget")
    low_games, high_games = low["games"], high["games"]
    if len(low_games) != len(high_games) or any(
        (lo["learned"]["seed"], lo["learned"]["seat"], lo["learned"]["persona"])
        != (hi["learned"]["seed"], hi["learned"]["seat"], hi["learned"]["persona"])
        for lo, hi in zip(low_games, high_games)
    ):
        raise ValueError("budget runs must use the same seeds, seats, and personas")
    metrics = ("win_share", "final_centers", "deals", "offers_received",
               "priced_offers", "accepted_offers", "betrayal_rate", "brier")
    out = {}
    for name in metrics:
        paired = []
        for lo, hi in zip(low_games, high_games):
            values = [pair[arm][name] for pair in (lo, hi)
                      for arm in ("learned", "constant")]
            if all(v is not None for v in values):
                paired.append((values[0] - values[1], values[2] - values[3]))
        out[name] = {
            "low_learned_minus_constant": (
                sum(a for a, _ in paired) / len(paired) if paired else None),
            "high_learned_minus_constant": (
                sum(b for _, b in paired) / len(paired) if paired else None),
            "high_minus_low_ci95": paired_interval([b - a for a, b in paired]),
            "paired_games": len(paired),
        }
    return out


def run(train_seeds: list[int], test_seeds: list[int],
        board_settings: dict, node_budget: int = 400,
        workers: Optional[int] = None) -> dict:
    """Fit the control on separate games, then run matched variant-seat pairs."""
    if set(train_seeds) & set(test_seeds):
        raise ValueError("training and test seeds must be disjoint")
    if not train_seeds or not test_seeds:
        raise ValueError("both training and test seeds are required")
    shared = {**board_settings, "node_budget": node_budget,
              "event_keyed_rng": True}
    training = play_many((GameConfig(seed=s, **shared) for s in train_seeds),
                         _training_result, workers=workers)
    kept = sum(k for k, _n in training)
    n = sum(n for _k, n in training)
    if not n:
        raise ValueError("training games contained no graded promises")
    baseline = kept / n

    board = GameConfig(**shared).make_board()
    learned_cfgs = [GameConfig(seed=s, **shared) for s in test_seeds]
    constant_cfgs = [GameConfig(
        seed=s, seat_keep_forecast={variant_seat(s, board.players): baseline},
        **shared) for s in test_seeds]
    learned = play_many(learned_cfgs, _game_result, workers=workers)
    constant = play_many(constant_cfgs, _game_result, workers=workers)
    by_persona = {
        persona: compare(
            [a for a in learned if a["persona"] == persona],
            [b for b in constant if b["persona"] == persona])
        for persona in sorted({row["persona"] for row in learned})
    }
    return {
        "design": "one rotating variant seat; learned versus constant P(keeps); paired seeds; event-keyed RNG",
        "training_seeds": train_seeds,
        "test_seeds": test_seeds,
        "training_predictions": n,
        "constant_keep_forecast": baseline,
        "config": GameConfig(seed=test_seeds[0], **shared).to_dict(),
        "metrics": compare(learned, constant),
        "by_persona": by_persona,
        "offer_summary": {"learned": summarize_opportunities(learned),
                          "constant": summarize_opportunities(constant)},
        "games": [{"learned": a, "constant": b}
                  for a, b in zip(learned, constant)],
    }
