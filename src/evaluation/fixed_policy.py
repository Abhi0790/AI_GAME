"""Calibration games whose proposals, replies and orders ignore trust forecasts.

These games test whether P(keeps) predicts behavior under a fixed policy. They
do not estimate calibration of the agents' usual self-selected deal population.
"""

import random
from typing import List

from src.agents.agent import Agent
from src.common.config import GameConfig
from src.common.schemas import (
    CommitmentType, DecisionTrace, Message, MessageType, OrderType, Player,
)
from src.engine.board import players
from src.engine.orders import generate_orders_for_unit
from src.engine.runner import GameRunner
from src.evaluation.metrics import brier_score, calibration_error, mean_ci95, reliability_bins
from src.harness import game_setup


class FixedPolicyAgent(Agent):
    """Keeps the normal trust model, but never consults it when playing."""

    def __init__(self, player: Player, seed: int):
        super().__init__(player)
        self.policy_rng = random.Random(f"fixed-policy:{seed}:{player.value}")

    def propose(self, state, commitments=None):
        roster = players()
        index = roster.index(self.player)
        alive = {u.player for u in state.units}
        partner = next((roster[(index + n) % len(roster)]
                        for n in range(1, len(roster))
                        if roster[(index + n) % len(roster)] in alive), None)
        if partner is None:
            return []
        return [Message(
            id=f"fixed-{state.turn}-{self.player.value}-{partner.value}",
            sender=self.player, receiver=partner,
            message_type=MessageType.PROPOSE,
            commitment_type=CommitmentType.ALLIANCE, turns=2,
        )]

    def reply(self, state, incoming, commitments=None):
        replies = []
        for msg in incoming:
            if msg.message_type != MessageType.PROPOSE or msg.receiver != self.player:
                continue
            accept = self.policy_rng.random() < 0.75
            replies.append(Message(
                id=f"reply-{msg.id}", sender=self.player, receiver=msg.sender,
                message_type=MessageType.ACCEPT if accept else MessageType.REJECT,
                reference_id=msg.id,
            ))
        return replies

    def act(self, state, active_commitments):
        orders = []
        for unit in state.units:
            if unit.player != self.player:
                continue
            options = generate_orders_for_unit(state, unit)
            moves = [o for o in options if o.order_type == OrderType.MOVE]
            supports = [o for o in options if o.order_type == OrderType.SUPPORT]
            choice = self.policy_rng.random()
            if choice < 0.4 or not moves:
                orders.append(options[0])
            elif choice < 0.9 or not supports:
                orders.append(self.policy_rng.choice(moves))
            else:
                orders.append(self.policy_rng.choice(supports))
        return orders, DecisionTrace(orders, 0.0, 0.0, "Fixed calibration policy")

    def deal_signed(self, commitment, turn):
        # The fixed policy has no negotiator price to record.
        pass


def play_fixed(cfg: GameConfig) -> GameRunner:
    with game_setup(cfg):
        random.seed(cfg.seed)
        agents = [FixedPolicyAgent(p, cfg.seed) for p in players()]
        runner = GameRunner(agents, negotiation_rounds=1)
        runner.config = cfg
        runner.run(verbose=False)
    return runner


def crossfit_baselines(games: List[list]) -> List[float]:
    """Leave one whole game out when fitting each base-rate comparator."""
    counts = [sum(p.observed is not None for p in game) for game in games]
    kept = [sum(p.observed is True for p in game) for game in games]
    total_count, total_kept = sum(counts), sum(kept)
    out = []
    for n, k in zip(counts, kept):
        other_n = total_count - n
        out.append((total_kept - k) / other_n if other_n else 0.5)
    return out


def summarise(games: List[list], seeds: List[int]) -> dict:
    bases = crossfit_baselines(games)
    rows = []
    for seed, points, base in zip(seeds, games, bases):
        graded = [p for p in points if p.observed is not None]
        if not graded:
            rows.append({"seed": seed, "predictions": 0, "model_brier": None,
                         "baseline_brier": None, "difference": None})
            continue
        model = brier_score(graded)
        baseline = sum((base - float(p.observed)) ** 2 for p in graded) / len(graded)
        rows.append({"seed": seed, "predictions": len(graded),
                     "model_brier": model, "baseline_brier": baseline,
                     "difference": model - baseline})
    all_points = [p for game in games for p in game if p.observed is not None]
    n = len(all_points)
    return {
        "policy": "fixed alliances; 75% acceptance; 40% hold, 50% move, 10% support",
        "seeds": seeds,
        "games": len(games),
        "predictions": n,
        "keep_rate": sum(p.observed is True for p in all_points) / n if n else None,
        "model_brier": brier_score(all_points),
        "baseline_brier": (
            sum(r["baseline_brier"] * r["predictions"] for r in rows
                if r["baseline_brier"] is not None) / n if n else None),
        "paired_difference_ci95": mean_ci95([r["difference"] for r in rows]),
        "calibration_error": calibration_error(all_points),
        "reliability_bins": reliability_bins(all_points, bins=8),
        "by_game": rows,
    }
