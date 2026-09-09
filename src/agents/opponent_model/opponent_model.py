"""
Opponent Model — tracks each opponent's historical behaviour to predict future moves.

Key capabilities:
  • Maintains per-opponent move statistics (aggression ratio, target preferences).
  • Estimates the most likely persona type based on observed trust-keeping rate.
  • Provides a weighted sampling function that the planner uses instead of
    uniform-random opponent order sampling.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

from src.common.schemas import (
    Player, GameState, Order, OrderType, Commitment, CommitmentType
)
from src.engine.board import get_adjacent, get_all_territories
from src.agents.planner.order_generator import generate_all_order_sets


class OpponentProfile:
    """Running statistics for a single opponent."""

    def __init__(self, player: Player):
        self.player = player
        # Counts
        self.total_orders = 0
        self.move_count = 0
        self.hold_count = 0
        self.support_count = 0
        # Territory-level targeting frequency
        self.target_freq: Dict[str, int] = defaultdict(int)
        # Commitment keeping rate
        self.commitments_seen = 0
        self.commitments_kept = 0

    # ------------------------------------------------------------------
    # Derived features
    # ------------------------------------------------------------------
    @property
    def aggression_ratio(self) -> float:
        """Fraction of orders that were MOVEs (vs HOLD/SUPPORT)."""
        if self.total_orders == 0:
            return 0.5  # uninformative prior
        return self.move_count / self.total_orders

    @property
    def cooperation_ratio(self) -> float:
        """Fraction of orders that were SUPPORTs."""
        if self.total_orders == 0:
            return 0.0
        return self.support_count / self.total_orders

    @property
    def trust_keeping_rate(self) -> float:
        if self.commitments_seen == 0:
            return 0.5
        return self.commitments_kept / self.commitments_seen

    def estimated_persona(self) -> str:
        """Rough persona estimate from observed behaviour."""
        tkr = self.trust_keeping_rate
        agg = self.aggression_ratio

        if tkr > 0.8:
            return "Honest"
        if agg > 0.7:
            return "Opportunist"
        if tkr < 0.4:
            return "Vengeful"
        return "Paranoid"

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def record_orders(self, orders: List[Order]):
        for o in orders:
            if o.player != self.player:
                continue
            self.total_orders += 1
            if o.order_type == OrderType.MOVE:
                self.move_count += 1
                if o.target:
                    self.target_freq[o.target] += 1
            elif o.order_type == OrderType.HOLD:
                self.hold_count += 1
            elif o.order_type == OrderType.SUPPORT:
                self.support_count += 1

    def record_commitment_outcome(self, kept: bool):
        self.commitments_seen += 1
        if kept:
            self.commitments_kept += 1


class OpponentModel:
    """Aggregates per-opponent profiles and exposes a weighted sampler."""

    def __init__(self, my_player: Player):
        self.my_player = my_player
        self.profiles: Dict[Player, OpponentProfile] = {
            p: OpponentProfile(p) for p in Player if p != my_player
        }

    # ------------------------------------------------------------------
    # Observation API
    # ------------------------------------------------------------------
    def observe_orders(self, orders: List[Order]):
        """Call after each turn with the joint orders that were played."""
        for profile in self.profiles.values():
            profile.record_orders(orders)

    def observe_commitment_outcomes(self, outcomes):
        """Call after each turn with CommitmentOutcome list."""
        for o in outcomes:
            for p in o.commitment.players:
                if p == self.my_player:
                    continue
                if p in self.profiles:
                    self.profiles[p].record_commitment_outcome(
                        p not in o.broken_by
                    )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_opponent_orders(
        self, state: GameState, samples: int = 3
    ) -> List[List[Order]]:
        """Return *samples* joint-opponent order sets, weighted by profiles."""
        opponents = [p for p in Player if p != self.my_player]
        all_samples: List[List[Order]] = []

        for _ in range(samples):
            joint: List[Order] = []
            for opp in opponents:
                sets = generate_all_order_sets(state, opp)
                if not sets:
                    continue
                profile = self.profiles[opp]
                chosen = self._weighted_choice(sets, profile)
                joint.extend(chosen)
            all_samples.append(joint)
        return all_samples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _weighted_choice(
        order_sets: List[List[Order]], profile: OpponentProfile
    ) -> List[Order]:
        """Pick an order set with probability proportional to a score
        derived from the opponent's observed tendencies."""
        if not order_sets:
            return []

        weights: List[float] = []
        agg = profile.aggression_ratio

        for oset in order_sets:
            score = 1.0  # base weight
            for o in oset:
                if o.order_type == OrderType.MOVE:
                    # Higher aggression → higher weight for move-heavy sets
                    freq = profile.target_freq.get(o.target, 0)
                    score += agg * 0.5 + freq * 0.3
                elif o.order_type == OrderType.SUPPORT:
                    score += profile.cooperation_ratio * 0.5
                # HOLD is neutral — leave score as is
            weights.append(max(score, 0.01))

        total = sum(weights)
        probs = [w / total for w in weights]

        # Use random.choices with weights
        idx = random.choices(range(len(order_sets)), weights=probs, k=1)[0]
        return order_sets[idx]
