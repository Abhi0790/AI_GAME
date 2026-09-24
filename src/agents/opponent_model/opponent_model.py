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
from src.engine.board import get_adjacent, get_all_territories, players
from src.engine.orders import generate_all_order_sets
from src.engine.adjudicator import verify_commitments


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
            p: OpponentProfile(p) for p in players() if p != my_player
        }
        # (board, opponent, their deals) -> which of their order sets break a
        # promise. Recomputed once per position rather than per playout.
        self._break_mask: Dict[tuple, List[bool]] = {}

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
    def retaliation_rate(self, player: Player) -> float:
        """How likely this player is to actually hit back, in [0, 1].

        Used to price a THREAT: a player who moves is frightening, a player
        who only talks is not.

        ponytail: aggression is a proxy for follow-through. Track
        threat-then-attack pairs directly if the deterrence figure needs it.
        """
        profile = self.profiles.get(player)
        if profile is None or profile.total_orders == 0:
            return 0.5
        return min(1.0, profile.aggression_ratio)

    def _breaks(self, state: GameState, opp: Player, sets: List[List[Order]],
                deals: List[Commitment]) -> List[bool]:
        """Which of this opponent's order sets break one of their promises."""
        key = (state.turn,
               tuple(sorted((u.territory, u.player.value) for u in state.units)),
               tuple(sorted((t, o.value) for t, o in state.supply_centers.items() if o)),
               opp, tuple(sorted(c.id for c in deals)))
        hit = self._break_mask.get(key)
        if hit is None:
            hit = [any(not o.kept and opp in o.broken_by
                       for o in verify_commitments(state, deals, s))
                   for s in sets]
            if len(self._break_mask) > 256:
                self._break_mask.clear()
            self._break_mask[key] = hit
        return hit

    def sample_opponent_orders(
        self, state: GameState, samples: int = 3,
        commitments: Optional[List[Commitment]] = None, trust_model=None,
        incentives: Optional[Dict[Player, float]] = None,
    ) -> List[List[Order]]:
        """Return *samples* joint-opponent order sets, weighted by profiles.

        Weights are built once per opponent and then drawn from `samples`
        times; scoring all 81 order sets inside the sample loop made 32
        sampled worlds cost 32x more than it needed to.

        With *commitments* and a *trust_model*, the draw is also conditioned on
        what the opponent has promised: order sets that honour their live deals
        are weighted by P(they keep), the rest by 1 - P(they keep). Without
        this the planner predicted exactly the same behaviour from a partner
        whether or not a deal existed, so a promise could only ever appear as a
        penalty on my own orders and never as a reason to expect theirs to
        change. Reciprocity becomes a belief instead of a fine.
        """
        opponents = [p for p in players() if p != self.my_player]
        draws: Dict[Player, List[List[Order]]] = {}
        for opp in opponents:
            sets = generate_all_order_sets(state, opp)
            if not sets:
                continue
            weights = self._weights(sets, self.profiles[opp])

            deals = [c for c in (commitments or []) if opp in c.players]
            if deals and trust_model is not None:
                # One P(keeps) for the player, averaged over their live deals:
                # the draw is over whole order sets, and a set that breaks
                # anything is already off the honouring branch.
                #
                # ponytail: one averaged probability and a binary
                # honours/breaks split. Weight each set by the product over the
                # specific deals it breaks if partial defection (keep the DMZ,
                # drop the support) turns out to matter.
                iota = (incentives or {}).get(opp, 0.0)
                p_keep = sum(trust_model.p_keeps(opp, c.commitment_type, iota,
                                                 toward=self.my_player)
                             for c in deals) / len(deals)
                # Each branch is normalised, so the draw breaks with probability
                # 1 - P(keeps) however many order sets fall on either side.
                breaks = self._breaks(state, opp, sets, deals)
                w_break = sum(w for w, b in zip(weights, breaks) if b)
                w_keep = sum(weights) - w_break
                if w_break and w_keep:
                    weights = [w * ((1.0 - p_keep) / w_break if b else p_keep / w_keep)
                               for w, b in zip(weights, breaks)]

            draws[opp] = random.choices(sets, weights=weights, k=samples)

        return [
            [o for opp in opponents for o in draws.get(opp, [[]])[i]]
            for i in range(samples)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _weights(
        order_sets: List[List[Order]], profile: OpponentProfile
    ) -> List[float]:
        """Score every order set by how much it looks like what this
        opponent has actually been doing."""
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

        return weights

    def _weighted_choice(
        self, order_sets: List[List[Order]], profile: OpponentProfile
    ) -> List[Order]:
        """Single draw. Kept for callers that want one order set."""
        if not order_sets:
            return []
        return random.choices(order_sets, weights=self._weights(order_sets, profile), k=1)[0]
