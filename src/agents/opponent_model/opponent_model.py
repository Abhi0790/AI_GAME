"""Opponent model: move statistics as a prior, reweighted by trust, stated
intentions and, at the root of a decision, level-1 value."""
import math

from typing import Dict, List, Optional
from collections import defaultdict
import random

from src.common.schemas import (
    Player, GameState, Order, OrderType, Commitment
)
from src.engine.board import players, active
from src.engine.orders import generate_all_order_sets
from src.engine.adjudicator import verify_commitments, resolve

# Rationality of the level-1 model, per centre of gain: a set worth one centre
# more than another is e^LEVEL1_BETA times as likely. 0 turns it off.
LEVEL1_BETA = 3.0

# (board and rules, valuation knobs, turn, position, opponent) -> value of each
# of their order sets. Pure in those, so shared by every agent and decision.
_GAIN_CACHE: dict = {}


def level1_values(state: GameState, opp: Player, sets: List[List[Order]]) -> List[float]:
    """Each order set's value to `opp` with everyone else holding.

    ponytail: one reference world (all hold). Average over sampled level-0
    worlds if opponents look too greedy.
    """
    from src.agents.planner import planner as P
    knobs = (P.CENTRE_VALUE, P.UNIT_VALUE, P.THREATENS_VALUE, P.UNDER_THREAT_VALUE,
             P.MOBILITY_VALUE, P.GAME_END_VALUE, P.SIEGE_CREDIT)
    key = (repr(active()), knobs, state.turn,
           tuple(sorted((u.territory, u.player.value) for u in state.units)),
           tuple(sorted((t, o.value) for t, o in state.supply_centers.items() if o)), opp)
    hit = _GAIN_CACHE.get(key)
    if hit is None:
        hit = [P.evaluate_state(resolve(state, s)[0], opp) for s in sets]
        if len(_GAIN_CACHE) > 256:
            _GAIN_CACHE.clear()
        _GAIN_CACHE[key] = hit
    return hit


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
        # Set by the Agent each turn: moves each player announced, and
        # P(their words match their orders).
        self.intents: Dict[Player, List[tuple]] = {}
        self.candour: Dict[Player, float] = {}

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
        """How likely this player is to hit back, in [0, 1].

        ponytail: aggression proxies follow-through; track threat-then-attack pairs if needed.
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
        incentives: Optional[Dict[Player, float]] = None, rational: bool = False,
    ) -> List[List[Order]]:
        """*samples* joint opponent order sets, weighted by profile (and level-1
        value if *rational*); honouring their deals gets P(keeps) of the mass,
        doing what they announced gets P(candour)."""
        opponents = [p for p in players() if p != self.my_player]
        draws: Dict[Player, List[List[Order]]] = {}
        for opp in opponents:
            sets = generate_all_order_sets(state, opp)
            if not sets:
                continue
            weights = self._weights(sets, self.profiles[opp])
            if rational and LEVEL1_BETA:
                values = level1_values(state, opp, sets)
                top = max(values)
                weights = [w * math.exp(LEVEL1_BETA * (v - top))
                           for w, v in zip(weights, values)]

            deals = [c for c in (commitments or []) if opp in c.players]
            if deals and trust_model is not None:
                # ponytail: one P(keeps) averaged over their deals and a binary
                # honours/breaks split; weight per broken deal if partial defection matters.
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

            said = self.intents.get(opp)
            if said:
                # Sets doing everything announced get P(candour) between them.
                c = self.candour.get(opp, 0.5)
                match = [all(any(o.unit_territory == f and o.order_type == OrderType.MOVE
                                 and o.target == t for o in s) for f, t in said)
                         for s in sets]
                w_yes = sum(w for w, m in zip(weights, match) if m)
                w_no = sum(weights) - w_yes
                if w_yes and w_no:
                    weights = [w * (c / w_yes if m else (1.0 - c) / w_no)
                               for w, m in zip(weights, match)]

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
