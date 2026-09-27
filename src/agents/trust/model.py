from typing import Dict, List, Tuple
from src.common.schemas import Player, CommitmentType, Commitment, CommitmentOutcome
import numpy as np

# How much of the reputation hit a profitable betrayal is forgiven.
# w = 1 - lambda * incentive  (slide 6). lambda = 0 makes every break cost the
# same; lambda = 1 makes a maximally profitable break free.
LAMBDA_INCENTIVE = 0.6
MIN_BREAK_WEIGHT = 0.1


def break_weight(incentive: float) -> float:
    """w in "broken: beta + w". A gratuitous betrayal costs a full point of
    reputation; a lucrative one costs less, because it explains itself."""
    incentive = min(1.0, max(0.0, incentive))
    return max(MIN_BREAK_WEIGHT, 1.0 - LAMBDA_INCENTIVE * incentive)


class TrustRecord:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        
    def get_expected_value(self) -> float:
        return self.alpha / (self.alpha + self.beta)
        
    def update(self, kept: bool, discount: float = 1.0):
        if kept:
            self.alpha += discount
        else:
            self.beta += discount

class TrustModel:
    def __init__(self, owner: Player, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.owner = owner
        # target_player -> commitment_type -> TrustRecord
        self.records: Dict[Player, Dict[CommitmentType, TrustRecord]] = {}
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
    def get_record(self, player: Player, c_type: CommitmentType) -> TrustRecord:
        if player not in self.records:
            self.records[player] = {}
        if c_type not in self.records[player]:
            self.records[player][c_type] = TrustRecord(self.prior_alpha, self.prior_beta)
        return self.records[player][c_type]
        
    def get_reliability(self, player: Player, c_type: CommitmentType) -> float:
        return self.get_record(player, c_type).get_expected_value()

    def update_from_outcome(self, outcome: CommitmentOutcome, incentive_to_defect: float = 0.0):
        """Apply one graded commitment to the Beta records.

        Own record included on purpose: the engine publishes every outcome, so
        every player scores my record off the same public evidence. That record
        is the best estimate I have of how much reputation I stand to lose,
        which is what the planner prices a betrayal against.
        """
        for p in outcome.commitment.players:
            record = self.get_record(p, outcome.commitment.commitment_type)
            if p in outcome.broken_by:
                record.update(kept=False, discount=break_weight(incentive_to_defect))
            else:
                record.update(kept=True, discount=1.0)

    def reputation_drop(self, player: Player, c_type: CommitmentType,
                        incentive: float = 0.0) -> float:
        """Delta-P: how far belief that *player* keeps this kind of promise
        falls if they break it now. Beta(a, b) -> Beta(a, b + w)."""
        r = self.get_record(player, c_type)
        w = break_weight(incentive)
        return r.get_expected_value() - r.alpha / (r.alpha + r.beta + w)
                
    def apply_gossip(self, gossip_sender: Player, accused: Player, c_type: CommitmentType):
        # Forward chaining rule base - Rule R3
        # If trust in gossip_sender is high, discount accused trust
        sender_trust = sum([r.get_expected_value() for r in self.records.get(gossip_sender, {}).values()]) / max(1, len(self.records.get(gossip_sender, {})))
        if sender_trust > 0.6:
            record = self.get_record(accused, c_type)
            record.alpha *= 0.9 # decrease alpha slightly to reflect suspicion
            record.beta += 0.1

