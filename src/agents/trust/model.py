from typing import Dict, List, Tuple
from src.common.schemas import Player, CommitmentType, Commitment
from src.engine.adjudicator import CommitmentOutcome
import numpy as np

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
        # Incentive to defect determines how much we update beta on a break
        # If incentive is high, it was a "rational" betrayal. We still penalize, but maybe less.
        # "A profitable betrayal should damage trust less than an inexplicable/gratuitous betrayal."
        # discount = 1.0 if gratuitous (incentive = 0)
        # discount = 0.5 if profitable (incentive > 0)
        
        for p in outcome.commitment.players:
            if p == self.owner: continue # don't track trust in ourselves this way
            
            # If the player broke the commitment
            if p in outcome.broken_by:
                discount = 0.5 if incentive_to_defect > 0 else 1.0
                record = self.get_record(p, outcome.commitment.commitment_type)
                record.update(kept=False, discount=discount)
                # Explanation logic goes here later
            else:
                record = self.get_record(p, outcome.commitment.commitment_type)
                record.update(kept=True, discount=1.0)
                
    def apply_gossip(self, gossip_sender: Player, accused: Player, c_type: CommitmentType):
        # Forward chaining rule base - Rule R3
        # If trust in gossip_sender is high, discount accused trust
        sender_trust = sum([r.get_expected_value() for r in self.records.get(gossip_sender, {}).values()]) / max(1, len(self.records.get(gossip_sender, {})))
        if sender_trust > 0.6:
            record = self.get_record(accused, c_type)
            record.alpha *= 0.9 # decrease alpha slightly to reflect suspicion
            record.beta += 0.1

