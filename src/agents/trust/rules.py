from typing import List, Dict, Any
from src.common.schemas import Player, CommitmentType, CommitmentOutcome, TrustTrace
from src.agents.trust.model import TrustModel, break_weight, LAMBDA_INCENTIVE

class TrustRule:
    def evaluate(self, model: TrustModel, facts: Dict[str, Any]) -> List[TrustTrace]:
        pass

class OutcomeRule(TrustRule):
    def evaluate(self, model: TrustModel, facts: Dict[str, Any]) -> List[TrustTrace]:
        traces = []
        if 'outcome' not in facts:
            return traces
            
        outcome: CommitmentOutcome = facts['outcome']
        incentive = facts.get('incentive_to_defect', 0.0)
        c_type = outcome.commitment.commitment_type
        
        # The owner's own record is kept too: it is public evidence, and it is
        # what the planner prices its own betrayals against.
        for p in outcome.commitment.players:
            record = model.get_record(p, c_type)
            old_a, old_b = record.alpha, record.beta
            
            if p in outcome.broken_by:
                w = break_weight(incentive)
                if incentive > 0.5:
                    rule_name = "R2: Profitable Betrayal"
                    explanation = (f"w = 1 - {LAMBDA_INCENTIVE}x{incentive:.2f} = {w:.2f}: "
                                   f"penalty reduced, the break paid for itself")
                else:
                    rule_name = "R1: Gratuitous Betrayal"
                    explanation = (f"w = 1 - {LAMBDA_INCENTIVE}x{incentive:.2f} = {w:.2f}: "
                                   f"near-full penalty, the break gained little")

                record.update(kept=False, discount=w)
                traces.append(TrustTrace(rule_name, f"{p} broke {c_type}", old_a, old_b, record.alpha, record.beta, explanation))
            else:
                rule_name = "R0: Kept Promise"
                record.update(kept=True, discount=1.0)
                traces.append(TrustTrace(rule_name, f"{p} kept {c_type}", old_a, old_b, record.alpha, record.beta, "Trust increased"))
                
        return traces

class GossipRule(TrustRule):
    def evaluate(self, model: TrustModel, facts: Dict[str, Any]) -> List[TrustTrace]:
        traces = []
        if 'gossip' not in facts:
            return traces
            
        sender = facts['gossip']['sender']
        accused = facts['gossip']['accused']
        c_type = facts['gossip']['commitment_type']
        
        if sender == model.owner or accused == model.owner:
            return traces
            
        # R3: Gossip acceptance depends on sender's general trust
        sender_trust = 0
        records = model.records.get(sender, {})
        if records:
            sender_trust = sum(r.get_expected_value() for r in records.values()) / len(records)
        else:
            sender_trust = model.prior_alpha / (model.prior_alpha + model.prior_beta)
            
        if sender_trust > 0.6:
            record = model.get_record(accused, c_type)
            old_a, old_b = record.alpha, record.beta
            record.alpha *= 0.9
            record.beta += 0.1
            traces.append(TrustTrace("R3: Trusted Gossip", f"{sender} accused {accused}", old_a, old_b, record.alpha, record.beta, f"Discounted trust by 0.1 because {sender} is highly trusted"))
            
        return traces

class RuleEngine:
    def __init__(self, rules: List[TrustRule]):
        self.rules = rules
        
    def process(self, model: TrustModel, facts: Dict[str, Any]) -> List[TrustTrace]:
        all_traces = []
        for rule in self.rules:
            traces = rule.evaluate(model, facts)
            all_traces.extend(traces)
        return all_traces
