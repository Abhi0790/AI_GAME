from typing import List, Dict, Any
from src.common.schemas import Player, CommitmentType
from src.engine.adjudicator import CommitmentOutcome
from src.agents.trust.model import TrustModel

class TrustTrace:
    def __init__(self, rule_name: str, facts: str, old_alpha: float, old_beta: float, new_alpha: float, new_beta: float, explanation: str):
        self.rule_name = rule_name
        self.facts = facts
        self.old_alpha = old_alpha
        self.old_beta = old_beta
        self.new_alpha = new_alpha
        self.new_beta = new_beta
        self.explanation = explanation

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
        
        for p in outcome.commitment.players:
            if p == model.owner:
                continue
                
            record = model.get_record(p, c_type)
            old_a, old_b = record.alpha, record.beta
            
            if p in outcome.broken_by:
                if incentive > 0.5:
                    rule_name = "R2: Profitable Betrayal"
                    discount = 0.5
                    explanation = f"Penalty reduced because expected gain was {incentive}"
                else:
                    rule_name = "R1: Gratuitous Betrayal"
                    discount = 1.0
                    explanation = "Full penalty applied because betrayal had little strategic gain"
                    
                record.update(kept=False, discount=discount)
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
