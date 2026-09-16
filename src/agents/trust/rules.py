from typing import List, Dict, Any
from src.common.schemas import Player, CommitmentType, CommitmentOutcome, TrustTrace
from src.agents.trust.model import TrustModel, break_weight, LAMBDA_INCENTIVE

# How hard a refuted accusation hits the accuser. Lying is graded harder than
# betrayal on purpose: a broken promise is one data point about one kind of
# deal, a proven lie is a direct observation of the speaker's honesty.
FALSE_ACCUSATION_WEIGHT = 2.0
GOSSIP_TRUST_THRESHOLD = 0.6


class TrustRule:
    def evaluate(self, model: TrustModel, facts: Dict[str, Any]) -> List[TrustTrace]:
        return []


class OutcomeRule(TrustRule):
    """R0/R1/R2 — first-hand evidence: a deal this agent was party to."""

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
    """R3 — second-hand evidence: somebody says X betrayed them.

    Only fires on an accusation the engine confirmed. An unverified or
    refuted claim moves nobody's opinion of the accused; that is what makes
    lying a strategy with a downside rather than free damage.
    """

    def evaluate(self, model: TrustModel, facts: Dict[str, Any]) -> List[TrustTrace]:
        traces = []
        if 'gossip' not in facts:
            return traces

        g = facts['gossip']
        sender, accused, c_type = g['sender'], g['accused'], g['commitment_type']
        verdict = g.get('verdict')

        if sender == model.owner or accused == model.owner:
            return traces
        if verdict == "REFUTED":
            return traces  # R4 handles the accuser; the accused is untouched

        sender_trust = model.general_trust(sender)
        if sender_trust <= GOSSIP_TRUST_THRESHOLD:
            return traces

        record = model.get_record(accused, c_type)
        old_a, old_b = record.alpha, record.beta
        record.alpha *= 0.9
        record.beta += 0.1
        traces.append(TrustTrace(
            "R3: Trusted Gossip", f"{sender} accused {accused} ({verdict or 'unverified'})",
            old_a, old_b, record.alpha, record.beta,
            f"Discounted trust in {accused.value}: {sender.value} is trusted at "
            f"{sender_trust:.2f} and the engine did not refute the claim"))
        return traces


class FalseAccusationRule(TrustRule):
    """R4 — the accuser is the one on trial when the engine refutes them.

    Without this there is no cost to lying, so no agent has to weigh whether
    an accusation is worth making. The hit lands on every kind of promise the
    liar has outstanding, because what was observed is the liar, not the deal.
    """

    def evaluate(self, model: TrustModel, facts: Dict[str, Any]) -> List[TrustTrace]:
        traces = []
        if 'gossip' not in facts:
            return traces
        g = facts['gossip']
        if g.get('verdict') != "REFUTED":
            return traces

        # The owner's own record is updated too, for the same reason
        # OutcomeRule keeps it: a refuted accusation is public, so this is
        # what my standing now *is*, and an agent that cannot see its own
        # reputation collapse will keep lying forever.
        sender = g['sender']

        for c_type in CommitmentType:
            record = model.get_record(sender, c_type)
            old_a, old_b = record.alpha, record.beta
            record.update(kept=False, discount=FALSE_ACCUSATION_WEIGHT)
            traces.append(TrustTrace(
                "R4: False Accusation", f"{sender} accused {g['accused']} falsely",
                old_a, old_b, record.alpha, record.beta,
                f"Engine refuted the claim: beta += {FALSE_ACCUSATION_WEIGHT} on "
                f"{c_type.value}, a proven lie outweighs one broken deal"))
        return traces


class RuleEngine:
    def __init__(self, rules: List[TrustRule]):
        self.rules = rules

    def process(self, model: TrustModel, facts: Dict[str, Any]) -> List[TrustTrace]:
        all_traces = []
        for rule in self.rules:
            all_traces.extend(rule.evaluate(model, facts))
        return all_traces
