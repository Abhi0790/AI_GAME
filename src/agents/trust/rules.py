from typing import List, Dict, Any, Optional
from src.common.schemas import (
    CommitmentOutcome, TrustTrace, obligated_parties,
)
from src.agents.trust.model import TrustModel, break_weight, LAMBDA_INCENTIVE


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
        members = outcome.commitment.players
        others = lambda p: [q for q in members if q != p]
        # Only a party that owed something can have kept or broken it. A
        # SUPPORT binds players[0]; an EXCHANGE binds one side per leg.
        owing = obligated_parties(outcome.commitment, facts.get('turn', 0))

        for p in owing:
            record = model.get_record(p, c_type)
            old_a, old_b = record.alpha, record.beta
            counterparties = ", ".join(q.value for q in others(p)) or "nobody"

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

                # Reputation and the specific relationships both move: the
                # people who were in the deal learn more about this player
                # than the table does.
                model.observe(p, members, c_type, kept=False, discount=w)
                traces.append(TrustTrace(
                    rule_name, f"{p.value} broke {c_type.value} with {counterparties}",
                    old_a, old_b, record.alpha, record.beta, explanation))
            else:
                model.observe(p, members, c_type, kept=True, discount=1.0)
                traces.append(TrustTrace(
                    "R0: Kept Promise",
                    f"{p.value} kept {c_type.value} with {counterparties}",
                    old_a, old_b, record.alpha, record.beta, "Trust increased"))

        return traces


class PublicRecordRule(TrustRule):
    """R5 — the table's evidence: a public deal between two other players.

    The engine publishes the verdict on every non-private commitment, so an
    agent does not need to be in a deal to see it broken. That is what makes
    a *reputation* distinct from a relationship: the general record absorbs
    everything anyone did in the open, while the pair record stays first-hand.

    Without this the two layers were built from identical evidence and the
    pair record could never disagree with the reputation. A private break is
    invisible to a third party and so moves nothing here.
    """

    def evaluate(self, model: TrustModel, facts: Dict[str, Any]) -> List[TrustTrace]:
        traces = []
        outcome: Optional[CommitmentOutcome] = facts.get('public_outcome')
        if outcome is None or outcome.commitment.private:
            return traces
        if model.owner in outcome.commitment.players:
            return traces  # first-hand; OutcomeRule owns it

        c_type = outcome.commitment.commitment_type
        for p in obligated_parties(outcome.commitment, facts.get('turn', 0)):
            record = model.get_record(p, c_type)
            old_a, old_b = record.alpha, record.beta
            broke = p in outcome.broken_by
            # Reputation only. Watching two other players deal says nothing
            # about how either of them treats *me*.
            record.update(kept=not broke, discount=1.0)
            traces.append(TrustTrace(
                "R5: Public Record",
                f"{p.value} {'broke' if broke else 'kept'} a public "
                f"{c_type.value} with "
                f"{', '.join(q.value for q in outcome.commitment.players if q != p)}",
                old_a, old_b, record.alpha, record.beta,
                "Seen in the open, so it moves their standing with everyone"))
        return traces


class RuleEngine:
    def __init__(self, rules: List[TrustRule]):
        self.rules = rules

    def process(self, model: TrustModel, facts: Dict[str, Any]) -> List[TrustTrace]:
        all_traces = []
        for rule in self.rules:
            all_traces.extend(rule.evaluate(model, facts))
        return all_traces
