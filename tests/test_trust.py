"""Tests for the Trust Model and Rule Engine."""

import pytest
from src.common.schemas import Player, CommitmentType, Commitment
from src.agents.trust.model import TrustModel, TrustRecord
from src.agents.trust.rules import RuleEngine, OutcomeRule, GossipRule, TrustTrace
from src.engine.adjudicator import CommitmentOutcome


# ── TrustRecord unit tests ──────────────────────────────────────────────

class TestTrustRecord:
    def test_initial_expected_value(self):
        """Default prior (1,1) → 0.5."""
        r = TrustRecord(1.0, 1.0)
        assert r.get_expected_value() == pytest.approx(0.5)

    def test_update_kept(self):
        r = TrustRecord(1.0, 1.0)
        r.update(kept=True)
        # alpha=2, beta=1 → 2/3
        assert r.get_expected_value() == pytest.approx(2.0 / 3.0)

    def test_update_broken(self):
        r = TrustRecord(1.0, 1.0)
        r.update(kept=False)
        # alpha=1, beta=2 → 1/3
        assert r.get_expected_value() == pytest.approx(1.0 / 3.0)

    def test_update_with_discount(self):
        r = TrustRecord(1.0, 1.0)
        r.update(kept=False, discount=0.5)
        # alpha=1, beta=1.5 → 1/2.5 = 0.4
        assert r.get_expected_value() == pytest.approx(0.4)

    def test_many_kept_drives_trust_high(self):
        r = TrustRecord(1.0, 1.0)
        for _ in range(20):
            r.update(kept=True)
        assert r.get_expected_value() > 0.9


# ── TrustModel tests ────────────────────────────────────────────────────

class TestTrustModel:
    def test_initial_reliability(self):
        m = TrustModel(Player.RED)
        assert m.get_reliability(Player.BLUE, CommitmentType.ALLIANCE) == pytest.approx(0.5)

    def test_paranoid_prior(self):
        """Paranoid prior (1, 3) → 0.25."""
        m = TrustModel(Player.RED, prior_alpha=1.0, prior_beta=3.0)
        assert m.get_reliability(Player.BLUE, CommitmentType.ALLIANCE) == pytest.approx(0.25)

    def test_update_from_kept_outcome(self):
        m = TrustModel(Player.RED)
        c = Commitment(id="1", commitment_type=CommitmentType.ALLIANCE,
                       players=[Player.RED, Player.BLUE], valid_until_turn=5)
        outcome = CommitmentOutcome(commitment=c, kept=True, broken_by=[])
        m.update_from_outcome(outcome)
        # BLUE kept → trust goes up from 0.5
        assert m.get_reliability(Player.BLUE, CommitmentType.ALLIANCE) > 0.5

    def test_update_from_broken_outcome(self):
        m = TrustModel(Player.RED)
        c = Commitment(id="1", commitment_type=CommitmentType.ALLIANCE,
                       players=[Player.RED, Player.BLUE], valid_until_turn=5)
        outcome = CommitmentOutcome(commitment=c, kept=False, broken_by=[Player.BLUE])
        m.update_from_outcome(outcome)
        assert m.get_reliability(Player.BLUE, CommitmentType.ALLIANCE) < 0.5

    def test_tracks_own_public_record(self):
        m = TrustModel(Player.RED)
        c = Commitment(id="1", commitment_type=CommitmentType.ALLIANCE,
                       players=[Player.RED, Player.BLUE], valid_until_turn=5)
        outcome = CommitmentOutcome(commitment=c, kept=False, broken_by=[Player.RED])
        m.update_from_outcome(outcome)
        # Outcomes are public, so an agent scores its own record off the same
        # evidence everyone else has. The planner needs it to price deltaP.
        own = m.get_record(Player.RED, CommitmentType.ALLIANCE)
        assert own.beta > 1.0
        assert m.reputation_drop(Player.RED, CommitmentType.ALLIANCE) > 0

    def test_gossip_reduces_trust(self):
        m = TrustModel(Player.RED)
        # First make sender trusted
        m.get_record(Player.GREEN, CommitmentType.ALLIANCE).alpha = 5.0
        m.get_record(Player.GREEN, CommitmentType.ALLIANCE).beta = 1.0
        # Apply gossip
        m.apply_gossip(Player.GREEN, Player.BLUE, CommitmentType.ALLIANCE)
        # BLUE's trust should decrease
        assert m.get_reliability(Player.BLUE, CommitmentType.ALLIANCE) < 0.5


# ── Rule Engine tests ────────────────────────────────────────────────────

class TestRuleEngine:
    def test_outcome_rule_kept(self):
        m = TrustModel(Player.RED)
        c = Commitment(id="1", commitment_type=CommitmentType.ALLIANCE,
                       players=[Player.RED, Player.BLUE], valid_until_turn=5)
        outcome = CommitmentOutcome(commitment=c, kept=True, broken_by=[])
        rule = OutcomeRule()
        traces = rule.evaluate(m, {'outcome': outcome, 'incentive_to_defect': 0.0})
        assert {t.rule_name for t in traces} == {"R0: Kept Promise"}

    def test_outcome_rule_gratuitous_betrayal(self):
        m = TrustModel(Player.RED)
        c = Commitment(id="1", commitment_type=CommitmentType.ALLIANCE,
                       players=[Player.RED, Player.BLUE], valid_until_turn=5)
        outcome = CommitmentOutcome(commitment=c, kept=False, broken_by=[Player.BLUE])
        rule = OutcomeRule()
        traces = rule.evaluate(m, {'outcome': outcome, 'incentive_to_defect': 0.1})
        blamed = [t for t in traces if "broke" in t.facts]
        assert len(blamed) == 1
        assert blamed[0].rule_name == "R1: Gratuitous Betrayal"

    def test_outcome_rule_profitable_betrayal(self):
        m = TrustModel(Player.RED)
        c = Commitment(id="1", commitment_type=CommitmentType.ALLIANCE,
                       players=[Player.RED, Player.BLUE], valid_until_turn=5)
        outcome = CommitmentOutcome(commitment=c, kept=False, broken_by=[Player.BLUE])
        rule = OutcomeRule()
        traces = rule.evaluate(m, {'outcome': outcome, 'incentive_to_defect': 0.8})
        blamed = [t for t in traces if "broke" in t.facts]
        assert len(blamed) == 1
        assert blamed[0].rule_name == "R2: Profitable Betrayal"

    def test_gossip_rule_trusted_sender(self):
        m = TrustModel(Player.RED)
        m.get_record(Player.GREEN, CommitmentType.ALLIANCE).alpha = 5.0
        m.get_record(Player.GREEN, CommitmentType.ALLIANCE).beta = 1.0
        rule = GossipRule()
        facts = {'gossip': {'sender': Player.GREEN, 'accused': Player.BLUE,
                            'commitment_type': CommitmentType.ALLIANCE}}
        traces = rule.evaluate(m, facts)
        assert len(traces) == 1
        assert traces[0].rule_name == "R3: Trusted Gossip"

    def test_gossip_rule_untrusted_sender_ignored(self):
        m = TrustModel(Player.RED)
        # Sender has low trust
        m.get_record(Player.GREEN, CommitmentType.ALLIANCE).alpha = 1.0
        m.get_record(Player.GREEN, CommitmentType.ALLIANCE).beta = 5.0
        rule = GossipRule()
        facts = {'gossip': {'sender': Player.GREEN, 'accused': Player.BLUE,
                            'commitment_type': CommitmentType.ALLIANCE}}
        traces = rule.evaluate(m, facts)
        assert len(traces) == 0  # Should be ignored

    def test_rule_engine_processes_all_rules(self):
        m = TrustModel(Player.RED)
        engine = RuleEngine([OutcomeRule(), GossipRule()])
        c = Commitment(id="1", commitment_type=CommitmentType.ALLIANCE,
                       players=[Player.RED, Player.BLUE], valid_until_turn=5)
        outcome = CommitmentOutcome(commitment=c, kept=True, broken_by=[])
        traces = engine.process(m, {'outcome': outcome})
        assert len(traces) >= 1
