"""Tests for the Trust Model and Rule Engine."""

import pytest
from src.common.schemas import Player, CommitmentType, Commitment
from src.agents.trust.model import (
    TrustModel, TrustRecord, EVIDENCE_DECAY, p_keeps_given,
)
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
        # Evidence decays before the new observation lands:
        # alpha = d + 1, beta = d.
        d = EVIDENCE_DECAY
        assert r.get_expected_value() == pytest.approx((d + 1) / (2 * d + 1))

    def test_update_broken(self):
        r = TrustRecord(1.0, 1.0)
        r.update(kept=False)
        d = EVIDENCE_DECAY
        assert r.get_expected_value() == pytest.approx(d / (2 * d + 1))

    def test_update_with_discount(self):
        r = TrustRecord(1.0, 1.0)
        r.update(kept=False, discount=0.5)
        d = EVIDENCE_DECAY
        assert r.get_expected_value() == pytest.approx(d / (2 * d + 0.5))

    def test_evidence_decays_so_a_record_never_hardens(self):
        """The bug behind issue #16: without decay a settled Beta moves by
        0.01 on a betrayal and deltaP stops deterring anything."""
        r = TrustRecord(1.0, 1.0)
        for _ in range(30):
            r.update(kept=True)
        before = r.get_expected_value()
        r.update(kept=False, discount=1.0)
        assert before - r.get_expected_value() > 0.1

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


# ── Bayesian network (issue #10) ─────────────────────────────────────────

class TestTrustNetwork:
    def test_incentive_changes_the_answer(self):
        """The whole point of the network: identical reliability, different
        temptation, different prediction. A Beta mean cannot do this."""
        m = TrustModel(Player.RED)
        calm = m.p_keeps(Player.BLUE, CommitmentType.ALLIANCE, 0.0)
        tempted = m.p_keeps(Player.BLUE, CommitmentType.ALLIANCE, 1.0)
        assert tempted < calm

    def test_reliability_still_moves_the_answer(self):
        m = TrustModel(Player.RED)
        m.get_record(Player.BLUE, CommitmentType.ALLIANCE).alpha = 9.0
        m.get_record(Player.GREEN, CommitmentType.ALLIANCE).beta = 9.0
        assert m.p_keeps(Player.BLUE, CommitmentType.ALLIANCE, 0.0) > \
               m.p_keeps(Player.GREEN, CommitmentType.ALLIANCE, 0.0)

    def test_probabilities_stay_in_range(self):
        for r in (0.0, 0.3, 0.5, 0.9, 1.0):
            for i in (0.0, 0.05, 0.3, 1.0):
                assert 0.0 <= p_keeps_given(r, i) <= 1.0

    def test_reputation_drop_is_never_negative(self):
        m = TrustModel(Player.RED)
        for _ in range(5):
            m.get_record(Player.BLUE, CommitmentType.DMZ).update(kept=True)
        for i in (0.0, 0.5, 1.0):
            assert m.reputation_drop(Player.BLUE, CommitmentType.DMZ, i) >= 0.0

    def test_snapshot_shows_priors_before_any_evidence(self):
        """The trust panel was blank on turn one; a prior is still a belief."""
        m = TrustModel(Player.RED, prior_alpha=1.0, prior_beta=3.0)
        snap = m.snapshot()
        assert len(snap) == len(Player) * len(CommitmentType)
        assert all(abs(b.expected_reliability - 0.25) < 1e-9 for b in snap)


# ── Lying and the false-accusation rule (issue #9) ───────────────────────

class TestFalseAccusation:
    def _fire(self, verdict, owner=Player.RED, sender=Player.BLUE):
        from src.agents.trust.rules import FalseAccusationRule
        m = TrustModel(owner)
        before = m.get_reliability(sender, CommitmentType.ALLIANCE)
        traces = FalseAccusationRule().evaluate(m, {'gossip': {
            'sender': sender, 'accused': Player.GREEN,
            'commitment_type': CommitmentType.ALLIANCE, 'verdict': verdict}})
        return m, before, traces

    def test_a_refuted_accuser_loses_standing(self):
        m, before, traces = self._fire("REFUTED")
        assert traces
        assert m.get_reliability(Player.BLUE, CommitmentType.ALLIANCE) < before

    def test_a_confirmed_accuser_is_untouched(self):
        m, before, traces = self._fire("CONFIRMED")
        assert not traces
        assert m.get_reliability(Player.BLUE, CommitmentType.ALLIANCE) == before

    def test_the_hit_lands_on_every_kind_of_promise(self):
        """What was observed is the liar, not the deal."""
        m, _before, _t = self._fire("REFUTED")
        for c_type in CommitmentType:
            assert m.get_reliability(Player.BLUE, c_type) < 0.5

    def test_a_liar_sees_their_own_reputation_fall(self):
        """Otherwise nothing stops them lying again next turn."""
        m, _b, traces = self._fire("REFUTED", owner=Player.BLUE, sender=Player.BLUE)
        assert traces
        assert m.get_reliability(Player.BLUE, CommitmentType.ALLIANCE) < 0.5

    def test_a_refuted_claim_does_not_damage_the_accused(self):
        from src.agents.trust.rules import GossipRule
        m = TrustModel(Player.RED)
        m.get_record(Player.BLUE, CommitmentType.ALLIANCE).alpha = 9.0  # trusted liar
        before = m.get_reliability(Player.GREEN, CommitmentType.ALLIANCE)
        GossipRule().evaluate(m, {'gossip': {
            'sender': Player.BLUE, 'accused': Player.GREEN,
            'commitment_type': CommitmentType.ALLIANCE, 'verdict': "REFUTED"}})
        assert m.get_reliability(Player.GREEN, CommitmentType.ALLIANCE) == before


class TestEngineVerifiesAccusations:
    def _runner(self):
        from src.engine.runner import GameRunner
        from src.agents.agent import Agent
        return GameRunner([Agent(p, "Honest") for p in Player])

    def test_a_true_accusation_is_confirmed(self):
        from src.common.schemas import (
            Commitment, CommitmentOutcome, Message, MessageType,
        )
        c = Commitment(id="c", commitment_type=CommitmentType.ALLIANCE,
                       players=[Player.RED, Player.BLUE], valid_until_turn=5)
        outcome = CommitmentOutcome(commitment=c, kept=False, broken_by=[Player.BLUE])
        msg = Message(id="m", sender=Player.RED, message_type=MessageType.BROADCAST,
                      broadcast_kind="BETRAYED", broadcast_target=Player.BLUE,
                      commitment_type=CommitmentType.ALLIANCE)
        assert self._runner().verify_accusation(msg, [outcome]) == "CONFIRMED"

    def test_a_kept_public_deal_refutes_the_accuser(self):
        from src.common.schemas import Commitment, CommitmentOutcome, Message, MessageType
        c = Commitment(id="c", commitment_type=CommitmentType.ALLIANCE,
                       players=[Player.RED, Player.BLUE], valid_until_turn=5)
        outcome = CommitmentOutcome(commitment=c, kept=True, broken_by=[])
        msg = Message(id="m", sender=Player.RED, message_type=MessageType.BROADCAST,
                      broadcast_kind="BETRAYED", broadcast_target=Player.BLUE,
                      commitment_type=CommitmentType.ALLIANCE)
        assert self._runner().verify_accusation(msg, [outcome]) == "REFUTED"

    def test_an_invented_betrayal_cannot_be_settled(self):
        """With no public deal between them the engine has nothing to check
        against, so the claim stands or falls on the accuser's reputation.
        That ambiguity is what gives lying an expected value."""
        from src.common.schemas import Message, MessageType
        msg = Message(id="m", sender=Player.RED, message_type=MessageType.BROADCAST,
                      broadcast_kind="BETRAYED", broadcast_target=Player.GREEN,
                      commitment_type=CommitmentType.ALLIANCE)
        assert self._runner().verify_accusation(msg, []) == "UNVERIFIED"

    def test_a_broken_private_deal_is_not_published(self):
        """The engine graded it, but saying so would make every private deal
        public the moment it was broken."""
        from src.common.schemas import Commitment, CommitmentOutcome, Message, MessageType
        c = Commitment(id="c", commitment_type=CommitmentType.ALLIANCE,
                       players=[Player.RED, Player.BLUE], valid_until_turn=5,
                       private=True)
        outcome = CommitmentOutcome(commitment=c, kept=False, broken_by=[Player.BLUE])
        msg = Message(id="m", sender=Player.RED, message_type=MessageType.BROADCAST,
                      broadcast_kind="BETRAYED", broadcast_target=Player.BLUE,
                      commitment_type=CommitmentType.ALLIANCE)
        assert self._runner().verify_accusation(msg, [outcome]) == "UNVERIFIED"

    def test_accusing_over_a_deal_you_were_not_in_is_unverifiable(self):
        from src.common.schemas import Commitment, CommitmentOutcome, Message, MessageType
        c = Commitment(id="c", commitment_type=CommitmentType.ALLIANCE,
                       players=[Player.GREEN, Player.BLUE], valid_until_turn=5)
        outcome = CommitmentOutcome(commitment=c, kept=False, broken_by=[Player.BLUE])
        msg = Message(id="m", sender=Player.RED, message_type=MessageType.BROADCAST,
                      broadcast_kind="BETRAYED", broadcast_target=Player.BLUE,
                      commitment_type=CommitmentType.ALLIANCE)
        assert self._runner().verify_accusation(msg, [outcome]) == "UNVERIFIED"


# ── Calibration (issue #8) ───────────────────────────────────────────────

class TestCalibration:
    def test_predictions_are_paired_with_outcomes(self):
        import random
        from src.engine.runner import GameRunner
        from src.agents.agent import Agent
        from src.evaluation.metrics import brier_score, reliability_bins
        random.seed(3)
        runner = GameRunner([
            Agent(Player.RED, "Opportunist"), Agent(Player.BLUE, "Honest"),
            Agent(Player.GREEN, "Paranoid"), Agent(Player.GOLD, "Vengeful")])
        runner.run(verbose=False)

        graded = [c for c in runner.calibration if c.observed is not None]
        assert graded, "no (prediction, outcome) pair was ever recorded"
        for c in graded:
            assert 0.0 <= c.predicted <= 1.0
            assert c.observer != c.subject
        score = brier_score(graded)
        assert 0.0 <= score <= 1.0
        assert sum(b["count"] for b in reliability_bins(graded)) == len(graded)

    def test_a_perfect_forecaster_scores_zero(self):
        from src.common.schemas import CalibrationPoint
        from src.evaluation.metrics import brier_score, calibration_error
        pts = [CalibrationPoint(turn=1, observer=Player.RED, subject=Player.BLUE,
                                commitment_type=CommitmentType.ALLIANCE,
                                predicted=1.0, observed=True),
               CalibrationPoint(turn=1, observer=Player.RED, subject=Player.GREEN,
                                commitment_type=CommitmentType.ALLIANCE,
                                predicted=0.0, observed=False)]
        assert brier_score(pts) == 0.0
        assert calibration_error(pts) == 0.0
