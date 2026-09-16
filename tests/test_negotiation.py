"""Tests for the Negotiation Strategy module.

The decision rule under test is no longer a reliability cut-off. A deal is
accepted when

    P(keep) * V_kept + (1 - P(keep)) * V_broken > V_none

so the tests stub the three values and check the arithmetic, rather than
poking a threshold and hoping.
"""

import pytest
from src.common.schemas import (
    Player, GameState, Unit, Message, MessageType, CommitmentType, Commitment,
    message_to_commitment, invalid_proposal_reason, describe_message,
)
from src.agents.trust.model import TrustModel
from src.agents.planner.planner import Planner, PlannerConfig
from src.agents.negotiation.strategy import NegotiationStrategy, THREAT_COOLDOWN


def _make_state(**overrides):
    defaults = dict(
        turn=1,
        units=[
            Unit(player=Player.RED, territory="R1"),
            Unit(player=Player.RED, territory="R2"),
            Unit(player=Player.BLUE, territory="B1"),
            Unit(player=Player.BLUE, territory="B2"),
        ],
        supply_centers={"R1": Player.RED, "R2": Player.RED,
                        "B1": Player.BLUE, "B2": Player.BLUE,
                        "N1": None, "N2": None},
        territory_owners={"R1": Player.RED, "R2": Player.RED,
                          "B1": Player.BLUE, "B2": Player.BLUE},
    )
    defaults.update(overrides)
    return GameState(**defaults)


def _strategy(player=Player.RED):
    return NegotiationStrategy(player, Planner(player, PlannerConfig()))


def _stub_values(ns, v_none, v_kept, v_broken):
    """Pin the counterfactual so the decision rule is the only thing tested.

    deal_totals is what the decision reads (deal_values scaled over the deal's
    life, plus Vcoop); stubbing both keeps the arithmetic under test rather
    than the board.
    """
    ns.deal_values = lambda *a, **k: (v_none, v_kept, v_broken)
    ns.deal_totals = lambda *a, **k: (v_none, v_kept, v_broken)


def _propose(commitment_type=CommitmentType.ALLIANCE, turns=3, **kw):
    return Message(id="m", sender=Player.BLUE, receiver=Player.RED,
                   message_type=MessageType.PROPOSE,
                   commitment_type=commitment_type, turns=turns, **kw)


class TestAcceptanceIsExpectedValue:
    def test_accepts_when_expected_value_beats_no_deal(self):
        ns = _strategy()
        trust = TrustModel(Player.RED)                 # P(keep) ≈ 0.5
        _stub_values(ns, v_none=1.0, v_kept=3.0, v_broken=0.5)
        # 0.5*3.0 + 0.5*0.5 = 1.75 > 1.0
        assert ns.evaluate_proposal(_make_state(), _propose(), trust).message_type \
            == MessageType.ACCEPT

    def test_rejects_when_exposure_outweighs_the_upside(self):
        ns = _strategy()
        trust = TrustModel(Player.RED)
        # A big upside that is not big enough to cover being walked over.
        _stub_values(ns, v_none=2.0, v_kept=2.6, v_broken=-2.0)
        ns._counter_terms = lambda msg: None
        assert ns.evaluate_proposal(_make_state(), _propose(), trust).message_type \
            == MessageType.REJECT

    def test_low_reliability_can_flip_the_same_deal(self):
        """Identical board values, different partner: the only thing that
        changed is P(keep), and that is enough to change the answer."""
        state, msg = _make_state(), _propose()
        values = dict(v_none=1.0, v_kept=3.0, v_broken=-1.0)

        trusted = TrustModel(Player.RED)
        trusted.get_record(Player.BLUE, CommitmentType.ALLIANCE).alpha = 9.0
        ns = _strategy(); _stub_values(ns, **values)
        assert ns.evaluate_proposal(state, msg, trusted).message_type == MessageType.ACCEPT

        shifty = TrustModel(Player.RED)
        shifty.get_record(Player.BLUE, CommitmentType.ALLIANCE).beta = 9.0
        ns = _strategy(); _stub_values(ns, **values)
        ns._counter_terms = lambda m: None
        assert ns.evaluate_proposal(state, msg, shifty).message_type == MessageType.REJECT


class TestCounterOffers:
    def test_counters_with_shorter_terms_before_walking_away(self):
        ns = _strategy()
        trust = TrustModel(Player.RED)
        # The 3-turn deal fails; the halved one passes.
        def values(state, deal, partner, commitments):
            return (1.0, 3.0, 2.0) if deal.valid_until_turn <= 2 else (1.0, 1.0, 0.0)
        ns.deal_values = values
        ns.deal_totals = values

        reply = ns.evaluate_proposal(_make_state(), _propose(turns=3), trust)
        assert reply.message_type == MessageType.COUNTER
        assert reply.turns == 1
        assert reply.reference_id == "m"

    def test_a_counter_is_never_countered_again(self):
        """Otherwise rounds 2 and 3 are an infinite haggle."""
        ns = _strategy()
        trust = TrustModel(Player.RED)
        _stub_values(ns, v_none=5.0, v_kept=0.0, v_broken=0.0)
        msg = _propose(turns=4)
        msg.message_type = MessageType.COUNTER
        assert ns.evaluate_proposal(_make_state(), msg, trust).message_type \
            == MessageType.REJECT


class TestGrammarCoverage:
    def test_support_hold_is_proposable(self):
        """Third sentence of the grammar; nothing used to generate it."""
        ns = _strategy()
        offers = ns._candidate_deals(_make_state(), Player.BLUE)
        support_holds = [m for m in offers
                         if m.commitment_type == CommitmentType.SUPPORT
                         and m.supported_from is None]
        assert support_holds, "no support-hold offer was ever generated"
        assert "hold" in describe_message(support_holds[0])

    def test_threat_is_rate_limited(self):
        ns = _strategy()
        ns.threats_made[Player.BLUE] = 5
        state = _make_state(turn=5 + THREAT_COOLDOWN - 1)
        threats = [m for m in ns.generate_proposals(state, TrustModel(Player.RED))
                   if m.message_type == MessageType.THREAT and m.receiver == Player.BLUE]
        assert not threats

    def test_threat_has_a_condition_and_an_action(self):
        ns = _strategy()
        state = _make_state()
        threats = [m for m in ns.generate_proposals(state, TrustModel(Player.RED))
                   if m.message_type == MessageType.THREAT]
        for t in threats:
            assert t.condition and t.action

    def test_no_proposals_to_self(self):
        ns = _strategy()
        for m in ns.generate_proposals(_make_state(), TrustModel(Player.RED)):
            assert m.receiver != Player.RED

    def test_malformed_proposals_are_rejected_not_priced(self):
        ns = _strategy()
        trust = TrustModel(Player.RED)
        junk = Message(id="x", sender=Player.BLUE, receiver=Player.RED,
                       message_type=MessageType.PROPOSE,
                       commitment_type=CommitmentType.DMZ,
                       dmz_territories=["NOWHERE"], turns=999)
        assert ns.evaluate_proposal(_make_state(), junk, trust).message_type \
            == MessageType.REJECT

    def test_threat_is_not_answered_with_accept_or_reject(self):
        ns = _strategy()
        threat = Message(id="t", sender=Player.BLUE, receiver=Player.RED,
                         message_type=MessageType.THREAT,
                         condition="you move", action="I retaliate")
        assert ns.evaluate_proposal(_make_state(), threat, TrustModel(Player.RED)) is None


class TestGrammarValidation:
    @pytest.mark.parametrize("msg,reason", [
        (_propose(turns=999), "duration"),
        (_propose(CommitmentType.DMZ, dmz_territories=[]), "no territories"),
        (_propose(CommitmentType.DMZ, dmz_territories=["NOWHERE"]), "unknown"),
        (_propose(CommitmentType.SUPPORT, target_territory="NOWHERE"), "unknown"),
        (_propose(CommitmentType.SUPPORT, target_territory="N1",
                  supported_from="G1"), "adjacent"),
    ])
    def test_rejected_sentences(self, msg, reason):
        got = invalid_proposal_reason(msg)
        assert got and reason in got

    @pytest.mark.parametrize("msg", [
        _propose(),
        _propose(CommitmentType.DMZ, turns=2, dmz_territories=["N1"]),
        _propose(CommitmentType.SUPPORT, turns=1, target_territory="N1",
                 supported_from="R2"),
        _propose(CommitmentType.SUPPORT, turns=1, target_territory="N1"),
    ])
    def test_accepted_sentences(self, msg):
        assert invalid_proposal_reason(msg) is None


class TestMessageToCommitment:
    def test_basic_conversion(self):
        msg = Message(id="m1", sender=Player.RED, receiver=Player.BLUE,
                      message_type=MessageType.PROPOSE,
                      commitment_type=CommitmentType.ALLIANCE, turns=3)
        c = message_to_commitment(msg, Player.BLUE, current_turn=4)
        assert c.commitment_type == CommitmentType.ALLIANCE
        assert Player.RED in c.players
        assert Player.BLUE in c.players
        assert c.valid_until_turn == 7  # 4 + 3

    def test_dmz_conversion(self):
        msg = Message(id="m2", sender=Player.RED, receiver=Player.BLUE,
                      message_type=MessageType.PROPOSE,
                      commitment_type=CommitmentType.DMZ, turns=2,
                      dmz_territories=["N1"])
        c = message_to_commitment(msg, Player.BLUE, current_turn=1)
        assert c.commitment_type == CommitmentType.DMZ
        assert c.dmz_territories == ["N1"]
        assert c.valid_until_turn == 3
