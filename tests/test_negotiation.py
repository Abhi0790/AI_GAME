"""Tests for the Negotiation Strategy module."""

import pytest
from src.common.schemas import (
    Player, GameState, Unit, Message, MessageType, CommitmentType, Commitment,
)
from src.agents.trust.model import TrustModel
from src.agents.planner.planner import Planner, PlannerConfig
from src.agents.negotiation.strategy import NegotiationStrategy, message_to_commitment


def _make_state(**overrides):
    defaults = dict(
        turn=1,
        units=[
            Unit(player=Player.RED, territory="R1"),
            Unit(player=Player.RED, territory="R2"),
            Unit(player=Player.BLUE, territory="B1"),
            Unit(player=Player.BLUE, territory="B2"),
        ],
        supply_centers={"R1": Player.RED, "B1": Player.BLUE,
                        "N1": None, "N2": None},
        territory_owners={"R1": Player.RED, "R2": Player.RED,
                          "B1": Player.BLUE, "B2": Player.BLUE},
    )
    defaults.update(overrides)
    return GameState(**defaults)


class TestProposalGeneration:
    def test_generates_alliance_for_trusted_player(self):
        state = _make_state()
        trust = TrustModel(Player.RED)
        # Ensure BLUE is trusted
        trust.get_record(Player.BLUE, CommitmentType.ALLIANCE).alpha = 5.0
        planner = Planner(Player.RED, PlannerConfig())
        ns = NegotiationStrategy(Player.RED, planner)
        proposals = ns.generate_proposals(state, trust)
        alliance_props = [m for m in proposals
                          if m.commitment_type == CommitmentType.ALLIANCE
                          and m.receiver == Player.BLUE]
        assert len(alliance_props) >= 1

    def test_generates_threat_for_untrusted_player(self):
        state = _make_state()
        trust = TrustModel(Player.RED)
        # Make BLUE very untrusted
        trust.get_record(Player.BLUE, CommitmentType.ALLIANCE).alpha = 1.0
        trust.get_record(Player.BLUE, CommitmentType.ALLIANCE).beta = 10.0
        planner = Planner(Player.RED, PlannerConfig())
        ns = NegotiationStrategy(Player.RED, planner)
        proposals = ns.generate_proposals(state, trust)
        threats = [m for m in proposals
                   if m.message_type == MessageType.THREAT
                   and m.receiver == Player.BLUE]
        assert len(threats) >= 1

    def test_no_proposals_to_self(self):
        state = _make_state()
        trust = TrustModel(Player.RED)
        planner = Planner(Player.RED, PlannerConfig())
        ns = NegotiationStrategy(Player.RED, planner)
        proposals = ns.generate_proposals(state, trust)
        for m in proposals:
            assert m.receiver != Player.RED


class TestProposalEvaluation:
    def test_accepts_alliance_from_trusted(self):
        state = _make_state()
        trust = TrustModel(Player.RED)
        trust.get_record(Player.BLUE, CommitmentType.ALLIANCE).alpha = 5.0
        planner = Planner(Player.RED, PlannerConfig())
        ns = NegotiationStrategy(Player.RED, planner)
        msg = Message(id="test-1", sender=Player.BLUE, receiver=Player.RED,
                      message_type=MessageType.PROPOSE,
                      commitment_type=CommitmentType.ALLIANCE, turns=3)
        reply = ns.evaluate_proposal(state, msg, trust)
        assert reply.message_type == MessageType.ACCEPT

    def test_rejects_alliance_from_untrusted(self):
        state = _make_state(
            supply_centers={"R1": Player.RED, "R2": Player.RED, "B1": Player.BLUE}
        )
        trust = TrustModel(Player.RED)
        trust.get_record(Player.BLUE, CommitmentType.ALLIANCE).alpha = 1.0
        trust.get_record(Player.BLUE, CommitmentType.ALLIANCE).beta = 10.0
        planner = Planner(Player.RED, PlannerConfig())
        ns = NegotiationStrategy(Player.RED, planner)
        msg = Message(id="test-2", sender=Player.BLUE, receiver=Player.RED,
                      message_type=MessageType.PROPOSE,
                      commitment_type=CommitmentType.ALLIANCE, turns=3)
        reply = ns.evaluate_proposal(state, msg, trust)
        assert reply.message_type == MessageType.REJECT

    def test_accepts_when_weak(self):
        """Even low trust is accepted when the agent holds few centres."""
        state = _make_state(
            supply_centers={"R1": Player.RED, "B1": Player.BLUE,
                            "N1": Player.BLUE, "N2": Player.BLUE},
        )
        trust = TrustModel(Player.RED)
        trust.get_record(Player.BLUE, CommitmentType.ALLIANCE).alpha = 1.0
        trust.get_record(Player.BLUE, CommitmentType.ALLIANCE).beta = 5.0
        planner = Planner(Player.RED, PlannerConfig())
        ns = NegotiationStrategy(Player.RED, planner)
        msg = Message(id="test-3", sender=Player.BLUE, receiver=Player.RED,
                      message_type=MessageType.PROPOSE,
                      commitment_type=CommitmentType.ALLIANCE, turns=3)
        reply = ns.evaluate_proposal(state, msg, trust)
        assert reply.message_type == MessageType.ACCEPT


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
