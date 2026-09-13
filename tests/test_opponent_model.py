"""Tests for the Opponent Model module."""

import pytest
from src.common.schemas import (
    Player, GameState, Order, OrderType, Unit, Commitment, CommitmentType,
)
from src.agents.opponent_model.opponent_model import OpponentModel, OpponentProfile
from src.engine.adjudicator import CommitmentOutcome


def _make_state():
    return GameState(
        turn=1,
        units=[
            Unit(player=Player.RED, territory="R1"),
            Unit(player=Player.BLUE, territory="B1"),
            Unit(player=Player.GREEN, territory="G1"),
        ],
        supply_centers={"R1": Player.RED, "B1": Player.BLUE},
        territory_owners={"R1": Player.RED, "B1": Player.BLUE, "G1": Player.GREEN},
    )


class TestOpponentProfile:
    def test_initial_aggression(self):
        p = OpponentProfile(Player.BLUE)
        assert p.aggression_ratio == 0.5  # uninformative prior

    def test_aggression_after_moves(self):
        p = OpponentProfile(Player.BLUE)
        orders = [
            Order(player=Player.BLUE, unit_territory="B1",
                  order_type=OrderType.MOVE, target="N1"),
            Order(player=Player.BLUE, unit_territory="B2",
                  order_type=OrderType.MOVE, target="B3"),
            Order(player=Player.BLUE, unit_territory="B3",
                  order_type=OrderType.HOLD),
        ]
        p.record_orders(orders)
        # 2 moves, 1 hold → 2/3
        assert p.aggression_ratio == pytest.approx(2.0 / 3.0)

    def test_cooperation_ratio(self):
        p = OpponentProfile(Player.BLUE)
        orders = [
            Order(player=Player.BLUE, unit_territory="B1",
                  order_type=OrderType.SUPPORT, target="R2"),
            Order(player=Player.BLUE, unit_territory="B2",
                  order_type=OrderType.HOLD),
        ]
        p.record_orders(orders)
        assert p.cooperation_ratio == pytest.approx(0.5)

    def test_trust_keeping_rate(self):
        p = OpponentProfile(Player.BLUE)
        p.record_commitment_outcome(kept=True)
        p.record_commitment_outcome(kept=True)
        p.record_commitment_outcome(kept=False)
        assert p.trust_keeping_rate == pytest.approx(2.0 / 3.0)

    def test_target_frequency(self):
        p = OpponentProfile(Player.BLUE)
        orders = [
            Order(player=Player.BLUE, unit_territory="B1",
                  order_type=OrderType.MOVE, target="N1"),
            Order(player=Player.BLUE, unit_territory="B2",
                  order_type=OrderType.MOVE, target="N1"),
        ]
        p.record_orders(orders)
        assert p.target_freq["N1"] == 2

    def test_persona_estimation(self):
        p = OpponentProfile(Player.BLUE)
        # Simulate high trust-keeping
        for _ in range(10):
            p.record_commitment_outcome(kept=True)
        assert p.estimated_persona() == "Honest"

    def test_ignores_other_player_orders(self):
        p = OpponentProfile(Player.BLUE)
        orders = [
            Order(player=Player.RED, unit_territory="R1",
                  order_type=OrderType.MOVE, target="R2"),
        ]
        p.record_orders(orders)
        assert p.total_orders == 0


class TestOpponentModel:
    def test_creates_profiles_for_opponents(self):
        om = OpponentModel(Player.RED)
        assert Player.BLUE in om.profiles
        assert Player.GREEN in om.profiles
        assert Player.GOLD in om.profiles
        assert Player.RED not in om.profiles

    def test_observe_orders(self):
        om = OpponentModel(Player.RED)
        orders = [
            Order(player=Player.BLUE, unit_territory="B1",
                  order_type=OrderType.MOVE, target="N1"),
        ]
        om.observe_orders(orders)
        assert om.profiles[Player.BLUE].total_orders == 1
        assert om.profiles[Player.BLUE].move_count == 1

    def test_observe_commitment_outcomes(self):
        om = OpponentModel(Player.RED)
        c = Commitment(id="1", commitment_type=CommitmentType.ALLIANCE,
                       players=[Player.RED, Player.BLUE], valid_until_turn=5)
        outcome = CommitmentOutcome(commitment=c, kept=False, broken_by=[Player.BLUE])
        om.observe_commitment_outcomes([outcome])
        assert om.profiles[Player.BLUE].commitments_seen == 1
        assert om.profiles[Player.BLUE].commitments_kept == 0

    def test_sample_opponent_orders_returns_correct_count(self):
        state = _make_state()
        om = OpponentModel(Player.RED)
        samples = om.sample_opponent_orders(state, samples=5)
        assert len(samples) == 5

    def test_sample_returns_valid_orders(self):
        state = _make_state()
        om = OpponentModel(Player.RED)
        samples = om.sample_opponent_orders(state, samples=3)
        for sample_set in samples:
            for o in sample_set:
                assert o.player != Player.RED
