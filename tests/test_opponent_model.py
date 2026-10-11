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


def test_draws_break_at_the_believed_rate():
    """The sampler's betrayal rate is 1 - P(keeps), however many order sets break."""
    import random
    from src.agents.trust.model import TrustModel
    state = _make_state()
    deal = Commitment(id="d", commitment_type=CommitmentType.DMZ,
                      players=[Player.RED, Player.BLUE], valid_until_turn=3,
                      dmz_territories=["N1"])
    model, trust = OpponentModel(Player.RED), TrustModel(Player.RED)
    p_keep = trust.p_keeps(Player.BLUE, CommitmentType.DMZ, toward=Player.RED)
    random.seed(0)
    draws = model.sample_opponent_orders(state, 4000, commitments=[deal], trust_model=trust)
    broke = sum(any(o.player == Player.BLUE and o.target == "N1" for o in w) for w in draws)
    assert abs(broke / len(draws) - (1 - p_keep)) < 0.03


def test_level1_model_expects_an_opponent_to_take_a_free_centre():
    """Blue stands on C2, a plain square, in autumn; every move off it takes a
    centre and holding takes nothing. A level-1 model expects the capture far
    more often than the move-frequency prior does."""
    import random
    from src.engine.board import get_supply_centers, get_all_territories
    state = GameState(
        turn=2,
        units=[Unit(player=Player.BLUE, territory="C2"), Unit(player=Player.RED, territory="R2")],
        supply_centers={t: None for t in get_supply_centers()}
        | {"R1": Player.RED, "R2": Player.RED, "B1": Player.BLUE},
        territory_owners={t: None for t in get_all_territories()},
    )
    model = OpponentModel(Player.RED)
    holds = lambda worlds: sum(all(o.order_type != OrderType.MOVE
                                   for o in w if o.player == Player.BLUE) for w in worlds)
    random.seed(0)
    prior = holds(model.sample_opponent_orders(state, 400))
    random.seed(0)
    rational = holds(model.sample_opponent_orders(state, 400, rational=True))
    assert rational < prior / 3


def test_level1_cache_is_keyed_on_valuation_knobs():
    from src.agents.opponent_model.opponent_model import level1_values
    from src.agents.planner import planner as P
    from src.engine.orders import generate_all_order_sets
    state = _make_state()
    sets = generate_all_order_sets(state, Player.BLUE)
    before = level1_values(state, Player.BLUE, sets)
    old, P.CENTRE_VALUE = P.CENTRE_VALUE, 3.0
    try:
        assert level1_values(state, Player.BLUE, sets) != before
    finally:
        P.CENTRE_VALUE = old
