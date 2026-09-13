"""Tests for the Planner module."""

import pytest
import random
from src.common.schemas import (
    GameState, Order, OrderType, Player, Unit, Commitment, CommitmentType,
)
from src.agents.planner.planner import (
    Planner, PlannerConfig, DecisionTrace,
    evaluate_state, calculate_commitment_penalty,
)
from src.agents.trust.model import TrustModel


def _make_state(**overrides):
    defaults = dict(
        turn=1,
        units=[
            Unit(player=Player.RED, territory="R1"),
            Unit(player=Player.RED, territory="R2"),
            Unit(player=Player.RED, territory="R3"),
            Unit(player=Player.BLUE, territory="B1"),
            Unit(player=Player.BLUE, territory="B2"),
            Unit(player=Player.BLUE, territory="B3"),
        ],
        supply_centers={"R1": Player.RED, "B1": Player.BLUE,
                        "N1": None, "N2": None,
                        "G1": Player.GREEN, "Y1": Player.GOLD},
        territory_owners={
            "R1": Player.RED, "R2": Player.RED, "R3": Player.RED,
            "B1": Player.BLUE, "B2": Player.BLUE, "B3": Player.BLUE,
        },
    )
    defaults.update(overrides)
    return GameState(**defaults)


class TestEvaluateState:
    def test_more_supply_centers_is_better(self):
        state1 = _make_state(
            territory_owners={"R1": Player.RED},
        )
        state2 = _make_state(
            territory_owners={"R1": Player.RED, "N1": Player.RED},
        )
        assert evaluate_state(state2, Player.RED) > evaluate_state(state1, Player.RED)

    def test_more_units_is_better(self):
        s1 = _make_state(units=[Unit(player=Player.RED, territory="R1")])
        s2 = _make_state(units=[
            Unit(player=Player.RED, territory="R1"),
            Unit(player=Player.RED, territory="R2"),
        ])
        assert evaluate_state(s2, Player.RED) > evaluate_state(s1, Player.RED)

    def test_score_is_positive(self):
        state = _make_state()
        score = evaluate_state(state, Player.RED)
        assert score > 0


class TestCommitmentPenalty:
    def test_no_penalty_when_keeping(self):
        state = _make_state()
        orders = [
            Order(player=Player.RED, unit_territory="R1", order_type=OrderType.HOLD),
        ]
        c = Commitment(id="1", commitment_type=CommitmentType.ALLIANCE,
                       players=[Player.RED, Player.BLUE], valid_until_turn=5)
        penalty = calculate_commitment_penalty(
            state, Player.RED, orders, [c], PlannerConfig()
        )
        assert penalty == 0.0

    def test_penalty_when_breaking_alliance(self):
        state = _make_state(
            territory_owners={"B1": Player.BLUE},
        )
        orders = [
            Order(player=Player.RED, unit_territory="R2",
                  order_type=OrderType.MOVE, target="B3"),
        ]
        c = Commitment(id="1", commitment_type=CommitmentType.ALLIANCE,
                       players=[Player.RED, Player.BLUE], valid_until_turn=5)
        # B3 belongs to BLUE via units, RED attacking it breaks alliance
        penalty = calculate_commitment_penalty(
            state, Player.RED, orders, [c], PlannerConfig()
        )
        assert penalty > 0.0

    def test_honest_persona_has_higher_penalty(self):
        state = _make_state(
            territory_owners={"B1": Player.BLUE},
        )
        orders = [
            Order(player=Player.RED, unit_territory="R2",
                  order_type=OrderType.MOVE, target="B3"),
        ]
        c = Commitment(id="1", commitment_type=CommitmentType.ALLIANCE,
                       players=[Player.RED, Player.BLUE], valid_until_turn=5)
        p_opp = calculate_commitment_penalty(
            state, Player.RED, orders, [c], PlannerConfig(reputation_cost_coefficient=1.0)
        )
        p_honest = calculate_commitment_penalty(
            state, Player.RED, orders, [c], PlannerConfig(reputation_cost_coefficient=100.0)
        )
        assert p_honest > p_opp


class TestPlanner:
    def test_returns_valid_orders(self):
        random.seed(42)
        state = _make_state()
        trust = TrustModel(Player.RED)
        planner = Planner(Player.RED, PlannerConfig())
        orders, trace = planner.find_best_orders(state, [], trust)
        assert len(orders) > 0
        for o in orders:
            assert o.player == Player.RED

    def test_returns_decision_trace(self):
        random.seed(42)
        state = _make_state()
        trust = TrustModel(Player.RED)
        planner = Planner(Player.RED, PlannerConfig())
        orders, trace = planner.find_best_orders(state, [], trust)
        assert isinstance(trace, DecisionTrace)
        assert trace.expected_value >= 0

    def test_fallback_to_hold_with_no_units(self):
        state = _make_state(units=[])
        trust = TrustModel(Player.RED)
        planner = Planner(Player.RED, PlannerConfig())
        orders, trace = planner.find_best_orders(state, [], trust)
        assert len(orders) == 0
        assert "Fallback" in trace.explanation or trace.expected_value == 0
