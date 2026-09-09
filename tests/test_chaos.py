import pytest
from src.common.schemas import GameState, Order, OrderType, Player, Unit
from src.engine.adjudicator import resolve
from src.agents.agent import Agent
from src.engine.runner import GameRunner

def test_chaos_agent_invalid_orders():
    # Provide orders that are completely malformed or reference non-existent territories
    state = GameState(turn=1, units=[Unit(player=Player.RED, territory="R1")], supply_centers={}, territory_owners={})
    orders = [
        Order(player=Player.RED, unit_territory="R1", order_type=OrderType.MOVE, target="NOT_A_TERRITORY"),
        Order(player=Player.BLUE, unit_territory="FAKE", order_type=OrderType.SUPPORT, target="NULL")
    ]
    # Engine should not crash, it should reject invalid orders and fallback to hold
    new_state, outcomes, log = resolve(state, orders)
    assert len(new_state.units) == 1
    assert new_state.units[0].territory == "R1"

