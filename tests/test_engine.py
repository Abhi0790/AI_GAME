import pytest
from src.common.schemas import GameState, Order, OrderType, Player, Unit, Commitment, CommitmentType
from src.engine.adjudicator import resolve, verify_commitments

def test_basic_move():
    state = GameState(
        turn=1,
        units=[Unit(player=Player.RED, territory="R1")],
        supply_centers={"R1": Player.RED},
        territory_owners={"R1": Player.RED}
    )
    orders = [
        Order(player=Player.RED, unit_territory="R1", order_type=OrderType.MOVE, target="R2")
    ]
    new_state, _, log = resolve(state, orders)
    
    assert len(new_state.units) == 1
    assert new_state.units[0].territory == "R2"
    assert new_state.territory_owners["R2"] == Player.RED

def test_bounce():
    state = GameState(
        turn=1,
        units=[
            Unit(player=Player.RED, territory="R1"),
            Unit(player=Player.BLUE, territory="B3")
        ],
        supply_centers={},
        territory_owners={}
    )
    orders = [
        Order(player=Player.RED, unit_territory="R1", order_type=OrderType.MOVE, target="R2"),
        Order(player=Player.BLUE, unit_territory="B3", order_type=OrderType.MOVE, target="R2")
    ]
    new_state, _, log = resolve(state, orders)
    
    # Both should bounce and stay where they are
    assert len(new_state.units) == 2
    territories = {u.territory for u in new_state.units}
    assert "R1" in territories
    assert "B3" in territories
    assert "R2" not in territories

def test_support_success():
    state = GameState(
        turn=1,
        units=[
            Unit(player=Player.RED, territory="R1"),
            Unit(player=Player.RED, territory="R3"),
            Unit(player=Player.BLUE, territory="B3") # B3 is adjacent to R2, R1 is adjacent to R2, R3 is adjacent to R2
        ],
        supply_centers={},
        territory_owners={}
    )
    orders = [
        Order(player=Player.RED, unit_territory="R1", order_type=OrderType.MOVE, target="R2"),
        Order(player=Player.RED, unit_territory="R3", order_type=OrderType.SUPPORT, supported_from="R1", target="R2"),
        Order(player=Player.BLUE, unit_territory="B3", order_type=OrderType.MOVE, target="R2")
    ]
    new_state, _, log = resolve(state, orders)
    
    # Red moves with support (strength 2), Blue moves with strength 1. Red should win.
    assert any(u.territory == "R2" and u.player == Player.RED for u in new_state.units)
    assert any(u.territory == "B3" and u.player == Player.BLUE for u in new_state.units)
    assert any(u.territory == "R3" and u.player == Player.RED for u in new_state.units)
    
