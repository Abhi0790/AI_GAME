import pytest
from src.common.schemas import GameState, Order, OrderType, Player, Unit, Commitment, CommitmentType
from src.engine.adjudicator import resolve, verify_commitments

def test_dmz_commitment():
    state = GameState(turn=1, units=[], supply_centers={}, territory_owners={})
    c = Commitment(id="1", commitment_type=CommitmentType.DMZ, players=[Player.RED, Player.BLUE], valid_until_turn=2, dmz_territories=["N1"])
    orders = [
        Order(player=Player.RED, unit_territory="R1", order_type=OrderType.MOVE, target="N1"),
        Order(player=Player.BLUE, unit_territory="B1", order_type=OrderType.HOLD)
    ]
    outcomes = verify_commitments(state, [c], orders)
    assert len(outcomes) == 1
    assert outcomes[0].kept == False
    assert Player.RED in outcomes[0].broken_by
    assert Player.BLUE not in outcomes[0].broken_by

def test_alliance_commitment():
    state = GameState(
        turn=1, 
        units=[], 
        supply_centers={"R1": Player.RED, "B1": Player.BLUE}, 
        territory_owners={}
    )
    c = Commitment(id="2", commitment_type=CommitmentType.ALLIANCE, players=[Player.RED, Player.BLUE], valid_until_turn=2)
    orders = [
        Order(player=Player.RED, unit_territory="R2", order_type=OrderType.MOVE, target="B1"),
    ]
    outcomes = verify_commitments(state, [c], orders)
    assert outcomes[0].kept == False
    assert Player.RED in outcomes[0].broken_by

