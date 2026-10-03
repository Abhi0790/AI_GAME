"""Order noise and fortresses."""
from src.common.config import GameConfig
from src.common.schemas import (
    Commitment, CommitmentType, GameState, Order, OrderType, Player, Unit,
)
from src.engine import runner as runner_module
from src.engine.adjudicator import resolve
from src.engine.board import Board, ring_board, use_board
from src.harness import play

R, B = Player.RED, Player.BLUE


def _state(units):
    board = ring_board()
    return GameState(
        turn=1, units=units,
        supply_centers={t: None for t in board.supply_centers},
        territory_owners={t: None for t in board.territories})


def test_fortress_needs_two_supports():
    # R1 is a fortress: one support bounces, two dislodge.
    with use_board(ring_board(fortified_homes=True)):
        s = _state([Unit(player=B, territory="R1"), Unit(player=R, territory="R2"),
                    Unit(player=R, territory="C2"), Unit(player=R, territory="Y2")])
        move = Order(player=R, unit_territory="R2", order_type=OrderType.MOVE, target="R1")
        support = [Order(player=R, unit_territory=t, order_type=OrderType.SUPPORT,
                         target="R1", supported_from="R2") for t in ("C2", "Y2")]
        after, _, _ = resolve(s, [move, support[0]])
        assert after.get_unit_at("R1").player == B
        after, _, _ = resolve(s, [move] + support)
        assert after.get_unit_at("R1").player == R


def test_plain_board_unchanged():
    s = _state([Unit(player=B, territory="R1"), Unit(player=R, territory="R2"),
                Unit(player=R, territory="C2")])
    with use_board(ring_board()):
        after, _, _ = resolve(s, [
            Order(player=R, unit_territory="R2", order_type=OrderType.MOVE, target="R1"),
            Order(player=R, unit_territory="C2", order_type=OrderType.SUPPORT,
                  target="R1", supported_from="R2")])
    assert after.get_unit_at("R1").player == R


def test_fortresses_round_trip_and_break_symmetry_when_lopsided():
    b = ring_board(fortified_homes=True)
    assert Board.from_dict(b.to_dict()).fortresses == b.fortresses
    d = b.to_dict()
    d["fortresses"] = ["R1"]
    try:
        Board.from_dict(d)
    except AssertionError:
        pass
    else:
        raise AssertionError("one fortress on a symmetric board should be rejected")


def test_lost_support_is_graded_as_broken():
    runner = runner_module.GameRunner.__new__(runner_module.GameRunner)
    runner.state = _state([Unit(player=R, territory="R2"), Unit(player=B, territory="B1")])
    orders = [Order(player=R, unit_territory="R2", order_type=OrderType.SUPPORT,
                    target="N1", supported_from="B1")]
    old = runner_module.ORDER_NOISE
    try:
        runner_module.ORDER_NOISE = 1.0
        executed, slipped = runner._slip(orders)
    finally:
        runner_module.ORDER_NOISE = old
    assert slipped == orders and executed[0].order_type == OrderType.HOLD
    deal = Commitment(id="d", commitment_type=CommitmentType.SUPPORT, players=[R, B],
                      valid_until_turn=1, target_territory="N1", supported_from="B1")
    _, outcomes, _ = resolve(runner.state, executed, [deal])
    assert not outcomes[0].kept


def test_noise_in_a_real_game():
    cfg = GameConfig(seed=3, max_turns=4, node_budget=150,
                     knobs={"ORDER_NOISE": 0.3})
    runner = play(cfg)
    slipped = [o for step in runner.history for o in step.slipped]
    assert slipped and all(o.order_type != OrderType.HOLD for o in slipped)
    for step in runner.history:
        assert len(step.intended) == len(step.orders)
