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


def test_siege_needs_a_unit_there_since_spring():
    Y = Player.GOLD
    with use_board(ring_board(sieges=True)):
        s = _state([Unit(player=Y, territory="C2")]).model_copy(update={"turn": 2})
        walk_in = [Order(player=Y, unit_territory="C2", order_type=OrderType.MOVE, target="R1")]
        after, _, _ = resolve(s, walk_in)
        assert after.supply_centers["R1"] is None
        sat = _state([Unit(player=Y, territory="R1")]).model_copy(update={"turn": 2})
        after, _, _ = resolve(sat, [])
        assert after.supply_centers["R1"] == Y
    with use_board(ring_board()):
        after, _, _ = resolve(s, walk_in)
        assert after.supply_centers["R1"] == Y


def test_regime_change_switches_persona_mid_game():
    seats = GameConfig(seed=2).seats()
    red = seats[Player.RED]
    new = "Opportunist" if red != "Opportunist" else "Honest"
    runner = play(GameConfig(seed=2, max_turns=4, node_budget=300,
                             regime_change={"seat": "Red", "turn": 3, "persona": new}))
    agent = runner.agents[Player.RED]
    assert agent.persona_name == new
    assert agent.negotiation.bound == (new != "Opportunist")
    assert any("Change of leadership" in line for line in runner.history[2].log.events)
    from src.harness import reputation_of
    carried = reputation_of(runner)
    assert carried and all(red not in about for about in carried.values())


def test_siege_occupation_counts_toward_value():
    from src.agents.planner.planner import evaluate_state
    Y = Player.GOLD
    on, off = _state([Unit(player=Y, territory="R1")]), _state([Unit(player=Y, territory="C2")])
    with use_board(ring_board(sieges=True)):
        siege_gap = evaluate_state(on, Y) - evaluate_state(off, Y)
    with use_board(ring_board()):
        plain_gap = evaluate_state(on, Y) - evaluate_state(off, Y)
    assert siege_gap > plain_gap


def test_reputation_carries_across_games_unless_erased():
    from src.harness import build_agents, game_setup, play_series, reputation_of
    ledger = {"Honest": {"Opportunist": {"Alliance": [9.0, 3.0], "tempted": [1.0, 4.0]}}}
    for erase, want in ((False, (9.0, 3.0)), (True, (1.0, 1.0))):
        cfg = GameConfig(seed=0, reputation=ledger, erase_reputation=erase)
        with game_setup(cfg):
            agents = {a.persona_name: a for a in build_agents(cfg)}
        opp = agents["Opportunist"].player
        rec = agents["Honest"].trust_model.get_record(opp, CommitmentType.ALLIANCE)
        assert (rec.alpha, rec.beta) == want
    first, second = play_series([GameConfig(seed=s, max_turns=2, node_budget=150) for s in (4, 5)])
    assert second.config.reputation == reputation_of(first)
