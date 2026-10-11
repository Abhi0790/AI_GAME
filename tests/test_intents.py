"""Stated intentions: announced, graded on executed orders, learned from."""
import random

from src.agents.agent import Agent
from src.agents.opponent_model.opponent_model import OpponentModel
from src.common.config import GameConfig
from src.common.schemas import GameState, Message, MessageType, Order, OrderType, Player, Unit, carried_out
from src.engine.board import get_all_territories, get_supply_centers
from src.harness import play

R, B = Player.RED, Player.BLUE


def _intent(frm="B2", to="C1"):
    return Message(id="i", sender=B, message_type=MessageType.INTENT,
                   supported_from=frm, target_territory=to)


def test_carried_out_reads_the_executed_move():
    move = Order(player=B, unit_territory="B2", order_type=OrderType.MOVE, target="C1")
    hold = Order(player=B, unit_territory="B2", order_type=OrderType.HOLD)
    assert carried_out(_intent(), [move]) and not carried_out(_intent(), [hold])


def test_candour_learns_from_graded_intents():
    a = Agent(R)
    a.observe_intents([(_intent(), False)] * 3)
    assert a.candour_of(B) < 0.25
    a.observe_intents([(_intent(), True)] * 6)
    assert a.candour_of(B) > 0.6


def test_a_candid_announcer_is_expected_to_do_what_it_said():
    state = GameState(turn=1, units=[Unit(player=B, territory="B2")],
                      supply_centers={t: None for t in get_supply_centers()},
                      territory_owners={t: None for t in get_all_territories()})
    model = OpponentModel(R)
    model.intents, model.candour = {B: [("B2", "C1")]}, {B: 0.9}
    random.seed(0)
    worlds = model.sample_opponent_orders(state, 500)
    said = sum(carried_out(_intent(), w) for w in worlds)
    assert 400 < said < 500


def test_games_announce_and_grade_intents():
    runner = play(GameConfig(seed=2, max_turns=4))
    graded = [g for step in runner.history for g in step.intents]
    assert graded and all(isinstance(done, bool) for _m, done in graded)
    assert any("Intent" in e for step in runner.history for e in step.log.events)
