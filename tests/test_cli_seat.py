"""The terminal seat's answers reach the engine."""
import importlib.util
import pathlib

from src.common.schemas import (CommitmentType, GameState, Message, MessageType, Player, Unit)
from src.engine.board import get_all_territories, get_supply_centers

_spec = importlib.util.spec_from_file_location(
    "play", pathlib.Path(__file__).parent.parent / "scripts" / "play.py")
play = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(play)


def _state():
    return GameState(turn=1, units=[Unit(player=Player.RED, territory="R1")],
                     supply_centers={t: None for t in get_supply_centers()},
                     territory_owners={t: None for t in get_all_territories()})


def test_accepting_in_the_terminal_sends_an_accept(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "1")
    seat = play.CLIHuman(Player.RED)
    offer = Message(id="m", sender=Player.BLUE, receiver=Player.RED,
                    message_type=MessageType.PROPOSE,
                    commitment_type=CommitmentType.ALLIANCE, turns=3)
    replies = seat.reply(_state(), [offer])
    assert [r.message_type for r in replies] == [MessageType.ACCEPT]


def test_terminal_draw_vote(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "1")
    assert play.CLIHuman(Player.RED).vote_draw(_state(), stalled=4)
