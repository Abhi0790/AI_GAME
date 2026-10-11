"""A unanimous draw vote ends the game shared among the survivors."""
from src.agents import agent as agent_mod
from src.agents.agent import Agent
from src.common.config import GameConfig
from src.common.schemas import GameState, Player, Unit
from src.harness import play


def test_stalled_board_with_shared_lead_is_drawn(monkeypatch):
    monkeypatch.setattr(agent_mod, "DRAW_STALL_TURNS", 1)
    runner = play(GameConfig(seed=1, draws=True, max_turns=12))
    assert runner.drawn and set(runner.drawn) == set(runner.survivors())
    assert any("Draw agreed" in e for e in runner.history[-1].log.events)
    assert len(runner.history) < 12 and runner.winner() is None


def test_draws_off_by_default(monkeypatch):
    monkeypatch.setattr(agent_mod, "DRAW_STALL_TURNS", 0)
    runner = play(GameConfig(seed=1, max_turns=3))
    assert runner.drawn is None and len(runner.history) == 3


def test_a_sole_leader_refuses_and_others_accept_a_stalemate():
    state = GameState(turn=5, units=[Unit(player=Player.RED, territory="R1")],
                      supply_centers={"R1": Player.RED, "R2": Player.RED, "B1": Player.BLUE},
                      territory_owners={})
    assert not Agent(Player.RED).vote_draw(state, stalled=10)
    assert Agent(Player.BLUE).vote_draw(state, stalled=10)
    assert not Agent(Player.BLUE).vote_draw(state, stalled=1)


def test_a_replay_records_the_draw_and_the_intents(monkeypatch, tmp_path):
    from src.engine.replay import load_replay, save_replay
    monkeypatch.setattr(agent_mod, "DRAW_STALL_TURNS", 1)
    runner = play(GameConfig(seed=1, draws=True, max_turns=12))
    replay = load_replay(save_replay(runner.history, runner.state, directory=str(tmp_path),
                                     filename="r.json", result=runner.result()))
    assert replay["version"] == 5
    assert replay["result"]["drawn"] and replay["result"]["winner"] is None
    assert replay["result"]["leaders"] == sorted(p.value for p in runner.drawn)
    saved = [i for t in replay["history"] for i in t["intents"]]
    assert len(saved) == sum(len(s.intents) for s in runner.history)
    assert all(set(i) == {"sender", "from", "to", "carried_out"} for i in saved)
