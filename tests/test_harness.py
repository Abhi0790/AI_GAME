"""The harness is the one way a game gets built, so it must be reproducible
and its knob overrides must not leak."""
from src.common.config import GameConfig, seating_for
from src.common.schemas import Player
from src.harness import apply_knobs, knob_values, play
from src.agents.trust import model as trust_model


def test_seating_rotates_and_round_trips():
    assert seating_for(0) != seating_for(1)
    assert sorted(seating_for(3).values()) == sorted(seating_for(0).values())
    cfg = GameConfig(seed=5, human_seat=Player.RED, knobs={"LAMBDA_INCENTIVE": 0.6},
                     seat_planner={Player.BLUE: {"search": "mcts"}})
    back = GameConfig.from_dict(cfg.to_dict())
    assert back.seats() == cfg.seats() and back.seat_planner == cfg.seat_planner


def test_knobs_apply_and_restore():
    before = trust_model.LAMBDA_INCENTIVE
    with apply_knobs({"LAMBDA_INCENTIVE": 0.99}):
        assert trust_model.LAMBDA_INCENTIVE == 0.99
    assert trust_model.LAMBDA_INCENTIVE == before
    assert "CENTRE_VALUE" in knob_values()


def test_same_seed_same_game():
    a = play(GameConfig(seed=11, node_budget=200))
    b = play(GameConfig(seed=11, node_budget=200))
    assert [str(h[1]) for h in a.history] == [str(h[1]) for h in b.history]


def test_every_adjudication_is_charged_to_a_seat():
    """Seats' totals plus the runner's one resolution per turn account for every
    call, and the search budget's count is part of each seat's total."""
    from src.engine import adjudicator
    before = adjudicator.resolve_calls
    runner = play(GameConfig(seed=5, max_turns=4, node_budget=300))
    made = adjudicator.resolve_calls - before
    charged = sum(sum(step.total_nodes.values()) for step in runner.history)
    assert made == charged + len(runner.history)
    for step in runner.history:
        for p, n in step.nodes.items():
            assert n <= step.total_nodes[p]


def test_seating_varies_who_sits_opposite_whom():
    opposite = {frozenset((s[Player.RED], s[Player.GREEN]))
                for s in map(seating_for, range(12))}
    assert len(opposite) == 6
