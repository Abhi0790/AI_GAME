import random
from types import SimpleNamespace

import pytest
from src.common.schemas import GameState, Order, OrderType, Player, Unit, Commitment, CommitmentType
from src.engine.adjudicator import resolve, verify_commitments
from src.engine.runner import GameRunner, AGENT_METHODS
from src.engine.board import players, get_all_territories
from src.agents.agent import Agent
from src.agents.planner.planner import PlannerConfig

PERSONAS = ("Opportunist", "Honest", "Paranoid", "Vengeful")

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
            Unit(player=Player.BLUE, territory="B1")
        ],
        supply_centers={},
        territory_owners={}
    )
    orders = [
        Order(player=Player.RED, unit_territory="R1", order_type=OrderType.MOVE, target="R2"),
        Order(player=Player.BLUE, unit_territory="B1", order_type=OrderType.MOVE, target="R2")
    ]
    new_state, _, log = resolve(state, orders)
    
    # Both should bounce and stay where they are
    assert len(new_state.units) == 2
    territories = {u.territory for u in new_state.units}
    assert "R1" in territories
    assert "B1" in territories
    assert "R2" not in territories

def test_support_success():
    state = GameState(
        turn=1,
        units=[
            Unit(player=Player.RED, territory="R1"),
            Unit(player=Player.RED, territory="N1"),
            Unit(player=Player.BLUE, territory="B1")  # R1, N1 and B1 are all adjacent to R2
        ],
        supply_centers={},
        territory_owners={}
    )
    orders = [
        Order(player=Player.RED, unit_territory="R1", order_type=OrderType.MOVE, target="R2"),
        Order(player=Player.RED, unit_territory="N1", order_type=OrderType.SUPPORT, supported_from="R1", target="R2"),
        Order(player=Player.BLUE, unit_territory="B1", order_type=OrderType.MOVE, target="R2")
    ]
    new_state, _, log = resolve(state, orders)
    
    # Red moves with support (strength 2), Blue moves with strength 1. Red should win.
    assert any(u.territory == "R2" and u.player == Player.RED for u in new_state.units)
    assert any(u.territory == "B1" and u.player == Player.BLUE for u in new_state.units)
    assert any(u.territory == "N1" and u.player == Player.RED for u in new_state.units)
    


# ── the runner ───────────────────────────────────────────────────────────

def _play(seed):
    """One whole game, reduced to what has to be reproducible."""
    random.seed(seed)
    runner = GameRunner([Agent(p, name, PlannerConfig(node_budget=200))
                         for p, name in zip(Player, PERSONAS)])
    runner.run(verbose=False)
    return [(str(h.orders), h.log.events) for h in runner.history]


def test_two_games_with_the_same_seed_are_identical():
    """Two games in ONE process, which is where the order-set cache used to
    leak the first game's random subsample into the second."""
    assert _play(3) == _play(3)


def _stub(player):
    return SimpleNamespace(player=player,
                           **{m: (lambda *a, **kw: []) for m in AGENT_METHODS})


def test_the_runner_rejects_an_agent_missing_a_method():
    table = [_stub(p) for p in players()]
    GameRunner(table)                       # the full shape is accepted
    del table[0].act
    with pytest.raises(TypeError, match="act"):
        GameRunner(table)


def test_the_runner_rejects_a_table_that_does_not_fill_the_board():
    """A short table would leave home centres unowned."""
    with pytest.raises(ValueError, match="board seats"):
        GameRunner([_stub(p) for p in players()[:-1]])


def _autumn_state(units, owned, turn=2):
    """An even (autumn) turn, so builds and removals run."""
    terr = get_all_territories()
    return GameState(
        turn=turn,
        units=[Unit(player=p, territory=t) for p, t in units],
        supply_centers={t: (Player.RED if t in owned else None) for t in terr},
        territory_owners={t: None for t in terr},
    )


def test_build_site_is_not_decided_by_territory_order():
    """Red is owed one build and has two empty centres to choose from.

    Taking the lowest-indexed one made autumn depend on territory order, which
    the board's own symmetry does not preserve -- worth 0.46 of a centre per
    seat. The choice is deterministic, but it must not track list order.
    """
    from src.engine.board import ring_board, use_board
    chosen = set()
    with use_board(ring_board(home_builds=False)):   # two legal sites to choose between
        for turn in range(2, 40, 2):
            state = _autumn_state([(Player.RED, "R1"), (Player.RED, "C1")],
                                  {"R1", "R2", "N1"}, turn=turn)
            new, _o, _l = resolve(state, [hold(Player.RED, "R1"),
                                          hold(Player.RED, "C1")])
            chosen |= {u.territory for u in new.units
                       if u.player == Player.RED} - {"R1", "C1"}
    assert chosen == {"R2", "N1"}, f"builds only ever landed on {chosen}"


def test_removal_is_not_decided_by_unit_creation_order():
    """Red must disband two of three units; which survives must not simply be
    whichever was created first."""
    survived = set()
    for turn in range(2, 40, 2):
        state = _autumn_state(
            [(Player.RED, "R1"), (Player.RED, "C1"), (Player.RED, "C2")],
            {"R1"}, turn=turn)
        new, _o, _l = resolve(
            state, [hold(Player.RED, t) for t in ("R1", "C1", "C2")])
        alive = {u.territory for u in new.units if u.player == Player.RED}
        assert len(alive) == 1
        survived |= alive
    assert len(survived) > 1, f"the same unit always survived: {survived}"


def test_autumn_resolution_does_not_touch_the_global_rng():
    """The planner resolves thousands of hypothetical autumns while searching.
    If builds drew from `random`, search effort would move the stream that
    decides the real ones."""
    import random as _random

    state = _autumn_state([(Player.RED, "R1"), (Player.RED, "C1")],
                          {"R1", "R2", "N1"})
    orders = [hold(Player.RED, "R1"), hold(Player.RED, "C1")]
    _random.seed(0)
    before = _random.getstate()
    for _ in range(20):
        resolve(state, orders)
    assert _random.getstate() == before


def hold(player, territory):
    return Order(player=player, unit_territory=territory, order_type=OrderType.HOLD)


def test_a_shared_lead_is_not_a_win_and_is_split_by_the_metrics():
    """80% of games end tied, so `max()` on the score dict was handing the
    result to whichever seat sorts first in Player order."""
    from src.common.schemas import leaders
    from src.evaluation.metrics import win_rates

    tied = {Player.RED: 3, Player.BLUE: 3, Player.GREEN: 2, Player.GOLD: 2}
    assert set(leaders(tied)) == {Player.RED, Player.BLUE}

    rates = win_rates([tied])
    assert rates[Player.RED] == rates[Player.BLUE] == 0.5
    assert rates[Player.GREEN] == rates[Player.GOLD] == 0.0
    assert abs(sum(rates.values()) - 1.0) < 1e-9

    clear = {Player.RED: 4, Player.BLUE: 3, Player.GREEN: 2, Player.GOLD: 1}
    assert leaders(clear) == [Player.RED]
    assert win_rates([clear])[Player.RED] == 1.0


def test_sampled_order_sets_do_not_touch_the_global_stream():
    """Past the cap the sample is seeded from the position, so the cache is pure."""
    import random
    from src.engine.orders import generate_all_order_sets, _SETS_CACHE, MAX_ORDER_SETS
    from src.engine.board import get_all_territories
    units = [Unit(player=Player.RED, territory=t) for t in ("R1", "R2", "N1", "C2", "Y2")]
    state = GameState(turn=1, units=units,
                      supply_centers={t: None for t in get_all_territories()},
                      territory_owners={t: None for t in get_all_territories()})
    _SETS_CACHE.clear()
    random.seed(1)
    first = generate_all_order_sets(state, Player.RED)
    after = random.random()
    random.seed(1)
    assert random.random() == after
    _SETS_CACHE.clear()
    assert len(first) == MAX_ORDER_SETS
    assert generate_all_order_sets(state, Player.RED) == first
