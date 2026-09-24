"""The board is configurable, so what used to be constants must hold at every
size, and the default must stay what it was."""
import pytest

from src.common.config import GameConfig
from src.common.schemas import Player
from src.engine.board import (
    Board, MAX_SEATS, MIN_SEATS, active, is_supply_center, players,
    ring_board, use_board, win_centers,
)
from src.harness import build_runner, game_setup, play


# The original hand-written map, up to the two corridors' names.
ORIGINAL = {
    "R1": {"R2", "Y2", "C2"}, "R2": {"R1", "B1", "N1"},
    "B1": {"R2", "B2", "N1"}, "B2": {"B1", "G1", "C1"},
    "G1": {"B2", "G2", "C1"}, "G2": {"G1", "Y1", "N2"},
    "Y1": {"G2", "Y2", "N2"}, "Y2": {"Y1", "R1", "C2"},
    "N1": {"R2", "B1", "C1", "C2"}, "N2": {"G2", "Y1", "C1", "C2"},
    "C1": {"B2", "G1", "N1", "N2"}, "C2": {"Y2", "R1", "N1", "N2"},
}


def test_default_board_is_the_original_map():
    b = active()
    assert {t: set(n) for t, n in b.adjacency.items()} == ORIGINAL
    assert set(b.supply_centers) == set(ORIGINAL) - {"C1", "C2"}
    assert b.win_centers == 6 and b.max_turns == 12
    assert b.players == [Player.RED, Player.BLUE, Player.GREEN, Player.GOLD]


@pytest.mark.parametrize("seats", range(MIN_SEATS, MAX_SEATS + 1))
def test_every_ring_size_is_a_legal_and_fair_board(seats):
    # __post_init__ runs the checks, so constructing one is most of the test.
    b = ring_board(seats)
    assert len(b.players) == seats
    assert len(b.territories) == seats * 3
    reach = {p: sorted(len(b.adjacency[t]) for t in homes)
             for p, homes in b.home_centers.items()}
    assert len({tuple(v) for v in reach.values()}) == 1


@pytest.mark.parametrize("bad", [
    dict(seats=1), dict(seats=MAX_SEATS + 1), dict(seats=99),
    dict(seats=4, homes_per_player=0),
])
def test_impossible_boards_are_refused(bad):
    with pytest.raises(ValueError):
        ring_board(**bad)


def test_unreachable_win_threshold_is_refused():
    with pytest.raises(AssertionError):
        ring_board(4, win_centers=999)


@pytest.mark.parametrize("seats", range(MIN_SEATS, MAX_SEATS + 1))
@pytest.mark.parametrize("homes", [1, 2, 3])
def test_no_board_is_won_before_anyone_moves(seats, homes):
    """Half the centres is already held at setup on a small board."""
    b = ring_board(seats, homes)
    assert b.win_centers > homes


def test_a_threshold_already_held_at_setup_is_refused():
    with pytest.raises(AssertionError, match="already held at setup"):
        ring_board(4, 2, win_centers=2)


def test_a_hand_written_board_need_not_be_a_ring():
    """A Board is just data; the ring generator is not required."""
    line = Board(
        adjacency={"A": ["B"], "B": ["A", "C"], "C": ["B"]},
        supply_centers=["A", "C"],
        home_centers={Player.RED: ["A"], Player.BLUE: ["C"]},
        win_centers=2,
        max_turns=2,
    )
    with use_board(line):
        assert players() == [Player.RED, Player.BLUE]
        assert is_supply_center("A") and not is_supply_center("B")


def test_use_board_restores_the_previous_board():
    before = active()
    with use_board(ring_board(7)):
        assert len(players()) == 7 and win_centers() == 12
    assert active() is before
    assert len(players()) == 4


@pytest.mark.parametrize("seats", [2, 3, 5, 8])
def test_a_game_of_any_size_plays_to_the_end(seats):
    cfg = GameConfig(seed=3, n_seats=seats, max_turns=3, node_budget=100)
    runner = play(cfg)
    assert set(runner.center_counts()) == set(cfg.make_board().players)
    assert len(runner.personas()) == seats
    assert len(runner.history) == 3 or runner.winner() is not None


def test_board_settings_reach_the_runner():
    cfg = GameConfig(seed=1, n_seats=3, homes_per_player=3,
                     win_centers=4, max_turns=7)
    with game_setup(cfg):
        runner = build_runner(cfg)
    assert runner.max_turns == 7 and runner.win_centers == 4
    assert len(runner.state.units) == 9          # 3 seats x 3 homes


def test_agents_must_fill_exactly_the_board_s_seats():
    from src.agents.agent import Agent
    from src.engine.runner import GameRunner
    with use_board(ring_board(4)):
        too_few = [Agent(p, "Honest") for p in players()[:3]]
        with pytest.raises(ValueError, match="board seats"):
            GameRunner(too_few)


def test_config_round_trips_the_board():
    cfg = GameConfig(seed=2, n_seats=6, homes_per_player=1, max_turns=9)
    back = GameConfig.from_dict(cfg.to_dict())
    assert back.make_board().to_dict() == cfg.make_board().to_dict()
    assert len(back.seats()) == 6


def test_an_explicit_board_overrides_the_ring_settings():
    explicit = ring_board(5, max_turns=4).to_dict()
    cfg = GameConfig(n_seats=2, board=explicit)       # n_seats is ignored
    assert len(cfg.make_board().players) == 5
    assert cfg.make_board().max_turns == 4


@pytest.mark.parametrize("seats", range(MIN_SEATS, MAX_SEATS + 1))
def test_every_seat_holds_every_persona_equally_often(seats):
    """Rotation must reach every (seat, persona) pair equally, at every seat
    count -- not just where seats and personas happen to be the same number."""
    from collections import Counter
    from src.common.config import PERSONA_ORDER, seating_for

    table = ring_board(seats).players
    tally = {p: Counter() for p in table}
    for seed in range(20 * len(PERSONA_ORDER)):
        for seat, persona in seating_for(seed, table).items():
            tally[seat][persona] += 1
    for seat, counts in tally.items():
        assert set(counts) == set(PERSONA_ORDER), seat
        assert len(set(counts.values())) == 1, (seat, counts)


def test_a_disconnected_board_is_refused():
    with pytest.raises(AssertionError, match="not connected"):
        Board(adjacency={"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]},
              supply_centers=["A", "B", "C", "D"],
              home_centers={Player.RED: ["A"], Player.BLUE: ["C"]},
              win_centers=2, max_turns=2)


def test_a_seat_may_hold_non_adjacent_homes():
    """A split seat is legal. It cannot support its own units, since support
    needs adjacency, so its two units never combine."""
    ring = ring_board(4)
    split = Board(
        adjacency={t: list(n) for t, n in ring.adjacency.items()},
        supply_centers=list(ring.supply_centers),
        home_centers={Player.RED: ["R1", "G1"], Player.BLUE: ["R2", "G2"],
                      Player.GREEN: ["B1", "Y1"], Player.GOLD: ["B2", "Y2"]},
        win_centers=ring.win_centers, max_turns=ring.max_turns)
    with use_board(split):
        assert "G1" not in split.adjacency["R1"]
        assert len(players()) == 4


@pytest.mark.parametrize("seats", range(MIN_SEATS, MAX_SEATS + 1))
def test_generated_layout_has_no_overlapping_nodes(seats):
    """The dashboard draws from `positions`, scaling node radius by
    sqrt(12/territories). Nodes must not collide at any board size."""
    import math

    b = ring_board(seats)
    scale = min(1.0, math.sqrt(12 / len(b.territories)))
    centres = set(b.supply_centers)
    radius = {t: (25 if t in centres else 20) * scale for t in b.territories}
    for i, a in enumerate(b.territories):
        for c in b.territories[i + 1:]:
            gap = math.dist(b.positions[a], b.positions[c])
            assert gap > radius[a] + radius[c], f"{a} and {c} overlap at {seats} seats"


@pytest.mark.parametrize("seats", range(MIN_SEATS, MAX_SEATS + 1))
def test_every_seat_has_a_distinct_code_for_the_ui(seats):
    """Ownership is drawn as a letter as well as a colour, so the letters have
    to be unique -- eight colours are not all distinguishable."""
    from src.engine.board import SEAT_CODES

    codes = [SEAT_CODES[p] for p in ring_board(seats).players]
    assert len(set(codes)) == len(codes)


def test_order_cache_is_not_blind_to_the_board():
    """Two boards can share territory names and differ in adjacency. Keying the
    order-set cache on unit placement alone served one board's legal moves for
    the other -- reachable with two dashboard games open on different maps."""
    from src.common.schemas import GameState, Unit
    from src.engine.board import get_adjacent
    from src.engine.orders import generate_all_order_sets, _SETS_CACHE

    base = ring_board(4)
    adj = {t: list(n) for t, n in base.adjacency.items()}
    adj["R1"].append("N2")
    adj["N2"].append("R1")
    edged = Board(adjacency=adj, supply_centers=list(base.supply_centers),
                  home_centers={p: list(h) for p, h in base.home_centers.items()},
                  win_centers=base.win_centers, max_turns=base.max_turns)

    terr = base.territories
    state = GameState(turn=1, units=[Unit(player=Player.RED, territory="R1")],
                      supply_centers={t: None for t in terr},
                      territory_owners={t: None for t in terr})
    _SETS_CACHE.clear()
    with use_board(base):
        generate_all_order_sets(state, Player.RED)
    with use_board(edged):
        offered = {o.target for s in generate_all_order_sets(state, Player.RED)
                   for o in s if o.target}
        assert "N2" in get_adjacent("R1")
        assert "N2" in offered, "cache served the previous board's legal moves"


def test_a_board_cannot_be_edited_after_construction():
    """`fingerprint` keys the order cache and `_centre_set` answers
    `is_supply_center`; both are derived once, so an edited board would serve
    stale legal moves."""
    b = ring_board(4)
    with pytest.raises(Exception):
        b.win_centers = 99


@pytest.mark.parametrize("homes", [3, 4])
def test_subsampled_order_sets_keep_the_hold_and_every_single_unit_action(homes):
    """Above MAX_ORDER_SETS the enumeration is sampled. The documented promise
    is that all-hold and every one-unit action survive the cut."""
    from src.common.schemas import GameState, OrderType, Unit
    from src.engine.orders import generate_all_order_sets, MAX_ORDER_SETS, _SETS_CACHE

    b = ring_board(4, homes)
    with use_board(b):
        _SETS_CACHE.clear()
        terr = b.territories
        units = [Unit(player=p, territory=t)
                 for p in (Player.RED, Player.BLUE) for t in b.home_centers[p]]
        state = GameState(turn=1, units=units,
                          supply_centers={t: None for t in terr},
                          territory_owners={t: None for t in terr})
        sets = generate_all_order_sets(state, Player.RED)
        assert len(sets) == MAX_ORDER_SETS, "expected this board to subsample"
        acting = [sum(1 for o in s if o.order_type != OrderType.HOLD) for s in sets]
        assert acting.count(0) == 1, "the all-hold set must survive"
        assert acting.count(1) >= homes, "every single-unit action must survive"


@pytest.mark.parametrize("seats", range(MIN_SEATS, MAX_SEATS + 1))
def test_gap_centres_holds_the_contested_pool_fixed_across_table_sizes(seats):
    """By default the middle alternates centre/corridor when the seat count is
    even and is all centres when it is odd, so the contested pool jumps with
    parity and confounds any comparison across table sizes."""
    homes = 2
    default = ring_board(seats, homes)
    allc = ring_board(seats, homes, gap_centres=True)
    none = ring_board(seats, homes, gap_centres=False)

    # Pinned: centres scale linearly with the table, with no parity jump.
    assert len(none.supply_centers) == seats * homes
    assert len(allc.supply_centers) == seats * homes + seats
    # The default sits at one end or the other depending on parity.
    expected = allc if seats % 2 else default
    assert len(default.supply_centers) == len(expected.supply_centers)
    # Every seat still faces the same middle, whichever setting is used.
    for b in (default, allc, none):
        reach = {p: sorted(len(b.adjacency[t]) for t in hs)
                 for p, hs in b.home_centers.items()}
        assert len({tuple(v) for v in reach.values()}) == 1


@pytest.mark.parametrize("home_only, expected", [(False, ["N1", "R1", "R2"]), (True, ["R1", "R2"])])
def test_build_rule(home_only, expected):
    """Build anywhere owned, or only on home centres as in standard Diplomacy."""
    from src.common.schemas import GameState, Order, OrderType, Unit
    from src.engine.adjudicator import resolve
    from src.engine.board import get_all_territories, get_supply_centers
    with game_setup(GameConfig(home_builds=home_only)):
        sc = {t: None for t in get_supply_centers()}
        sc.update({"N1": Player.RED, "R1": Player.RED, "R2": Player.RED})
        state = GameState(turn=2, units=[Unit(player=Player.RED, territory="R2")],
                          supply_centers=sc,
                          territory_owners={t: None for t in get_all_territories()})
        hold = Order(player=Player.RED, unit_territory="R2", order_type=OrderType.HOLD)
        new = resolve(state, [hold])[0]
    assert sorted(u.territory for u in new.units) == expected


@pytest.mark.parametrize("fraction, expected", [(0.5, 5), (0.6, 6), (0.7, 7), (0.1, 3)])
def test_win_fraction_sets_the_default_threshold(fraction, expected):
    """A share of the board's centres, never below one more than a seat starts with."""
    assert GameConfig(win_fraction=fraction).make_board().win_centers == expected
    assert GameConfig(win_fraction=fraction, win_centers=4).make_board().win_centers == 4


def test_win_fraction_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        GameConfig(win_fraction=0).make_board()
