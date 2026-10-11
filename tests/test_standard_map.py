"""Fleets, coasts and convoys on the standard map."""
from src.common.schemas import GameState, Order, OrderType, Unit, UnitType
from src.engine.adjudicator import resolve, default_choice
from src.engine.board import use_board, build_options
from src.engine.orders import generate_orders_for_unit
from src.engine.standard_map import standard_board, POWERS

E, F, G, R = (POWERS[k] for k in ("England", "France", "Germany", "Russia"))
BOARD = standard_board()


def unit(p, loc, kind="A"):
    t, _, coast = loc.partition("/")
    return Unit(player=p, territory=t, coast=coast or None,
                unit_type=UnitType.FLEET if kind == "F" else UnitType.ARMY)


def play(units, orders, turn=1):
    with use_board(BOARD):
        state = GameState(turn=turn, units=units,
                          supply_centers={t: None for t in BOARD.supply_centers},
                          territory_owners={})
        new, _o, log = resolve(state, orders)
    return {u.location: u.player for u in new.units}, log.events


def mv(p, a, b, coast=None):
    return Order(player=p, unit_territory=a, order_type=OrderType.MOVE, target=b, coast=coast)


def sup(p, a, b, frm=None):
    return Order(player=p, unit_territory=a, order_type=OrderType.SUPPORT, target=b,
                 supported_from=frm)


def convoy(p, fleet, frm, to):
    return Order(player=p, unit_territory=fleet, order_type=OrderType.CONVOY, target=to,
                 supported_from=frm)


def test_the_standard_setup():
    assert len(BOARD.territories) == 75 and len(BOARD.supply_centers) == 34
    assert len(BOARD.starting_units()) == 22
    assert BOARD.win_centers == 18


def test_fleets_stay_at_sea_and_on_coasts_armies_on_land():
    at, _ = play([unit(G, "Kie", "F"), unit(E, "Lon")],
                 [mv(G, "Kie", "Mun"), mv(E, "Lon", "NTH")])
    assert at == {"Kie": G, "Lon": E}


def test_coasts():
    # 6.B.1: Portugal reaches both Spanish coasts, so the coast must be named.
    at, _ = play([unit(F, "Por", "F")], [mv(F, "Por", "Spa")])
    assert at == {"Por": F}
    # 6.B.2: from Gascony only the north coast is reachable, so it is implied.
    at, _ = play([unit(F, "Gas", "F")], [mv(F, "Gas", "Spa")])
    assert at == {"Spa/nc": F}
    # 6.B.3: a coast the fleet cannot reach is refused.
    at, _ = play([unit(F, "Gas", "F")], [mv(F, "Gas", "Spa", coast="sc")])
    assert at == {"Gas": F}
    # 6.B.4: support into a province counts whatever coast the mover is aimed at.
    at, _ = play([unit(F, "Gas", "F"), unit(F, "Mar", "F"), unit(G, "Spa/sc", "F")],
                 [sup(F, "Gas", "Spa", "Mar"), mv(F, "Mar", "Spa", coast="sc")])
    assert at["Spa/sc"] == F


def test_convoy_carries_an_army_over_the_sea():
    at, _ = play([unit(E, "Lon"), unit(E, "NTH", "F")],
                 [mv(E, "Lon", "Nwy"), convoy(E, "NTH", "Lon", "Nwy")])
    assert at == {"Nwy": E, "NTH": E}


def test_dislodged_convoy_fleet_strands_the_army():
    at, log = play([unit(E, "Lon"), unit(E, "NTH", "F"), unit(G, "Den", "F"), unit(G, "HEL", "F")],
                   [mv(E, "Lon", "Bel"), convoy(E, "NTH", "Lon", "Bel"),
                    mv(G, "Den", "NTH"), sup(G, "HEL", "NTH", "Den")])
    assert at["Lon"] == E and at["NTH"] == G
    assert "Convoy fails: Lon -> Bel" in log


def test_convoyed_army_does_not_cut_support_against_its_own_convoy():
    """The army bound for Brest would cut Brest's support, but that support
    is aimed at the fleet carrying it, so the convoy fails instead."""
    at, _ = play([unit(E, "Lon"), unit(E, "ENG", "F"), unit(F, "Bre", "F"), unit(F, "MAO", "F")],
                 [mv(E, "Lon", "Bre"), convoy(E, "ENG", "Lon", "Bre"),
                  sup(F, "Bre", "ENG", "MAO"), mv(F, "MAO", "ENG")])
    assert at["Lon"] == E and at["ENG"] == F and at["Bre"] == F


def test_a_convoyed_army_swaps_without_a_head_to_head():
    """Armies swapping, one by its own convoy, both arrive."""
    at, _ = play([unit(E, "Nwy"), unit(E, "SKA", "F"), unit(R, "Swe")],
                 [mv(E, "Nwy", "Swe"), convoy(E, "SKA", "Nwy", "Swe"), mv(R, "Swe", "Nwy")])
    assert at["Swe"] == E and at["Nwy"] == R


def test_fleet_retreats_along_the_coast():
    at, log = play([unit(F, "Spa/sc", "F"), unit(G, "Mar"), unit(G, "Gas")],
                   [mv(G, "Mar", "Spa"), sup(G, "Gas", "Spa", "Mar")])
    assert at["Spa"] == G
    retreat = next(e for e in log if e.startswith("Retreat: Spa"))
    assert retreat.split("-> ")[1] in {"Por", "MAO", "WES", "LYO"}


def test_builds_offer_armies_and_fleets_one_per_province():
    with use_board(BOARD):
        assert build_options("StP") == ["StP", "F StP/nc", "F StP/sc"]
        assert build_options("Mos") == ["Mos"]
        picks = default_choice(R, "build", build_options("StP") + build_options("Sev"), 3, None, 2)
    assert len(picks) == 2


def test_generated_orders_include_convoys_and_are_legal():
    with use_board(BOARD):
        state = GameState(turn=1, units=[unit(E, "Lon"), unit(E, "NTH", "F")],
                          supply_centers={t: None for t in BOARD.supply_centers},
                          territory_owners={})
        fleet = generate_orders_for_unit(state, state.units[1])
        army = generate_orders_for_unit(state, state.units[0])
    assert any(o.order_type == OrderType.CONVOY and o.target == "Nwy" for o in fleet)
    assert any(o.order_type == OrderType.MOVE and o.target == "Nwy" for o in army)
    assert not any(o.target == "ENG" for o in army if o.order_type == OrderType.MOVE)


def test_order_sets_only_move_by_sea_with_own_convoy():
    from src.engine.orders import generate_all_order_sets
    with use_board(BOARD):
        state = GameState(turn=1, units=[unit(E, "Lon"), unit(E, "NTH", "F")],
                          supply_centers={t: None for t in BOARD.supply_centers},
                          territory_owners={})
        sets = generate_all_order_sets(state, E)
    by_sea = [s for s in sets if any(o.unit_territory == "Lon" and o.target == "Nwy" for o in s)]
    assert by_sea and all(any(o.order_type == OrderType.CONVOY and o.target == "Nwy" for o in s)
                          for s in by_sea)


def test_a_standard_game_plays_and_replays(tmp_path):
    from src.common.config import GameConfig
    from src.harness import build_runner, game_setup
    from src.engine.replay import save_replay, load_replay, reconstruct_states
    cfg = GameConfig(seed=1, map="standard", max_turns=2, node_budget=100)
    with game_setup(cfg):
        runner = build_runner(cfg)
        runner.run()
        path = save_replay(runner.history, runner.state, directory=str(tmp_path),
                           result=runner.result())
    states = reconstruct_states(load_replay(path))
    assert len(runner.history) == 2
    assert {u.location: u.unit_type for u in states[-1].units} == \
        {u.location: u.unit_type for u in runner.state.units}
    assert any(u.unit_type == UnitType.FLEET for u in states[-1].units)
