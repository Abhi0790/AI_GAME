"""Adjudicator conformance: DATC cases, adapted, plus property tests.

The DATC is the reference test set for Diplomacy adjudicators. This game has
no fleets, coasts, convoys or retreats, so the sections covering them do not
apply; what is left is section 6.A (illegal orders), 6.C (circular movement
and swaps), 6.D (supports and dislodges) and 6.E (head-to-head battles), which is what this file walks
through on the twelve-territory board.

Adjacency used below — the default board, `ring_board(4)`:
    R1: R2 Y2 C2      R2: R1 B1 N1      B1: R2 B2 N1      B2: B1 G1 C1
    G1: B2 G2 C1      G2: G1 Y1 N2      Y1: G2 Y2 N2      Y2: Y1 R1 C2
    N1: R2 B1 C1 C2   N2: G2 Y1 C1 C2   C1: B2 G1 N1 N2   C2: Y2 R1 N1 N2
`test_board_adjacency_is_symmetric` reads the real thing, so this is a map for
the reader, not a second source of truth.
"""

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from src.common.schemas import (
    GameState, Order, OrderType, Player, Unit, Commitment, CommitmentType,
)
from src.engine.adjudicator import resolve
import random

from src.engine.board import (
    get_all_territories, get_adjacent, adjacency, players, active,
)
from src.engine.orders import generate_all_order_sets

TERRITORIES = get_all_territories()


def board(*units, turn=1):
    """units: (player, territory) pairs."""
    us = [Unit(player=p, territory=t) for p, t in units]
    return GameState(
        turn=turn, units=us,
        supply_centers={t: None for t in TERRITORIES},
        territory_owners={t: None for t in TERRITORIES},
    )


def move(p, frm, to):
    return Order(player=p, unit_territory=frm, order_type=OrderType.MOVE, target=to)


def hold(p, at):
    return Order(player=p, unit_territory=at, order_type=OrderType.HOLD)


def support_move(p, at, frm, to):
    return Order(player=p, unit_territory=at, order_type=OrderType.SUPPORT,
                 target=to, supported_from=frm)


def support_hold(p, at, target):
    return Order(player=p, unit_territory=at, order_type=OrderType.SUPPORT, target=target)


def positions(state):
    return {u.territory: u.player for u in state.units}


R, B, G, Y = Player.RED, Player.BLUE, Player.GREEN, Player.GOLD


# ── 6.A — orders that are not legal at all ──────────────────────────────

class TestIllegalOrders:
    def test_6A_move_to_non_adjacent_is_ignored(self):
        """6.A.3 — an army cannot move to a territory it does not border."""
        new, _o, log = resolve(board((R, "R1")), [move(R, "R1", "G1")])
        assert positions(new) == {"R1": R}
        assert any("not adjacent" in e for e in log.events)

    def test_6A_move_to_own_square_is_ignored(self):
        new, _o, _l = resolve(board((R, "R1")), [move(R, "R1", "R1")])
        assert positions(new) == {"R1": R}

    def test_6A_support_to_non_adjacent_is_ignored(self):
        """6.A.7 — a unit can only support into a territory it borders."""
        state = board((R, "R1"), (R, "R2"), (B, "B1"))
        new, _o, log = resolve(state, [
            move(R, "R2", "B1"),
            support_move(R, "R1", "R2", "B1"),   # R1 does not border B1
            hold(B, "B1"),
        ])
        assert positions(new)["B1"] == B  # unsupported attack bounces
        assert any("Invalid support" in e for e in log.events)

    def test_6A_orders_for_a_unit_that_is_not_mine(self):
        new, _o, _l = resolve(board((R, "R1"), (B, "R2")), [move(B, "R1", "Y2")])
        assert positions(new) == {"R1": R, "R2": B}

    def test_6A_orders_for_a_territory_that_does_not_exist(self):
        new, _o, _l = resolve(board((R, "R1")), [move(R, "VOID", "ALSO_VOID")])
        assert positions(new) == {"R1": R}

    def test_6A_second_order_for_the_same_unit_is_ignored(self):
        """Contradictory orders in one turn: the first one stands."""
        state = board((R, "R1"))
        new, _o, log = resolve(state, [hold(R, "R1"), move(R, "R1", "R2")])
        assert positions(new) == {"R1": R}
        assert any("Duplicate order" in e for e in log.events)


# ── 6.C — circular movement and swaps ───────────────────────────────────

class TestMovementCycles:
    def test_6C_two_units_cannot_swap_places(self):
        """6.C.4 — a head-to-head of equal strength bounces. Both units stay
        where they started; they do not walk through each other."""
        state = board((R, "R1"), (B, "R2"))
        new, _o, _l = resolve(state, [move(R, "R1", "R2"), move(B, "R2", "R1")])
        assert positions(new) == {"R1": R, "R2": B}

    def test_6C_an_illegal_support_does_not_win_a_head_to_head(self):
        """C2 borders R1 but not R2, so it cannot support an attack into R2.
        The order is dropped and the head-to-head bounces as usual."""
        state = board((R, "R1"), (B, "R2"), (R, "C2"))
        new, _o, _l = resolve(state, [
            move(R, "R1", "R2"),
            move(B, "R2", "R1"),
            support_move(R, "C2", "R1", "R2"),
        ])
        assert positions(new) == {"R1": R, "R2": B, "C2": R}

    def test_6C_head_to_head_with_legal_support_dislodges(self):
        state = board((R, "R2"), (B, "N1"), (R, "B1"))
        new, _o, _l = resolve(state, [
            move(R, "R2", "N1"),
            move(B, "N1", "R2"),
            support_move(R, "B1", "R2", "N1"),   # B1 borders N1
        ])
        assert positions(new)["N1"] == R
        assert "R2" not in positions(new)        # Blue was dislodged, not swapped

    def test_6C_three_unit_circle_all_move(self):
        """6.C.1 — everybody follows the unit in front and the circle turns."""
        state = board((R, "R1"), (B, "R2"), (G, "B1"))
        new, _o, _l = resolve(state, [
            move(R, "R1", "R2"), move(B, "R2", "B1"), move(G, "B1", "R2"),
        ])
        # R2 is contested by Red and Green, so the circle is not clean: bounce.
        assert positions(new) == {"R1": R, "R2": B, "B1": G}

    def test_6C_chain_of_moves_into_a_vacated_square(self):
        state = board((R, "R1"), (B, "R2"), (G, "B1"))
        new, _o, _l = resolve(state, [
            move(R, "R1", "R2"), move(B, "R2", "N1"), move(G, "B1", "B2"),
        ])
        assert positions(new) == {"R2": R, "N1": B, "B2": G}

    def test_6C_follower_bounces_when_the_unit_ahead_bounces(self):
        """The bug this replaced: a unit whose own move failed used to be
        dislodged by whoever was queued behind it."""
        state = board((R, "R1"), (B, "R2"), (G, "B1"))
        new, _o, _l = resolve(state, [
            move(R, "R1", "R2"),
            move(B, "R2", "B1"),   # bounces off Green
            hold(G, "B1"),
        ])
        assert positions(new) == {"R1": R, "R2": B, "B1": G}


# ── 6.D — supports and dislodges ────────────────────────────────────────

class TestSupportsAndDislodges:
    def test_6D_supported_attack_dislodges(self):
        state = board((R, "R2"), (R, "B1"), (B, "N1"))
        new, _o, _l = resolve(state, [
            move(R, "R2", "N1"), support_move(R, "B1", "R2", "N1"), hold(B, "N1"),
        ])
        assert positions(new)["N1"] == R
        assert B not in positions(new).values()

    def test_6D_supported_hold_stops_a_supported_attack(self):
        """Attack 2 against defence 2 bounces: strength must exceed defence."""
        state = board((R, "R2"), (R, "B1"), (B, "N1"), (B, "C2"))
        new, _o, _l = resolve(state, [
            move(R, "R2", "N1"), support_move(R, "B1", "R2", "N1"),
            hold(B, "N1"), support_hold(B, "C2", "N1"),
        ])
        assert positions(new)["N1"] == B

    def test_6D_support_is_cut_by_an_attack(self):
        state = board((R, "R2"), (R, "B1"), (B, "N1"), (B, "B2"))
        new, _o, _l = resolve(state, [
            move(R, "R2", "N1"),
            support_move(R, "B1", "R2", "N1"),
            move(B, "B2", "B1"),          # cuts B1's support
            hold(B, "N1"),
        ])
        assert positions(new)["N1"] == B   # attack is back to strength 1

    def test_6D_support_is_not_cut_from_the_square_being_attacked(self):
        """6.D.15 — an attack coming out of the supported territory does not
        cut the support aimed at it."""
        state = board((R, "R2"), (R, "B1"), (B, "N1"))
        new, _o, _l = resolve(state, [
            move(R, "R2", "N1"),
            support_move(R, "B1", "R2", "N1"),
            move(B, "N1", "B1"),           # head to head with the supporter
        ])
        assert positions(new)["N1"] == R

    def test_6D_no_self_dislodgement(self):
        """6.D.10 — a player cannot dislodge their own unit, with or without
        support."""
        state = board((R, "R2"), (R, "N1"), (R, "B1"))
        new, _o, _l = resolve(state, [
            move(R, "R2", "N1"), support_move(R, "B1", "R2", "N1"), hold(R, "N1"),
        ])
        assert positions(new) == {"R2": R, "N1": R, "B1": R}

    def test_6D_two_equal_attacks_bounce_and_the_defender_survives(self):
        state = board((R, "R2"), (B, "B1"), (G, "C1"))
        new, _o, _l = resolve(state, [
            move(R, "R2", "N1"), move(B, "B1", "N1"), move(G, "C1", "N1"),
        ])
        assert "N1" not in positions(new)

    def test_6D_a_bouncing_attacker_still_blocks_the_square(self):
        state = board((R, "R2"), (B, "B1"), (G, "C1"), (G, "C2"))
        new, _o, _l = resolve(state, [
            move(R, "R2", "N1"),
            move(B, "B1", "N1"),
            move(G, "C1", "N1"), support_move(G, "C2", "C1", "N1"),
        ])
        assert positions(new)["N1"] == G   # strength 2 beats two strength-1s

    def test_6D_own_support_cannot_dislodge_own_unit(self):
        """6.D.12 — supporting a foreign attack on your own unit does not count."""
        s = board((R, "R1"), (B, "R2"), (R, "C2"))
        out = resolve(s, [hold(R, "R1"), move(B, "R2", "R1"),
                          support_move(R, "C2", "R2", "R1")])[0]
        assert positions(out) == {"R1": R, "R2": B, "C2": R}

    def test_6D_foreign_support_is_still_counted_by_others(self):
        """6.D.14 — with a support of its own on top, the attack does dislodge."""
        s = board((R, "R1"), (B, "R2"), (R, "C2"), (G, "Y2"))
        out = resolve(s, [hold(R, "R1"), move(B, "R2", "R1"),
                          support_move(R, "C2", "R2", "R1"),
                          support_move(G, "Y2", "R2", "R1")])[0]
        assert positions(out) == {"R1": B, "C2": R, "Y2": G}

    def test_6E_own_support_does_not_win_a_head_to_head_against_own_unit(self):
        """The same rule in a head-to-head: Red's support for Blue does not beat Red."""
        s = board((R, "R1"), (B, "R2"), (R, "C2"))
        out = resolve(s, [move(R, "R1", "R2"), move(B, "R2", "R1"),
                          support_move(R, "C2", "R2", "R1")])[0]
        assert positions(out) == {"R1": R, "R2": B, "C2": R}

    def test_6D_support_hold_for_a_unit_that_moves_does_not_help_it(self):
        state = board((R, "R2"), (R, "B1"), (B, "N1"), (B, "C2"))
        new, _o, _l = resolve(state, [
            move(R, "R2", "N1"),
            support_move(R, "B1", "R2", "N1"),
            move(B, "N1", "C1"),
            support_hold(B, "C2", "N1"),   # N1 is leaving; nothing to hold
        ])
        assert positions(new)["N1"] == R
        assert positions(new)["C1"] == B


# ── 6.E — head-to-head battles ──────────────────────────────────────────

class TestHeadToHead:
    def test_6E_dislodged_unit_has_no_effect_on_attackers_area(self):
        """6.E.1 — a unit beaten head to head does not block the square its
        attacker left, so a third unit walks in."""
        s = board((R, "R2"), (R, "B1"), (B, "N1"), (G, "R1"))
        out = resolve(s, [move(R, "R2", "N1"), support_move(R, "B1", "R2", "N1"),
                          move(B, "N1", "R2"), move(G, "R1", "R2")])[0]
        assert positions(out) == {"N1": R, "B1": R, "R2": G}

    def test_6E_head_to_head_bounce_still_blocks(self):
        """An equal head-to-head bounces both, and the square stays blocked."""
        s = board((R, "R2"), (B, "N1"), (G, "R1"))
        out = resolve(s, [move(R, "R2", "N1"), move(B, "N1", "R2"),
                          move(G, "R1", "R2")])[0]
        assert positions(out) == {"R2": R, "N1": B, "R1": G}


def _map(edges, homes):
    """A small board with the DATC's own geography, so a case reads as published."""
    from src.engine.board import Board
    adj = {}
    for e in edges.split():
        a, b = e.split("-")
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    return Board(adjacency=adj, supply_centers=[t for ts in homes.values() for t in ts],
                 home_centers=homes, win_centers=2)


# The North Sea and its neighbours, plus Berlin/Munich. Armies stand in for fleets.
NORTH_SEA = _map("NTH-NWG NTH-NOR NTH-SKA NTH-DEN NTH-HEL NTH-HOL NTH-BEL NTH-ENG NTH-YOR "
                 "NTH-EDI NWG-NOR NWG-EDI NOR-SKA SKA-DEN HEL-HOL HEL-KIE HEL-DEN HOL-BEL "
                 "HOL-KIE HOL-RUH BEL-ENG BEL-RUH YOR-EDI KIE-RUH KIE-DEN KIE-BER KIE-MUN "
                 "BER-MUN MUN-RUH",
                 {G: ["EDI"], R: ["KIE"], Y: ["NOR"], B: ["BEL"]})
BALKANS = _map("BUD-VIE BUD-GAL BUD-RUM BUD-SER BUD-TRI VIE-GAL VIE-TRI RUM-GAL RUM-SER SER-TRI",
               {R: ["TRI"], B: ["VIE"], G: ["SER"], Y: ["GAL"]})


def _datc(board_, units, orders):
    from src.engine.board import use_board
    with use_board(board_):
        state = GameState(turn=1, units=[Unit(player=p, territory=t) for p, t in units],
                          supply_centers={t: None for t in board_.supply_centers},
                          territory_owners={t: None for t in board_.territories})
        return positions(resolve(state, orders)[0])


class TestHeadToHeadDATC:
    """6.E on the published geography. Germany R, France B, England G, Russia Y;
    in the Balkans case Austria R, Italy B, Russia Y."""

    def test_6E2_no_self_dislodgement_in_head_to_head(self):
        units = [(R, "BER"), (R, "KIE"), (R, "MUN")]
        orders = [move(R, "BER", "KIE"), move(R, "KIE", "BER"), support_move(R, "MUN", "BER", "KIE")]
        assert _datc(NORTH_SEA, units, orders) == {t: p for p, t in units}

    def test_6E3_no_help_in_dislodging_own_unit(self):
        units = [(R, "BER"), (R, "MUN"), (G, "KIE")]
        orders = [move(R, "BER", "KIE"), support_move(R, "MUN", "KIE", "BER"), move(G, "KIE", "BER")]
        assert _datc(NORTH_SEA, units, orders) == {t: p for p, t in units}

    def test_6E6_not_dislodged_because_of_own_support_still_has_effect(self):
        units = [(R, "HOL"), (R, "HEL"), (B, "NTH"), (B, "BEL"), (B, "ENG"), (Y, "KIE"), (Y, "RUH")]
        orders = [move(R, "HOL", "NTH"), support_move(R, "HEL", "HOL", "NTH"),
                  move(B, "NTH", "HOL"), support_move(B, "BEL", "NTH", "HOL"),
                  support_move(B, "ENG", "HOL", "NTH"),
                  support_move(Y, "KIE", "RUH", "HOL"), move(Y, "RUH", "HOL")]
        assert _datc(NORTH_SEA, units, orders) == {t: p for p, t in units}

    def test_6E7_no_self_dislodgement_with_beleaguered_garrison(self):
        units = [(G, "NTH"), (G, "YOR"), (R, "HOL"), (R, "HEL"), (Y, "SKA"), (Y, "NOR")]
        orders = [hold(G, "NTH"), support_move(G, "YOR", "NOR", "NTH"),
                  support_move(R, "HOL", "HEL", "NTH"), move(R, "HEL", "NTH"),
                  support_move(Y, "SKA", "NOR", "NTH"), move(Y, "NOR", "NTH")]
        assert _datc(NORTH_SEA, units, orders) == {t: p for p, t in units}

    def test_6E8_beleaguered_garrison_and_head_to_head(self):
        units = [(G, "NTH"), (G, "YOR"), (R, "HOL"), (R, "HEL"), (Y, "SKA"), (Y, "NOR")]
        orders = [move(G, "NTH", "NOR"), support_move(G, "YOR", "NOR", "NTH"),
                  support_move(R, "HOL", "HEL", "NTH"), move(R, "HEL", "NTH"),
                  support_move(Y, "SKA", "NOR", "NTH"), move(Y, "NOR", "NTH")]
        assert _datc(NORTH_SEA, units, orders) == {t: p for p, t in units}

    def test_6E9_almost_self_dislodgement_the_garrison_leaves(self):
        """The same attack succeeds once the garrison moves out: the support counts again."""
        units = [(G, "NTH"), (G, "YOR"), (R, "HOL"), (R, "HEL"), (Y, "SKA"), (Y, "NOR")]
        orders = [move(G, "NTH", "NWG"), support_move(G, "YOR", "NOR", "NTH"),
                  support_move(R, "HOL", "HEL", "NTH"), move(R, "HEL", "NTH"),
                  support_move(Y, "SKA", "NOR", "NTH"), move(Y, "NOR", "NTH")]
        assert _datc(NORTH_SEA, units, orders) == {
            "NWG": G, "YOR": G, "HOL": R, "HEL": R, "SKA": Y, "NTH": Y}

    def test_6E12_support_on_attack_on_own_unit_still_prevents(self):
        """Austria's support cannot dislodge Austria, but it still blocks Russia."""
        units = [(R, "BUD"), (R, "SER"), (B, "VIE"), (Y, "GAL"), (Y, "RUM")]
        orders = [move(R, "BUD", "RUM"), support_move(R, "SER", "VIE", "BUD"),
                  move(B, "VIE", "BUD"), move(Y, "GAL", "BUD"), support_move(Y, "RUM", "GAL", "BUD")]
        assert _datc(BALKANS, units, orders) == {t: p for p, t in units}

    def test_6E13_three_way_beleaguered_garrison(self):
        units = [(G, "EDI"), (G, "YOR"), (B, "BEL"), (B, "ENG"), (R, "NTH"), (Y, "NWG"), (Y, "NOR")]
        orders = [support_move(G, "EDI", "YOR", "NTH"), move(G, "YOR", "NTH"),
                  move(B, "BEL", "NTH"), support_move(B, "ENG", "BEL", "NTH"), hold(R, "NTH"),
                  move(Y, "NWG", "NTH"), support_move(Y, "NOR", "NWG", "NTH")]
        assert _datc(NORTH_SEA, units, orders) == {t: p for p, t in units}


# ── Commitment grading ──────────────────────────────────────────────────

class TestCommitmentGrading:
    def test_dmz_broken_by_the_intent_not_the_outcome(self):
        """A break is graded on the order given, so a betrayal that bounces
        still counts. Otherwise incompetence would launder bad faith."""
        state = board((R, "R2"), (B, "B1"))
        dmz = Commitment(id="c", commitment_type=CommitmentType.DMZ,
                         players=[R, B], valid_until_turn=5, dmz_territories=["N1"])
        _new, outcomes, _l = resolve(state, [move(R, "R2", "N1"), move(B, "B1", "N1")], [dmz])
        assert not outcomes[0].kept
        assert set(outcomes[0].broken_by) == {R, B}

    def test_support_commitment_kept_when_the_support_is_given(self):
        state = board((R, "B1"), (B, "R2"))
        c = Commitment(id="c", commitment_type=CommitmentType.SUPPORT, players=[R, B],
                       valid_until_turn=5, target_territory="N1", supported_from="R2")
        _new, outcomes, _l = resolve(
            state, [support_move(R, "B1", "R2", "N1"), move(B, "R2", "N1")], [c])
        assert outcomes[0].kept

    def test_support_hold_commitment_is_gradeable(self):
        """A SUPPORT deal with no origin is a promise to support a hold — the
        grammar could not express it, but the engine always could grade it."""
        state = board((R, "B1"), (B, "N1"))
        c = Commitment(id="c", commitment_type=CommitmentType.SUPPORT, players=[R, B],
                       valid_until_turn=5, target_territory="N1")
        _new, kept, _l = resolve(state, [support_hold(R, "B1", "N1"), hold(B, "N1")], [c])
        assert kept[0].kept
        _new, broken, _l = resolve(state, [hold(R, "B1"), hold(B, "N1")], [c])
        assert not broken[0].kept and R in broken[0].broken_by


# ── Property tests ──────────────────────────────────────────────────────

_players = st.sampled_from(players())
_territories = st.sampled_from(TERRITORIES)


@st.composite
def random_position(draw):
    """A legal board: at most one unit per territory."""
    occupied = draw(st.lists(_territories, min_size=1, max_size=6, unique=True))
    return board(*[(draw(_players), t) for t in occupied])


@st.composite
def random_orders(draw, state):
    orders = []
    for u in state.units:
        kind = draw(st.sampled_from([OrderType.HOLD, OrderType.MOVE, OrderType.SUPPORT]))
        if kind == OrderType.HOLD:
            orders.append(hold(u.player, u.territory))
        else:
            target = draw(st.sampled_from(get_adjacent(u.territory)))
            if kind == OrderType.MOVE:
                orders.append(move(u.player, u.territory, target))
            else:
                origins = [o for o in get_adjacent(target) if o != u.territory]
                frm = draw(st.sampled_from(origins + [None])) if origins else None
                orders.append(Order(player=u.player, unit_territory=u.territory,
                                    order_type=OrderType.SUPPORT, target=target,
                                    supported_from=frm))
    return orders


@settings(max_examples=250, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(st.data())
def test_resolution_invariants(data):
    """Whatever the orders, the board stays a board.

    Three things must hold after any turn, and all three were reachable
    failures before the adjudicator was rewritten: no two units share a
    square, no unit teleports, and nobody gains units out of nowhere.
    """
    state = data.draw(random_position())
    orders = data.draw(random_orders(state))
    new, _outcomes, _log = resolve(state, orders)

    squares = [u.territory for u in new.units]
    assert len(squares) == len(set(squares)), "two units in one territory"
    assert all(t in TERRITORIES for t in squares), "unit left the board"
    assert len(new.units) <= len(state.units), "units appeared from nowhere"

    before = {u.territory: u.player for u in state.units}
    for u in new.units:
        origins = [t for t in [u.territory] + get_adjacent(u.territory)
                   if before.get(t) == u.player]
        assert origins, f"{u.player} appeared at {u.territory} from nowhere"


@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(st.data())
def test_every_generated_order_set_is_legal(data):
    """The planner only ever considers orders the engine will accept —
    otherwise its search is scoring moves that cannot be played."""
    state = data.draw(random_position())
    player = data.draw(_players)
    for order_set in generate_all_order_sets(state, player)[:20]:
        held = {u.territory for u in state.units if u.player == player}
        assert {o.unit_territory for o in order_set} == held
        for o in order_set:
            if o.order_type != OrderType.HOLD:
                assert o.target in get_adjacent(o.unit_territory)


@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(st.data())
def test_holding_everything_changes_nothing(data):
    state = data.draw(random_position())
    new, _o, _l = resolve(state, [hold(u.player, u.territory) for u in state.units])
    assert positions(new) == positions(state)


def test_board_adjacency_is_symmetric():
    adj = adjacency()
    for t, neighbours in adj.items():
        for n in neighbours:
            assert t in adj[n], f"{t} -> {n} is one-way"


# ── symmetry: the engine must not read meaning into list order ───────────

def _sigma_and_pi():
    """The board's reflection, and the seat permutation it induces."""
    b = active()
    sigma = b.symmetries[0]
    home_of = {t: p for p, hs in b.home_centers.items() for t in hs}
    pi = {p: home_of[sigma[hs[0]]] for p, hs in b.home_centers.items()}
    return sigma, pi


def _mirror_state(st, sigma, pi):
    return GameState(
        turn=st.turn,
        units=[Unit(player=pi[u.player], territory=sigma[u.territory])
               for u in st.units],
        supply_centers={sigma[t]: (pi[o] if o else None)
                        for t, o in st.supply_centers.items()},
        territory_owners={sigma[t]: (pi[o] if o else None)
                          for t, o in st.territory_owners.items()})


def _mirror_order(o, sigma, pi):
    return Order(player=pi[o.player], unit_territory=sigma[o.unit_territory],
                 order_type=o.order_type,
                 target=sigma[o.target] if o.target else None,
                 supported_from=sigma[o.supported_from] if o.supported_from else None)


@pytest.mark.parametrize("turn", [1, 2])
def test_adjudication_is_equivariant_under_the_board_symmetry(turn):
    """Mirror a position and its orders, and the outcome must mirror too.

    Run at both parities, since builds and removals only fire on even turns.

    Compares how many units each seat ends with, not which squares they sit on:
    build sites and disbands are deliberately random, so exact positions cannot
    mirror. That makes this a guard on movement, capture and dislodgement --
    NOT on build-site fairness, which `test_build_site_is_not_decided_by_
    territory_order` covers instead.
    """
    sigma, pi = _sigma_and_pi()
    rng = random.Random(turn)
    terr = get_all_territories()

    for trial in range(150):
        squares = rng.sample(terr, rng.randint(2, 6))
        units = [Unit(player=rng.choice(players()), territory=t) for t in squares]
        owners = {t: rng.choice(players() + [None]) for t in terr}
        st = GameState(turn=turn, units=units,
                       supply_centers=dict(owners),
                       territory_owners={t: None for t in terr})
        orders = []
        for u in units:
            adj = get_adjacent(u.territory)
            kind = rng.choice(["hold", "move", "support"])
            if kind == "move":
                orders.append(move(u.player, u.territory, rng.choice(adj)))
            elif kind == "support":
                orders.append(support_hold(u.player, u.territory, rng.choice(adj)))
            else:
                orders.append(hold(u.player, u.territory))

        # Builds and removals are deliberately randomised, so compare the
        # multiset of (owner, was_built) rather than exact squares: what must
        # mirror is how many units each seat ends with, not which coin landed.
        seed = rng.randrange(1 << 30)
        random.seed(seed)
        direct, _o, _l = resolve(st, orders)
        random.seed(seed)
        mirrored, _o, _l = resolve(_mirror_state(st, sigma, pi),
                                   [_mirror_order(o, sigma, pi) for o in orders])

        want = sorted(pi[p].value for p in (u.player for u in direct.units))
        got = sorted(u.player.value for u in mirrored.units)
        assert want == got, (
            f"turn {turn} trial {trial}: mirrored position gave {got}, "
            f"mirror of the original is {want}")


class TestDislodgedSupport:
    """6.D.17 — a dislodged unit gives no support.

    The attack that dislodges the supporter comes OUT OF the very square the
    support is aimed at, so the ordinary cut rule does not apply. Only the
    dislodgement voids it.
    """

    def test_support_from_a_dislodged_unit_does_not_count(self):
        # Red C2->N1 at strength 2 thanks to R2; Blue dislodges R2 from N1;
        # Green contests N1 at strength 1. Voiding the support makes it 1 v 1.
        state = board((R, "C2"), (R, "R2"), (B, "N1"), (B, "B1"), (G, "C1"))
        new, _o, _l = resolve(state, [
            move(R, "C2", "N1"),
            support_move(R, "R2", "C2", "N1"),
            move(B, "N1", "R2"),
            support_move(B, "B1", "N1", "R2"),
            move(G, "C1", "N1"),
        ])
        pos = positions(new)
        assert pos.get("R2") == B, "the supporter should have been dislodged"
        assert "N1" not in pos, f"N1 should be empty, got {pos.get('N1')}"

    def test_a_surviving_supporter_still_counts(self):
        # Same shape, but Blue's attack on the supporter is unsupported, so R2
        # holds and its support stands.
        state = board((R, "C2"), (R, "R2"), (B, "N1"), (G, "C1"))
        new, _o, _l = resolve(state, [
            move(R, "C2", "N1"),
            support_move(R, "R2", "C2", "N1"),
            move(B, "N1", "R2"),
            move(G, "C1", "N1"),
        ])
        assert positions(new).get("N1") == R, "unbroken support should still win N1"
