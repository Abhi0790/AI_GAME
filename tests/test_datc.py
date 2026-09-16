"""Adjudicator conformance: DATC cases, adapted, plus property tests.

The DATC is the reference test set for Diplomacy adjudicators. This game has
no fleets, coasts, convoys or retreats, so the sections covering them do not
apply; what is left is section 6.A (illegal orders), 6.C (circular movement
and swaps) and 6.D (supports and dislodges), which is what this file walks
through on the twelve-territory board.

Adjacency used below (src/engine/board.py):
    R1: R2 Y2 C1      R2: R1 B1 N1      B1: R2 B2 N1      B2: B1 G1 C2
    G1: B2 G2 C2      G2: G1 Y1 N2      Y1: G2 Y2 N2      Y2: Y1 R1 C1
    N1: R2 B1 C1 C2   N2: G2 Y1 C1 C2   C1: R1 Y2 N1 N2   C2: B2 G1 N1 N2
"""

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from src.common.schemas import (
    GameState, Order, OrderType, Player, Unit, Commitment, CommitmentType,
)
from src.engine.adjudicator import resolve
from src.engine.board import get_all_territories, get_adjacent, ADJACENCY
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
        """C1 borders R1 but not R2, so it cannot support an attack into R2.
        The order is dropped and the head-to-head bounces as usual."""
        state = board((R, "R1"), (B, "R2"), (R, "C1"))
        new, _o, _l = resolve(state, [
            move(R, "R1", "R2"),
            move(B, "R2", "R1"),
            support_move(R, "C1", "R1", "R2"),
        ])
        assert positions(new) == {"R1": R, "R2": B, "C1": R}

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

_players = st.sampled_from(list(Player))
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
    for t, neighbours in ADJACENCY.items():
        for n in neighbours:
            assert t in ADJACENCY[n], f"{t} -> {n} is one-way"
