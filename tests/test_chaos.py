"""Robustness: a real game with a player who behaves badly on purpose.

The old version of this file asserted that `resolve()` ignored two junk
orders. That is not the requirement — the requirement is that a full game
survives a participant sending contradictory orders, nothing at all,
malformed messages, acceptances of proposals that were never made, and a
flood of out-of-grammar spam. So the chaos agent takes a seat.
"""

import random
import pytest

from src.common.schemas import (
    GameState, Order, OrderType, Player, Unit, Message, MessageType,
    CommitmentType, invalid_proposal_reason,
)
from src.engine.adjudicator import resolve
from src.engine.runner import GameRunner
from src.engine.board import get_all_territories, MAX_TURNS
from src.agents.agent import Agent
from src.agents.chaos import ChaosAgent, MODES


def _game(mode=None, seed=11):
    random.seed(seed)
    return GameRunner([
        ChaosAgent(Player.RED, mode, seed=seed),
        Agent(Player.BLUE, "Honest"),
        Agent(Player.GREEN, "Paranoid"),
        Agent(Player.GOLD, "Vengeful"),
    ])


class TestAdjudicatorAgainstJunk:
    def test_orders_referencing_nothing_are_ignored(self):
        state = GameState(turn=1, units=[Unit(player=Player.RED, territory="R1")],
                          supply_centers={}, territory_owners={})
        orders = [
            Order(player=Player.RED, unit_territory="R1",
                  order_type=OrderType.MOVE, target="NOT_A_TERRITORY"),
            Order(player=Player.BLUE, unit_territory="FAKE",
                  order_type=OrderType.SUPPORT, target="NULL"),
        ]
        new_state, _outcomes, _log = resolve(state, orders)
        assert len(new_state.units) == 1
        assert new_state.units[0].territory == "R1"

    def test_a_flood_of_junk_orders_changes_nothing(self):
        rng = random.Random(3)
        territories = get_all_territories() + ["VOID", "", "R1;DROP"]
        state = GameState(
            turn=1,
            units=[Unit(player=Player.RED, territory="R1"),
                   Unit(player=Player.BLUE, territory="B1")],
            supply_centers={}, territory_owners={})
        orders = [
            Order(player=rng.choice(list(Player)),
                  unit_territory=rng.choice(territories),
                  order_type=rng.choice(list(OrderType)),
                  target=rng.choice(territories))
            for _ in range(400)
        ]
        new_state, _o, _l = resolve(state, orders)
        squares = [u.territory for u in new_state.units]
        assert len(squares) == len(set(squares))
        assert all(t in get_all_territories() for t in squares)


class TestChaosSeat:
    @pytest.mark.parametrize("mode", MODES)
    def test_full_game_survives_each_mode(self, mode):
        runner = _game(mode)
        winner = runner.run(verbose=False)
        assert winner in list(Player)
        assert 1 <= len(runner.history) <= MAX_TURNS

    def test_full_game_survives_a_rotating_saboteur(self):
        """A different kind of misbehaviour every turn."""
        runner = _game(None)
        runner.run(verbose=False)
        assert runner.history

    def test_board_stays_legal_throughout(self):
        runner = _game(None, seed=5)
        runner.run(verbose=False)
        territories = set(get_all_territories())
        for step in runner.history:
            squares = [u.territory for u in step.state.units]
            assert len(squares) == len(set(squares)), "two units in one territory"
            assert set(squares) <= territories

    def test_a_silent_player_is_not_a_crash(self):
        runner = _game("silent")
        runner.run(verbose=False)
        # A seat that never orders anything holds; it should still be on the
        # board unless somebody took its squares off it.
        assert len(runner.history) >= 1

    def test_phantom_accepts_do_not_create_commitments(self):
        """Accepting a proposal nobody made must not put a deal on the table
        that the engine will then grade against a real player."""
        runner = _game("phantom_accept")
        runner.run(verbose=False)
        for c in runner.commitments:
            assert len(set(c.players)) == 2, "a commitment with a phantom party"

    def test_out_of_grammar_proposals_never_become_commitments(self):
        runner = _game("out_of_grammar")
        runner.run(verbose=False)
        territories = set(get_all_territories())
        for step in runner.history:
            for c in step.commitments:
                assert c.commitment_type is not None
                assert 1 <= c.valid_until_turn <= MAX_TURNS + MAX_TURNS
                for t in (c.dmz_territories or []):
                    assert t in territories
                if c.commitment_type == CommitmentType.SUPPORT:
                    assert c.target_territory in territories

    def test_spam_does_not_stall_the_turn_loop(self):
        runner = _game("spam_proposals")
        runner.run(verbose=False)
        assert len(runner.history) >= 1
        for step in runner.history:
            # Every seat still gets orders adjudicated, spam or no spam.
            assert step.orders is not None

    def test_a_liar_is_refuted_not_believed(self):
        """The chaos seat accuses somebody every turn. The engine grades the
        claims, so a false one must come back REFUTED."""
        runner = _game("garbage_orders", seed=2)
        runner.run(verbose=False)
        accusations = [m for step in runner.history for m in step.messages
                       if m.broadcast_kind == "BETRAYED" and m.sender == Player.RED]
        assert accusations, "the chaos seat never accused anyone"
        assert any(m.engine_verdict == "REFUTED" for m in accusations)


class TestGrammarGate:
    """The guard that stops a hostile client getting an unenforceable deal
    onto the table. The chaos agent found this the first time it ran."""

    def test_an_accept_of_a_typeless_proposal_is_refused(self):
        msg = Message(id="x", sender=Player.RED, receiver=Player.BLUE,
                      message_type=MessageType.PROPOSE)
        assert invalid_proposal_reason(msg) is not None

    def test_a_999_turn_alliance_is_refused(self):
        msg = Message(id="x", sender=Player.RED, receiver=Player.BLUE,
                      message_type=MessageType.PROPOSE,
                      commitment_type=CommitmentType.ALLIANCE, turns=999)
        assert "duration" in invalid_proposal_reason(msg)

    def test_a_proposal_to_yourself_is_refused(self):
        msg = Message(id="x", sender=Player.RED, receiver=Player.RED,
                      message_type=MessageType.PROPOSE,
                      commitment_type=CommitmentType.ALLIANCE, turns=2)
        assert invalid_proposal_reason(msg) is not None
