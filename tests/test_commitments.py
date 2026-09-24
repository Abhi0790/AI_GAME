import pytest
from src.common.schemas import (
    GameState, Order, OrderType, Player, Unit, Commitment, CommitmentType,
    Message, MessageType, message_to_commitment,
)
from src.engine.adjudicator import resolve, verify_commitments
from src.engine.runner import GameRunner
from src.engine.board import players

def test_dmz_commitment():
    state = GameState(turn=1, units=[], supply_centers={}, territory_owners={})
    c = Commitment(id="1", commitment_type=CommitmentType.DMZ, players=[Player.RED, Player.BLUE], valid_until_turn=2, dmz_territories=["N1"])
    orders = [
        Order(player=Player.RED, unit_territory="R1", order_type=OrderType.MOVE, target="N1"),
        Order(player=Player.BLUE, unit_territory="B1", order_type=OrderType.HOLD)
    ]
    outcomes = verify_commitments(state, [c], orders)
    assert len(outcomes) == 1
    assert outcomes[0].kept == False
    assert Player.RED in outcomes[0].broken_by
    assert Player.BLUE not in outcomes[0].broken_by

def test_alliance_commitment():
    state = GameState(
        turn=1, 
        units=[], 
        supply_centers={"R1": Player.RED, "B1": Player.BLUE}, 
        territory_owners={}
    )
    c = Commitment(id="2", commitment_type=CommitmentType.ALLIANCE, players=[Player.RED, Player.BLUE], valid_until_turn=2)
    orders = [
        Order(player=Player.RED, unit_territory="R2", order_type=OrderType.MOVE, target="B1"),
    ]
    outcomes = verify_commitments(state, [c], orders)
    assert outcomes[0].kept == False
    assert Player.RED in outcomes[0].broken_by



# ── duration ─────────────────────────────────────────────────────────────

class _Mute:
    """An agent that negotiates nothing and holds everything.

    Keeps the board still, so the only thing the turn loop below can vary is
    how long a commitment survives.
    """
    def __init__(self, player):
        self.player = player

    def propose(self, state, commitments=None):
        return []

    def reply(self, state, incoming, commitments=None):
        return []

    def act(self, state, commitments):
        return [Order(player=self.player, unit_territory=u.territory,
                      order_type=OrderType.HOLD)
                for u in state.units if u.player == self.player], None

    def update_beliefs_from_outcomes(self, prev_state, new_state, outcomes):
        pass

    def beliefs(self):
        return []

    def predict_keep(self, state, commitment, subject):
        return 0.5


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("start", [1, 2])  # struck in spring, and in autumn
def test_a_k_turn_deal_is_graded_exactly_k_times(k, start):
    """valid_until_turn is the last turn graded, inclusive.

    The deal must be graded on every one of its k turns and on none after,
    including when its life spans the spring/autumn boundary where units are
    rebuilt.
    """
    runner = GameRunner([_Mute(p) for p in players()])
    while runner.state.turn < start:
        runner.step()

    msg = Message(id="deal", sender=Player.RED, receiver=Player.BLUE,
                  message_type=MessageType.PROPOSE,
                  commitment_type=CommitmentType.DMZ, turns=k,
                  dmz_territories=["N1"])
    c = message_to_commitment(msg, Player.BLUE, runner.state.turn)
    assert c.valid_until_turn == start + k - 1
    runner.commitments.append(c)

    graded = []
    for _ in range(k + 2):
        turn = runner.state.turn
        if any(o.commitment.id == "deal" for o in runner.step().outcomes):
            graded.append(turn)
    assert graded == list(range(start, start + k))


class TestObligatedParties:
    """A deal's participants are not all its debtors. Crediting the passive
    side with a kept promise inflates trust and deflates betrayal rates."""

    def _deal(self, kind, **kw):
        from src.common.schemas import Commitment
        return Commitment(id="x", commitment_type=kind,
                          players=[Player.RED, Player.BLUE],
                          valid_until_turn=9, **kw)

    def test_support_obligates_only_the_supporter(self):
        from src.common.schemas import obligated_parties

        d = self._deal(CommitmentType.SUPPORT, target_territory="N1",
                       supported_from="B1")
        assert obligated_parties(d, 1) == [Player.RED]

    def test_an_exchange_obligates_one_side_per_leg(self):
        from src.common.schemas import obligated_parties

        d = self._deal(CommitmentType.EXCHANGE, target_territory="N1",
                       supported_from="R2", dmz_territories=["C1"], repay_turn=5)
        assert obligated_parties(d, 2) == [Player.RED], "giver owes the support leg"
        assert obligated_parties(d, 5) == [Player.BLUE], "payer owes the repay leg"

    def test_alliance_and_dmz_bind_everyone(self):
        from src.common.schemas import obligated_parties

        for d in (self._deal(CommitmentType.ALLIANCE),
                  self._deal(CommitmentType.DMZ, dmz_territories=["N1"])):
            assert set(obligated_parties(d, 1)) == {Player.RED, Player.BLUE}

    def test_one_move_breaking_two_deals_is_one_betrayal(self):
        from types import SimpleNamespace
        from src.common.schemas import CommitmentOutcome
        from src.evaluation.metrics import betrayal_rate_per_player

        both = [CommitmentOutcome(self._deal(CommitmentType.ALLIANCE), kept=False,
                                  broken_by=[Player.RED]),
                CommitmentOutcome(self._deal(CommitmentType.DMZ, dmz_territories=["N1"]),
                                  kept=False, broken_by=[Player.RED])]
        step = SimpleNamespace(outcomes=both, state=SimpleNamespace(turn=1))
        rates = betrayal_rate_per_player([step])
        assert rates[Player.RED] == 1.0 and rates[Player.BLUE] == 0.0

    def test_betrayal_denominators_skip_the_passive_side(self):
        from types import SimpleNamespace
        from src.common.schemas import CommitmentOutcome
        from src.evaluation.metrics import betrayal_rate_per_player

        kept = CommitmentOutcome(
            self._deal(CommitmentType.SUPPORT, target_territory="N1",
                       supported_from="B1"), kept=True, broken_by=[])
        step = SimpleNamespace(outcomes=[kept],
                              state=SimpleNamespace(turn=1))
        rates = betrayal_rate_per_player([step])
        # Blue owed nothing, so it must not be credited with a kept promise.
        assert rates[Player.BLUE] == 0.0
        assert rates[Player.RED] == 0.0
