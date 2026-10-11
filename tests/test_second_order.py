"""Second-order beliefs: what others have seen of me, and who will refuse me."""
from src.agents.agent import Agent
from src.common.schemas import Commitment, CommitmentOutcome, CommitmentType, Player
from src.engine.board import players

R, B, G = Player.RED, Player.BLUE, Player.GREEN


def _outcome(private, broke):
    c = Commitment(id="d", commitment_type=CommitmentType.ALLIANCE, players=[R, B],
                   valid_until_turn=5, private=private)
    return CommitmentOutcome(commitment=c, kept=not broke, broken_by=[R] if broke else [])


def test_a_private_break_is_seen_only_by_the_partner():
    a = Agent(R)
    a._witnessed(_outcome(private=True, broke=True), turn=1)
    assert a.standing_with(B) < 0.5 and a.standing_with(G) == 0.5


def test_a_public_break_is_seen_by_everyone():
    a = Agent(R)
    a._witnessed(_outcome(private=False, broke=True), turn=1)
    assert all(a.standing_with(p) < 0.5 for p in players() if p != R)


def test_breaking_in_view_makes_me_expect_refusal_and_keeping_repairs_it():
    a = Agent(R)
    a._witnessed(_outcome(private=True, broke=True), turn=1)
    assert a.expects_refusal(B) and not a.expects_refusal(G)
    for _ in range(3):
        a._witnessed(_outcome(private=True, broke=False), turn=1)
    assert not a.expects_refusal(B)
