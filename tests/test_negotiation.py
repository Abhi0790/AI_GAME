"""Tests for the Negotiation Strategy module.

The decision rule under test is no longer a reliability cut-off. A deal is
accepted when

    P(keep) * V_kept + (1 - P(keep)) * V_broken > V_none

so the tests stub the three values and check the arithmetic, rather than
poking a threshold and hoping.
"""

import pytest
from src.common.schemas import (
    Player, GameState, Unit, Message, MessageType, CommitmentType, Commitment,
    message_to_commitment, invalid_proposal_reason, describe_message,
)
from src.agents.trust.model import TrustModel
from src.agents.planner.planner import Planner, PlannerConfig
from src.agents.negotiation.strategy import NegotiationStrategy, THREAT_COOLDOWN


def _make_state(**overrides):
    defaults = dict(
        turn=1,
        units=[
            Unit(player=Player.RED, territory="R1"),
            Unit(player=Player.RED, territory="R2"),
            Unit(player=Player.BLUE, territory="B1"),
            Unit(player=Player.BLUE, territory="B2"),
        ],
        supply_centers={"R1": Player.RED, "R2": Player.RED,
                        "B1": Player.BLUE, "B2": Player.BLUE,
                        "N1": None, "N2": None},
        territory_owners={"R1": Player.RED, "R2": Player.RED,
                          "B1": Player.BLUE, "B2": Player.BLUE},
    )
    defaults.update(overrides)
    return GameState(**defaults)


def _strategy(player=Player.RED):
    return NegotiationStrategy(player, Planner(player, PlannerConfig()))


def _stub_values(ns, v_none, v_kept, v_broken):
    """Pin the counterfactual so the decision rule is the only thing tested.

    deal_totals is what the decision reads (deal_values scaled over the deal's
    life, plus Vcoop); stubbing both keeps the arithmetic under test rather
    than the board.
    """
    ns.deal_values = lambda *a, **k: (v_none, v_kept, v_broken)
    ns.deal_totals = lambda *a, **k: (v_none, v_kept, v_broken)


def _propose(commitment_type=CommitmentType.ALLIANCE, turns=3, **kw):
    return Message(id="m", sender=Player.BLUE, receiver=Player.RED,
                   message_type=MessageType.PROPOSE,
                   commitment_type=commitment_type, turns=turns, **kw)


class TestAcceptanceIsExpectedValue:
    def test_accepts_when_expected_value_beats_no_deal(self):
        ns = _strategy()
        trust = TrustModel(Player.RED)                 # a stranger, P(keep) ≈ 0.9
        _stub_values(ns, v_none=1.0, v_kept=3.0, v_broken=0.5)
        # 0.9*3.0 + 0.1*0.5 = 2.75 > 1.0
        assert ns.evaluate_proposal(_make_state(), _propose(), trust).message_type \
            == MessageType.ACCEPT

    def test_rejects_when_exposure_outweighs_the_upside(self):
        ns = _strategy()
        trust = TrustModel(Player.RED)
        # A big upside that is not big enough to cover being walked over, even
        # by a stranger trusted at 0.9: 0.9*2.6 + 0.1*-10 = 1.34 < 2.0.
        _stub_values(ns, v_none=2.0, v_kept=2.6, v_broken=-10.0)
        ns._counter_terms = lambda msg: None
        assert ns.evaluate_proposal(_make_state(), _propose(), trust).message_type \
            == MessageType.REJECT

    def test_low_reliability_can_flip_the_same_deal(self):
        """Identical board values, different partner: the only thing that
        changed is P(keep), and that is enough to change the answer."""
        state, msg = _make_state(), _propose()
        values = dict(v_none=1.0, v_kept=3.0, v_broken=-1.0)

        trusted = TrustModel(Player.RED)
        trusted.get_record(Player.BLUE, CommitmentType.ALLIANCE).alpha = 9.0
        ns = _strategy(); _stub_values(ns, **values)
        assert ns.evaluate_proposal(state, msg, trusted).message_type == MessageType.ACCEPT

        shifty = TrustModel(Player.RED)
        shifty.get_record(Player.BLUE, CommitmentType.ALLIANCE).beta = 9.0
        ns = _strategy(); _stub_values(ns, **values)
        ns._counter_terms = lambda m: None
        assert ns.evaluate_proposal(state, msg, shifty).message_type == MessageType.REJECT


class TestCounterOffers:
    def test_counters_with_shorter_terms_before_walking_away(self):
        ns = _strategy()
        trust = TrustModel(Player.RED)
        # The 3-turn deal fails; the halved one passes.
        def values(state, deal, partner, commitments):
            return (1.0, 3.0, 2.0) if deal.valid_until_turn <= 2 else (1.0, 1.0, 0.0)
        ns.deal_values = values
        ns.deal_totals = values

        reply = ns.evaluate_proposal(_make_state(), _propose(turns=3), trust)
        assert reply.message_type == MessageType.COUNTER
        assert reply.turns == 1
        assert reply.reference_id == "m"

    def test_a_counter_is_never_countered_again(self):
        """Otherwise rounds 2 and 3 are an infinite haggle."""
        ns = _strategy()
        trust = TrustModel(Player.RED)
        _stub_values(ns, v_none=5.0, v_kept=0.0, v_broken=0.0)
        msg = _propose(turns=4)
        msg.message_type = MessageType.COUNTER
        assert ns.evaluate_proposal(_make_state(), msg, trust).message_type \
            == MessageType.REJECT


class TestGrammarCoverage:
    def test_support_hold_is_proposable(self):
        """Third sentence of the grammar; nothing used to generate it."""
        ns = _strategy()
        offers = ns._candidate_deals(_make_state(), Player.BLUE)
        support_holds = [m for m in offers
                         if m.commitment_type == CommitmentType.SUPPORT
                         and m.supported_from is None]
        assert support_holds, "no support-hold offer was ever generated"
        assert "hold" in describe_message(support_holds[0])

    def test_threat_is_rate_limited(self):
        ns = _strategy()
        ns.threats_made[Player.BLUE] = 5
        state = _make_state(turn=5 + THREAT_COOLDOWN - 1)
        threats = [m for m in ns.generate_proposals(state, TrustModel(Player.RED))
                   if m.message_type == MessageType.THREAT and m.receiver == Player.BLUE]
        assert not threats

    def test_threat_has_a_condition_and_an_action(self):
        ns = _strategy()
        state = _make_state()
        threats = [m for m in ns.generate_proposals(state, TrustModel(Player.RED))
                   if m.message_type == MessageType.THREAT]
        for t in threats:
            assert t.condition and t.action

    def test_no_proposals_to_self(self):
        ns = _strategy()
        for m in ns.generate_proposals(_make_state(), TrustModel(Player.RED)):
            assert m.receiver != Player.RED

    def test_malformed_proposals_are_rejected_not_priced(self):
        ns = _strategy()
        trust = TrustModel(Player.RED)
        junk = Message(id="x", sender=Player.BLUE, receiver=Player.RED,
                       message_type=MessageType.PROPOSE,
                       commitment_type=CommitmentType.DMZ,
                       dmz_territories=["NOWHERE"], turns=999)
        assert ns.evaluate_proposal(_make_state(), junk, trust).message_type \
            == MessageType.REJECT

    def test_threat_is_not_answered_with_accept_or_reject(self):
        ns = _strategy()
        threat = Message(id="t", sender=Player.BLUE, receiver=Player.RED,
                         message_type=MessageType.THREAT,
                         condition="you move", action="I retaliate")
        assert ns.evaluate_proposal(_make_state(), threat, TrustModel(Player.RED)) is None


class TestGrammarValidation:
    @pytest.mark.parametrize("msg,reason", [
        (_propose(turns=999), "duration"),
        (_propose(CommitmentType.DMZ, dmz_territories=[]), "no territories"),
        (_propose(CommitmentType.DMZ, dmz_territories=["NOWHERE"]), "unknown"),
        (_propose(CommitmentType.SUPPORT, target_territory="NOWHERE"), "unknown"),
        (_propose(CommitmentType.SUPPORT, target_territory="N1",
                  supported_from="G1"), "adjacent"),
    ])
    def test_rejected_sentences(self, msg, reason):
        got = invalid_proposal_reason(msg)
        assert got and reason in got

    @pytest.mark.parametrize("msg", [
        _propose(),
        _propose(CommitmentType.DMZ, turns=2, dmz_territories=["N1"]),
        _propose(CommitmentType.SUPPORT, turns=1, target_territory="N1",
                 supported_from="R2"),
        _propose(CommitmentType.SUPPORT, turns=1, target_territory="N1"),
    ])
    def test_accepted_sentences(self, msg):
        assert invalid_proposal_reason(msg) is None


class TestMessageToCommitment:
    def test_basic_conversion(self):
        msg = Message(id="m1", sender=Player.RED, receiver=Player.BLUE,
                      message_type=MessageType.PROPOSE,
                      commitment_type=CommitmentType.ALLIANCE, turns=3)
        c = message_to_commitment(msg, Player.BLUE, current_turn=4)
        assert c.commitment_type == CommitmentType.ALLIANCE
        assert Player.RED in c.players
        assert Player.BLUE in c.players
        assert c.valid_until_turn == 6  # turns 4,5,6 — last graded turn, inclusive

    def test_dmz_conversion(self):
        msg = Message(id="m2", sender=Player.RED, receiver=Player.BLUE,
                      message_type=MessageType.PROPOSE,
                      commitment_type=CommitmentType.DMZ, turns=2,
                      dmz_territories=["N1"])
        c = message_to_commitment(msg, Player.BLUE, current_turn=1)
        assert c.commitment_type == CommitmentType.DMZ
        assert c.dmz_territories == ["N1"]
        assert c.valid_until_turn == 2  # turns 1,2


def test_a_quote_is_not_a_signed_price():
    """Pricing a deal every turn must not reset the price it was signed at."""
    from src.agents.agent import Agent
    from src.common.schemas import commitment_key
    agent = Agent(Player.RED, "Honest")
    ns = agent.negotiation
    deal = Commitment(id="d", commitment_type=CommitmentType.ALLIANCE,
                      players=[Player.RED, Player.BLUE], valid_until_turn=4)
    key = commitment_key(deal)
    state = lambda t: GameState(turn=t, units=[], supply_centers={}, territory_owners={})
    ns._record_price(state(2), deal, 1.5)
    agent.deal_signed(deal, 2)
    ns._record_price(state(3), deal, 9.0)
    assert ns.deal_prices[key] == 1.5 and ns.deal_signed_turn[key] == 2


def test_no_deals_with_a_player_about_to_win():
    """One centre from winning, a player's offers are refused and none are made to them."""
    from src.agents.agent import Agent
    from src.common.schemas import MessageType
    red = Agent(Player.RED, "Honest")
    units = [Unit(player=Player.RED, territory="R1"), Unit(player=Player.RED, territory="R2"),
             Unit(player=Player.BLUE, territory="B1"), Unit(player=Player.BLUE, territory="N1")]
    near = {"R1": Player.RED, "R2": Player.RED, "B1": Player.BLUE, "B2": Player.BLUE,
            "N1": Player.BLUE, "N2": Player.BLUE, "G1": Player.BLUE}
    state = GameState(turn=3, units=units, supply_centers=near,
                      territory_owners={u.territory: u.player for u in units})
    offer = Message(id="m", sender=Player.BLUE, receiver=Player.RED,
                    message_type=MessageType.PROPOSE,
                    commitment_type=CommitmentType.ALLIANCE, turns=3)
    [answer] = red.reply(state, [offer])
    assert answer.message_type == MessageType.REJECT
    assert not any(m.receiver == Player.BLUE and m.message_type == MessageType.PROPOSE
                   for m in red.propose(state))



def test_nobody_is_about_to_win_at_the_start_of_a_two_seat_game():
    """The threshold there is one above the starting centres; deals must still be possible."""
    from src.agents.planner.planner import near_win
    from src.common.config import GameConfig
    from src.harness import build_runner, game_setup
    cfg = GameConfig(seed=1, n_seats=2)
    with game_setup(cfg):
        runner = build_runner(cfg)
        assert not any(near_win(runner.state, p) for p in runner.agents)


def test_a_bound_player_does_not_sign_away_its_growth():
    """One two-way alliance at a time, and no promise to stay out of a centre it
    can take. The Opportunist is not bound and signs either."""
    from src.agents.agent import Agent
    units = [Unit(player=Player.RED, territory="R1"), Unit(player=Player.RED, territory="R2"),
             Unit(player=Player.BLUE, territory="B1"), Unit(player=Player.BLUE, territory="B2")]
    state = GameState(turn=1, units=units,
                      supply_centers={"R1": Player.RED, "R2": Player.RED,
                                      "B1": Player.BLUE, "B2": Player.BLUE, "N1": None},
                      territory_owners={u.territory: u.player for u in units})
    dmz = Commitment(id="d", commitment_type=CommitmentType.DMZ,
                     players=[Player.RED, Player.BLUE], valid_until_turn=3,
                     dmz_territories=["N1"])
    ally = lambda other: Commitment(id=other.value, commitment_type=CommitmentType.ALLIANCE,
                                    players=[Player.RED, other], valid_until_turn=3)
    honest, opp = Agent(Player.RED, "Honest").negotiation, Agent(Player.RED, "Opportunist").negotiation
    honest._live_commitments = opp._live_commitments = [ally(Player.GREEN)]
    assert honest.overcommits(state, dmz) and honest.overcommits(state, ally(Player.BLUE))
    assert not honest.overcommits(state, ally(Player.GREEN)), "renewing the one alliance is fine"
    assert not opp.overcommits(state, dmz) and not opp.overcommits(state, ally(Player.BLUE))
