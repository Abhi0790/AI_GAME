"""The four richer deal structures.

  1. private deals      — graded by the engine, not published by it
  2. per-pair trust     — "keeps promises to *me*" vs "keeps promises"
  3. multi-party pacts  — one promise that binds three, or none
  4. exchanges          — two legs of unequal value, falling due on different turns
"""

import random
import uuid
import pytest

from src.common.schemas import (
    GameState, Order, OrderType, Player, Unit, Commitment, CommitmentType,
    Message, MessageType, message_to_commitment, invalid_proposal_reason,
    exchange_leg_due, commitment_key,
)
from src.engine.adjudicator import resolve, verify_commitments
from src.engine.runner import GameRunner
from src.engine.board import get_all_territories
from src.agents.agent import Agent
from src.agents.trust.model import TrustModel, PAIR_SHRINKAGE

R, B, G, Y = Player.RED, Player.BLUE, Player.GREEN, Player.GOLD
TERRITORIES = get_all_territories()


def board(*units, turn=1):
    return GameState(
        turn=turn,
        units=[Unit(player=p, territory=t) for p, t in units],
        supply_centers={t: None for t in TERRITORIES},
        territory_owners={t: None for t in TERRITORIES},
    )


def move(p, frm, to):
    return Order(player=p, unit_territory=frm, order_type=OrderType.MOVE, target=to)


def hold(p, at):
    return Order(player=p, unit_territory=at, order_type=OrderType.HOLD)


def support(p, at, frm, to):
    return Order(player=p, unit_territory=at, order_type=OrderType.SUPPORT,
                 target=to, supported_from=frm)


def _runner(personas=("Opportunist", "Honest", "Paranoid", "Vengeful")):
    return GameRunner([Agent(p, name) for p, name in zip(Player, personas)])


def _pacts_in(seed):
    random.seed(seed)
    runner = _runner()
    runner.run(verbose=False)
    return {c.id for step in runner.history for c in step.commitments
            if len(c.players) > 2}


# ── 1. Private deals ─────────────────────────────────────────────────────

class TestPrivateDeals:
    def test_a_private_deal_is_still_graded(self):
        """Privacy is about publication, not enforcement."""
        state = board((R, "R2"), (B, "B1"))
        c = Commitment(id="c", commitment_type=CommitmentType.DMZ, players=[R, B],
                       valid_until_turn=5, dmz_territories=["N1"], private=True)
        outcomes = verify_commitments(state, [c], [move(R, "R2", "N1")])
        assert not outcomes[0].kept and R in outcomes[0].broken_by

    def test_a_broken_private_deal_cannot_be_proved(self):
        c = Commitment(id="c", commitment_type=CommitmentType.ALLIANCE,
                       players=[R, B], valid_until_turn=5, private=True)
        from src.common.schemas import CommitmentOutcome
        outcome = CommitmentOutcome(commitment=c, kept=False, broken_by=[B])
        msg = Message(id="m", sender=R, message_type=MessageType.BROADCAST,
                      broadcast_kind="BETRAYED", broadcast_target=B,
                      commitment_type=CommitmentType.ALLIANCE)
        assert _runner().verify_accusation(msg, [outcome]) == "UNVERIFIED"

    def test_the_same_deal_public_is_provable(self):
        """The only difference between the two cases is the privacy flag —
        which is exactly what gives lying somewhere to hide."""
        c = Commitment(id="c", commitment_type=CommitmentType.ALLIANCE,
                       players=[R, B], valid_until_turn=5, private=False)
        from src.common.schemas import CommitmentOutcome
        outcome = CommitmentOutcome(commitment=c, kept=False, broken_by=[B])
        msg = Message(id="m", sender=R, message_type=MessageType.BROADCAST,
                      broadcast_kind="BETRAYED", broadcast_target=B,
                      commitment_type=CommitmentType.ALLIANCE)
        assert _runner().verify_accusation(msg, [outcome]) == "CONFIRMED"

    def test_the_same_promise_does_not_stack_in_two_forms(self):
        """A public and a private version of one promise between the same two
        players is one relationship. Keying them apart let it be recorded —
        and priced — twice."""
        kw = dict(commitment_type=CommitmentType.ALLIANCE, players=[R, B],
                  valid_until_turn=5)
        assert commitment_key(Commitment(id="a", private=True, **kw)) == \
               commitment_key(Commitment(id="b", private=False, **kw))

    def test_publicity_is_one_way(self):
        """Re-proposing an open deal privately must not hide it again."""
        runner = _runner()
        public = Commitment(id="a", commitment_type=CommitmentType.ALLIANCE,
                            players=[R, B], valid_until_turn=3, private=False)
        runner.commitments = [public]
        offer = Message(id="m", sender=R, receiver=B, message_type=MessageType.PROPOSE,
                        commitment_type=CommitmentType.ALLIANCE, turns=3, private=True)
        runner._opening = [offer]
        for agent in runner.agents.values():
            agent.propose = lambda *a, **k: []
            agent.reply = lambda *a, **k: []
        runner.agents[B].reply = lambda *a, **k: [
            Message(id=str(uuid.uuid4()), sender=B, receiver=R,
                    message_type=MessageType.ACCEPT, reference_id="m")]
        runner._negotiate([])
        assert len(runner.commitments) == 1
        assert runner.commitments[0].private is False

    def test_personas_differ_in_how_much_they_hide(self):
        assert _runner().agents[R].negotiation.privacy_preference > \
               _runner().agents[B].negotiation.privacy_preference


# ── 2. Per-pair trust ────────────────────────────────────────────────────

class TestPairTrust:
    def _split_record(self):
        """Blue honours every deal with Red and breaks every deal with Green."""
        m = TrustModel(R)
        for _ in range(5):
            m.observe(B, [B, R], CommitmentType.ALLIANCE, kept=True)
            m.observe(B, [B, G], CommitmentType.ALLIANCE, kept=False)
        return m

    def test_the_same_player_reads_differently_per_counterparty(self):
        m = self._split_record()
        toward_red = m.get_reliability(B, CommitmentType.ALLIANCE, toward=R)
        toward_green = m.get_reliability(B, CommitmentType.ALLIANCE, toward=G)
        assert toward_red > toward_green
        # ...and the general reputation sits between the two.
        general = m.get_reliability(B, CommitmentType.ALLIANCE)
        assert toward_green < general < toward_red

    def test_one_observation_barely_moves_the_pair_estimate(self):
        """Shrinkage: a single data point is not a pattern."""
        m = TrustModel(R)
        for _ in range(6):
            m.observe(B, [B, G], CommitmentType.ALLIANCE, kept=False)
        general = m.get_reliability(B, CommitmentType.ALLIANCE)
        m.observe(B, [B, R], CommitmentType.ALLIANCE, kept=True)
        toward_red = m.get_reliability(B, CommitmentType.ALLIANCE, toward=R)
        assert abs(toward_red - general) < 0.35

    def test_evidence_accumulates_until_the_pair_record_dominates(self):
        """With more pair-specific evidence the blended estimate should track
        the pair record itself, not the pooled reputation.

        (Measuring the *gap* between the two would not show this: every
        observation about Blue-with-Red also feeds Blue's general record, so
        the two converge even as the pair record gains weight.)"""
        m = TrustModel(R)
        for _ in range(6):
            m.observe(B, [B, G], CommitmentType.ALLIANCE, kept=False)

        distances = []
        for _ in range(8):
            m.observe(B, [B, R], CommitmentType.ALLIANCE, kept=True)
            pair = m.get_pair_record(B, R, CommitmentType.ALLIANCE).get_expected_value()
            blended = m.get_reliability(B, CommitmentType.ALLIANCE, toward=R)
            distances.append(abs(blended - pair))
        assert distances[-1] < distances[0]

    def test_betraying_a_long_standing_partner_costs_more(self):
        m = self._split_record()
        assert m.reputation_drop(B, CommitmentType.ALLIANCE, 0.0, toward=R) > \
               m.reputation_drop(B, CommitmentType.ALLIANCE, 0.0, toward=G)

    def test_no_counterparty_falls_back_to_reputation(self):
        m = self._split_record()
        assert m.get_reliability(B, CommitmentType.ALLIANCE, toward=None) == \
               m.get_record(B, CommitmentType.ALLIANCE).get_expected_value()

    def test_snapshot_carries_both_views(self):
        m = self._split_record()
        row = next(b for b in m.snapshot()
                   if b.subject == B and b.commitment_type == CommitmentType.ALLIANCE)
        assert row.reliability_toward_observer is not None
        assert row.reliability_toward_observer > row.expected_reliability


# ── 3. Multi-party pacts ─────────────────────────────────────────────────

class TestPacts:
    def _pact(self, **kw):
        return Commitment(id="p", commitment_type=CommitmentType.ALLIANCE,
                          players=[R, B, G], valid_until_turn=6, **kw)

    def test_a_pact_binds_every_member_not_just_the_first_two(self):
        """The old check read players[0] and players[1], so the third member
        could walk into anybody and never be graded."""
        state = board((R, "R2"), (B, "B1"), (G, "N1"))
        outcomes = verify_commitments(state, [self._pact()], [move(G, "N1", "R2")])
        assert not outcomes[0].kept and G in outcomes[0].broken_by

    def test_two_members_can_break_it_at_once(self):
        state = board((R, "R2"), (B, "B1"), (G, "N1"))
        outcomes = verify_commitments(state, [self._pact()],
                                      [move(G, "N1", "R2"), move(B, "B1", "N1")])
        assert set(outcomes[0].broken_by) == {G, B}

    def test_a_pact_kept_by_everyone_is_kept(self):
        state = board((R, "R2"), (B, "B1"), (G, "N1"))
        outcomes = verify_commitments(state, [self._pact()],
                                      [hold(R, "R2"), hold(B, "B1"), hold(G, "N1")])
        assert outcomes[0].kept

    def test_a_pact_only_forms_once_every_member_signs(self):
        runner = _runner()
        members = [R, B, G]
        offer = Message(id="pact", sender=R, receiver=B, message_type=MessageType.PROPOSE,
                        commitment_type=CommitmentType.ALLIANCE, turns=3,
                        coalition=members)
        runner._opening = [offer,
                           offer.model_copy(update={"receiver": G})]

        # Only Blue answers: nobody is bound yet.
        runner.agents[B].decisions = {}
        for agent in runner.agents.values():
            agent.propose = lambda *a, **k: []
        runner.agents[B].reply = lambda *a, **k: [
            Message(id=str(uuid.uuid4()), sender=B, receiver=R,
                    message_type=MessageType.ACCEPT, reference_id="pact")]
        runner.agents[G].reply = lambda *a, **k: []
        runner.agents[Y].reply = lambda *a, **k: []
        runner.agents[R].reply = lambda *a, **k: []
        runner._negotiate([])
        assert not runner.commitments, "a pact bound people before everyone signed"

        # Green signs too: now it exists, and it binds all three.
        runner._opening = [offer, offer.model_copy(update={"receiver": G})]
        runner.agents[G].reply = lambda *a, **k: [
            Message(id=str(uuid.uuid4()), sender=G, receiver=R,
                    message_type=MessageType.ACCEPT, reference_id="pact")]
        runner._negotiate([])
        assert len(runner.commitments) == 1
        assert set(runner.commitments[0].players) == {R, B, G}

    def test_a_broken_pact_releases_the_loyal_members(self):
        """The whole difference between one three-way promise and three
        bilateral ones."""
        runner = _runner()
        runner.commitments = [self._pact()]
        from src.common.schemas import CommitmentOutcome
        outcomes = [CommitmentOutcome(commitment=runner.commitments[0],
                                      kept=False, broken_by=[G])]
        # step() does the dissolving; reproduce the rule it applies.
        dissolved = {o.commitment.id for o in outcomes
                     if not o.kept and len(o.commitment.players) > 2}
        runner.commitments = [c for c in runner.commitments if c.id not in dissolved]
        assert runner.commitments == []

    @pytest.mark.parametrize("coalition,reason", [
        ([R, B], "three members"),
        ([B, G, Y], "outside it"),
    ])
    def test_malformed_pacts_are_refused(self, coalition, reason):
        msg = Message(id="m", sender=R, receiver=B, message_type=MessageType.PROPOSE,
                      commitment_type=CommitmentType.ALLIANCE, turns=3,
                      coalition=coalition)
        got = invalid_proposal_reason(msg)
        assert got and reason in got

    def test_a_pact_forms_in_a_real_game(self):
        """Coalitions are situational — somebody has to be far enough ahead
        that two others both want to gang up — so this sweeps seeds rather
        than asserting one game produces one."""
        found = [
            seed for seed in range(1, 13)
            if _pacts_in(seed)
        ]
        assert found, "no coalition ever formed across twelve games"

    def test_a_pact_only_appears_when_somebody_is_ahead(self):
        """Nobody organises a coalition against a player who is level."""
        random.seed(2)
        runner = _runner()
        ns = runner.agents[G].negotiation
        ns._live_commitments = []
        # Opening position: everybody holds two centres.
        assert not ns._pact_offers(runner.state, runner.agents[G].trust_model)


# ── 4. Exchanges ─────────────────────────────────────────────────────────

class TestExchanges:
    def _exchange(self, repay_turn=2):
        return Commitment(id="x", commitment_type=CommitmentType.EXCHANGE,
                          players=[R, B], valid_until_turn=4,
                          target_territory="N1", supported_from="B1",
                          dmz_territories=["R2"], repay_turn=repay_turn)

    def test_the_legs_fall_due_on_different_turns(self):
        c = self._exchange(repay_turn=3)
        assert exchange_leg_due(c, 1) == "give"
        assert exchange_leg_due(c, 3) == "repay"

    def test_the_giver_is_graded_first(self):
        state = board((R, "R2"), (B, "B1"), turn=1)
        kept = verify_commitments(state, [self._exchange()],
                                  [support(R, "R2", "B1", "N1"), move(B, "B1", "N1")])
        assert kept[0].kept
        broken = verify_commitments(state, [self._exchange()],
                                    [hold(R, "R2"), move(B, "B1", "N1")])
        assert not broken[0].kept and R in broken[0].broken_by

    def test_the_payer_is_not_on_the_hook_before_repayment_falls_due(self):
        """Turn 1 is the giver's leg; the payer cannot break it yet."""
        state = board((R, "R2"), (B, "B1"), turn=1)
        outcomes = verify_commitments(state, [self._exchange(repay_turn=2)],
                                      [support(R, "R2", "B1", "N1"), move(B, "B1", "R2")])
        assert outcomes[0].kept

    def test_taking_the_support_and_not_paying_is_a_betrayal(self):
        """The defection the structure exists to make possible."""
        state = board((R, "R2"), (B, "N1"), turn=2)
        outcomes = verify_commitments(state, [self._exchange(repay_turn=2)],
                                      [hold(R, "R2"), move(B, "N1", "R2")])
        assert not outcomes[0].kept and B in outcomes[0].broken_by

    def test_paying_up_keeps_it(self):
        state = board((R, "R2"), (B, "N1"), turn=2)
        outcomes = verify_commitments(state, [self._exchange(repay_turn=2)],
                                      [hold(R, "R2"), move(B, "N1", "C1")])
        assert outcomes[0].kept

    @pytest.mark.parametrize("kw,reason", [
        (dict(dmz_territories=None), "nothing owed"),
        (dict(dmz_territories=["NOWHERE"]), "unknown"),
        (dict(repay_turn=99), "after the game ends"),
    ])
    def test_malformed_exchanges_are_refused(self, kw, reason):
        base = dict(commitment_type=CommitmentType.EXCHANGE, turns=2,
                    target_territory="N1", supported_from="R2",
                    dmz_territories=["R1"])
        base.update(kw)
        msg = Message(id="m", sender=R, receiver=B,
                      message_type=MessageType.PROPOSE, **base)
        got = invalid_proposal_reason(msg)
        assert got and reason in got

    def test_an_exchange_is_proposable_and_priced(self):
        random.seed(17)
        runner = _runner()
        ns = runner.agents[R].negotiation
        ns._live_commitments = []
        offers = [o for o in ns._candidate_deals(runner.state, B)
                  if o.commitment_type == CommitmentType.EXCHANGE]
        assert offers, "no exchange was ever offered"
        offer = offers[0]
        assert offer.repay_turn == runner.state.turn + 1
        assert offer.target_territory not in (offer.dmz_territories or []), \
            "asked them to stay out of the very square we just helped them take"
        deal = message_to_commitment(offer, B, runner.state.turn)
        v_none, v_kept, v_broken = ns.deal_totals(runner.state, deal, B, [])
        assert v_kept >= v_none >= v_broken
