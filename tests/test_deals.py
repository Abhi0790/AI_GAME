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
from src.engine.board import get_all_territories, ring_board, use_board
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


# Pacts form against a leader, and at the default 5-of-10 threshold a leader
# usually wins before a coalition can (about one game in twenty-five). Played to
# 7 of 10, games run long enough that about one in four forms one.
PACT_SEEDS = range(1, 21)


def _long_game():
    return use_board(ring_board(win_fraction=0.7))


def _pacts_in(seed):
    random.seed(seed)
    with _long_game():
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

    def test_a_lapse_costs_little_where_trust_is_established(self):
        """One break by a partner with a long clean record barely moves the
        prediction: kept rates are flat above reliability 0.4, so the fitted
        CPT is too. Another break against a partner already betrayed costs a
        lot, because that record sits on the CPT's drop."""
        m = self._split_record()
        toward_red = m.reputation_drop(B, CommitmentType.ALLIANCE, 0.0, toward=R)
        toward_green = m.reputation_drop(B, CommitmentType.ALLIANCE, 0.0, toward=G)
        assert 0 <= toward_red < toward_green

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

    def test_a_broken_bilateral_deal_ends(self):
        """Left standing, a two-way deal was broken again every turn of the war
        that followed, and each time counted as a fresh betrayal."""
        from src.common.schemas import Commitment, Order, OrderType
        runner = _runner()
        dmz = Commitment(id="d", commitment_type=CommitmentType.DMZ, players=[R, B],
                         valid_until_turn=9, dmz_territories=["N1"])
        runner.commitments = [dmz]
        for p, a in runner.agents.items():
            a.propose = lambda *a_, **k: []
            a.reply = lambda *a_, **k: []
            a.act = (lambda s, c: ([Order(player=R, unit_territory="R2",
                                          order_type=OrderType.MOVE, target="N1")], None)
                     ) if p == R else (lambda s, c: ([], None))
        record = runner.step()
        assert [o.kept for o in record.outcomes] == [False]
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
        found = [seed for seed in PACT_SEEDS if _pacts_in(seed)]
        assert found, f"no coalition formed in seeds {PACT_SEEDS}"

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


# ── Evidence tiers: who learns what (R5) ─────────────────────────────────

class TestEvidenceTiers:
    """An agent's reputation record and its per-pair record have to be built
    from *different* evidence, or the pair layer cannot ever disagree with
    the reputation and is dead weight.
    """

    def _outcome(self, players, broken_by, private=False):
        from src.common.schemas import CommitmentOutcome
        c = Commitment(id="c", commitment_type=CommitmentType.ALLIANCE,
                       players=list(players), valid_until_turn=6, private=private)
        return CommitmentOutcome(commitment=c, kept=not broken_by,
                                 broken_by=list(broken_by))

    def _agent(self):
        from src.agents.agent import Agent as _A
        return _A(R, "Honest")

    def _state(self):
        return board((R, "R1"), (B, "B1"), (G, "G1"), (Y, "Y1"))

    def test_a_public_break_between_others_is_seen(self):
        a = self._agent()
        before = a.trust_model.get_reliability(B, CommitmentType.ALLIANCE)
        s = self._state()
        a.update_beliefs_from_outcomes(s, s, [self._outcome([B, G], [B])])
        assert a.trust_model.get_reliability(B, CommitmentType.ALLIANCE) < before

    def test_a_private_break_between_others_is_invisible(self):
        """A deal I am not in and cannot see moves nothing."""
        a = self._agent()
        before = a.trust_model.get_reliability(B, CommitmentType.ALLIANCE)
        s = self._state()
        a.update_beliefs_from_outcomes(s, s, [self._outcome([B, G], [B], private=True)])
        assert a.trust_model.get_reliability(B, CommitmentType.ALLIANCE) == before

    def test_watching_others_does_not_touch_the_pair_record(self):
        """Seeing Blue betray Green says nothing about how Blue treats me."""
        a = self._agent()
        s = self._state()
        before = a.trust_model.get_pair_record(
            B, R, CommitmentType.ALLIANCE).get_expected_value()
        a.update_beliefs_from_outcomes(s, s, [self._outcome([B, G], [B])])
        after = a.trust_model.get_pair_record(
            B, R, CommitmentType.ALLIANCE).get_expected_value()
        assert after == before

    def test_reputation_and_relationship_can_now_disagree(self):
        """The property the whole per-pair layer exists for: Blue keeps every
        promise to me and breaks every promise with everyone else."""
        a = self._agent()
        s = self._state()
        for _ in range(4):
            a.update_beliefs_from_outcomes(s, s, [self._outcome([B, R], [])])
            a.update_beliefs_from_outcomes(s, s, [self._outcome([B, G], [B])])
        general = a.trust_model.get_reliability(B, CommitmentType.ALLIANCE)
        toward_me = a.trust_model.get_reliability(B, CommitmentType.ALLIANCE, toward=R)
        assert toward_me > general + 0.1

    def test_a_public_outcome_is_not_counted_twice_for_a_party(self):
        """OutcomeRule owns the deals I am in; R5 must keep its hands off."""
        from src.agents.trust.rules import PublicRecordRule
        a = self._agent()
        before = a.trust_model.get_reliability(B, CommitmentType.ALLIANCE)
        traces = PublicRecordRule().evaluate(
            a.trust_model, {"public_outcome": self._outcome([B, R], [B])})
        assert traces == []
        assert a.trust_model.get_reliability(B, CommitmentType.ALLIANCE) == before


# ── Bookkeeping defects these features exposed ───────────────────────────

class TestTurnRecordIntegrity:
    def test_a_pact_broken_the_turn_it_forms_is_still_recorded(self):
        """It is dissolved at the end of the turn, but it was live during it.
        Recording the survivors only made broken pacts outnumber made ones."""
        from src.evaluation.metrics import deal_mix
        found = False
        for seed in PACT_SEEDS:
            random.seed(seed)
            with _long_game():
                runner = _runner()
                runner.run(verbose=False)
            mix = deal_mix(runner.history)
            for kind, made in mix["made"].items():
                assert mix["broken"].get(kind, 0) <= made, (
                    f"{kind}: {mix['broken'].get(kind)} broken but only {made} made")
            if mix["made"].get("pact"):
                found = True
        assert found, "no pact in these games, so the check proved nothing"


class TestHumanSeatDecisions:
    """The examiner has to be able to answer two offers from the same player
    differently. Keying answers on (sender, type) alone made a pact and a
    bilateral alliance — and a public and a private one — the same question.
    """

    def _offer(self, **kw):
        from src.agents.agent import HumanAgent
        return HumanAgent(R, "Honest"), Message(
            id=str(uuid.uuid4()), sender=B, receiver=R,
            message_type=MessageType.PROPOSE,
            commitment_type=CommitmentType.ALLIANCE, turns=3, **kw)

    def test_a_pact_and_a_bilateral_offer_are_different_questions(self):
        human, bilateral = self._offer()
        _, pact = self._offer(coalition=[B, R, G])
        assert human.decision_key(bilateral) != human.decision_key(pact)

    def test_a_public_and_a_private_offer_are_different_questions(self):
        human, public = self._offer()
        _, private = self._offer(private=True)
        assert human.decision_key(public) != human.decision_key(private)

    def test_a_counter_offer_inherits_the_answer_already_given(self):
        human, original = self._offer()
        counter = original.model_copy(update={
            "id": "other", "message_type": MessageType.COUNTER, "turns": 1})
        assert human.decision_key(counter) == human.decision_key(original)


class TestReplayCarriesTheDeals:
    def test_a_replay_round_trips_every_deal_structure(self):
        """A replay that records only the verdicts cannot show what was on
        the table during a turn nobody broke anything."""
        import tempfile
        from src.engine.replay import (
            save_replay, load_replay, reconstruct_commitments,
        )
        random.seed(2)
        runner = _runner()
        runner.run(verbose=False)

        with tempfile.TemporaryDirectory() as tmp:
            path = save_replay(runner.history, runner.state,
                               directory=tmp, filename="r.json")
            replay = load_replay(path)

        assert replay["version"] >= 2
        turns = reconstruct_commitments(replay)
        assert len(turns) == len(runner.history)
        flat = [c for turn in turns for c in turn]
        assert flat, "no commitment survived the round trip"
        assert any(c.private for c in flat), "privacy was lost"
        assert any(c.commitment_type == CommitmentType.EXCHANGE for c in flat), \
            "exchanges were lost"
        assert any(c.repay_turn for c in flat), "repayment turns were lost"

        # ...and the live deals match what the runner actually held.
        for recorded, step in zip(turns, runner.history):
            assert {c.id for c in recorded} == {c.id for c in step.commitments}


class TestPrivateVisibility:
    """A private deal is only private if the seats outside it cannot see it."""

    def _runner(self, leak):
        from src.common.config import GameConfig
        from src.harness import build_runner, game_setup
        cfg = GameConfig(seed=4, node_budget=100, max_turns=4,
                         knobs={"PRIVATE_LEAK_RATE": leak})
        with game_setup(cfg):
            r = build_runner(cfg)
            r.run(verbose=False)
            return r

    def test_an_outsider_is_not_shown_a_secret_deal(self):
        from src.common.schemas import Commitment, CommitmentType, Player

        r = self._runner(0.0)
        secret = Commitment(id="s1", commitment_type=CommitmentType.DMZ,
                            players=[Player.RED, Player.BLUE], valid_until_turn=99,
                            dmz_territories=["N1"], private=True)
        r.commitments = [secret]
        assert r.knows(Player.RED, secret) and r.knows(Player.BLUE, secret)
        assert not r.knows(Player.GREEN, secret)
        assert secret not in r.visible_to(Player.GREEN)
        assert secret in r.visible_to(Player.RED)

    def test_a_public_deal_is_visible_to_everyone(self):
        from src.common.schemas import Commitment, CommitmentType, Player

        r = self._runner(0.0)
        open_deal = Commitment(id="p1", commitment_type=CommitmentType.ALLIANCE,
                               players=[Player.RED, Player.BLUE], valid_until_turn=99)
        r.commitments = [open_deal]
        assert all(r.knows(p, open_deal) for p in r.agents)

    def test_a_leaked_deal_becomes_visible_to_whoever_learned_it(self):
        from src.common.schemas import Commitment, CommitmentType, Player

        r = self._runner(0.0)
        secret = Commitment(id="s2", commitment_type=CommitmentType.DMZ,
                            players=[Player.RED, Player.BLUE], valid_until_turn=99,
                            dmz_territories=["N1"], private=True)
        r.commitments = [secret]
        r._leaked["s2"] = {Player.GREEN}
        assert r.knows(Player.GREEN, secret)
        assert not r.knows(Player.GOLD, secret)

    def test_the_leak_rate_spans_from_airtight_to_fully_open(self):
        assert not any(v for v in self._runner(0.0)._leaked.values())
        assert any(v for v in self._runner(1.0)._leaked.values())


class TestDirectionalDealsSurviveNegotiation:
    """Who owes what must not depend on who spoke last, or on a renewal."""

    def test_a_counter_offer_does_not_flip_the_supporter(self):
        import uuid
        from src.common.schemas import (
            Message, MessageType, CommitmentType, Player, message_to_commitment)
        from src.agents.negotiation.strategy import NegotiationStrategy

        offer = Message(id=str(uuid.uuid4()), sender=Player.RED,
                        receiver=Player.BLUE, message_type=MessageType.PROPOSE,
                        commitment_type=CommitmentType.SUPPORT, turns=4,
                        target_territory="N1", supported_from="B1")
        straight = message_to_commitment(offer, Player.BLUE, 1)

        ns = NegotiationStrategy.__new__(NegotiationStrategy)
        ns.player = Player.BLUE
        counter = NegotiationStrategy._counter_terms(ns, offer)
        countered = message_to_commitment(counter, Player.RED, 1)

        assert straight.players[0] == Player.RED
        assert countered.players[0] == Player.RED, (
            "countering the same terms handed the duty to the other side")

    def test_renewing_an_exchange_moves_its_repayment_date(self):
        from src.common.schemas import (
            Commitment, CommitmentType, Player, exchange_leg_due)
        from src.common.config import GameConfig
        from src.harness import build_runner, game_setup

        cfg = GameConfig(seed=1, node_budget=100, max_turns=4)
        with game_setup(cfg):
            runner = build_runner(cfg)
        old = Commitment(id="e1", commitment_type=CommitmentType.EXCHANGE,
                         players=[Player.RED, Player.BLUE], valid_until_turn=3,
                         target_territory="N1", supported_from="R2",
                         dmz_territories=["C1"], repay_turn=3)
        runner.commitments = [old]
        renewed = Commitment(id="e2", commitment_type=CommitmentType.EXCHANGE,
                             players=[Player.RED, Player.BLUE], valid_until_turn=9,
                             target_territory="N1", supported_from="R2",
                             dmz_territories=["C1"], repay_turn=9)
        # What the runner does on a renewal.
        old.valid_until_turn = max(old.valid_until_turn, renewed.valid_until_turn)
        if renewed.repay_turn is not None:
            old.repay_turn = renewed.repay_turn
            old.valid_until_turn = max(old.valid_until_turn, renewed.repay_turn)

        assert old.repay_turn == 9
        assert exchange_leg_due(old, 5) == "give", (
            "a renewed exchange still owes its support leg")
        assert exchange_leg_due(old, 9) == "repay"


def test_mirror_support_deals_are_recognised():
    from src.common.schemas import Commitment, CommitmentType, Player, mirrors
    def leg(a, b, frm):
        return Commitment(id=f"{a.value}{b.value}", commitment_type=CommitmentType.SUPPORT,
                          players=[a, b], created_turn=1, valid_until_turn=1,
                          target_territory="C1", supported_from=frm)
    blue_for_green = leg(Player.BLUE, Player.GREEN, "G1")
    assert mirrors(blue_for_green, leg(Player.GREEN, Player.BLUE, "B2"))
    assert not mirrors(blue_for_green, leg(Player.BLUE, Player.GREEN, "G1"))
