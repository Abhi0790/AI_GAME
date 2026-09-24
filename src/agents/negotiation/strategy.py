"""Negotiation — the closed grammar, priced by the planner.

Nothing here decides anything on a trust threshold any more. A deal is worth
proposing when the planner says the board is better with it honoured than
without it, and worth accepting when

    P(keep) * V_kept + (1 - P(keep)) * V_broken  >  V_none

with my own exposure counted inside V_broken: in that branch I am the one
keeping my side of a deal the other player is walking away from.
"""

from typing import List, Dict, Optional, Tuple
import uuid

from src.common.schemas import (
    GameState, Message, MessageType, Player, CommitmentType,
    Commitment, Order, OrderType, message_to_commitment,
    invalid_proposal_reason, commitment_key,
)
from src.agents.planner.planner import (
    Planner, evaluate_state, cooperation_value, hostile_world,
    breaks_commitment, defection_incentive, near_win, CENTRE_VALUE,
)
from src.agents.trust.model import TrustModel
from src.engine.adjudicator import resolve
from src.engine.orders import generate_all_order_sets
from src.engine.board import get_adjacent, is_supply_center, max_turns, players

# Deals whose counterfactual gain is smaller than this are noise, not offers.
MIN_DEAL_VALUE = 0.05
MAX_PROPOSALS_PER_OPPONENT = 2
# How far behind the leader this player has to be before a coalition against
# them is worth organising.
PACT_DEFICIT = 2
# A threat repeated every turn is noise, and the receiver stops pricing it.
THREAT_COOLDOWN = 3


class NegotiationStrategy:
    def __init__(self, player: Player, planner: Planner):
        self.player = player
        self.planner = planner
        # Threats I have sent, so I know which ones I am on the hook for.
        self.threats_made: Dict[Player, int] = {}
        # The deals already on the table this turn. The Agent refreshes it
        # before every call; pricing a new offer means pricing it against
        # what I am already committed to.
        self._live_commitments: List[Commitment] = []
        # Pruning the candidate set is the expensive part of pricing a deal,
        # and it is identical for every deal considered in the same position.
        # Memoised per (turn, live deals); a dozen offers then cost one prune.
        self._candidate_memo: Tuple = (None, None)
        # Set by the Agent from the persona: how readily this player keeps a
        # deal off the public record.
        self.privacy_preference: float = 0.0
        # What I thought each deal was worth when I put my name to it, keyed
        # on the deal's identity rather than the message id so a renewal or a
        # counter-offer still finds the price I originally paid. Read back by
        # the runner against what the planner does with the same deal in the
        # same turn.
        self.deal_prices: Dict[tuple, float] = {}
        # ...and on which turn I put my name to it, so the planner can charge
        # the *unused* share of that price back when it walks away.
        self.deal_signed_turn: Dict[tuple, int] = {}
        # Every deal priced, signed or not: key -> (turn, gain). Only a quote
        # from the turn a deal is actually signed becomes its price.
        self.quotes: Dict[tuple, Tuple[int, float]] = {}
        # Players the persona's policy signs nothing with this turn. The Agent
        # refreshes it before every call (Agent.policy).
        self.refuse: set = set()
        # Whether my persona owes its partners loyalty (Agent.policy). A bound
        # player cannot sign with everyone and still grow, so it signs less.
        self.bound: bool = False

    # ------------------------------------------------------------------
    # Counterfactual machinery — one small search, reused by every branch
    # ------------------------------------------------------------------
    def _candidates(self, state: GameState, commitments: List[Commitment]) -> List[List[Order]]:
        """My plausible order sets, pruned by the planner's own dominance
        filter so pricing a deal costs a handful of adjudications, not 81."""
        key = (state.turn, tuple(sorted(c.id for c in commitments)),
               tuple(sorted((u.territory, u.player.value) for u in state.units)))
        if self._candidate_memo[0] == key:
            return self._candidate_memo[1]
        sets = generate_all_order_sets(state, self.player)
        kept, _pruned = self.planner.prune(state, sets, commitments)
        self._candidate_memo = (key, kept)
        return kept

    def _value_of(self, state: GameState, plan: List[Order], world: List[Order],
                  commitments: List[Commitment]) -> float:
        """What one plan of mine is worth against one picture of the others."""
        new_state, _o, _l = resolve(state, plan + world)
        return evaluate_state(new_state, self.player)

    def _best_plan(self, state: GameState, candidates: List[List[Order]],
                   worlds: List[List[Order]], weights: List[float],
                   commitments: List[Commitment]) -> Tuple[List[Order], float]:
        """The single plan I would actually play, and what it scores.

        One plan across all the worlds, not a fresh best reply per world. The
        difference matters: choosing with hindsight inside each world makes a
        partner's peace worth exactly nothing, because I could always have
        defended perfectly against an attack I already knew was coming. It is
        committing to *one* plan that a promise changes.
        """
        best, best_score = None, float("-inf")
        for cand in candidates:
            score = sum(w * self._value_of(state, cand, world, commitments)
                        for world, w in zip(worlds, weights))
            if score > best_score:
                best, best_score = cand, score
        if best is None:
            return [], evaluate_state(state, self.player)
        return best, best_score

    def _partners_of(self, deal: Optional[Commitment], partner: Player) -> List[Player]:
        """Everyone whose behaviour the deal is supposed to change.

        For a pact that is all the other members, not just whoever proposed
        it: a three-way alliance is worth roughly twice a two-way one, and
        pricing it against a single partner understated it so badly that
        coalitions were almost never accepted.
        """
        if deal is not None and len(deal.players) > 2:
            return [p for p in deal.players if p != self.player]
        return [partner]

    def _worlds(self, state: GameState, partner: Player, deal: Optional[Commitment]
                ) -> Tuple[List[Order], List[Order]]:
        """Two pictures of the other side: honouring the deal, and exploiting
        it. Everyone outside the deal is assumed hostile in both, so the
        comparison isolates the members' choice rather than the weather.
        """
        partners = self._partners_of(deal, partner)
        others = [o for o in hostile_world(state, self.player)
                  if o.player not in partners]
        partner_units = [u for u in state.units if u.player in partners]

        kept: List[Order] = list(others)
        broken: List[Order] = list(others)
        mine = {u.territory for u in state.units if u.player == self.player}
        mine |= {t for t, owner in state.supply_centers.items() if owner == self.player}

        # What "honouring" means for the partner depends on which side of the
        # deal they are on. In an exchange I give the support and they repay
        # later by staying out of somewhere, so their kept-world behaviour is
        # the repayment, not a support order.
        owed = set(deal.dmz_territories or []) if (
            deal is not None and deal.commitment_type == CommitmentType.EXCHANGE) else set()

        for u in partner_units:
            partner = u.player
            if owed and partner == (deal.players[1] if deal else partner):
                # Honouring the repayment: anywhere but the squares they owe
                # me. Breaking it: straight into one of them, which is the
                # whole reason the repayment was worth asking for.
                clear = [a for a in get_adjacent(u.territory) if a not in owed]
                kept.append(Order(player=partner, unit_territory=u.territory,
                                  order_type=OrderType.MOVE if clear else OrderType.HOLD,
                                  target=clear[0] if clear else None))
                grab = next((a for a in get_adjacent(u.territory) if a in owed), None)
                broken.append(Order(player=partner, unit_territory=u.territory,
                                    order_type=OrderType.MOVE if grab else OrderType.HOLD,
                                    target=grab))
                continue

            # Honouring: stays off my squares. For a support deal, actually
            # gives the support that was promised.
            if (deal is not None and deal.commitment_type == CommitmentType.SUPPORT
                    and deal.supported_from == u.territory):
                kept.append(Order(player=partner, unit_territory=u.territory,
                                  order_type=OrderType.MOVE, target=deal.target_territory))
            elif (deal is not None and deal.commitment_type == CommitmentType.SUPPORT
                  and deal.target_territory in get_adjacent(u.territory)):
                kept.append(Order(player=partner, unit_territory=u.territory,
                                  order_type=OrderType.SUPPORT,
                                  target=deal.target_territory,
                                  supported_from=deal.supported_from))
            else:
                safe = [a for a in get_adjacent(u.territory) if a not in mine]
                target = next((a for a in safe if is_supply_center(a)), None)
                kept.append(Order(player=partner, unit_territory=u.territory,
                                  order_type=OrderType.MOVE if target else OrderType.HOLD,
                                  target=target))

            # Exploiting: walks straight onto whatever of mine it can reach.
            grab = next((a for a in get_adjacent(u.territory) if a in mine), None)
            broken.append(Order(player=partner, unit_territory=u.territory,
                                order_type=OrderType.MOVE if grab else OrderType.HOLD,
                                target=grab))
        return kept, broken

    def deal_values(self, state: GameState, deal: Commitment, partner: Player,
                    commitments: List[Commitment]) -> Tuple[float, float, float]:
        """(V_none, V_kept, V_broken) for one candidate deal.

        V_none is what I can do with no deal at all — every order set open to
        me. V_kept and V_broken are restricted to the order sets that honour
        the deal, because that restriction *is* the price of signing, and it
        is exactly what makes V_broken worse than V_none.
        """
        free = self._candidates(state, commitments)
        with_deal = commitments + [deal]
        honouring = [c for c in free
                     if not breaks_commitment(state, self.player, c, with_deal)]
        if not honouring:
            honouring = free

        kept_world, broken_world = self._worlds(state, partner, deal)
        worlds = [kept_world, broken_world]

        # No deal: the partner is committed to nothing, so I hedge — one plan,
        # chosen against both pictures of them at once, with my own orders
        # unconstrained.
        _plan, v_none = self._best_plan(state, free, worlds, [0.5, 0.5], commitments)

        # With the deal I plan for the partner honouring it, and I give up the
        # order sets that would break my own side. V_broken is that same plan
        # meeting a partner who walks away: my exposure is the plan I can no
        # longer un-choose, which is exactly what signing costs.
        plan, _ = self._best_plan(state, honouring, worlds, [1.0, 0.0], with_deal)
        v_kept = self._value_of(state, plan, kept_world, with_deal)
        v_broken = self._value_of(state, plan, broken_world, with_deal)
        return v_none, v_kept, v_broken

    def deal_totals(self, state: GameState, deal: Commitment, partner: Player,
                    commitments: List[Commitment]) -> Tuple[float, float, float]:
        """(V_none, V_kept, V_broken) over the whole life of the deal.

        The per-turn counterfactual cannot see a three-turn alliance: it
        prices one turn of a promise that is supposed to buy several, which
        is why nothing was ever worth proposing at the opening. Two changes:

          * the per-turn gap is carried over the turns the deal runs for;
          * the kept branch picks up Vcoop, discounted by the same
            turns-left/12 horizon the planner charges for breaking it.

        Making a promise and breaking one are then priced with the same
        quantity, which is the only way the two decisions can be consistent.
        """
        v_none, v_kept, v_broken = self.deal_values(state, deal, partner, commitments)
        # valid_until_turn is the last turn the deal is graded, inclusive.
        turns = max(1, deal.valid_until_turn - state.turn + 1)
        coop = sum(cooperation_value(state, self.player, q)
                   for q in self._partners_of(deal, partner)) * turns / max_turns()
        return turns * v_none, turns * v_kept + coop, turns * v_broken

    def exposure(self, state: GameState, partner: Player) -> float:
        """How much of mine sits inside their reach, normalised.

        A standing position, not a temptation: it says whether this player is
        worth talking to at all, and it is what a threat is aimed at.
        """
        return min(1.0, cooperation_value(state, partner, self.player) / (2 * CENTRE_VALUE))

    def their_incentive(self, state: GameState, partner: Player,
                        deal: Optional[Commitment] = None) -> float:
        """What breaking is worth to *them* this turn — the incentive node of
        the trust network.

        The same estimator the calibration log records, so the probability a
        deal is accepted on is the probability that gets graded afterwards.
        Exposure was standing in for this and turned out to carry no signal
        about whether anyone actually broke anything.
        """
        live = list(self._live_commitments) + ([deal] if deal is not None else [])
        return defection_incentive(state, partner, live)

    def overcommits(self, state: GameState, deal: Commitment) -> bool:
        """Would a bound player be signing away its growth?

        Two ways, and only for a bound player: a second two-way alliance (a
        pact against a leader is exempt), or a promise to stay out of a centre
        it does not own but can reach -- a DMZ, or the repay leg of an exchange.
        With every neighbour allied and every contested centre demilitarised,
        a player who keeps its word has nothing left to take.
        """
        if not self.bound:
            return False
        if deal.commitment_type == CommitmentType.ALLIANCE and len(deal.players) == 2:
            others = set(deal.players) - {self.player}
            return any(c.commitment_type == CommitmentType.ALLIANCE and len(c.players) == 2
                       and self.player in c.players and not others & set(c.players)
                       for c in self._live_commitments)
        keep_out = []
        if deal.commitment_type == CommitmentType.DMZ:
            keep_out = deal.dmz_territories or []
        elif (deal.commitment_type == CommitmentType.EXCHANGE
              and deal.players[1:2] == [self.player]):
            keep_out = deal.dmz_territories or []
        reach = {a for u in state.units if u.player == self.player
                 for a in get_adjacent(u.territory)}
        return any(is_supply_center(t) and state.supply_centers.get(t) != self.player
                   and t in reach for t in keep_out)

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------
    def _candidate_deals(self, state: GameState, partner: Player) -> List[Message]:
        """Every sentence of the grammar that is legal to say right now."""
        my_units = [u for u in state.units if u.player == self.player]
        their_units = [u for u in state.units if u.player == partner]
        offers: List[Message] = []

        def msg(**kw) -> Message:
            return Message(id=str(uuid.uuid4()), sender=self.player, receiver=partner,
                           message_type=MessageType.PROPOSE, **kw)

        offers.append(msg(commitment_type=CommitmentType.ALLIANCE, turns=3))

        # DMZ on the centres we both border and *neither of us owns*. Without
        # the second half this proposed demilitarising the other player's own
        # home centre, which is not contested ground — it is just asking them
        # to stop reinforcing their border.
        contested = {
            t for u in my_units for t in get_adjacent(u.territory)
            if is_supply_center(t)
            and state.supply_centers.get(t) not in (self.player, partner)
            and any(t in get_adjacent(ou.territory) for ou in their_units)
        }
        if contested:
            offers.append(msg(commitment_type=CommitmentType.DMZ, turns=2,
                              dmz_territories=sorted(contested)))

        # Support a move: their unit into a square I also border.
        for ou in their_units:
            for adj in get_adjacent(ou.territory):
                if any(adj in get_adjacent(mu.territory) for mu in my_units) \
                        and state.supply_centers.get(adj) not in (self.player, partner):
                    offers.append(msg(commitment_type=CommitmentType.SUPPORT, turns=1,
                                      target_territory=adj, supported_from=ou.territory))
                    break

        # Support a hold: I promise to prop up a centre of theirs that is
        # under pressure. Third sentence of the grammar, previously unsayable
        # because nothing ever generated a SUPPORT with no origin.
        for ou in their_units:
            if any(ou.territory in get_adjacent(mu.territory) for mu in my_units) \
                    and is_supply_center(ou.territory):
                offers.append(msg(commitment_type=CommitmentType.SUPPORT, turns=1,
                                  target_territory=ou.territory))
                break

        # An exchange: I support their move this turn, and next turn they
        # keep out of somewhere I want. The two halves are worth different
        # amounts and fall due on different turns, which is the only way to
        # trade something I have now for something I want later.
        repayment = sorted({
            t for u in my_units for t in get_adjacent(u.territory)
            if is_supply_center(t)
            and state.supply_centers.get(t) != partner
            and any(t in get_adjacent(ou.territory) for ou in their_units)
        })
        for ou in their_units:
            gift = next((adj for adj in get_adjacent(ou.territory)
                         if any(adj in get_adjacent(mu.territory) for mu in my_units)
                         and state.supply_centers.get(adj) not in (self.player, partner)),
                        None)
            owed = [t for t in repayment if t != gift]
            if gift and owed:
                offers.append(msg(commitment_type=CommitmentType.EXCHANGE, turns=2,
                                  target_territory=gift, supported_from=ou.territory,
                                  dmz_territories=owed[:2],
                                  repay_turn=state.turn + 1))
                break

        # Anything aimed at a third party is worth keeping quiet: announcing
        # it warns the target. A private deal also cannot be proved either way
        # afterwards, which is a cost to the honest and an opportunity to the
        # rest -- so the willingness to go private tracks the persona.
        for offer in offers:
            targets_a_third_party = (
                offer.commitment_type in (CommitmentType.SUPPORT, CommitmentType.EXCHANGE)
                and offer.target_territory is not None
                and state.supply_centers.get(offer.target_territory)
                not in (None, self.player, partner))
            if targets_a_third_party or self.privacy_preference > 0.5:
                offer.private = True

        return offers

    # ------------------------------------------------------------------
    # Multi-party pacts
    # ------------------------------------------------------------------
    def _pact_offers(self, state: GameState, trust_model: TrustModel) -> List[Message]:
        """A three-way pact against whoever is running away with the game.

        Two players behind the leader can each hold their own bilateral deal
        and still be picked off one at a time. The point of a pact is that it
        is a single promise: it only forms if everybody signs, and the moment
        one member breaks it the others are released rather than left bound
        to a coalition that no longer exists.
        """
        counts = {p: 0 for p in players()}
        for owner in state.supply_centers.values():
            if owner:
                counts[owner] += 1
        alive = [p for p in players() if any(u.player == p for u in state.units)]
        leader = max(alive, key=lambda p: counts[p], default=None)
        if leader is None or leader == self.player:
            return []
        if counts[leader] - counts[self.player] < PACT_DEFICIT:
            return []

        # Everyone else who is also behind, most trustworthy first.
        allies = sorted(
            (p for p in alive if p not in (self.player, leader)
             and counts[p] < counts[leader] and p not in self.refuse),
            key=lambda p: -trust_model.get_reliability(
                p, CommitmentType.ALLIANCE, toward=self.player))
        if len(allies) < 2:
            return []

        members = [self.player] + allies[:2]
        # One id shared by every copy: the members are answering the *same*
        # proposal, and the runner only forms the pact once all of them have
        # signed that one id. Giving each copy its own id meant the
        # signatures never met and no pact could ever form.
        pact_id = str(uuid.uuid4())
        return [
            Message(id=pact_id, sender=self.player, receiver=m,
                    message_type=MessageType.PROPOSE,
                    commitment_type=CommitmentType.ALLIANCE, turns=3,
                    coalition=members)
            for m in members if m != self.player
        ]

    def generate_proposals(self, state: GameState, trust_model: TrustModel) -> List[Message]:
        proposals: List[Message] = []
        commitments = self._live_commitments

        # A pact is proposed on the board position, not on any one partner's
        # value to me, so it is generated once rather than per opponent.
        if not any(len(c.players) > 2 and self.player in c.players
                   for c in commitments):
            proposals.extend(self._pact_offers(state, trust_model))

        for p in players():
            if p == self.player:
                continue
            their_units = [u for u in state.units if u.player == p]
            if not their_units:
                continue

            ranked: List[Tuple[float, Message]] = []
            barred = near_win(state, p) or p in self.refuse
            for offer in ([] if barred else self._candidate_deals(state, p)):
                deal = message_to_commitment(offer, p, state.turn)
                if self.overcommits(state, deal):
                    continue
                v_none, v_kept, _v_broken = self.deal_totals(state, deal, p, commitments)
                # Ranked by the counterfactual the deck asks for: value if
                # honoured minus value without the deal.
                self._record_price(state, deal, v_kept - v_none)
                ranked.append((v_kept - v_none, offer))

            ranked.sort(key=lambda r: r[0], reverse=True)
            for gain, offer in ranked[:MAX_PROPOSALS_PER_OPPONENT]:
                if gain > MIN_DEAL_VALUE:
                    offer.rationale = {"gain": gain, "considered": len(ranked),
                                       "floor": MIN_DEAL_VALUE}
                    proposals.append(offer)

            # A threat is what is left when there is nothing worth trading:
            # they can reach my centres and no deal with them pays.
            exposed = self.exposure(state, p)
            cooled = state.turn - self.threats_made.get(p, -THREAT_COOLDOWN) >= THREAT_COOLDOWN
            if not any(m.receiver == p for m in proposals) and exposed > 0.3 and cooled:
                proposals.append(Message(
                    id=str(uuid.uuid4()), sender=self.player, receiver=p,
                    message_type=MessageType.THREAT,
                    condition=f"you move into a centre of {self.player.value}",
                    action="I spend the rest of the game taking yours"))
                self.threats_made[p] = state.turn

        return proposals

    # ------------------------------------------------------------------
    # Proposal evaluation
    # ------------------------------------------------------------------
    def evaluate_proposal(self, state: GameState, msg: Message,
                          trust_model: TrustModel) -> Optional[Message]:
        """ACCEPT, COUNTER or REJECT one incoming sentence."""
        if msg.message_type == MessageType.THREAT:
            # Not a deal, so there is nothing to accept. It is priced instead:
            # the agent records it and the planner charges more for attacking
            # a player who has promised to retaliate (see Agent.deterrence).
            return None

        if invalid_proposal_reason(msg) is not None:
            return self._reject(msg)  # malformed or out of grammar
        if any(p != self.player and (near_win(state, p) or p in self.refuse)
               for p in (msg.coalition or [msg.sender])):
            return self._reject(msg)  # about to win, or barred by my policy

        commitments = self._live_commitments
        deal = message_to_commitment(msg, self.player, state.turn)
        if self.overcommits(state, deal):
            return self._reject(msg)
        v_none, v_kept, v_broken = self.deal_totals(state, deal, msg.sender, commitments)
        self._record_price(state, deal, v_kept - v_none)
        # toward=self.player: what matters is whether they keep promises
        # *to me*, which can differ sharply from their general reputation.
        p_keep = trust_model.p_keeps(msg.sender, msg.commitment_type,
                                     self.their_incentive(state, msg.sender, deal),
                                     toward=self.player)
        ev = p_keep * v_kept + (1 - p_keep) * v_broken
        why = {"p_keep": p_keep, "v_none": v_none, "v_kept": v_kept,
               "v_broken": v_broken, "ev": ev}

        if ev > v_none:
            return self._accept(msg, why)

        # Not at these terms — but a shorter deal exposes me for fewer turns,
        # so try the same sentence with a smaller number in it before walking
        # away. This is what makes negotiation rounds 2 and 3 do any work.
        counter = self._counter_terms(msg)
        if counter is not None and msg.message_type != MessageType.COUNTER:
            c_deal = message_to_commitment(counter, self.player, state.turn)
            c_none, c_kept, c_broken = self.deal_totals(state, c_deal, msg.sender, commitments)
            c_ev = p_keep * c_kept + (1 - p_keep) * c_broken
            if c_ev > c_none:
                # Not priced into deal_prices: c_deal is built from my own
                # counter, so both "players" are me and its key is not the
                # key the accepted deal will have. The planner re-prices it.
                counter.rationale = {**why, "ev_countered": c_ev,
                                     "v_none_countered": c_none}
                return counter

        return self._reject(msg, why)

    def _counter_terms(self, msg: Message) -> Optional[Message]:
        """The same deal, smaller. Halve the term; for a DMZ, give back the
        territories I most want to keep my hands free on.

        A pact is not counter-offerable: its members are answering one shared
        proposal, and a bilateral counter would quietly drop the coalition and
        turn it into a two-way deal nobody agreed to.
        """
        if msg.coalition:
            return None
        turns = max(1, (msg.turns or 1) // 2)
        dmz = msg.dmz_territories
        if msg.commitment_type == CommitmentType.DMZ and dmz and len(dmz) > 1:
            dmz = sorted(dmz)[:len(dmz) - 1]
        elif turns == (msg.turns or 1) and msg.commitment_type == CommitmentType.DMZ:
            return None
        if turns == (msg.turns or 1) and dmz == msg.dmz_territories:
            return None

        return Message(
            id=str(uuid.uuid4()), sender=self.player, receiver=msg.sender,
            message_type=MessageType.COUNTER, reference_id=msg.id,
            commitment_type=msg.commitment_type, turns=turns,
            target_territory=msg.target_territory,
            supported_from=msg.supported_from, dmz_territories=dmz,
            private=msg.private, repay_turn=msg.repay_turn,
            # The terms are unchanged, so the duty stays where it was.
            obligated=msg.obligated or msg.sender,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _record_price(self, state: GameState, deal: Commitment, gain: float):
        """Quote a deal. Pricing is not signing, so this leaves the signed
        price alone."""
        self.quotes[commitment_key(deal)] = (state.turn, gain)

    def signed(self, deal: Commitment, turn: int):
        """A deal was signed or renewed: this turn's quote becomes its price.
        Without one, the price is dropped and the planner re-prices it."""
        key = commitment_key(deal)
        quote = self.quotes.get(key)
        if quote is not None and quote[0] == turn:
            self.deal_prices[key] = quote[1]
            self.deal_signed_turn[key] = turn
        else:
            self.deal_prices.pop(key, None)
            self.deal_signed_turn.pop(key, None)

    def _answer(self, msg: Message, kind: MessageType, why=None) -> Message:
        return Message(
            id=str(uuid.uuid4()),
            sender=self.player,
            receiver=msg.sender,
            message_type=kind,
            reference_id=msg.id,
            rationale=why,
        )

    def _accept(self, msg: Message, why=None) -> Message:
        return self._answer(msg, MessageType.ACCEPT, why)

    def _reject(self, msg: Message, why=None) -> Message:
        return self._answer(msg, MessageType.REJECT, why)
