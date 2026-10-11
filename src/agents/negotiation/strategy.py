"""Negotiation: the closed grammar, priced by the planner.

Accept when P(keep) * V_kept + (1 - P(keep)) * V_broken > V_none.
"""

from itertools import combinations
from math import prod
from typing import List, Dict, Optional, Tuple
import uuid

from src.common.schemas import (
    GameState, Message, MessageType, Player, CommitmentType,
    Commitment, Order, OrderType, message_to_commitment,
    invalid_proposal_reason, commitment_key,
)
from src.agents.planner.planner import (
    Planner, PlannerConfig, evaluate_state, cooperation_value, hostile_world,
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
# Most-trusted allies considered for a pact; every pair of them is priced.
PACT_SHORTLIST = 4


class NegotiationStrategy:
    def __init__(self, player: Player, planner: Planner):
        self.player = player
        self.planner = planner
        # Threats I have sent, so I know which ones I am on the hook for.
        self.threats_made: Dict[Player, int] = {}
        # The deals already on the table this turn, refreshed by the Agent.
        self._live_commitments: List[Commitment] = []
        # Pruned candidates, memoised per (turn, live deals, position).
        self._candidate_memo: Tuple = (None, None)
        # Set by the Agent from the persona: how readily this player keeps a
        # deal off the public record.
        self.privacy_preference: float = 0.0
        # What each deal was worth when I signed it, keyed on the deal's identity.
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
        # Partners I expect to refuse me (Agent.expects_refusal): no offers.
        self.rebuffed: set = set()
        # My promise-keeping as each partner sees it (Agent.standing_with),
        # and a negotiator per partner seat for pricing a deal from their side.
        self.standing: Dict[Player, float] = {}
        self._mirrors: Dict[Player, "NegotiationStrategy"] = {}
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
        """The one plan I would play across all the weighted worlds, and its score."""
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
        """Everyone whose behaviour the deal changes: all other members of a pact."""
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

        # In an exchange the partner honours by repaying (staying out), not by supporting.
        owed = set(deal.dmz_territories or []) if (
            deal is not None and deal.commitment_type == CommitmentType.EXCHANGE) else set()

        for u in partner_units:
            partner = u.player
            if owed and partner == (deal.players[1] if deal else partner):
                # Honouring: anywhere but the owed squares. Breaking: into one of them.
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
        """(V_none, V_kept, V_broken) for one candidate deal; the last two are
        restricted to order sets that honour it."""
        free = self._candidates(state, commitments)
        with_deal = commitments + [deal]
        honouring = [c for c in free
                     if not breaks_commitment(state, self.player, c, with_deal)]
        if not honouring:
            honouring = free

        kept_world, broken_world = self._worlds(state, partner, deal)
        worlds = [kept_world, broken_world]

        # No deal: one unconstrained plan hedged against both pictures of them.
        _plan, v_none = self._best_plan(state, free, worlds, [0.5, 0.5], commitments)

        # With the deal: an honouring plan for a partner who keeps; V_broken is
        # that same plan against a partner who walks away.
        plan, _ = self._best_plan(state, honouring, worlds, [1.0, 0.0], with_deal)
        v_kept = self._value_of(state, plan, kept_world, with_deal)
        v_broken = self._value_of(state, plan, broken_world, with_deal)
        return v_none, v_kept, v_broken

    def deal_totals(self, state: GameState, deal: Commitment, partner: Player,
                    commitments: List[Commitment]) -> Tuple[float, float, float]:
        """(V_none, V_kept, V_broken) over the deal's life: per-turn values times its
        turns, plus Vcoop x turns / max turns on the kept branch."""
        v_none, v_kept, v_broken = self.deal_values(state, deal, partner, commitments)
        # valid_until_turn is the last turn the deal is graded, inclusive.
        turns = max(1, deal.valid_until_turn - state.turn + 1)
        coop = sum(cooperation_value(state, self.player, q)
                   for q in self._partners_of(deal, partner)) * turns / max_turns()
        return turns * v_none, turns * v_kept + coop, turns * v_broken

    def their_surplus(self, state: GameState, deal: Commitment, partner: Player,
                      commitments: List[Commitment]) -> float:
        """What the deal is worth to the partner over no deal, priced from
        their seat with my standing in their eyes as P(I keep). Positive is
        what their acceptance test (evaluate_proposal) asks for."""
        mirror = self._mirrors.get(partner)
        if mirror is None:
            mirror = self._mirrors[partner] = NegotiationStrategy(
                partner, Planner(partner, PlannerConfig()))
        mirror._live_commitments = commitments
        v_none, v_kept, v_broken = mirror.deal_totals(state, deal, self.player, commitments)
        s = self.standing.get(partner, 0.5)
        return s * v_kept + (1 - s) * v_broken - v_none

    def exposure(self, state: GameState, partner: Player) -> float:
        """How much of mine sits inside their reach, normalised to [0, 1]."""
        return min(1.0, cooperation_value(state, partner, self.player) / (2 * CENTRE_VALUE))

    def their_incentive(self, state: GameState, partner: Player,
                        deal: Optional[Commitment] = None) -> float:
        """What breaking is worth to them this turn: the trust network's incentive node."""
        live = list(self._live_commitments) + ([deal] if deal is not None else [])
        return defection_incentive(state, partner, live)

    def overcommits(self, state: GameState, deal: Commitment) -> bool:
        """For a bound player: a second two-way alliance, or a promise to stay out
        of a reachable centre it does not own."""
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

        # DMZ on centres we both border and neither of us owns.
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

        # Support a hold on a centre of theirs next to my units.
        for ou in their_units:
            if any(ou.territory in get_adjacent(mu.territory) for mu in my_units) \
                    and is_supply_center(ou.territory):
                offers.append(msg(commitment_type=CommitmentType.SUPPORT, turns=1,
                                  target_territory=ou.territory))
                break

        # An exchange: I support their move now; next turn they keep out of somewhere I want.
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

        # Deals aimed at a third party go private; otherwise privacy follows the persona.
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
        """A three-way pact against a runaway leader, formed only if all sign."""
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
             and counts[p] < counts[leader] and p not in self.refuse | self.rebuffed),
            key=lambda p: -trust_model.get_reliability(
                p, CommitmentType.ALLIANCE, toward=self.player))[:PACT_SHORTLIST]

        # The pair whose pact has the largest Nash product of all three surpluses.
        commitments = self._live_commitments
        pact_id = str(uuid.uuid4())
        best = None
        for pair in combinations(allies, 2):
            members = [self.player, *pair]
            probe = Message(id=pact_id, sender=self.player, receiver=pair[0],
                            message_type=MessageType.PROPOSE,
                            commitment_type=CommitmentType.ALLIANCE, turns=3,
                            coalition=members)
            deal = message_to_commitment(probe, pair[0], state.turn)
            v_none, v_kept, _v_broken = self.deal_totals(state, deal, pair[0], commitments)
            gain = v_kept - v_none
            if gain <= MIN_DEAL_VALUE:
                continue
            theirs = [self.their_surplus(state, deal, m, commitments) for m in pair]
            if min(theirs) <= 0:
                continue
            nash = gain * prod(theirs)
            if best is None or nash > best[0]:
                best = (nash, gain, min(theirs), members)
        if best is None:
            return []

        nash, gain, theirs, members = best
        # One id shared by every copy, so the runner can collect all signatures.
        return [
            Message(id=pact_id, sender=self.player, receiver=m,
                    message_type=MessageType.PROPOSE,
                    commitment_type=CommitmentType.ALLIANCE, turns=3,
                    coalition=members,
                    rationale={"gain": gain, "their_gain": theirs, "nash": nash})
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

            # Nash bargaining: rank deals both sides gain from by the product of
            # the surpluses; skip any the partner is predicted to refuse.
            ranked: List[Tuple[float, float, float, Message]] = []
            barred = near_win(state, p) or p in self.refuse or p in self.rebuffed
            for offer in ([] if barred else self._candidate_deals(state, p)):
                deal = message_to_commitment(offer, p, state.turn)
                if self.overcommits(state, deal):
                    continue
                v_none, v_kept, _v_broken = self.deal_totals(state, deal, p, commitments)
                gain = v_kept - v_none
                self._record_price(state, deal, gain)
                if gain <= MIN_DEAL_VALUE:
                    continue
                theirs = self.their_surplus(state, deal, p, commitments)
                if theirs > 0:
                    ranked.append((gain * theirs, gain, theirs, offer))

            ranked.sort(key=lambda r: r[0], reverse=True)
            for nash, gain, theirs, offer in ranked[:MAX_PROPOSALS_PER_OPPONENT]:
                offer.rationale = {"gain": gain, "their_gain": theirs, "nash": nash,
                                   "considered": len(ranked), "floor": MIN_DEAL_VALUE}
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
            # Not a deal; the Agent records it as deterrence.
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

        # Not at these terms; offer the smaller version with the largest Nash product.
        best = None
        if msg.message_type != MessageType.COUNTER:
            for counter in self._counter_offers(msg):
                c_deal = message_to_commitment(counter, self.player, state.turn)
                c_none, c_kept, c_broken = self.deal_totals(state, c_deal, msg.sender, commitments)
                mine = p_keep * c_kept + (1 - p_keep) * c_broken - c_none
                if mine <= 0:
                    continue
                theirs = self.their_surplus(state, c_deal, msg.sender, commitments)
                if theirs > 0 and (best is None or mine * theirs > best[0]):
                    best = (mine * theirs, counter, c_none + mine, c_none, theirs)
        if best is not None:
            nash, counter, c_ev, c_none, theirs = best
            # Not recorded in deal_prices: its key differs from the accepted deal's.
            counter.rationale = {**why, "ev_countered": c_ev, "v_none_countered": c_none,
                                 "their_gain": theirs, "nash": nash}
            return counter

        return self._reject(msg, why)

    def _counter_offers(self, msg: Message) -> List[Message]:
        """The same deal, smaller: each shorter term, or one DMZ square dropped. Pacts get none."""
        if msg.coalition:
            return []
        base, dmz = msg.turns or 1, msg.dmz_territories
        shapes = [(t, dmz) for t in range(1, base)]
        if msg.commitment_type == CommitmentType.DMZ and dmz and len(dmz) > 1:
            shapes += [(base, [d for d in dmz if d != drop]) for drop in dmz]
        return [Message(
            id=str(uuid.uuid4()), sender=self.player, receiver=msg.sender,
            message_type=MessageType.COUNTER, reference_id=msg.id,
            commitment_type=msg.commitment_type, turns=turns,
            target_territory=msg.target_territory,
            supported_from=msg.supported_from, dmz_territories=squares,
            private=msg.private, repay_turn=msg.repay_turn,
            # The terms are unchanged, so the duty stays where it was.
            obligated=msg.obligated or msg.sender,
        ) for turns, squares in shapes]

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
            private=msg.private,
            rationale=why,
        )

    def _accept(self, msg: Message, why=None) -> Message:
        return self._answer(msg, MessageType.ACCEPT, why)

    def _reject(self, msg: Message, why=None) -> Message:
        return self._answer(msg, MessageType.REJECT, why)
