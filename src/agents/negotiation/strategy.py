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
    invalid_proposal_reason,
)
from src.agents.planner.planner import (
    Planner, evaluate_state, cooperation_value, hostile_world,
    breaks_commitment, defection_incentive, CENTRE_VALUE,
)
from src.agents.trust.model import TrustModel
from src.engine.adjudicator import resolve
from src.engine.orders import generate_all_order_sets
from src.engine.board import get_adjacent, is_supply_center, MAX_TURNS

# Deals whose counterfactual gain is smaller than this are noise, not offers.
MIN_DEAL_VALUE = 0.05
MAX_PROPOSALS_PER_OPPONENT = 2
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
        new_state, _o, _l = resolve(state, plan + world, commitments)
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

    def _worlds(self, state: GameState, partner: Player, deal: Optional[Commitment]
                ) -> Tuple[List[Order], List[Order]]:
        """Two pictures of the partner: honouring the deal, and exploiting it.

        Everyone else is assumed hostile in both, so the comparison isolates
        the partner's choice rather than the weather.
        """
        others = [o for o in hostile_world(state, self.player) if o.player != partner]
        partner_units = [u for u in state.units if u.player == partner]

        kept: List[Order] = list(others)
        broken: List[Order] = list(others)
        mine = {u.territory for u in state.units if u.player == self.player}
        mine |= {t for t, owner in state.supply_centers.items() if owner == self.player}

        for u in partner_units:
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
        turns = max(1, deal.valid_until_turn - state.turn)
        coop = cooperation_value(state, self.player, partner) * turns / MAX_TURNS
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

        return offers

    def generate_proposals(self, state: GameState, trust_model: TrustModel) -> List[Message]:
        proposals: List[Message] = []
        commitments = self._live_commitments

        for p in Player:
            if p == self.player:
                continue
            their_units = [u for u in state.units if u.player == p]
            if not their_units:
                continue

            ranked: List[Tuple[float, Message]] = []
            for offer in self._candidate_deals(state, p):
                deal = message_to_commitment(offer, p, state.turn)
                v_none, v_kept, _v_broken = self.deal_totals(state, deal, p, commitments)
                # Ranked by the counterfactual the deck asks for: value if
                # honoured minus value without the deal.
                ranked.append((v_kept - v_none, offer))

            ranked.sort(key=lambda r: r[0], reverse=True)
            for gain, offer in ranked[:MAX_PROPOSALS_PER_OPPONENT]:
                if gain > MIN_DEAL_VALUE:
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

        commitments = self._live_commitments
        deal = message_to_commitment(msg, self.player, state.turn)
        v_none, v_kept, v_broken = self.deal_totals(state, deal, msg.sender, commitments)
        p_keep = trust_model.p_keeps(msg.sender, msg.commitment_type,
                                     self.their_incentive(state, msg.sender, deal))
        ev = p_keep * v_kept + (1 - p_keep) * v_broken

        if ev > v_none:
            return self._accept(msg)

        # Not at these terms — but a shorter deal exposes me for fewer turns,
        # so try the same sentence with a smaller number in it before walking
        # away. This is what makes negotiation rounds 2 and 3 do any work.
        counter = self._counter_terms(msg)
        if counter is not None and msg.message_type != MessageType.COUNTER:
            c_deal = message_to_commitment(counter, self.player, state.turn)
            c_none, c_kept, c_broken = self.deal_totals(state, c_deal, msg.sender, commitments)
            if p_keep * c_kept + (1 - p_keep) * c_broken > c_none:
                return counter

        return self._reject(msg)

    def _counter_terms(self, msg: Message) -> Optional[Message]:
        """The same deal, smaller. Halve the term; for a DMZ, give back the
        territories I most want to keep my hands free on."""
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
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _accept(self, msg: Message) -> Message:
        return Message(
            id=str(uuid.uuid4()),
            sender=self.player,
            receiver=msg.sender,
            message_type=MessageType.ACCEPT,
            reference_id=msg.id,
        )

    def _reject(self, msg: Message) -> Message:
        return Message(
            id=str(uuid.uuid4()),
            sender=self.player,
            receiver=msg.sender,
            message_type=MessageType.REJECT,
            reference_id=msg.id,
        )
