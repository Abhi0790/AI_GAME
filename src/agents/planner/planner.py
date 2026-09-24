from typing import List, Dict, Tuple, Optional
from src.common.schemas import (
    GameState, Order, OrderType, Player, Commitment, DecisionTrace,
    commitment_key,
)
from src.engine.adjudicator import resolve, verify_commitments
from src.engine.orders import generate_all_order_sets
from src.engine.board import (
    is_supply_center, get_adjacent, max_turns, players, win_centers, home_centers,
)
from src.agents.trust.model import TrustModel, break_weight
import math
import random


class PlannerConfig:
    """Search budget and the one persona knob the planner reads.

    Depth-2 expectiminimax over a pruned candidate set, with every call to the
    adjudicator counted. The budget is in adjudications, not seconds, so the
    MCTS variant can be given exactly the same one.
    """

    def __init__(self, reputation_cost_coefficient: float = 1.0, depth: int = 2,
                 opponent_samples: int = 32, beam_width: int = 8,
                 depth2_candidates: int = 4, depth2_samples: int = 6,
                 depth2_replies: int = 6, node_budget: int = 1500,
                 vengeance: float = 0.0, search: str = "expectiminimax",
                 retaliation_samples: int = 1):
        self.reputation_cost_coefficient = reputation_cost_coefficient
        self.depth = depth                    # 1 = one ply, 2 = my move, their reply
        self.opponent_samples = opponent_samples
        self.beam_width = beam_width          # candidates surviving dominance pruning
        self.depth2_candidates = depth2_candidates
        self.depth2_samples = depth2_samples
        self.depth2_replies = depth2_replies
        self.node_budget = node_budget        # adjudications per decision
        self.vengeance = vengeance
        self.search = search                  # "expectiminimax" | "mcts"
        # Extra depth-2 worlds in which the players I just betrayed come for
        # me. 0 restores the old search, where defecting had no consequence
        # inside the tree at all.
        self.retaliation_samples = retaliation_samples


# Everything below is denominated in centres: one supply centre = 1.0, so the
# numbers in a decision trace read the same way as the worked example.
CENTRE_VALUE = 1.0
UNIT_VALUE = 0.3
THREATENS_VALUE = 0.2
UNDER_THREAT_VALUE = -0.2
MOBILITY_VALUE = 0.01
# Reaching win_centers ends the game: worth this much to the winner and costs
# each other player the same.
GAME_END_VALUE = 2.0
# A player this many centres or fewer from winning is treated as about to win:
# no deals with them, and breaking one to stop them is not charged as a betrayal.
NEAR_WIN_MARGIN = 1

# How much of the price a deal was signed at is forfeited by walking away from
# it. 1.0 means signing and breaking are priced with the same number, so the
# negotiator and the planner cannot disagree about an unchanged board; 0.0
# restores the old behaviour, where breaking was charged only the reputation
# term and was therefore ~20x cheaper than signing had been worth.
FORFEIT_WEIGHT = 1.0

# P(a betrayed player hits back) before anything has been seen of them. One in
# three is the weight the retaliation world carried when it was one sample of three.
RETALIATION_PRIOR = 1 / 3

# Charged, in centres, per partner in Planner.loyal_to that an order set breaks
# with first, so a loyal persona breaks first only for more than this. It needs
# selective deals (NegotiationStrategy.overcommits): while bound players signed
# with nearly everyone, any duty here handed the one persona without it 83-95%
# of wins.
LOYALTY_COST = 1.0


def evaluate_state(state: GameState, player: Player) -> float:
    """Centres held, units, centres under threat, centres threatened, mobility,
    and whether anyone has reached win_centers."""
    my_units = [u for u in state.units if u.player == player]
    my_squares = {u.territory for u in my_units}

    # 1. Centres held. Ownership persists between autumns, so read it from
    #    supply_centers, not from who happens to be standing there.
    score = CENTRE_VALUE * sum(
        1 for owner in state.supply_centers.values() if owner == player
    )

    # 2. Centres we threaten: not ours, and one move away.
    threatening = {
        adj for u in my_units for adj in get_adjacent(u.territory)
        if is_supply_center(adj) and state.supply_centers.get(adj) != player
    }
    score += len(threatening) * THREATENS_VALUE

    # 3. Our centres under threat: an enemy unit is one move away.
    enemy_reach = {
        adj for u in state.units if u.player != player
        for adj in get_adjacent(u.territory)
    }
    under_threat = {
        t for t, owner in state.supply_centers.items()
        if owner == player and t in enemy_reach
    }
    score += len(under_threat) * UNDER_THREAT_VALUE

    # 4. Units and mobility.
    score += len(my_units) * UNIT_VALUE
    score += MOBILITY_VALUE * sum(
        1 for u in my_units for adj in get_adjacent(u.territory)
        if adj not in my_squares
    )

    # 5. The game ends at win_centers.
    held = {}
    for owner in state.supply_centers.values():
        if owner:
            held[owner] = held.get(owner, 0) + 1
    win = win_centers()
    if held.get(player, 0) >= win:
        score += GAME_END_VALUE
    score -= GAME_END_VALUE * sum(1 for p, n in held.items() if p != player and n >= win)

    return score


def near_win(state: GameState, player: Player) -> bool:
    """Is this player within NEAR_WIN_MARGIN centres of winning, having grown to get
    there? Without the second test, a board whose threshold sits one above the
    starting centres (every 2-seat board) marks everyone from turn 1."""
    held = sum(1 for owner in state.supply_centers.values() if owner == player)
    return held >= win_centers() - NEAR_WIN_MARGIN and held > len(home_centers().get(player, []))


def cooperation_value(state: GameState, me: Player, partner: Player) -> float:
    """Vcoop: what the partner's cooperation is worth to me over the rest of
    the game, in the same units as evaluate_state.

    Two halves, both read straight off the board:
      - defensive: my centres their units are sitting next to. If they turn on
        me those are what I lose. A centre I already garrison myself counts for
        much less -- that is how Vcoop collapses once I no longer need them.
      - offensive: centres neither of us owns that we can both reach, i.e. the
        ones their support could win me.
    """
    partner_units = [u for u in state.units if u.player == partner]
    my_units = [u for u in state.units if u.player == me]
    if not partner_units or not my_units:
        return 0.0

    partner_reach = {t for u in partner_units for t in get_adjacent(u.territory)}
    my_reach = {t for u in my_units for t in get_adjacent(u.territory)}
    garrisoned = {u.territory for u in my_units}

    value = 0.0
    for t, owner in state.supply_centers.items():
        if owner == me and t in partner_reach:
            value += CENTRE_VALUE * (0.3 if t in garrisoned else 1.0)
        elif owner != me and t in my_reach and t in partner_reach:
            value += CENTRE_VALUE * 0.5
    return value


def sample_opponent_orders(
    state: GameState,
    my_player: Player,
    trust_model: Optional[TrustModel],
    samples: int = 3,
    opponent_model=None,
    commitments: Optional[List[Commitment]] = None,
    incentives: Optional[Dict[Player, float]] = None,
) -> List[List[Order]]:
    """Sample likely opponent orders.

    If an *opponent_model* is provided, delegate to its weighted sampler, which
    conditions the draw on what these players have promised when a trust model
    is supplied. Otherwise fall back to uniform-random sampling.
    """
    if opponent_model is not None:
        return opponent_model.sample_opponent_orders(
            state, samples, commitments=commitments, trust_model=trust_model,
            incentives=incentives)

    # ── Legacy uniform-random fallback ──────────────────────────────
    opponents = [p for p in players() if p != my_player]
    per_opponent = {opp: generate_all_order_sets(state, opp) for opp in opponents}

    all_samples = []
    for _ in range(samples):
        joint_orders = []
        for opp in opponents:
            sets = per_opponent[opp]
            if sets:
                joint_orders.extend(random.choice(sets))
        all_samples.append(joint_orders)

    return all_samples


def penalty_breakdown(state: GameState, player: Player, my_orders: List[Order],
                      commitments: List[Commitment], config: PlannerConfig,
                      trust_model: Optional[TrustModel] = None, incentive: float = 0.0,
                      outcomes=None, weights: Optional[Dict[Player, float]] = None):
    """Price every commitment this order set breaks.

        penalty = Vcoop(partner) x deltaP(partner keeps) x (turns left / 12)

    There is no "betray" flag anywhere: near the horizon the last term shrinks,
    and mid-game Vcoop collapses once the partner stops being useful. Either way
    the same arithmetic stops paying for loyalty.

    *weights* scales the charge per partner, from the persona's policy
    (Agent.policy); a partner missing from it is charged in full.

    Returns one (partner, vcoop, delta_p, horizon, amount) row per break.
    """
    rows = []
    weights = weights or {}
    horizon = max(0, max_turns() - state.turn) / max_turns()

    if outcomes is None:
        outcomes = verify_commitments(state, commitments, my_orders)
    for o in outcomes:
        if o.kept or player not in o.broken_by:
            continue

        c_type = o.commitment.commitment_type
        if trust_model is not None:
            # What breaking costs *my* standing, scored off the public record.
            delta_p = trust_model.reputation_drop(player, c_type, incentive)
        else:
            # No model supplied: fall back to an uninformative Beta(1, 1).
            w = break_weight(incentive)
            delta_p = 0.5 - 1.0 / (2.0 + w)

        for partner in o.commitment.players:
            # Breaking a deal to stop a player about to win is defence, not betrayal.
            if partner == player or near_win(state, partner):
                continue
            v_coop = cooperation_value(state, player, partner)
            amount = (v_coop * delta_p * horizon * config.reputation_cost_coefficient
                      * weights.get(partner, 1.0))
            rows.append((partner, v_coop, delta_p, horizon, amount))

    return rows


def forfeit_breakdown(state: GameState, player: Player, my_orders: List[Order],
                      commitments: List[Commitment],
                      forfeits: Dict[tuple, Tuple[float, float, float]],
                      outcomes=None):
    """What this order set gives back of the prices it paid for its deals.

    The reputation term above prices the *belief* a break moves. It does not
    price the thing the negotiator actually bought, and on this board deltaP
    is around 0.04, so breaking came out ~20x cheaper than signing had been
    worth. Charging the unused share of the signed price as well is what makes
    the two halves of the agent quote one number for one promise.

    *forfeits* is `Planner._forfeits`, computed once per decision because the
    figure depends on the deal and the board, not on which order set broke it.

    Returns one (commitment_key, partner, signed_price, fraction_remaining,
    amount) row per broken deal.
    """
    rows = []
    if not forfeits:
        return rows
    if outcomes is None:
        outcomes = verify_commitments(state, commitments, my_orders)
    for o in outcomes:
        if o.kept or player not in o.broken_by:
            continue
        f = forfeits.get(commitment_key(o.commitment))
        if f is None or any(p != player and near_win(state, p) for p in o.commitment.players):
            continue
        partner = next((p for p in o.commitment.players if p != player), player)
        rows.append((commitment_key(o.commitment), partner) + f)
    return rows


def calculate_commitment_penalty(state: GameState, player: Player, my_orders: List[Order],
                                 commitments: List[Commitment], config: PlannerConfig,
                                 trust_model: Optional[TrustModel] = None,
                                 incentive: float = 0.0) -> float:
    """Total reputation cost of the commitments this order set breaks."""
    return sum(row[-1] for row in penalty_breakdown(
        state, player, my_orders, commitments, config, trust_model, incentive))


def breaks_commitment(state: GameState, player: Player, my_orders: List[Order],
                      commitments: List[Commitment]) -> bool:
    return any(
        not o.kept and player in o.broken_by
        for o in verify_commitments(state, commitments, my_orders)
    )


def hostile_world(state: GameState, me: Player) -> List[Order]:
    """A reference world: every opponent unit that can step onto something of
    mine does. Used only for pruning, never for scoring a decision."""
    mine = {u.territory for u in state.units if u.player == me}
    mine |= {t for t, owner in state.supply_centers.items() if owner == me}
    orders = []
    for u in state.units:
        if u.player == me:
            continue
        target = next((a for a in get_adjacent(u.territory) if a in mine), None)
        orders.append(Order(
            player=u.player, unit_territory=u.territory,
            order_type=OrderType.MOVE if target else OrderType.HOLD, target=target))
    return orders


def defection_incentive(state: GameState, subject: Player,
                        commitments: List[Commitment], samples: int = 60,
                        resolver=None) -> float:
    """iota — what the board is offering *this player* to break, right now.

    The best they can do while breaking a live promise, minus the best they
    can do while keeping every one, both against a quiet reference world. It
    is the same quantity the planner computes for itself when it decides
    whether a betrayal pays, computed from the outside: every order is
    public, so an observer can run it.

    The previous stand-in was "how much of my stuff is within their reach",
    which turned out to carry no information about whether they actually
    broke anything (kept 0.544 vs broke 0.549 over 932 graded promises).
    """
    their_deals = [c for c in commitments if subject in c.players]
    if not their_deals:
        return 0.0

    sets = generate_all_order_sets(state, subject)
    if len(sets) > samples:
        sets = random.sample(sets, samples)

    best_keep = best_break = None
    for candidate in sets:
        breaks = breaks_commitment(state, subject, candidate, their_deals)
        value = evaluate_state(resolver(state, candidate) if resolver
                               else resolve(state, candidate)[0], subject)
        if breaks:
            best_break = value if best_break is None else max(best_break, value)
        else:
            best_keep = value if best_keep is None else max(best_keep, value)

    if best_break is None or best_keep is None:
        return 0.0
    return min(1.0, max(0.0, (best_break - best_keep) / CENTRE_VALUE))


class Planner:
    def __init__(self, player: Player, config: PlannerConfig):
        self.player = player
        self.config = config
        self.opponent_model = None  # set by Agent after construction
        # The negotiator that signed this seat's deals, set by Agent. Read for
        # `deal_prices` — what each promise was credited with at signature —
        # so breaking one can be charged the same number. No import: the
        # dependency runs the other way round.
        self.negotiation = None
        self.nodes = 0              # adjudications spent on the last decision
        self._incentives: Dict[Player, float] = {}
        # Set by Agent.act from the persona's policy, each decision:
        self.loyal_to: set = set()                    # partners owed loyalty (LOYALTY_COST)
        self.partner_weight: Dict[Player, float] = {} # reputation-cost scale per partner
        self.retaliation: Dict[Player, float] = {}    # P(they hit back once betrayed)

    def _opponent_incentives(self, state: GameState, commitments: List[Commitment]
                             ) -> Dict[Player, float]:
        """iota for every opponent with a live deal, once per decision.

        ponytail: reused for the hypothetical states deeper in the search;
        recompute per state if depth-2 predictions need it.
        """
        # Counted against this decision's budget: it is work done for the decision.
        count = lambda s, orders: self._resolve(s, orders, [])
        return {p: defection_incentive(state, p, commitments, resolver=count)
                for p in sorted({q for c in commitments for q in c.players},
                                key=lambda q: q.value)
                if p != self.player}

    def set_opponent_model(self, model):
        """Attach an OpponentModel for informed sampling."""
        self.opponent_model = model

    # ── forfeiture ──────────────────────────────────────────────────────
    def _forfeits(self, state: GameState, commitments: List[Commitment]
                  ) -> Dict[tuple, Tuple[float, float, float]]:
        """{commitment_key: (signed_price, fraction_remaining, amount)} for my
        live deals.

            forfeit = FORFEIT_WEIGHT x signed_price x turns_left / deal_length

        A deal I priced at signature and am a third of the way through is
        worth two thirds of that price to keep; walking away now gives that
        back. Negative prices clamp to zero — a deal cannot pay me to break
        it. A live deal I never priced (inherited, or signed as a counter) is
        re-priced on today's board, where every remaining turn is still ahead.
        """
        out: Dict[tuple, Tuple[float, float, float]] = {}
        if not FORFEIT_WEIGHT or self.negotiation is None:
            return out
        for c in commitments:
            if self.player not in c.players:
                continue
            key = commitment_key(c)
            price = self.negotiation.deal_prices.get(key)
            signed = self.negotiation.deal_signed_turn.get(key, state.turn)
            if price is None:
                partner = next((p for p in c.players if p != self.player), None)
                if partner is None:
                    continue
                others = [o for o in commitments if o.id != c.id]
                v_none, v_kept, _v_broken = self.negotiation.deal_totals(
                    state, c, partner, others)
                price, signed = v_kept - v_none, state.turn
            length = max(1, c.valid_until_turn - signed + 1)
            remaining = max(0, min(length, c.valid_until_turn - state.turn + 1))
            frac = remaining / length
            out[key] = (price, frac, max(0.0, FORFEIT_WEIGHT * price * frac))
        return out

    def _loyalty(self, graded) -> float:
        """LOYALTY_COST per partner in `loyal_to` this order set breaks with."""
        return LOYALTY_COST * len({
            p for o in graded if not o.kept and self.player in o.broken_by
            for p in o.commitment.players if p in self.loyal_to})

    # ── node accounting ─────────────────────────────────────────────────
    def _resolve(self, state, orders, commitments):
        """Every call to the adjudicator, counted. The search-variant figure
        puts this number on the x-axis, so it has to be the real one. Commitments
        do not move units, so they are not graded here."""
        self.nodes += 1
        return resolve(state, orders)[0]

    # ── dominance pruning ───────────────────────────────────────────────
    def prune(self, state: GameState, candidates: List[List[Order]],
              commitments: List[Commitment]) -> Tuple[List[List[Order]], int]:
        """Drop order sets no rational player would pick, then beam.

        Each candidate is scored in two reference worlds — opponents all hold,
        and opponents all attack me. A candidate is *dominated* when another
        candidate does at least as well in both and better in one; a dominated
        set cannot be the best reply to any mixture of the two. Loyal and
        treacherous sets are pruned in separate pools, so an option is never
        thrown away merely for keeping a promise.

        Replaces `random.sample(49)`, which could discard the best move.
        """
        quiet: List[Order] = []
        hostile = hostile_world(state, self.player)

        pools: Dict[bool, List[Tuple[List[Order], float, float]]] = {True: [], False: []}
        for cand in candidates:
            a = evaluate_state(self._resolve(state, cand + quiet, commitments), self.player)
            b = evaluate_state(self._resolve(state, cand + hostile, commitments), self.player)
            pools[breaks_commitment(state, self.player, cand, commitments)].append((cand, a, b))

        beams = []
        for pool in pools.values():
            front = [
                (c, a, b) for (c, a, b) in pool
                if not any(a2 >= a and b2 >= b and (a2 > a or b2 > b)
                           for (_c2, a2, b2) in pool)
            ]
            front.sort(key=lambda r: r[1] + r[2], reverse=True)
            beams.extend(front[:self.config.beam_width])
        # Merit order across both pools, so a truncated search is not biased to one.
        beams.sort(key=lambda r: r[1] + r[2], reverse=True)
        kept: List[List[Order]] = [c for c, _a, _b in beams]

        if not kept:
            kept = candidates[:self.config.beam_width]
        return kept, len(candidates) - len(kept)

    # ── vengeance ───────────────────────────────────────────────────────
    def _stance_bonus(self, before: GameState, after: GameState,
                      stance: Dict[Player, float]) -> float:
        """Value this agent attaches to hurting (or sparing) a given player,
        on top of what the board is worth to itself.

        One dict carries both halves of the social position:
          positive weight — a grudge. A Vengeful persona will take a move that
            costs it a centre if it costs the betrayer more, which is what
            "retaliates beyond value" has to mean in code.
          negative weight — a threat it believes. Attacking that player is
            priced up by the retaliation they promised, so a THREAT message
            changes behaviour instead of being dropped on the floor.
        """
        if not stance:
            return 0.0
        return sum(
            weight * (evaluate_state(before, p) - evaluate_state(after, p))
            for p, weight in stance.items() if weight
        )

    # ── depth-2 ─────────────────────────────────────────────────────────
    def _depth2_value(self, state: GameState, my_set: List[Order],
                      worlds: List[List[Order]], commitments: List[Commitment],
                      trust_model: Optional[TrustModel] = None) -> float:
        """Expectiminimax, two plies: chance over their orders, max over my
        reply, chance over their reply.

        Depth-1 cannot see a move that wins a centre this turn and hands back
        two the next; PlannerConfig.depth existed but was never read.
        """
        cfg = self.config

        # What this line of play does to my promises, and to whom. Everything
        # below is the consequence the search could not previously see: a deal
        # I break stops restraining the other side, and the people I broke it
        # with have a reason to come for me next turn.
        broken = [o.commitment for o in verify_commitments(state, commitments, my_set)
                  if not o.kept and self.player in o.broken_by]
        betrayed = {p for c in broken for p in c.players if p != self.player}
        broken_ids = {c.id for c in broken}
        # Released, exactly as GameRunner.step releases the members of a pact
        # somebody walked out of. Conditioning their reply on a promise I have
        # already broken would be modelling them as still bound by it.
        surviving = [c for c in commitments if c.id not in broken_ids]

        totals = []
        for world in worlds[:cfg.depth2_samples]:
            if self.nodes >= cfg.node_budget:
                break
            s1 = self._resolve(state, my_set + world, commitments)
            replies = generate_all_order_sets(s1, self.player)
            if len(replies) > cfg.depth2_replies:
                replies = random.sample(replies, cfg.depth2_replies)
            their_replies = sample_opponent_orders(
                s1, self.player, trust_model, samples=2,
                opponent_model=self.opponent_model, commitments=surviving,
                incentives=self._incentives)

            # The punishment channel. Betrayal was priced only by the scalar
            # Vcoop x deltaP x horizon; nothing in the tree showed the
            # betrayed player doing anything about it, so the search saw a
            # free centre and the penalty had to be large enough to outweigh
            # it on its own. One world per betrayed party in which they spend
            # their reply taking my things makes the cost a *position* rather
            # than a fine -- which is what the grudge machinery already does
            # once the turn is over (Agent.grudges, Planner._stance_bonus).
            # Weighted by how often the betrayed have actually hit back when
            # wronged (Agent.retaliation), so crossing a known avenger costs
            # more than crossing a player who lets it go.
            weights = [1.0] * len(their_replies)
            if betrayed and cfg.retaliation_samples:
                calm = [o for o in their_replies[0] if o.player not in betrayed]
                revenge = [o for o in hostile_world(s1, self.player)
                           if o.player in betrayed]
                r = max(self.retaliation.get(p, RETALIATION_PRIOR) for p in betrayed)
                n = len(their_replies)
                weights = [(1 - r) / n] * n + [r / cfg.retaliation_samples] * cfg.retaliation_samples
                their_replies = their_replies + [calm + revenge] * cfg.retaliation_samples

            best_reply = None
            for reply in replies:
                vals, ws = [], []
                for tr, w in zip(their_replies, weights):
                    if self.nodes >= cfg.node_budget:
                        break
                    s2 = self._resolve(s1, reply + tr, commitments)
                    vals.append(evaluate_state(s2, self.player))
                    ws.append(w)
                if vals and sum(ws) > 0:
                    mean = sum(v * w for v, w in zip(vals, ws)) / sum(ws)
                    best_reply = mean if best_reply is None else max(best_reply, mean)
            if best_reply is not None:
                totals.append(best_reply)

        if not totals:
            return None
        return sum(totals) / len(totals)

    # ── the decision ────────────────────────────────────────────────────
    def find_best_orders(self, state: GameState, commitments: List[Commitment],
                         trust_model: TrustModel,
                         stance: Optional[Dict[Player, float]] = None
                         ) -> Tuple[List[Order], DecisionTrace]:
        self.nodes = 0
        stance = stance or {}

        hold_set = [Order(player=self.player, unit_territory=u.territory, order_type=OrderType.HOLD)
                    for u in state.units if u.player == self.player]

        if self.config.search == "mcts":
            return self.mcts_best_orders(state, commitments, trust_model, stance)

        all_sets = generate_all_order_sets(state, self.player)
        my_order_sets, pruned = self.prune(state, all_sets, commitments)

        self._incentives = self._opponent_incentives(state, commitments)
        worlds = sample_opponent_orders(
            state, self.player, trust_model,
            samples=self.config.opponent_samples, opponent_model=self.opponent_model,
            commitments=commitments, incentives=self._incentives,
        )

        # Pass 1: expected value of every surviving candidate, depth 1.
        scored = []
        for my_set in my_order_sets:
            ev_total, veng_total, n = 0.0, 0.0, 0
            for opp_set in worlds:
                # Every candidate sees at least one world, as in MCTS, so a
                # budget spent on pruning does not fall through to HOLD.
                if n and self.nodes >= self.config.node_budget:
                    break
                new_state = self._resolve(state, my_set + opp_set, commitments)
                ev_total += evaluate_state(new_state, self.player)
                veng_total += self._stance_bonus(state, new_state, stance)
                n += 1
            if n == 0:
                continue
            breaks = breaks_commitment(state, self.player, my_set, commitments)
            scored.append([my_set, ev_total / n, veng_total / n, breaks])

        if not scored:
            return hold_set, DecisionTrace(hold_set, 0, 0, "Fallback to HOLD")

        # Pass 2 (depth 2): only the leaders are worth the adjudications.
        if self.config.depth >= 2:
            order = sorted(range(len(scored)), key=lambda i: scored[i][1] + scored[i][2],
                           reverse=True)[:self.config.depth2_candidates]
            for i in order:
                v = self._depth2_value(state, scored[i][0], worlds, commitments,
                                       trust_model)
                if v is not None:
                    # Vengeance stays the depth-1 estimate: it prices a grudge,
                    # not a line of play, and doubling the search for it is not
                    # worth the adjudications.
                    scored[i][1] = v

        # The incentive to defect is what a break buys over the best honest
        # option. That is the iota the trust model discounts the penalty by.
        loyal_evs = [ev for _s, ev, _v, breaks in scored if not breaks]
        best_loyal_ev = max(loyal_evs) if loyal_evs else 0.0

        forfeits = self._forfeits(state, commitments)

        best_orders, best_score, best_trace = [], float('-inf'), None
        nets: List[float] = []
        for my_set, avg_ev, veng, breaks in scored:
            incentive = 0.0
            if breaks:
                incentive = min(1.0, max(0.0, (avg_ev - best_loyal_ev) / CENTRE_VALUE))

            graded = verify_commitments(state, commitments, my_set)
            rows = penalty_breakdown(state, self.player, my_set, commitments,
                                     self.config, trust_model, incentive,
                                     outcomes=graded, weights=self.partner_weight)
            f_rows = forfeit_breakdown(state, self.player, my_set, commitments,
                                       forfeits, outcomes=graded)
            loyalty = self._loyalty(graded)
            penalty = sum(r[-1] for r in rows) + sum(r[-1] for r in f_rows) + loyalty
            net_score = avg_ev + veng - penalty
            nets.append(net_score)

            if net_score > best_score:
                best_score = net_score
                best_orders = my_set
                best_trace = DecisionTrace(
                    my_set, avg_ev, penalty,
                    self._explain(avg_ev, penalty, net_score, rows, incentive, veng,
                                  f_rows, loyalty),
                    bool(rows), penalty_rows=rows, nodes=self.nodes, vengeance=veng,
                    search=self.config.search, candidates=len(my_order_sets), pruned=pruned,
                    forfeit_rows=f_rows,
                )

        if not best_orders:
            return hold_set, DecisionTrace(hold_set, 0, 0, "Fallback to HOLD")

        best_trace.nodes = self.nodes
        best_trace.reversals = self._break_advantages(state, commitments, scored, nets)
        return best_orders, best_trace

    def _break_advantages(self, state: GameState, commitments: List[Commitment],
                          scored, nets: List[float]) -> List[tuple]:
        """Per live deal of mine: best net score breaking it, minus best net
        keeping it.

        Both sides are already-computed candidate scores, so this costs one
        `verify_commitments` per (candidate, deal) and no adjudications. A
        positive number is the planner saying it would rather break a promise
        the negotiator signed in the same turn.
        """
        rows = []
        for c in commitments:
            if self.player not in c.players:
                continue
            best_break = best_keep = None
            for (my_set, _ev, _veng, _b), net in zip(scored, nets):
                if breaks_commitment(state, self.player, my_set, [c]):
                    best_break = net if best_break is None else max(best_break, net)
                else:
                    best_keep = net if best_keep is None else max(best_keep, net)
            if best_break is not None and best_keep is not None:
                rows.append((commitment_key(c), c.commitment_type,
                             best_break - best_keep))
        return rows

    # ── determinised MCTS, same budget ──────────────────────────────────
    def mcts_best_orders(self, state: GameState, commitments: List[Commitment],
                         trust_model: TrustModel,
                         stance: Optional[Dict[Player, float]] = None
                         ) -> Tuple[List[Order], DecisionTrace]:
        """UCT over my order sets, one determinisation per playout.

        Each iteration fixes a sample of what the opponents do (the
        determinisation), plays my candidate against it, then rolls out one
        more ply at random. Selection is UCB1, so the budget concentrates on
        the candidates that keep looking good — the comparison the report
        wants is this against expectiminimax at *equal adjudications*, which
        is why both count through `self._resolve`.
        """
        stance = stance or {}
        cfg = self.config
        all_sets = generate_all_order_sets(state, self.player)
        candidates, pruned = self.prune(state, all_sets, commitments)

        counts = [0] * len(candidates)
        totals = [0.0] * len(candidates)
        self._incentives = self._opponent_incentives(state, commitments)
        worlds = sample_opponent_orders(
            state, self.player, trust_model, samples=max(8, cfg.opponent_samples // 2),
            opponent_model=self.opponent_model, commitments=commitments,
            incentives=self._incentives)

        i = 0
        # Pruning has already spent part of the budget. Every surviving
        # candidate still gets at least one playout, or an exhausted budget
        # would leave the whole arm table empty and fall through to HOLD.
        while self.nodes < cfg.node_budget or i < len(candidates):
            # UCB1: every arm once, then exploit with an exploration bonus.
            if i < len(candidates):
                arm = i
            else:
                total_n = sum(counts) or 1
                arm = max(range(len(candidates)), key=lambda k: (
                    totals[k] / counts[k] + 1.4 * math.sqrt(math.log(total_n) / counts[k])
                    if counts[k] else float('inf')))
            i += 1

            world = random.choice(worlds)
            s1 = self._resolve(state, candidates[arm] + world, commitments)
            value = evaluate_state(s1, self.player) + self._stance_bonus(state, s1, stance)

            # One random rollout ply.
            if self.nodes < cfg.node_budget:
                my_next = random.choice(generate_all_order_sets(s1, self.player))
                their_next = random.choice(sample_opponent_orders(
                    s1, self.player, trust_model, samples=1,
                    opponent_model=self.opponent_model, commitments=commitments,
                    incentives=self._incentives))
                s2 = self._resolve(s1, my_next + their_next, commitments)
                value = 0.5 * value + 0.5 * (evaluate_state(s2, self.player)
                                             + self._stance_bonus(state, s2, stance))

            counts[arm] += 1
            totals[arm] += value

        best_orders, best_score, best_trace = [], float('-inf'), None
        loyal = [totals[k] / counts[k] for k in range(len(candidates))
                 if counts[k] and not breaks_commitment(state, self.player, candidates[k], commitments)]
        best_loyal_ev = max(loyal) if loyal else 0.0
        forfeits = self._forfeits(state, commitments)

        for k, cand in enumerate(candidates):
            if not counts[k]:
                continue
            avg_ev = totals[k] / counts[k]
            breaks = breaks_commitment(state, self.player, cand, commitments)
            incentive = min(1.0, max(0.0, (avg_ev - best_loyal_ev) / CENTRE_VALUE)) if breaks else 0.0
            graded = verify_commitments(state, commitments, cand)
            rows = penalty_breakdown(state, self.player, cand, commitments,
                                     self.config, trust_model, incentive,
                                     outcomes=graded, weights=self.partner_weight)
            f_rows = forfeit_breakdown(state, self.player, cand, commitments,
                                       forfeits, outcomes=graded)
            loyalty = self._loyalty(graded)
            penalty = sum(r[-1] for r in rows) + sum(r[-1] for r in f_rows) + loyalty
            net = avg_ev - penalty
            if net > best_score:
                best_score, best_orders = net, cand
                best_trace = DecisionTrace(
                    cand, avg_ev, penalty,
                    self._explain(avg_ev, penalty, net, rows, incentive, 0.0, f_rows,
                                  loyalty)
                    + f" [MCTS, {sum(counts)} playouts]",
                    bool(rows), penalty_rows=rows, nodes=self.nodes,
                    search="mcts", candidates=len(candidates), pruned=pruned,
                    forfeit_rows=f_rows)

        if not best_orders:
            hold_set = [Order(player=self.player, unit_territory=u.territory,
                              order_type=OrderType.HOLD)
                        for u in state.units if u.player == self.player]
            return hold_set, DecisionTrace(hold_set, 0, 0, "Fallback to HOLD")
        return best_orders, best_trace

    @staticmethod
    def _explain(avg_ev, penalty, net_score, rows, incentive, vengeance=0.0,
                 forfeit_rows=(), loyalty=0.0) -> str:
        """One sentence an examiner can read off the screen."""
        veng = (f" Revenge worth {vengeance:.2f} counted in."
                if abs(vengeance) >= 0.005 else "")
        forfeit = "".join(
            f" Forfeits {amt:.2f} of the {price:.2f} it signed with "
            f"{partner.value} for, {frac:.0%} of the term still to run."
            for _key, partner, price, frac, amt in (forfeit_rows or ()))
        forfeit += (f" Breaks first with a partner owed loyalty: {loyalty:.2f}."
                    if loyalty else "")
        if not rows:
            return (f"Value {avg_ev:.2f}, no commitment broken, net {net_score:.2f}. "
                    f"Keeping every live promise was already the best move.{veng}")

        parts = "; ".join(
            f"Vcoop({partner.value}) {v:.2f} x dP {dp:.2f} x horizon {h:.2f} = {amt:.2f}"
            for partner, v, dp, h, amt in rows
        )
        return (f"Value {avg_ev:.2f} (gain over staying loyal {incentive:.2f}), "
                f"reputation cost {parts}, total {penalty:.2f}, net {net_score:.2f}. "
                f"{'Breaking' if net_score > 0 else 'Holding'} priced out at these "
                f"numbers.{forfeit}{veng}")
