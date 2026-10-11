from typing import List, Dict, Tuple, Optional
from src.common.schemas import (
    GameState, Order, OrderType, Player, Commitment, CommitmentType, DecisionTrace,
    commitment_key, exchange_leg_due, obligated_parties,
)
from src.engine.adjudicator import resolve, verify_commitments
from src.engine.orders import generate_all_order_sets, generate_orders_for_unit
from src.engine.board import (
    is_supply_center, get_adjacent, max_turns, players, win_centers, home_centers, sieges,
)
from src.agents.trust.model import TrustModel, break_weight
from src.agents.planner.regret import regret_matching, exploitability
import math
import random


class PlannerConfig:
    """Search budget (in adjudications) and the persona knobs the planner reads.

    "expectimax" searches to depth 2 against sampled opponents.
    """

    def __init__(self, reputation_cost_coefficient: float = 1.0, depth: int = 2,
                 opponent_samples: int = 32, beam_width: int = 8,
                 depth2_candidates: int = 4, depth2_samples: int = 6,
                 depth2_replies: int = 6, node_budget: int = 1500,
                 vengeance: float = 0.0, search: str = "expectimax",
                 retaliation_samples: int = 1,
                 trace_coverage: bool = False,
                 reserve_keep_candidate: bool = False,
                 regret_candidates: int = 6, regret_anchor: float = 0.3,
                 regret_iterations: int = 300, reputation_carries: bool = False):
        self.reputation_cost_coefficient = reputation_cost_coefficient
        self.depth = depth                    # 1 = one ply, 2 = my move, their reply
        self.opponent_samples = opponent_samples
        self.beam_width = beam_width          # candidates surviving dominance pruning
        self.depth2_candidates = depth2_candidates
        self.depth2_samples = depth2_samples
        self.depth2_replies = depth2_replies
        self.node_budget = node_budget        # adjudications per decision
        self.vengeance = vengeance
        if search not in ("expectimax", "mcts", "regret"):
            raise ValueError(f"unknown search {search!r}")
        self.search = search                  # "expectimax" | "mcts" | "regret"
        self.regret_candidates = regret_candidates  # order sets per seat in the solve
        self.regret_anchor = regret_anchor    # P(an opponent plays its modelled policy)
        self.regret_iterations = regret_iterations  # cap; cached joints cost no budget
        self.reputation_carries = reputation_carries  # reputation outlives this game
        # Extra depth-2 worlds in which the players I just betrayed come for me.
        self.retaliation_samples = retaliation_samples
        self.trace_coverage = trace_coverage
        self.reserve_keep_candidate = reserve_keep_candidate


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
# Under sieges, a unit on a centre it does not own yet counts this share of
# one. A capture there takes up to three turns, past the search horizon.
SIEGE_CREDIT = 0.5
# A player this many centres or fewer from winning is treated as about to win:
# no deals with them, and breaking one to stop them is not charged as a betrayal.
NEAR_WIN_MARGIN = 1

# Share of a deal's signed price forfeited by walking away from it.
FORFEIT_WEIGHT = 1.0

# P(a betrayed player hits back) before anything has been seen of them. One in
# three is the weight the retaliation world carried when it was one sample of three.
RETALIATION_PRIOR = 1 / 3

# Charged, in centres, per partner in Planner.loyal_to an order set breaks with first.
LOYALTY_COST = 1.0

# Charged, in centres times the persona's reputation coefficient, per announced
# move (INTENT) an order set does not make: Honest pays 0.5, the Opportunist 0.1.
INTENT_COST = 0.1


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
    if sieges():
        score += SIEGE_CREDIT * CENTRE_VALUE * sum(
            1 for t in my_squares
            if is_supply_center(t) and state.supply_centers.get(t) != player)

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
    """Vcoop: my centres within the partner's reach (garrisoned ones count less)
    plus centres neither owns that both can reach, in evaluate_state units."""
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
    rational: bool = False,
) -> List[List[Order]]:
    """Sample joint opponent orders from the opponent model, else uniformly."""
    if opponent_model is not None:
        return opponent_model.sample_opponent_orders(
            state, samples, commitments=commitments, trust_model=trust_model,
            incentives=incentives, rational=rational)

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
    """Vcoop x deltaP x horizon per broken deal, scaled by *weights*; horizon is
    turns left / max turns, or 1 when reputation carries into the next game.

    Returns one (partner, vcoop, delta_p, horizon, amount) row per break.
    """
    rows = []
    weights = weights or {}
    horizon = 1.0 if config.reputation_carries else max(0, max_turns() - state.turn) / max_turns()

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
    """The unused share of each broken deal's signed price (*forfeits* from Planner._forfeits).

    Returns one (commitment_key, partner, signed_price, fraction_remaining, amount) row per break.
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


def keeping_order_set(state: GameState, player: Player,
                      commitments: List[Commitment]) -> Optional[List[Order]]:
    """Construct a legal order set that keeps all obligations due this turn."""
    units = [u for u in state.units if u.player == player]
    holds = {u.territory: Order(player=player, unit_territory=u.territory,
                                order_type=OrderType.HOLD) for u in units}
    baseline = list(holds.values())
    if not breaks_commitment(state, player, baseline, commitments):
        return baseline

    required = set()
    for c in commitments:
        if player not in obligated_parties(c, state.turn):
            continue
        if c.commitment_type == CommitmentType.SUPPORT or (
            c.commitment_type == CommitmentType.EXCHANGE
            and exchange_leg_due(c, state.turn) == "give"
        ):
            required.add((c.supported_from, c.target_territory))
    if len(required) > len(units):
        return None

    options = {
        need: [order for u in units for order in generate_orders_for_unit(state, u)
               if order.order_type == OrderType.SUPPORT
               and (order.supported_from, order.target) == need]
        for need in required
    }
    if any(not choices for choices in options.values()):
        return None
    needs = sorted(required, key=lambda need: (len(options[need]), str(need)))

    def assign(index: int, chosen: dict) -> Optional[List[Order]]:
        if index == len(needs):
            orders = [chosen.get(u.territory, holds[u.territory]) for u in units]
            return orders if not breaks_commitment(state, player, orders, commitments) else None
        for order in options[needs[index]]:
            if order.unit_territory in chosen:
                continue
            chosen[order.unit_territory] = order
            result = assign(index + 1, chosen)
            if result is not None:
                return result
            del chosen[order.unit_territory]
        return None

    return assign(0, {})


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
    """iota: this player's best value breaking a live promise minus their best
    keeping every one, against a quiet world."""
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
        # The negotiator that signed this seat's deals (set by Agent), for deal_prices.
        self.negotiation = None
        self.nodes = 0              # adjudications spent on the last decision
        self._incentives: Dict[Player, float] = {}
        # Set by Agent.act from the persona's policy, each decision:
        self.loyal_to: set = set()                    # partners owed loyalty (LOYALTY_COST)
        self.partner_weight: Dict[Player, float] = {} # reputation-cost scale per partner
        self.retaliation: Dict[Player, float] = {}    # P(they hit back once betrayed)
        self.announced: List[Tuple[str, str]] = []   # moves I said I would make (INTENT)

    def _opponent_incentives(self, state: GameState, commitments: List[Commitment]
                             ) -> Dict[Player, float]:
        """iota for each opponent with a deal, within this decision's budget."""
        opponents = sorted({q for c in commitments for q in c.players
                            if q != self.player}, key=lambda q: q.value)
        # Leave at least a quarter of the budget for evaluating actual moves.
        reserve = max(1, self.config.node_budget // 4)
        available = max(0, self.config.node_budget - self.nodes - reserve)
        samples = min(60, available // max(1, len(opponents)))
        if samples < 2:
            return {p: 0.0 for p in opponents}
        count = lambda s, orders: self._resolve(s, orders, [])
        return {p: defection_incentive(state, p, commitments,
                                      samples=samples, resolver=count)
                for p in opponents}

    def set_opponent_model(self, model):
        """Attach an OpponentModel for informed sampling."""
        self.opponent_model = model

    # ── forfeiture ──────────────────────────────────────────────────────
    def _forfeits(self, state: GameState, commitments: List[Commitment]
                  ) -> Dict[tuple, Tuple[float, float, float]]:
        """{commitment_key: (signed_price, fraction_remaining, amount)} for my live deals,
        amount = FORFEIT_WEIGHT x max(0, price) x turns_left / deal_length."""
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

    def _costs(self, state, commitments, cand, trust_model, incentive, forfeits):
        """(penalty, reputation rows, forfeit rows, loyalty, candour) of one order set."""
        graded = verify_commitments(state, commitments, cand)
        rows = penalty_breakdown(state, self.player, cand, commitments,
                                 self.config, trust_model, incentive,
                                 outcomes=graded, weights=self.partner_weight)
        f_rows = forfeit_breakdown(state, self.player, cand, commitments,
                                   forfeits, outcomes=graded)
        loyalty = self._loyalty(graded)
        said = [(f, t) for f, t in self.announced
                if not any(o.unit_territory == f and o.order_type == OrderType.MOVE
                           and o.target == t for o in cand)]
        candour = INTENT_COST * self.config.reputation_cost_coefficient * len(said)
        penalty = sum(r[-1] for r in rows) + sum(r[-1] for r in f_rows) + loyalty + candour
        return penalty, rows, f_rows, loyalty, candour

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
              commitments: List[Commitment],
              max_candidates: Optional[int] = None) -> Tuple[List[List[Order]], int]:
        """Drop sets dominated in both reference worlds (all hold, all attack me),
        loyal and treacherous sets in separate pools, then beam."""
        original_count = len(candidates)
        if max_candidates is not None and len(candidates) > max_candidates:
            # Always keep one promise-keeping and one promise-breaking set.
            loyal = [c for c in candidates
                     if not breaks_commitment(state, self.player, c, commitments)]
            breaking = [c for c in candidates
                        if breaks_commitment(state, self.player, c, commitments)]
            selected = [pool[0] for pool in (loyal, breaking) if pool]
            selected = selected[:max_candidates]
            remaining = [c for c in candidates if c not in selected]
            slots = max_candidates - len(selected)
            candidates = selected + random.sample(remaining, slots)

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
        return kept, original_count - len(kept)

    def _prepare_candidates(self, state: GameState,
                            commitments: List[Commitment]):
        """Sample, prune, and optionally reserve one feasible keeping line."""
        all_sets = generate_all_order_sets(state, self.player)
        coverage = {}
        keep_set = None
        if self.config.trace_coverage or self.config.reserve_keep_candidate:
            due = [c for c in commitments
                   if self.player in obligated_parties(c, state.turn)]
            if due:
                keep_set = keeping_order_set(state, self.player, commitments)
                generated_keep = any(not breaks_commitment(
                    state, self.player, orders, commitments) for orders in all_sets)
                coverage = {
                    "due_obligations": len(due),
                    "keepable": keep_set is not None,
                    "generated_keep": generated_keep,
                }
                if (self.config.reserve_keep_candidate and keep_set is not None
                        and not generated_keep):
                    all_sets = all_sets + [keep_set]

        candidates, pruned = self.prune(
            state, all_sets, commitments,
            max_candidates=max(1, self.config.node_budget // 4))
        if coverage:
            loyal = next((orders for orders in candidates if not breaks_commitment(
                state, self.player, orders, commitments)), None)
            coverage["pruned_keep"] = loyal is not None
            if self.config.reserve_keep_candidate and loyal is not None:
                # Pruning sorts by raw position value. At low budgets that can
                # put every loyal option after the search runs out of nodes.
                candidates = [loyal] + [orders for orders in candidates if orders != loyal]
        return candidates, pruned, coverage

    # ── vengeance ───────────────────────────────────────────────────────
    def _stance_bonus(self, before: GameState, after: GameState,
                      stance: Dict[Player, float]) -> float:
        """Value of hurting each player: positive weight is a grudge, negative a believed threat."""
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
        """Expectimax, two plies: chance over their orders, max over my reply,
        chance over their reply."""
        cfg = self.config

        # Deals this line breaks, and whom it betrays.
        broken = [o.commitment for o in verify_commitments(state, commitments, my_set)
                  if not o.kept and self.player in o.broken_by]
        betrayed = {p for c in broken for p in c.players if p != self.player}
        broken_ids = {c.id for c in broken}
        # Broken deals release their members, as in GameRunner.step.
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

            # Retaliation worlds: the betrayed spend their reply on me,
            # weighted by how often they have hit back before.
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
        if self.config.node_budget < 4:
            return hold_set, DecisionTrace(hold_set, 0, 0, "Search budget too small")

        if self.config.search == "mcts":
            return self.mcts_best_orders(state, commitments, trust_model, stance)
        if self.config.search == "regret":
            return self.regret_best_orders(state, commitments, trust_model, stance)

        my_order_sets, pruned, coverage = self._prepare_candidates(state, commitments)

        self._incentives = self._opponent_incentives(state, commitments)
        worlds = sample_opponent_orders(
            state, self.player, trust_model,
            samples=self.config.opponent_samples, opponent_model=self.opponent_model,
            commitments=commitments, incentives=self._incentives, rational=True,
        )

        # Pass 1: expected value of every surviving candidate, depth 1.
        scored = []
        for my_set in my_order_sets:
            ev_total, veng_total, n = 0.0, 0.0, 0
            for opp_set in worlds:
                if self.nodes >= self.config.node_budget:
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
            return hold_set, DecisionTrace(hold_set, 0, 0, "Fallback to HOLD",
                                           coverage=coverage)

        if coverage:
            coverage["evaluated_keep"] = any(not row[3] for row in scored)

        # Pass 2 (depth 2): only the leaders are worth the adjudications.
        if self.config.depth >= 2:
            order = sorted(range(len(scored)), key=lambda i: scored[i][1] + scored[i][2],
                           reverse=True)[:self.config.depth2_candidates]
            for i in order:
                v = self._depth2_value(state, scored[i][0], worlds, commitments,
                                       trust_model)
                if v is not None:
                    # Vengeance stays the depth-1 estimate.
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

            penalty, rows, f_rows, loyalty, candour = self._costs(
                state, commitments, my_set, trust_model, incentive, forfeits)
            net_score = avg_ev + veng - penalty
            nets.append(net_score)

            if net_score > best_score:
                best_score = net_score
                best_orders = my_set
                best_trace = DecisionTrace(
                    my_set, avg_ev, penalty,
                    self._explain(avg_ev, penalty, net_score, rows,
                                  avg_ev - best_loyal_ev if loyal_evs else None,
                                  veng, f_rows, loyalty, candour),
                    bool(rows), penalty_rows=rows, nodes=self.nodes, vengeance=veng,
                    search=self.config.search, candidates=len(my_order_sets), pruned=pruned,
                    forfeit_rows=f_rows,
                )

        if not best_orders:
            return hold_set, DecisionTrace(hold_set, 0, 0, "Fallback to HOLD",
                                           coverage=coverage)

        best_trace.nodes = self.nodes
        best_trace.reversals = self._break_advantages(state, commitments, scored, nets)
        if coverage:
            keeping_nets = [net for row, net in zip(scored, nets) if not row[3]]
            breaking_nets = [net for row, net in zip(scored, nets) if row[3]]
            coverage.update(
                best_keep_net=max(keeping_nets) if keeping_nets else None,
                best_break_net=max(breaking_nets) if breaking_nets else None,
                chosen_break=breaks_commitment(state, self.player, best_orders,
                                                commitments))
            best_trace.coverage = coverage
        return best_orders, best_trace

    def _break_advantages(self, state: GameState, commitments: List[Commitment],
                          scored, nets: List[float]) -> List[tuple]:
        """Per live deal of mine: best net score breaking it minus best net keeping it."""
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
        """UCT over my order sets: one sampled opponent world per playout, then
        one random rollout ply."""
        stance = stance or {}
        cfg = self.config
        candidates, pruned, coverage = self._prepare_candidates(state, commitments)

        counts = [0] * len(candidates)
        totals = [0.0] * len(candidates)
        self._incentives = self._opponent_incentives(state, commitments)
        worlds = sample_opponent_orders(
            state, self.player, trust_model, samples=max(8, cfg.opponent_samples // 2),
            opponent_model=self.opponent_model, commitments=commitments,
            incentives=self._incentives, rational=True)

        i = 0
        while self.nodes < cfg.node_budget:
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
        if coverage:
            coverage["evaluated_keep"] = bool(loyal)
        best_loyal_ev = max(loyal) if loyal else 0.0
        forfeits = self._forfeits(state, commitments)

        for k, cand in enumerate(candidates):
            if not counts[k]:
                continue
            avg_ev = totals[k] / counts[k]
            breaks = breaks_commitment(state, self.player, cand, commitments)
            incentive = min(1.0, max(0.0, (avg_ev - best_loyal_ev) / CENTRE_VALUE)) if breaks else 0.0
            penalty, rows, f_rows, loyalty, candour = self._costs(
                state, commitments, cand, trust_model, incentive, forfeits)
            net = avg_ev - penalty
            if net > best_score:
                best_score, best_orders = net, cand
                best_trace = DecisionTrace(
                    cand, avg_ev, penalty,
                    self._explain(avg_ev, penalty, net, rows,
                                  avg_ev - best_loyal_ev if loyal else None,
                                  0.0, f_rows, loyalty, candour)
                    + f" [MCTS, {sum(counts)} playouts]",
                    bool(rows), penalty_rows=rows, nodes=self.nodes,
                    search="mcts", candidates=len(candidates), pruned=pruned,
                    forfeit_rows=f_rows)

        if not best_orders:
            hold_set = [Order(player=self.player, unit_territory=u.territory,
                              order_type=OrderType.HOLD)
                        for u in state.units if u.player == self.player]
            return hold_set, DecisionTrace(hold_set, 0, 0, "Fallback to HOLD")
        if coverage:
            coverage["chosen_break"] = breaks_commitment(
                state, self.player, best_orders, commitments)
            best_trace.coverage = coverage
        return best_orders, best_trace

    # ── regret matching, same budget ────────────────────────────────────
    def regret_best_orders(self, state: GameState, commitments: List[Commitment],
                           trust_model: TrustModel,
                           stance: Optional[Dict[Player, float]] = None
                           ) -> Tuple[List[Order], DecisionTrace]:
        """RM+ over a candidate set per seat, opponents anchored toward the
        opponent model; plays a draw from my average strategy."""
        stance = stance or {}
        cfg = self.config
        ranked, pruned, coverage = self._prepare_candidates(state, commitments)
        # One candidate per movement plan, best-ranked support variant first.
        # ponytail: keyed on moves alone; key on supports too if they start to matter.
        plans: Dict[frozenset, List[Order]] = {}
        for cand in ranked:
            plans.setdefault(frozenset((o.unit_territory, o.target) for o in cand
                                       if o.order_type == OrderType.MOVE), cand)
        mine = list(plans.values())[:cfg.regret_candidates]
        # Always offer the solve a way to keep every promise due.
        if not any(not breaks_commitment(state, self.player, c, commitments) for c in mine):
            loyal = next((c for c in ranked
                          if not breaks_commitment(state, self.player, c, commitments)),
                         None) or keeping_order_set(state, self.player, commitments)
            if loyal is not None:
                mine = mine[:cfg.regret_candidates - 1] + [loyal]
        self._incentives = self._opponent_incentives(state, commitments)
        forfeits = self._forfeits(state, commitments)

        draws = sample_opponent_orders(
            state, self.player, trust_model, samples=cfg.opponent_samples,
            opponent_model=self.opponent_model, commitments=commitments,
            incentives=self._incentives, rational=True)
        seats = [self.player] + [p for p in players() if p != self.player
                                 and any(u.player == p for u in state.units)]
        sets: List[List[List[Order]]] = [mine]
        anchors: List[Optional[List[float]]] = [None]
        for q in seats[1:]:
            counts: Dict[tuple, int] = {}
            for world in draws:
                key = tuple(o for o in world if o.player == q)
                if not key:
                    continue
                counts[key] = counts.get(key, 0) + 1
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:cfg.regret_candidates]
            sets.append([list(k) for k, _n in top])
            total = sum(n for _k, n in top)
            anchors.append([n / total for _k, n in top])

        cost = [self._costs(state, commitments, c, trust_model, 0.0, forfeits)[0]
                for c in mine]
        resolved: Dict[tuple, GameState] = {}

        def utility(p: int, joint: tuple) -> float:
            if joint not in resolved:
                orders = [o for seat, a in enumerate(joint) for o in sets[seat][a]]
                resolved[joint] = self._resolve(state, orders, commitments)
            after = resolved[joint]
            if p:
                return evaluate_state(after, seats[p])
            return (evaluate_state(after, self.player)
                    + self._stance_bonus(state, after, stance) - cost[joint[0]])

        per_iteration = sum(len(x) for x in sets)
        strategies, values, iterations = regret_matching(
            [len(x) for x in sets], utility, iterations=cfg.regret_iterations,
            anchors=anchors, anchor_weight=cfg.regret_anchor,
            keep_going=lambda: self.nodes + per_iteration <= cfg.node_budget)
        if not iterations:
            hold_set = [Order(player=self.player, unit_territory=u.territory,
                              order_type=OrderType.HOLD)
                        for u in state.units if u.player == self.player]
            return hold_set, DecisionTrace(hold_set, 0, 0, "Fallback to HOLD")

        pick = random.choices(range(len(mine)), weights=strategies[0])[0]
        chosen = mine[pick]
        # Action values are net of the zero-incentive cost; add it back for EV.
        ev = [values[0][k] + cost[k] for k in range(len(mine))]
        loyal = [ev[k] for k, c in enumerate(mine)
                 if not breaks_commitment(state, self.player, c, commitments)]
        best_loyal = max(loyal) if loyal else 0.0
        breaks = breaks_commitment(state, self.player, chosen, commitments)
        incentive = min(1.0, max(0.0, (ev[pick] - best_loyal) / CENTRE_VALUE)) if breaks else 0.0
        penalty, rows, f_rows, loyalty, candour = self._costs(
            state, commitments, chosen, trust_model, incentive, forfeits)
        gap = exploitability(strategies, values)
        trace = DecisionTrace(
            chosen, ev[pick], penalty,
            self._explain(ev[pick], penalty, ev[pick] - penalty, rows,
                          ev[pick] - best_loyal if loyal else None, 0.0, f_rows, loyalty, candour)
            + f" [regret matching, {iterations} iterations, played with "
              f"P={strategies[0][pick]:.2f}, exploitability {gap:.2f}]",
            bool(rows), penalty_rows=rows, nodes=self.nodes, search="regret",
            candidates=len(mine), pruned=pruned, forfeit_rows=f_rows, coverage=coverage)
        trace.mixed = [(c, p) for c, p in zip(mine, strategies[0])]
        trace.exploitability = gap
        return chosen, trace

    @staticmethod
    def _explain(avg_ev, penalty, net_score, rows, gain, vengeance=0.0,
                 forfeit_rows=(), loyalty=0.0, candour=0.0) -> str:
        """One sentence an examiner can read off the screen. `gain` is the raw
        value over the best loyal order set, None when every option breaks."""
        veng = (f" Revenge worth {vengeance:.2f} counted in."
                if abs(vengeance) >= 0.005 else "")
        forfeit = "".join(
            f" Forfeits {amt:.2f} of the {price:.2f} it signed with "
            f"{partner.value} for, {frac:.0%} of the term still to run."
            for _key, partner, price, frac, amt in (forfeit_rows or ()))
        forfeit += (f" Breaks first with a partner owed loyalty: {loyalty:.2f}."
                    if loyalty else "")
        forfeit += (f" Goes back on a move it announced: {candour:.2f}." if candour else "")
        if not rows:
            return (f"Value {avg_ev:.2f}, no commitment broken, net {net_score:.2f}. "
                    f"Keeping every live promise was already the best move.{veng}")

        parts = "; ".join(
            f"Vcoop({partner.value}) {v:.2f} x dP {dp:.2f} x horizon {h:.2f} = {amt:.2f}"
            for partner, v, dp, h, amt in rows
        )
        loyal = (f"gain over staying loyal {gain:.2f}" if gain is not None
                 else "no loyal option")
        verdict = ("Every option breaks a promise; this one costs least."
                   if gain is None else
                   "Breaking still scored higher than every loyal order set.")
        return (f"Value {avg_ev:.2f} ({loyal}), "
                f"reputation cost {parts}, total {penalty:.2f}, net {net_score:.2f}. "
                f"{verdict}{forfeit}{veng}")
