from typing import List, Dict, Tuple, Any, Optional
from src.common.schemas import GameState, Order, OrderType, Player, Commitment, DecisionTrace
from src.engine.adjudicator import resolve, verify_commitments
from src.engine.orders import generate_all_order_sets
from src.engine.board import is_supply_center, get_adjacent, MAX_TURNS
from src.agents.trust.model import TrustModel, break_weight
import random
import copy

class PlannerConfig:
    def __init__(self, reputation_cost_coefficient: float = 1.0, depth: int = 1,
                 opponent_samples: int = 8):
        self.reputation_cost_coefficient = reputation_cost_coefficient
        self.depth = depth
        # ponytail: 8 sampled worlds, not the 32 of slide 7, and depth is not
        # read yet (search is depth-1). Raise both once the search is faster.
        self.opponent_samples = opponent_samples

# Everything below is denominated in centres: one supply centre = 1.0, so the
# numbers in a decision trace read the same way as the worked example.
CENTRE_VALUE = 1.0
UNIT_VALUE = 0.3
THREATENS_VALUE = 0.2
UNDER_THREAT_VALUE = -0.2
MOBILITY_VALUE = 0.01


def evaluate_state(state: GameState, player: Player) -> float:
    """Centres held, units, centres under threat, centres threatened, mobility."""
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

    return score


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
    trust_model: TrustModel,
    samples: int = 3,
    opponent_model=None,
) -> List[List[Order]]:
    """Sample likely opponent orders.

    If an *opponent_model* is provided, delegate to its weighted sampler.
    Otherwise fall back to uniform-random sampling (original behaviour).
    """
    if opponent_model is not None:
        return opponent_model.sample_opponent_orders(state, samples)

    # ── Legacy uniform-random fallback ──────────────────────────────
    opponents = [p for p in Player if p != my_player]
    
    all_samples = []
    for _ in range(samples):
        joint_orders = []
        for opp in opponents:
            sets = generate_all_order_sets(state, opp)
            if sets:
                joint_orders.extend(random.choice(sets))
        all_samples.append(joint_orders)
        
    return all_samples

def penalty_breakdown(state: GameState, player: Player, my_orders: List[Order],
                      commitments: List[Commitment], config: PlannerConfig,
                      trust_model: TrustModel = None, incentive: float = 0.0):
    """Price every commitment this order set breaks.

        penalty = Vcoop(partner) x deltaP(partner keeps) x (turns left / 12)

    There is no "betray" flag anywhere: near the horizon the last term shrinks,
    and mid-game Vcoop collapses once the partner stops being useful. Either way
    the same arithmetic stops paying for loyalty.

    Returns one (partner, vcoop, delta_p, horizon, amount) row per break.
    """
    rows = []
    horizon = max(0, MAX_TURNS - state.turn) / MAX_TURNS

    for o in verify_commitments(state, commitments, my_orders):
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
            if partner == player:
                continue
            v_coop = cooperation_value(state, player, partner)
            amount = v_coop * delta_p * horizon * config.reputation_cost_coefficient
            rows.append((partner, v_coop, delta_p, horizon, amount))

    return rows


def calculate_commitment_penalty(state: GameState, player: Player, my_orders: List[Order],
                                 commitments: List[Commitment], config: PlannerConfig,
                                 trust_model: TrustModel = None,
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


class Planner:
    def __init__(self, player: Player, config: PlannerConfig):
        self.player = player
        self.config = config
        self.opponent_model = None  # set by Agent after construction

    def set_opponent_model(self, model):
        """Attach an OpponentModel for informed sampling."""
        self.opponent_model = model
        
    def find_best_orders(self, state: GameState, commitments: List[Commitment], trust_model: TrustModel) -> Tuple[List[Order], DecisionTrace]:
        my_order_sets = generate_all_order_sets(state, self.player)

        hold_set = [Order(player=self.player, unit_territory=u.territory, order_type=OrderType.HOLD)
                    for u in state.units if u.player == self.player]

        # ponytail: random subsample, not the dominance pruning the deck asks
        # for. Swap it in here if the candidate count ever gets past ~50.
        if len(my_order_sets) > 50:
            my_order_sets = random.sample(my_order_sets, 49)
            my_order_sets.append(hold_set)

        sampled_opp_orders = sample_opponent_orders(
            state, self.player, trust_model,
            samples=self.config.opponent_samples, opponent_model=self.opponent_model,
        )

        # Pass 1: expected value of every candidate, and whether it breaks a deal.
        scored = []
        for my_set in my_order_sets:
            total_ev = 0.0
            for opp_set in sampled_opp_orders:
                new_state, _, _ = resolve(state, my_set + opp_set, commitments)
                total_ev += evaluate_state(new_state, self.player)
            avg_ev = total_ev / max(1, len(sampled_opp_orders))
            breaks = breaks_commitment(state, self.player, my_set, commitments)
            scored.append((my_set, avg_ev, breaks))

        if not scored:
            return hold_set, DecisionTrace(hold_set, 0, 0, "Fallback to HOLD")

        # The incentive to defect is what a break buys over the best honest
        # option. That is the iota the trust model discounts the penalty by.
        loyal_evs = [ev for _, ev, breaks in scored if not breaks]
        best_loyal_ev = max(loyal_evs) if loyal_evs else 0.0

        best_orders, best_score, best_trace = [], float('-inf'), None
        for my_set, avg_ev, breaks in scored:
            incentive = 0.0
            if breaks:
                incentive = min(1.0, max(0.0, (avg_ev - best_loyal_ev) / CENTRE_VALUE))

            rows = penalty_breakdown(state, self.player, my_set, commitments,
                                     self.config, trust_model, incentive)
            penalty = sum(r[-1] for r in rows)
            net_score = avg_ev - penalty

            if net_score > best_score:
                best_score = net_score
                best_orders = my_set
                best_trace = DecisionTrace(
                    my_set, avg_ev, penalty,
                    self._explain(avg_ev, penalty, net_score, rows, incentive),
                    bool(rows),
                )

        if not best_orders:
            return hold_set, DecisionTrace(hold_set, 0, 0, "Fallback to HOLD")

        return best_orders, best_trace

    @staticmethod
    def _explain(avg_ev, penalty, net_score, rows, incentive) -> str:
        """One sentence an examiner can read off the screen."""
        if not rows:
            return (f"Value {avg_ev:.2f}, no commitment broken, net {net_score:.2f}. "
                    f"Keeping every live promise was already the best move.")

        parts = "; ".join(
            f"Vcoop({partner.value}) {v:.2f} x dP {dp:.2f} x horizon {h:.2f} = {amt:.2f}"
            for partner, v, dp, h, amt in rows
        )
        return (f"Value {avg_ev:.2f} (gain over staying loyal {incentive:.2f}), "
                f"reputation cost {parts}, total {penalty:.2f}, net {net_score:.2f}. "
                f"{'Breaking' if net_score > 0 else 'Holding'} priced out at these numbers.")
