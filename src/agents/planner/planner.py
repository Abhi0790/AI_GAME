from typing import List, Dict, Tuple, Any, Optional
from src.common.schemas import GameState, Order, OrderType, Player, Commitment
from src.engine.adjudicator import CommitmentOutcome
from src.engine.adjudicator import resolve, verify_commitments
from src.agents.planner.order_generator import generate_all_order_sets
from src.engine.board import is_supply_center, get_adjacent
from src.agents.trust.model import TrustModel
import random
import copy

MAX_TURNS = 12

class DecisionTrace:
    def __init__(self, candidate_orders: List[Order], expected_value: float, penalty: float, explanation: str, commitment_broken: bool = False):
        self.candidate_orders = candidate_orders
        self.expected_value = expected_value
        self.penalty = penalty
        self.explanation = explanation
        self.commitment_broken = commitment_broken

class PlannerConfig:
    def __init__(self, reputation_cost_coefficient: float = 1.0, depth: int = 1):
        self.reputation_cost_coefficient = reputation_cost_coefficient
        self.depth = depth

def evaluate_state(state: GameState, player: Player) -> float:
    score = 0.0
    
    # 1. Centers held
    owned_centers = 0
    for loc, owner in state.territory_owners.items():
        if owner == player and is_supply_center(loc):
            owned_centers += 1
            score += 10.0 # High value for holding SC
            
    # 2. Centers threatened (adjacent to a unit, but not owned)
    player_units = [u for u in state.units if u.player == player]
    threatened = set()
    for u in player_units:
        for adj in get_adjacent(u.territory):
            if is_supply_center(adj) and state.territory_owners.get(adj) != player:
                threatened.add(adj)
    score += len(threatened) * 2.0
    
    # 3. Unit safety (not adjacent to more enemies than we have adjacent)
    # Simple proxy: number of units
    score += len(player_units) * 3.0
    
    # 4. Mobility
    # More adjacent territories = better
    for u in player_units:
        score += len(get_adjacent(u.territory)) * 0.1
        
    return score

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

def calculate_commitment_penalty(state: GameState, player: Player, my_orders: List[Order], commitments: List[Commitment], config: PlannerConfig) -> float:
    """Calculate the reputation penalty for breaking a commitment."""
    # Check what breaks
    outcomes = verify_commitments(state, commitments, my_orders)
    
    penalty = 0.0
    turns_left = max(0, MAX_TURNS - state.turn)
    
    for o in outcomes:
        if not o.kept and player in o.broken_by:
            # We broke it
            # Penalty = expected future value of cooperation * chance of losing it
            # Future value scales with turns left.
            future_val = turns_left * 2.0 # Arbitrary unit value per turn
            
            # The penalty is proportional to the reputation cost coefficient (Persona)
            penalty += future_val * config.reputation_cost_coefficient
            
    return penalty

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
        
        # Limit to a reasonable number to prevent combinatorial explosion
        # If there are > 50, sample 50 random ones + all hold
        hold_set = [Order(player=self.player, unit_territory=u.territory, order_type=OrderType.HOLD) for u in state.units if u.player == self.player]
        
        if len(my_order_sets) > 50:
            my_order_sets = random.sample(my_order_sets, 49)
            my_order_sets.append(hold_set)
            
        sampled_opp_orders = sample_opponent_orders(
            state, self.player, trust_model,
            samples=3, opponent_model=self.opponent_model,
        )
        
        best_orders = []
        best_score = float('-inf')
        best_trace = None
        
        for my_set in my_order_sets:
            total_ev = 0.0
            
            for opp_set in sampled_opp_orders:
                joint_orders = my_set + opp_set
                new_state, _, _ = resolve(state, joint_orders, commitments)
                val = evaluate_state(new_state, self.player)
                total_ev += val
                
            avg_ev = total_ev / max(1, len(sampled_opp_orders))
            
            penalty = calculate_commitment_penalty(state, self.player, my_set, commitments, self.config)
            
            net_score = avg_ev - penalty
            
            if net_score > best_score:
                best_score = net_score
                best_orders = my_set
                
                broke = penalty > 0
                exp = f"Expected Value: {avg_ev:.2f}, Penalty: {penalty:.2f}, Net: {net_score:.2f}. "
                if broke:
                    exp += f"Breaking commitment was considered optimal because {avg_ev:.2f} > {penalty:.2f}."
                else:
                    exp += "Keeping commitments or no commitments broken."
                    
                best_trace = DecisionTrace(my_set, avg_ev, penalty, exp, broke)
                
        if not best_orders:
            return hold_set, DecisionTrace(hold_set, 0, 0, "Fallback to HOLD")
            
        return best_orders, best_trace
