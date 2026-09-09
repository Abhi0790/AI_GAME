from typing import List, Set
import itertools
from src.common.schemas import GameState, Order, OrderType, Player, Unit
from src.engine.board import get_adjacent

def generate_orders_for_unit(state: GameState, unit: Unit) -> List[Order]:
    orders = []
    # Hold
    orders.append(Order(player=unit.player, unit_territory=unit.territory, order_type=OrderType.HOLD))
    
    adj = get_adjacent(unit.territory)
    # Move
    for a in adj:
        orders.append(Order(player=unit.player, unit_territory=unit.territory, order_type=OrderType.MOVE, target=a))
        
    # Support
    for a in adj:
        # Can support a hold in an adjacent territory if there is a unit there
        target_unit = state.get_unit_at(a)
        if target_unit:
            orders.append(Order(player=unit.player, unit_territory=unit.territory, order_type=OrderType.SUPPORT, target=a))
            
        # Can support a move INTO an adjacent territory
        # We need to know who might move there. Since we generate independent orders, we can just say "Support any move into A from any adjacent of A"
        # To constrain action space, we only generate supports for actual units that could move there.
        adj_of_a = get_adjacent(a)
        for origin in adj_of_a:
            if origin != unit.territory:
                origin_unit = state.get_unit_at(origin)
                if origin_unit:
                    orders.append(Order(player=unit.player, unit_territory=unit.territory, order_type=OrderType.SUPPORT, target=a, supported_from=origin))
                    
    return orders

def generate_all_order_sets(state: GameState, player: Player) -> List[List[Order]]:
    units = [u for u in state.units if u.player == player]
    if not units:
        return [[]]
        
    unit_orders = [generate_orders_for_unit(state, u) for u in units]
    # Cartesian product of all possible orders for each unit
    all_combinations = list(itertools.product(*unit_orders))
    
    # Prune combinations? For now just return them. 
    # With 3 units, 1 hold + ~3 moves + ~5 supports = ~9 orders per unit. 9^3 = 729 combinations.
    # Pruning will be important.
    return [list(comb) for comb in all_combinations]
