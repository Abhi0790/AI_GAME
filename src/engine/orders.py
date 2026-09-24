from typing import List, Set, Tuple
import itertools
import random
import zlib
from src.common.schemas import GameState, Order, OrderType, Player, Unit
from src.engine.board import active, get_adjacent

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


def _signature(state: GameState) -> Tuple:
    """Everything order generation depends on: who is standing where, and on
    which map. Two boards can share territory names and differ in adjacency."""
    return (tuple(sorted((u.territory, u.player.value) for u in state.units)),
            active().fingerprint)


# Enumeration is pure in (occupancy, player) and the planner asks for the same
# player's sets dozens of times per turn while sampling. One dict turns that
# back into one call.
_SETS_CACHE: dict = {}
_CACHE_LIMIT = 256

# Two units give ~81 joint order sets; five give ~59,000, and the planner
# scores every one of them twice while pruning. The cartesian product is only
# tractable at the opening, so past the cap the set is sampled instead of
# enumerated — the whole-hold set and every single-unit action are always kept,
# because those are the ones a human would check first.
#
# The sample is seeded from the position, so the cache is a pure function and
# consuming it never shifts the game's random stream.
#
# ponytail: uniform sampling above the cap. If late-game play looks weak,
# bias the sample by the opponent model before making the cap bigger.
MAX_ORDER_SETS = 300


def generate_all_order_sets(state: GameState, player: Player) -> List[List[Order]]:
    key = (_signature(state), player)
    hit = _SETS_CACHE.get(key)
    if hit is not None:
        return hit

    units = [u for u in state.units if u.player == player]
    if not units:
        result = [[]]
    else:
        unit_orders = [generate_orders_for_unit(state, u) for u in units]
        total = 1
        for opts in unit_orders:
            total *= len(opts)

        if total <= MAX_ORDER_SETS:
            result = [list(comb) for comb in itertools.product(*unit_orders)]
        else:
            baseline = [opts[0] for opts in unit_orders]  # every unit holds
            seen = {tuple(baseline)}
            result = [list(baseline)]
            for i, opts in enumerate(unit_orders):       # one unit acts alone
                for o in opts[1:]:
                    comb = list(baseline)
                    comb[i] = o
                    if tuple(comb) not in seen:
                        seen.add(tuple(comb))
                        result.append(comb)
            rng = random.Random(zlib.crc32(repr((key[0][0], player.value)).encode()))
            while len(result) < MAX_ORDER_SETS:
                comb = [rng.choice(opts) for opts in unit_orders]
                if tuple(comb) not in seen:
                    seen.add(tuple(comb))
                    result.append(comb)

    if len(_SETS_CACHE) > _CACHE_LIMIT:
        _SETS_CACHE.clear()
    _SETS_CACHE[key] = result
    return result
