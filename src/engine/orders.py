from typing import List, Tuple
import itertools
import random
import zlib
from src.common.schemas import GameState, Order, OrderType, Player, Unit, UnitType
from src.engine.board import active, province, unit_moves, can_reach, convoy_reach, is_land
from src.engine.adjudicator import fleet_seas, could_move


def generate_orders_for_unit(state: GameState, unit: Unit, by_sea: bool = True) -> List[Order]:
    """Hold, every move, every support, and for a fleet at sea every convoy.
    `by_sea=False` leaves out an army's convoyed moves (paired with a fleet instead)."""
    def order(kind, target=None, frm=None, coast=None):
        return Order(player=unit.player, unit_territory=unit.territory, order_type=kind,
                     target=target, supported_from=frm, coast=coast)

    orders = [order(OrderType.HOLD)]
    steps = unit_moves(unit)
    for loc in steps:
        coast = loc.partition("/")[2] or None
        orders.append(order(OrderType.MOVE, province(loc), coast=coast))
    seas = fleet_seas(state)
    if unit.unit_type == UnitType.ARMY and by_sea:
        reached = {province(loc) for loc in steps}
        orders += [order(OrderType.MOVE, t) for t in sorted(convoy_reach(unit.territory, seas))
                   if t not in reached]

    reach = list(dict.fromkeys(province(loc) for loc in steps))
    for a in reach:
        if state.get_unit_at(a):
            orders.append(order(OrderType.SUPPORT, a))
        for origin in active().adjacency[a]:
            other = state.get_unit_at(origin)
            if origin != unit.territory and other and can_reach(other, a):
                orders.append(order(OrderType.SUPPORT, a, origin))
        if seas:
            orders += [order(OrderType.SUPPORT, a, other.territory) for other in state.units
                       if other.unit_type == UnitType.ARMY and other.territory != unit.territory
                       and not can_reach(other, a) and could_move(state, other.territory, a)]

    if unit.unit_type == UnitType.FLEET and not is_land(unit.territory):
        for army in state.units:
            if army.unit_type == UnitType.ARMY and unit.territory in _chain(army.territory, seas):
                orders += [order(OrderType.CONVOY, t, army.territory)
                           for t in sorted(convoy_reach(army.territory, seas))]
    return orders


def order_label(o: Order) -> str:
    """One order in words."""
    to = f"{o.target}/{o.coast}" if o.coast else o.target
    if o.order_type == OrderType.HOLD:
        return f"{o.unit_territory} holds"
    if o.order_type == OrderType.MOVE:
        return f"{o.unit_territory} -> {to}"
    if o.order_type == OrderType.CONVOY:
        return f"{o.unit_territory} convoys {o.supported_from} -> {o.target}"
    if o.supported_from:
        return f"{o.unit_territory} supports {o.supported_from} -> {o.target}"
    return f"{o.unit_territory} supports {o.target} holding"


def _convoy_pairs(unit_orders: List[List[Order]], seen: set) -> List[List[Order]]:
    """Every unit holds but one army and the fleet carrying it, both ordered."""
    baseline = [opts[0] for opts in unit_orders]
    at = {opts[0].unit_territory: j for j, opts in enumerate(unit_orders)}
    out = []
    for i, opts in enumerate(unit_orders):
        for o in opts:
            j = at.get(o.supported_from) if o.order_type == OrderType.CONVOY else None
            if j is None or o.target not in convoy_reach(o.supported_from, {o.unit_territory}):
                continue
            comb = list(baseline)
            comb[i] = o
            comb[j] = Order(player=o.player, unit_territory=o.supported_from,
                            order_type=OrderType.MOVE, target=o.target)
            if tuple(comb) not in seen:
                seen.add(tuple(comb))
                out.append(comb)
    return out


def _chain(land: str, seas: set) -> set:
    """Fleet-held seas connected to a coastal province."""
    adj = active().adjacency
    reached = {s for s in adj.get(land, []) if s in seas}
    frontier = list(reached)
    while frontier:
        for n in adj[frontier.pop()]:
            if n in seas and n not in reached:
                reached.add(n)
                frontier.append(n)
    return reached


def _signature(state: GameState) -> Tuple:
    """Everything order generation depends on: who is standing where, and on
    which map. Two boards can share territory names and differ in adjacency."""
    return (tuple(sorted((u.territory, u.player.value) for u in state.units)),
            active().fingerprint)


# Enumeration is pure in (occupancy, player), so it is cached.
_SETS_CACHE: dict = {}
_CACHE_LIMIT = 256

# Past this many joint sets, sample (seeded by position) instead of enumerating,
# always keeping the all-hold set and every single-unit action.
# ponytail: uniform sampling above the cap; bias by the opponent model if late-game play looks weak.
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
        unit_orders = [generate_orders_for_unit(state, u, by_sea=False) for u in units]
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

        result += _convoy_pairs(unit_orders, {tuple(c) for c in result})

    if len(_SETS_CACHE) > _CACHE_LIMIT:
        _SETS_CACHE.clear()
    _SETS_CACHE[key] = result
    return result
