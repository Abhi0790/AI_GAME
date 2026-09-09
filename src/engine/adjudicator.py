from typing import List, Dict, Tuple, Set
from src.common.schemas import GameState, Order, OrderType, Player, Commitment, CommitmentType, Unit
from src.engine.board import is_adjacent, is_supply_center

class ResolutionLog:
    def __init__(self):
        self.events = []
    def add(self, event: str):
        self.events.append(event)

class CommitmentOutcome:
    def __init__(self, commitment: Commitment, kept: bool, broken_by: List[Player]):
        self.commitment = commitment
        self.kept = kept
        self.broken_by = broken_by

def verify_commitments(state: GameState, commitments: List[Commitment], orders: List[Order]) -> List[CommitmentOutcome]:
    outcomes = []
    
    # Precompute player territories for ALLIANCE check
    player_territories = {p: set() for p in Player}
    for t, owner in state.territory_owners.items():
        if owner:
            player_territories[owner].add(t)
    for u in state.units:
        player_territories[u.player].add(u.territory)
            
    for c in commitments:
        broken_by = set()
        
        if c.commitment_type == CommitmentType.DMZ:
            for p in c.players:
                for o in orders:
                    if o.player == p and o.order_type == OrderType.MOVE and o.target in c.dmz_territories:
                        broken_by.add(p)
                        
        elif c.commitment_type == CommitmentType.SUPPORT:
            supporter = c.players[0]
            supported_unit = c.supported_from
            target = c.target_territory
            
            gave_support = False
            for o in orders:
                if o.player == supporter and o.order_type == OrderType.SUPPORT and o.supported_from == supported_unit and o.target == target:
                    gave_support = True
            if not gave_support:
                broken_by.add(supporter)
                
        elif c.commitment_type == CommitmentType.ALLIANCE:
            # Neither attacks the other's units or centres
            p1, p2 = c.players[0], c.players[1]
            for o in orders:
                if o.order_type == OrderType.MOVE:
                    if o.player == p1 and o.target in player_territories[p2]:
                        broken_by.add(p1)
                    if o.player == p2 and o.target in player_territories[p1]:
                        broken_by.add(p2)

        kept = len(broken_by) == 0
        outcomes.append(CommitmentOutcome(commitment=c, kept=kept, broken_by=list(broken_by)))
        
    return outcomes

def resolve(state: GameState, orders: List[Order], commitments: List[Commitment] = None) -> Tuple[GameState, List[CommitmentOutcome], ResolutionLog]:
    if commitments is None:
        commitments = []
    log = ResolutionLog()
    
    # Verify commitments before resolution based on intended orders
    outcomes = verify_commitments(state, commitments, orders)
    
    valid_orders = {}
    for o in orders:
        unit = state.get_unit_at(o.unit_territory)
        if unit and unit.player == o.player:
            if o.order_type == OrderType.MOVE and not is_adjacent(o.unit_territory, o.target):
                log.add(f"Invalid move: {o.unit_territory} to {o.target} (not adjacent)")
                continue
            if o.order_type == OrderType.SUPPORT and not is_adjacent(o.unit_territory, o.target):
                log.add(f"Invalid support: {o.unit_territory} cannot support into {o.target} (not adjacent)")
                continue
            valid_orders[o.unit_territory] = o

    for unit in state.units:
        if unit.territory not in valid_orders:
            valid_orders[unit.territory] = Order(player=unit.player, unit_territory=unit.territory, order_type=OrderType.HOLD)

    attacks_on = {t: [] for t in valid_orders.keys()}
    for loc, o in valid_orders.items():
        if o.order_type == OrderType.MOVE:
            attacks_on.setdefault(o.target, []).append(loc)
            
    cut_supports = set()
    for loc, o in valid_orders.items():
        if o.order_type == OrderType.SUPPORT:
            attackers = attacks_on.get(loc, [])
            for att in attackers:
                if att != o.target:
                    cut_supports.add(loc)
                    log.add(f"Support cut: {loc} (attacked by {att})")
                    break

    supports_for_move = {}
    supports_for_hold = {}
    for loc, o in valid_orders.items():
        if o.order_type == OrderType.SUPPORT and loc not in cut_supports:
            if o.supported_from:
                sup_o = valid_orders.get(o.supported_from)
                if sup_o and sup_o.order_type == OrderType.MOVE and sup_o.target == o.target:
                    supports_for_move.setdefault((o.supported_from, o.target), []).append(loc)
            else:
                supports_for_hold.setdefault(o.target, []).append(loc)
                
    attack_strengths = {}
    for loc, o in valid_orders.items():
        if o.order_type == OrderType.MOVE:
            attack_strengths[loc] = 1 + len(supports_for_move.get((loc, o.target), []))

    defend_strengths = {}
    for loc, o in valid_orders.items():
        if o.order_type != OrderType.MOVE:
            defend_strengths[loc] = 1 + len(supports_for_hold.get(loc, []))

    new_units = []
    dislodged = set()
    
    all_targets = set([o.target for o in valid_orders.values() if o.order_type == OrderType.MOVE])
    for loc in valid_orders.keys():
        if valid_orders[loc].order_type != OrderType.MOVE:
            all_targets.add(loc)
            
    for target in all_targets:
        contenders = [loc for loc, o in valid_orders.items() if o.order_type == OrderType.MOVE and o.target == target]
        defender = target if target in valid_orders and valid_orders[target].order_type != OrderType.MOVE else None
        
        if not contenders:
            if defender:
                new_units.append(Unit(player=valid_orders[defender].player, territory=target))
            continue
            
        max_str = 0
        best_attackers = []
        for c in contenders:
            if attack_strengths[c] > max_str:
                max_str = attack_strengths[c]
                best_attackers = [c]
            elif attack_strengths[c] == max_str:
                best_attackers.append(c)
                
        def_str = defend_strengths.get(defender, 0) if defender else 0
        
        if len(best_attackers) == 1:
            attacker = best_attackers[0]
            if defender and valid_orders[attacker].player == valid_orders[defender].player:
                log.add(f"Bounce: {attacker} cannot dislodge friendly {defender}")
                if defender:
                    new_units.append(Unit(player=valid_orders[defender].player, territory=target))
            elif max_str > def_str:
                log.add(f"Move succeeds: {attacker} -> {target}")
                new_units.append(Unit(player=valid_orders[attacker].player, territory=target))
                if defender:
                    log.add(f"Dislodged: {defender}")
                    dislodged.add(defender)
            else:
                log.add(f"Bounce: {attacker} -> {target} (attack {max_str} <= defense {def_str})")
                if defender:
                    new_units.append(Unit(player=valid_orders[defender].player, territory=target))
        else:
            log.add(f"Bounce: Multiple attackers for {target} with strength {max_str}")
            if defender:
                new_units.append(Unit(player=valid_orders[defender].player, territory=target))

    for loc, o in valid_orders.items():
        if o.order_type == OrderType.MOVE:
            if loc not in dislodged and not any(u.territory == o.target and u.player == o.player for u in new_units):
                origin_taken = any(u.territory == loc for u in new_units)
                if origin_taken:
                    log.add(f"Dislodged while moving: {loc}")
                    dislodged.add(loc)
                else:
                    new_units.append(Unit(player=o.player, territory=loc))

    new_owners = dict(state.territory_owners)
    for u in new_units:
        new_owners[u.territory] = u.player
        
    final_units = list(new_units)
    # Builds/Removals in Autumn (Even turns? Or let's just say every turn for simplicity of the 12-turn game, or strictly autumn)
    # The requirement: "units gained or lost each autumn based on centres held."
    # Let's say odd turns are Spring, even turns are Autumn. After Autumn turn (turn % 2 == 0), adjust units.
    if state.turn % 2 == 0:
        new_sc = dict(state.supply_centers)
        for u in new_units:
            if is_supply_center(u.territory):
                new_sc[u.territory] = u.player
                
        sc_counts = {p: 0 for p in Player}
        for owner in new_sc.values():
            if owner:
                sc_counts[owner] += 1
                
        player_units = {p: [] for p in Player}
        for u in final_units:
            player_units[u.player].append(u)
            
        final_units = []
        for p in Player:
            units = player_units[p]
            diff = sc_counts[p] - len(units)
            if diff < 0:
                log.add(f"Removal: {p} loses {-diff} units")
                units = units[:diff]
            elif diff > 0:
                owned_scs = [t for t, owner in new_sc.items() if owner == p]
                empty_scs = [t for t in owned_scs if not any(u.territory == t for u in units)]
                for i in range(min(diff, len(empty_scs))):
                    log.add(f"Build: {p} builds in {empty_scs[i]}")
                    units.append(Unit(player=p, territory=empty_scs[i]))
            final_units.extend(units)
    else:
        # Supply centers only change hands in Autumn
        new_sc = state.supply_centers

    new_state = GameState(
        turn=state.turn + 1,
        units=final_units,
        supply_centers=new_sc,
        territory_owners=new_owners
    )

    return new_state, outcomes, log
