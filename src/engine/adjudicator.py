from typing import List, Dict, Tuple, Set, Optional
from src.common.schemas import (
    GameState, Order, OrderType, Player, Commitment, CommitmentType, Unit,
    CommitmentOutcome, exchange_leg_due,
)
from src.engine.board import is_adjacent, is_supply_center, get_all_territories

class ResolutionLog:
    def __init__(self):
        self.events = []
    def add(self, event: str):
        self.events.append(event)

def _gave_support(orders: List[Order], supporter: Player,
                  supported_from: Optional[str], target: Optional[str]) -> bool:
    """Did this player actually write the support order they promised?

    supported_from of None is a promise to support a *hold*, and matches an
    order with no origin.
    """
    return any(
        o.player == supporter and o.order_type == OrderType.SUPPORT
        and o.supported_from == supported_from and o.target == target
        for o in orders
    )


def verify_commitments(state: GameState, commitments: List[Commitment], orders: List[Order]) -> List[CommitmentOutcome]:
    outcomes = []

    # What an alliance protects (slide 4): "neither moves into the other's
    # units or centres". Occupied squares plus owned supply centres — NOT
    # every territory the player has ever passed through.
    player_territories = {p: set() for p in Player}
    for t, owner in state.supply_centers.items():
        if owner:
            player_territories[owner].add(t)
    for u in state.units:
        player_territories[u.player].add(u.territory)

    for c in commitments:
        broken_by = set()

        if c.commitment_type == CommitmentType.DMZ:
            for p in c.players:
                for o in orders:
                    if o.player == p and o.order_type == OrderType.MOVE and o.target in (c.dmz_territories or []):
                        broken_by.add(p)

        elif c.commitment_type == CommitmentType.SUPPORT:
            if not _gave_support(orders, c.players[0], c.supported_from, c.target_territory):
                broken_by.add(c.players[0])

        elif c.commitment_type == CommitmentType.EXCHANGE:
            # Two legs, each owed by a different player on a different turn.
            # Only the leg that is due can be broken, so the giver is off the
            # hook once they have paid and the payer is not yet on it.
            leg = exchange_leg_due(c, state.turn)
            if leg == "give":
                if not _gave_support(orders, c.players[0], c.supported_from,
                                     c.target_territory):
                    broken_by.add(c.players[0])
            elif leg == "repay":
                payer = c.players[1]
                for o in orders:
                    if (o.player == payer and o.order_type == OrderType.MOVE
                            and o.target in (c.dmz_territories or [])):
                        broken_by.add(payer)

        elif c.commitment_type == CommitmentType.ALLIANCE:
            # Nobody moves into any other member's units or centres. Written
            # over every ordered pair rather than players[0] and players[1],
            # so a three-way pact binds all three and not just the first two.
            for o in orders:
                if o.order_type != OrderType.MOVE or o.player not in c.players:
                    continue
                for other in c.players:
                    if other != o.player and o.target in player_territories[other]:
                        broken_by.add(o.player)
                        break

        kept = len(broken_by) == 0
        outcomes.append(CommitmentOutcome(commitment=c, kept=kept, broken_by=list(broken_by)))

    return outcomes


def _collect_valid_orders(state: GameState, orders: List[Order], log: ResolutionLog
                          ) -> Dict[str, Order]:
    """One order per unit, with everything illegal thrown away.

    A misbehaving player can send orders for units it does not own, for
    territories that do not exist, two orders for the same unit, or a move to
    the other side of the board. The first order a unit receives is the one
    that counts; anything left unordered holds.
    """
    valid: Dict[str, Order] = {}
    for o in orders:
        unit = state.get_unit_at(o.unit_territory)
        if not unit or unit.player != o.player:
            continue
        if o.unit_territory in valid:
            log.add(f"Duplicate order for {o.unit_territory} ignored")
            continue
        if o.order_type == OrderType.MOVE:
            if not o.target or not is_adjacent(o.unit_territory, o.target):
                log.add(f"Invalid move: {o.unit_territory} to {o.target} (not adjacent)")
                continue
        if o.order_type == OrderType.SUPPORT:
            if not o.target or not is_adjacent(o.unit_territory, o.target):
                log.add(f"Invalid support: {o.unit_territory} cannot support into {o.target} (not adjacent)")
                continue
            if o.supported_from and not is_adjacent(o.supported_from, o.target):
                log.add(f"Invalid support: {o.supported_from} cannot reach {o.target}")
                continue
        valid[o.unit_territory] = o

    for unit in state.units:
        if unit.territory not in valid:
            valid[unit.territory] = Order(player=unit.player, unit_territory=unit.territory,
                                          order_type=OrderType.HOLD)
    return valid


def _adjudicate_moves(valid_orders: Dict[str, Order], log: ResolutionLog
                      ) -> Tuple[Set[str], Set[str]]:
    """Decide which moves go through. Returns (successful origins, dislodged).

    Written as a fixpoint rather than one pass, because whether a move
    succeeds depends on whether the square ahead of it empties, which depends
    on another move. Start with every move optimistically succeeding and
    knock them down until nothing changes; a move can only ever go from
    "succeeds" to "bounces", so this terminates, and a circle of units each
    following the one in front is left standing, which is the correct answer.

    Two rules the previous single pass got wrong:
      * two units ordered at each other do not swap places (a head-to-head is
        won by the stronger, or it bounces);
      * a unit whose own move bounces is not dislodged by whoever was
        following it — the follower bounces instead.
    """
    moves = {loc: o for loc, o in valid_orders.items() if o.order_type == OrderType.MOVE}

    # Supports are cut by any attack except one coming from the very square
    # the support is aimed at.
    attackers_of: Dict[str, List[str]] = {}
    for loc, o in moves.items():
        attackers_of.setdefault(o.target, []).append(loc)

    cut: Set[str] = set()
    for loc, o in valid_orders.items():
        if o.order_type != OrderType.SUPPORT:
            continue
        if any(att != o.target for att in attackers_of.get(loc, [])):
            cut.add(loc)
            log.add(f"Support cut: {loc}")

    move_support: Dict[str, int] = {loc: 0 for loc in moves}
    hold_support: Dict[str, int] = {loc: 0 for loc in valid_orders if loc not in moves}
    for loc, o in valid_orders.items():
        if o.order_type != OrderType.SUPPORT or loc in cut:
            continue
        if o.supported_from:
            target_order = valid_orders.get(o.supported_from)
            if target_order and target_order.order_type == OrderType.MOVE \
                    and target_order.target == o.target:
                move_support[o.supported_from] = move_support.get(o.supported_from, 0) + 1
        elif o.target in hold_support:
            hold_support[o.target] += 1

    strength = {loc: 1 + move_support.get(loc, 0) for loc in moves}
    defence = {loc: 1 + hold_support.get(loc, 0) for loc in hold_support}

    succeeds = {loc: True for loc in moves}
    reasons: Dict[str, str] = {}

    changed = True
    while changed:
        changed = False
        for loc, o in moves.items():
            if not succeeds[loc]:
                continue
            target = o.target

            # 1. Somebody else is pushing just as hard into the same square.
            #    A rival that bounces still blocks, so every mover counts.
            rivals = [r for r in attackers_of[target] if r != loc]
            if any(strength[r] >= strength[loc] for r in rivals):
                succeeds[loc] = False
                reasons[loc] = f"Bounce: {loc} -> {target} (matched by another attacker)"
                changed = True
                continue

            occupant = valid_orders.get(target)
            if occupant is None:
                continue

            # 2. Head to head: the two are walking through each other.
            if occupant.order_type == OrderType.MOVE and occupant.target == loc:
                if occupant.player == o.player:
                    succeeds[loc] = False
                    reasons[loc] = f"Bounce: {loc} and {target} are friendly and cannot swap"
                elif strength[loc] <= strength[target]:
                    succeeds[loc] = False
                    reasons[loc] = (f"Bounce: {loc} <-> {target} head to head "
                                    f"({strength[loc]} vs {strength[target]})")
                if not succeeds[loc]:
                    changed = True
                continue

            # 3. The occupant is leaving, so the square will be empty.
            if occupant.order_type == OrderType.MOVE and succeeds.get(target):
                continue

            # 4. The occupant is staying. Dislodging it needs more strength
            #    than it has defence, and nobody may dislodge their own unit.
            if occupant.player == o.player:
                succeeds[loc] = False
                reasons[loc] = f"Bounce: {loc} cannot dislodge friendly {target}"
                changed = True
            elif strength[loc] <= defence.get(target, 1):
                succeeds[loc] = False
                reasons[loc] = (f"Bounce: {loc} -> {target} "
                                f"(attack {strength[loc]} <= defence {defence.get(target, 1)})")
                changed = True

    dislodged: Set[str] = set()
    for loc in moves:
        if succeeds[loc]:
            log.add(f"Move succeeds: {loc} -> {moves[loc].target}")
            target = moves[loc].target
            occupant = valid_orders.get(target)
            if occupant is not None and not (occupant.order_type == OrderType.MOVE
                                             and succeeds.get(target)):
                dislodged.add(target)
                log.add(f"Dislodged: {target}")
        else:
            log.add(reasons.get(loc, f"Bounce: {loc} -> {moves[loc].target}"))

    return {loc for loc in moves if succeeds[loc]}, dislodged


def resolve(state: GameState, orders: List[Order], commitments: List[Commitment] = None) -> Tuple[GameState, List[CommitmentOutcome], ResolutionLog]:
    if commitments is None:
        commitments = []
    log = ResolutionLog()

    # Verify commitments before resolution based on intended orders
    outcomes = verify_commitments(state, commitments, orders)

    valid_orders = _collect_valid_orders(state, orders, log)
    moved, dislodged = _adjudicate_moves(valid_orders, log)

    new_units: List[Unit] = []
    for loc, o in valid_orders.items():
        if loc in dislodged:
            continue
        destination = o.target if loc in moved else loc
        new_units.append(Unit(player=o.player, territory=destination))

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
                log.add(f"Removal: {p.value} loses {-diff} units")
                units = units[:diff]
            elif diff > 0:
                owned_scs = [t for t, owner in new_sc.items() if owner == p]
                empty_scs = [t for t in owned_scs if not any(u.territory == t for u in units)]
                for i in range(min(diff, len(empty_scs))):
                    log.add(f"Build: {p.value} builds in {empty_scs[i]}")
                    units.append(Unit(player=p, territory=empty_scs[i]))
            final_units.extend(units)
    else:
        # Supply centers only change hands in Autumn
        new_sc = state.supply_centers

    # Occupancy only, and computed last so autumn builds and removals are in
    # it. Supply-centre ownership is the thing that persists; a vacated
    # territory reverts to nobody.
    new_owners = {t: None for t in get_all_territories()}
    for u in final_units:
        new_owners[u.territory] = u.player

    new_state = GameState(
        turn=state.turn + 1,
        units=final_units,
        supply_centers=new_sc,
        territory_owners=new_owners
    )

    return new_state, outcomes, log
