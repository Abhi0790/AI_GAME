import zlib
from typing import List, Dict, Tuple, Set, Optional
from src.common.schemas import (
    GameState, Order, OrderType, Player, Commitment, CommitmentType, Unit,
    CommitmentOutcome, exchange_leg_due,
)
from src.engine.board import (
    is_adjacent, is_supply_center, get_all_territories, players, home_builds, home_centers,
    fortress_bonus, sieges, has_fleets, is_land, unit_moves, can_reach, convoy_reach,
    province, build_options, built_unit,
)
from src.common.schemas import UnitType

class ResolutionLog:
    def __init__(self):
        self.events = []
    def add(self, event: str):
        self.events.append(event)

def _gave_support(orders: List[Order], supporter: Player,
                  supported_from: Optional[str], target: Optional[str]) -> bool:
    """Did this player write the promised support (supported_from None = a hold)?"""
    return any(
        o.player == supporter and o.order_type == OrderType.SUPPORT
        and o.supported_from == supported_from and o.target == target
        for o in orders
    )


def _shuffle_key(name: str, player: Player, turn: int):
    """A stable pseudo-random key (crc32) that leaves the game's random stream alone."""
    return zlib.crc32(f"{name}|{player.value}|{turn}".encode())


def verify_commitments(state: GameState, commitments: List[Commitment], orders: List[Order]) -> List[CommitmentOutcome]:
    if not commitments:
        return []
    outcomes = []

    # What an alliance protects (occupied squares and owned centres), built lazily.
    player_territories = None

    def territories_of(p):
        nonlocal player_territories
        if player_territories is None:
            player_territories = {q: set() for q in players()}
            for t, owner in state.supply_centers.items():
                if owner:
                    player_territories[owner].add(t)
            for u in state.units:
                player_territories[u.player].add(u.territory)
        return player_territories[p]

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
            # Only the leg that is due can be broken.
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
            # Nobody moves into any other member's units or centres.
            for o in orders:
                if o.order_type != OrderType.MOVE or o.player not in c.players:
                    continue
                for other in c.players:
                    if other != o.player and o.target in territories_of(other):
                        broken_by.add(o.player)
                        break

        kept = len(broken_by) == 0
        outcomes.append(CommitmentOutcome(commitment=c, kept=kept, broken_by=list(broken_by)))

    return outcomes


def fleet_seas(state: GameState) -> Set[str]:
    """Seas holding a fleet, the squares a convoy can pass through."""
    return {u.territory for u in state.units
            if u.unit_type == UnitType.FLEET and not is_land(u.territory)}


def could_move(state: GameState, origin: str, target: str) -> bool:
    """Could the unit at `origin` reach `target`, by convoy if it must?"""
    unit = state.get_unit_at(origin)
    if unit is None:
        return is_adjacent(origin, target)
    return can_reach(unit, target) or (
        unit.unit_type == UnitType.ARMY and target in convoy_reach(origin, fleet_seas(state)))


def _legal_move(state: GameState, unit: Unit, o: Order) -> Optional[Order]:
    """The move with its coast filled in, or None if the unit cannot make it."""
    if unit.unit_type == UnitType.ARMY:
        return o if o.target and could_move(state, unit.territory, o.target) else None
    coasts = [loc for loc in unit_moves(unit) if province(loc) == o.target]
    if o.coast:
        return o if f"{o.target}/{o.coast}" in coasts else None
    if len(coasts) != 1:
        return None
    return o.model_copy(update={"coast": coasts[0].partition("/")[2] or None})


def _collect_valid_orders(state: GameState, orders: List[Order], log: ResolutionLog
                          ) -> Dict[str, Order]:
    """One order per unit (the first it receives), illegal ones dropped; the rest hold."""
    valid: Dict[str, Order] = {}
    for o in orders:
        unit = state.get_unit_at(o.unit_territory)
        if not unit or unit.player != o.player:
            continue
        if o.unit_territory in valid:
            log.add(f"Duplicate order for {o.unit_territory} ignored")
            continue
        if o.order_type == OrderType.MOVE:
            legal = _legal_move(state, unit, o)
            if legal is None:
                log.add(f"Invalid move: {o.unit_territory} to {o.target} (not adjacent)")
                continue
            o = legal
        if o.order_type == OrderType.SUPPORT:
            if not o.target or not can_reach(unit, o.target):
                log.add(f"Invalid support: {o.unit_territory} cannot support into {o.target} (not adjacent)")
                continue
            if o.supported_from and not could_move(state, o.supported_from, o.target):
                log.add(f"Invalid support: {o.supported_from} cannot reach {o.target}")
                continue
        if o.order_type == OrderType.CONVOY:
            army = state.get_unit_at(o.supported_from or "")
            if (unit.unit_type != UnitType.FLEET or is_land(unit.territory)
                    or army is None or army.unit_type != UnitType.ARMY
                    or not o.target or not is_land(o.target)):
                log.add(f"Invalid convoy: {o.unit_territory}")
                continue
        valid[o.unit_territory] = o

    for unit in state.units:
        if unit.territory not in valid:
            valid[unit.territory] = Order(player=unit.player, unit_territory=unit.territory,
                                          order_type=OrderType.HOLD)
    return valid


def _convoy_route(valid_orders: Dict[str, Order], loc: str, broken: Set[str]) -> Set[str]:
    """The fleets convoying the army at `loc`, or empty if they do not reach its target."""
    o = valid_orders[loc]
    fleets = {f for f, c in valid_orders.items() if c.order_type == OrderType.CONVOY
              and c.supported_from == loc and c.target == o.target and f not in broken}
    return fleets if o.target in convoy_reach(loc, fleets) else set()


def _movement(state: GameState, valid_orders: Dict[str, Order], log: ResolutionLog
              ) -> Tuple[Dict[str, Order], Set[str], Set[str], Set[str]]:
    """Moves with convoys: (orders as played, moved, dislodged, convoyed).

    An army goes by sea when it cannot walk there or its own side orders the
    convoy. A convoy whose fleet is dislodged fails, and the army stays put.
    """
    convoyed = set()
    if has_fleets():
        for loc, o in valid_orders.items():
            unit = state.get_unit_at(loc)
            if o.order_type == OrderType.MOVE and unit.unit_type == UnitType.ARMY and (
                    not can_reach(unit, o.target) or any(
                        c.order_type == OrderType.CONVOY and c.player == o.player
                        and c.supported_from == loc and c.target == o.target
                        for c in valid_orders.values())):
                convoyed.add(loc)

    broken: Set[str] = set()
    while True:
        played = dict(valid_orders)
        routes = {loc: _convoy_route(valid_orders, loc, broken) for loc in convoyed}
        for loc, route in routes.items():
            if not route:
                played[loc] = Order(player=played[loc].player, unit_territory=loc,
                                    order_type=OrderType.HOLD)
        routes = {loc: r for loc, r in routes.items() if r}
        pass_log = ResolutionLog()
        moved, dislodged = _adjudicate_moves(played, pass_log, routes)
        newly = {f for r in routes.values() for f in r if f in dislodged} - broken
        if not newly:
            break
        broken |= newly
    for loc in sorted(convoyed - set(routes)):
        log.add(f"Convoy fails: {loc} -> {valid_orders[loc].target}")
    log.events += pass_log.events
    return played, moved, dislodged, set(routes)


def _adjudicate_moves(valid_orders: Dict[str, Order], log: ResolutionLog,
                      routes: Optional[Dict[str, Set[str]]] = None
                      ) -> Tuple[Set[str], Set[str]]:
    """Decide which moves go through. Returns (successful origins, dislodged).

    `routes`: convoyed army -> its convoying fleets. A fixpoint: every move
    starts as succeeding and is knocked down until nothing changes; it is
    rerun with dislodged supporters' support voided.
    """
    routes = routes or {}
    moves = {loc: o for loc, o in valid_orders.items() if o.order_type == OrderType.MOVE}

    # Supports are cut by any foreign attack except one coming from the very
    # square the support is aimed at, and a convoyed attack on a support aimed
    # at one of its own convoying fleets.
    attackers_of: Dict[str, List[str]] = {}
    for loc, o in moves.items():
        attackers_of.setdefault(o.target, []).append(loc)

    cut: Set[str] = set()
    for loc, o in valid_orders.items():
        if o.order_type != OrderType.SUPPORT:
            continue
        if any(att != o.target and moves[att].player != o.player
               and o.target not in routes.get(att, ())
               for att in attackers_of.get(loc, [])):
            cut.add(loc)
            log.add(f"Support cut: {loc}")

    voided: Set[str] = set()      # supporters found dislodged on a previous pass
    while True:
        succeeds, reasons, dislodged = _settle(valid_orders, moves, attackers_of,
                                               cut | voided, set(routes))
        # Support from a unit that ends up dislodged never happened.
        newly = {loc for loc in dislodged
                 if valid_orders[loc].order_type == OrderType.SUPPORT
                 and loc not in cut and loc not in voided}
        if not newly:
            break
        voided |= newly

    for loc in sorted(voided):
        log.add(f"Support voided: {loc} was dislodged")
    for loc in moves:
        if succeeds[loc]:
            log.add(f"Move succeeds: {loc} -> {moves[loc].target}")
            if moves[loc].target in dislodged:
                log.add(f"Dislodged: {moves[loc].target}")
        else:
            log.add(reasons.get(loc, f"Bounce: {loc} -> {moves[loc].target}"))
    return {loc for loc in moves if succeeds[loc]}, dislodged


def _settle(valid_orders, moves, attackers_of, cut, convoyed=frozenset()):
    """One pass of the bounce fixpoint with a fixed set of cut supports."""

    def swapping(a: str, b: str) -> bool:
        """a and b walk into each other; a convoyed unit passes by sea instead."""
        return (a in moves and b in moves and moves[a].target == b
                and moves[b].target == a and not {a, b} & convoyed)

    move_support: Dict[str, int] = {loc: 0 for loc in moves}
    supporters: Dict[str, List[Player]] = {loc: [] for loc in moves}
    hold_support: Dict[str, int] = {loc: 0 for loc in valid_orders if loc not in moves}
    for loc, o in valid_orders.items():
        if o.order_type != OrderType.SUPPORT or loc in cut:
            continue
        if o.supported_from:
            target_order = valid_orders.get(o.supported_from)
            if target_order and target_order.order_type == OrderType.MOVE \
                    and target_order.target == o.target:
                move_support[o.supported_from] = move_support.get(o.supported_from, 0) + 1
                supporters[o.supported_from].append(o.player)
        elif o.target in hold_support:
            hold_support[o.target] += 1

    strength = {loc: 1 + move_support.get(loc, 0) for loc in moves}
    defence = {loc: 1 + hold_support.get(loc, 0) for loc in hold_support}

    def attack(loc: str, defender: Player) -> int:
        """Strength against a unit: its owner's supports never help dislodge it (DATC 6.D.12)."""
        return 1 + sum(1 for p in supporters[loc] if p != defender)

    succeeds = {loc: True for loc in moves}
    reasons: Dict[str, str] = {}

    changed = True
    while changed:
        changed = False
        for loc, o in moves.items():
            if not succeeds[loc]:
                continue
            target = o.target
            occupant = valid_orders.get(target)

            # 1. Somebody else is pushing just as hard into the same square.
            #    A rival that bounces still blocks, unless it lost a head-to-head
            #    with the unit leaving that square (DATC 6.E.1).
            rivals = [r for r in attackers_of[target] if r != loc and not (
                swapping(target, r) and succeeds.get(target, False))]
            # Against a unit that stays, its owner's supports do not count here
            # either, or they would beat a rival they cannot dislodge past (6.E.7).
            leaving = (occupant is not None and occupant.order_type == OrderType.MOVE
                       and not swapping(loc, target) and succeeds.get(target, False))
            mine = strength[loc] if occupant is None or leaving else attack(loc, occupant.player)
            if any(strength[r] >= mine for r in rivals):
                succeeds[loc] = False
                reasons[loc] = f"Bounce: {loc} -> {target} (matched by another attacker)"
                changed = True
                continue

            if occupant is None:
                continue

            # 2. Head to head: the two are walking through each other.
            if swapping(loc, target):
                if occupant.player == o.player:
                    succeeds[loc] = False
                    reasons[loc] = f"Bounce: {loc} and {target} are friendly and cannot swap"
                elif attack(loc, occupant.player) <= strength[target]:
                    succeeds[loc] = False
                    reasons[loc] = (f"Bounce: {loc} <-> {target} head to head "
                                    f"({attack(loc, occupant.player)} vs {strength[target]})")
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
            elif attack(loc, occupant.player) <= defence.get(target, 1) + fortress_bonus(target):
                succeeds[loc] = False
                reasons[loc] = (f"Bounce: {loc} -> {target} "
                                f"(attack {attack(loc, occupant.player)} <= defence "
                                f"{defence.get(target, 1) + fortress_bonus(target)})")
                changed = True

    dislodged: Set[str] = set()
    for loc in moves:
        if not succeeds[loc]:
            continue
        target = moves[loc].target
        occupant = valid_orders.get(target)
        if occupant is not None and not (occupant.order_type == OrderType.MOVE
                                         and succeeds.get(target)):
            dislodged.add(target)

    return succeeds, reasons, dislodged


def _retreats(state: GameState, valid_orders: Dict[str, Order], moved: Set[str],
              dislodged: Set[str], survivors: List[Unit], convoyed: Set[str], choose,
              log: ResolutionLog) -> List[Unit]:
    """Each dislodged unit retreats or disbands; two retreating to one province both disband."""
    occupied = {u.territory for u in survivors}
    standoffs = {o.target for loc, o in valid_orders.items()
                 if o.order_type == OrderType.MOVE and loc not in moved}
    origin_of = {valid_orders[loc].target: loc for loc in moved if loc not in convoyed}
    wanted: Dict[str, List[Unit]] = {}
    for loc in sorted(dislodged):
        unit = state.get_unit_at(loc)
        options = retreat_options(unit, occupied, origin_of.get(loc), standoffs)
        pick = choose(unit.player, "retreat", options, 1,
                      state.model_copy(update={"units": survivors}), state.turn)[:1] if options else []
        if pick and pick[0] in options:
            to = pick[0]
            wanted.setdefault(province(to), []).append(unit.model_copy(update={
                "territory": province(to), "coast": to.partition("/")[2] or None}))
            log.add(f"Retreat: {loc} -> {to}")
        else:
            log.add(f"Disbanded: {loc} (no retreat)")
    back = []
    for square, units in wanted.items():
        if len(units) > 1:
            log.add(f"Disbanded: {len(units)} units retreating to {square} collide")
        else:
            back.extend(units)
    return back


# Every call from anywhere. The runner reads the difference around each agent
# call to charge that seat's total work, budgeted or not.
resolve_calls = 0


def default_choice(player: Player, kind: str, options: List[str], count: int,
                   state: GameState, turn: int) -> List[str]:
    """Pick `count` of `options` in a stable pseudo-random order (hypothetical
    turns), at most one build per province."""
    ranked = sorted(options, key=lambda t: _shuffle_key(t, player, turn))
    if kind != "build":
        return ranked[:count]
    picks: Dict[str, str] = {}
    for t in ranked:
        picks.setdefault(province(t.removeprefix("F ")), t)
    return list(picks.values())[:count]


def retreat_options(unit: Unit, occupied: Set[str], attacker_origin: Optional[str],
                    standoffs: Set[str]) -> List[str]:
    """Where a dislodged unit may go: a square it could move to that is empty,
    not where its attacker came from, and not left empty by a standoff."""
    return [t for t in unit_moves(unit) if province(t) not in occupied
            and province(t) != attacker_origin and province(t) not in standoffs]


def resolve(state: GameState, orders: List[Order], commitments: List[Commitment] = None,
            chooser=None) -> Tuple[GameState, List[CommitmentOutcome], ResolutionLog]:
    """Movement, retreats, then (in autumn) ownership and adjustments.

    `chooser(player, kind, options, count, state, turn)` returns the squares a
    player picks: kind is "retreat" (count 1; empty list disbands), "build" or
    "disband" (up to count; building fewer waives the rest).
    """
    global resolve_calls
    resolve_calls += 1
    if commitments is None:
        commitments = []
    choose = chooser or default_choice
    log = ResolutionLog()

    # Verify commitments before resolution based on intended orders
    outcomes = verify_commitments(state, commitments, orders)

    valid_orders = _collect_valid_orders(state, orders, log)
    valid_orders, moved, dislodged, convoyed = _movement(state, valid_orders, log)

    new_units: List[Unit] = []
    for loc, o in valid_orders.items():
        if loc in dislodged:
            continue
        unit = state.get_unit_at(loc)
        new_units.append(unit.model_copy(update={"territory": o.target, "coast": o.coast})
                         if loc in moved else unit)

    if dislodged:
        new_units += _retreats(state, valid_orders, moved, dislodged, new_units, convoyed,
                               choose, log)

    final_units = list(new_units)
    if state.turn % 2 == 0:
        new_sc = dict(state.supply_centers)
        for u in new_units:
            sitting = state.get_unit_at(u.territory)
            if is_supply_center(u.territory) and (
                    not sieges() or (sitting is not None and sitting.player == u.player)):
                new_sc[u.territory] = u.player

        sc_counts = {p: 0 for p in players()}
        for owner in new_sc.values():
            if owner:
                sc_counts[owner] += 1

        player_units = {p: [] for p in players()}
        for u in final_units:
            player_units[u.player].append(u)

        interim = GameState(turn=state.turn, units=final_units, supply_centers=new_sc,
                            territory_owners=state.territory_owners)
        final_units = []
        for p in players():
            units = player_units[p]
            diff = sc_counts[p] - len(units)
            if diff < 0:
                squares = [u.territory for u in units]
                gone = set(choose(p, "disband", squares, -diff, interim, state.turn))
                # A chooser that names too few, or squares it does not hold,
                # is made up with the default so the count is always right.
                gone = {t for t in gone if t in squares}
                if len(gone) < -diff:
                    gone |= set(default_choice(p, "disband", [t for t in squares if t not in gone],
                                               -diff - len(gone), interim, state.turn))
                for t in sorted(gone):
                    log.add(f"Removal: {p.value} disbands {t}")
                units = [u for u in units if u.territory not in gone]
            elif diff > 0:
                owned_scs = [t for t, owner in new_sc.items() if owner == p
                             and (not home_builds() or t in home_centers()[p])]
                # Any unit, not just my own: under sieges an enemy can sit on my centre.
                standing = {u.territory for u in interim.units}
                empty_scs = [t for t in owned_scs if t not in standing]
                options = [b for t in empty_scs for b in build_options(t)]
                picks = {}
                for b in choose(p, "build", options, diff, interim, state.turn):
                    if b in options and len(picks) < diff:
                        picks.setdefault(province(b.removeprefix("F ")), b)
                for b in picks.values():
                    unit = built_unit(p, b)
                    log.add(f"Build: {p.value} builds in {unit.territory}" if b == unit.territory
                            else f"Build: {p.value} builds a fleet in {unit.location}")
                    units.append(unit)
                if len(picks) < min(diff, len(empty_scs)):
                    log.add(f"Build: {p.value} waives {min(diff, len(empty_scs)) - len(picks)}")
            final_units.extend(units)
    else:
        # Supply centers only change hands in Autumn
        new_sc = state.supply_centers

    # Occupancy after builds and removals; centre ownership is what persists.
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
