"""The map, the seats and the victory condition.

One `Board` is active at a time; `use_board` swaps it for a block. The default
is `ring_board(4)`, which reproduces the original 12-territory map.

    home ring     R1 - R2 - B1 - B2 - G1 - G2 - Y1 - Y2 - (back to R1)
    middle ring   N1 - C1 - N2 - C2 - (back to N1)
    N1 between R2 and B1        C1 between B2 and G1
    N2 between G2 and Y1        C2 between Y2 and R1

`_check` verifies the map is symmetric and the adjudicator is equivariant under
that symmetry. It does not establish that seats win equally often: measured over
60 games with identical personas, the {Blue, Gold} positions finish ~0.45
centres ahead of {Red, Green}. That asymmetry is in the agent layer.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import cos, pi, sin
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from src.common.schemas import Player

# Gold takes Y because Green already has G.
SEAT_CODES = {
    Player.RED: "R", Player.BLUE: "B", Player.GREEN: "G", Player.GOLD: "Y",
    Player.PURPLE: "P", Player.ORANGE: "O", Player.TEAL: "T", Player.PINK: "K",
}

MIN_SEATS = 2
MAX_SEATS = len(Player)

DEFAULT_SEATS = 4
DEFAULT_HOMES_PER_PLAYER = 2
DEFAULT_MAX_TURNS = 12
# Share of the board's centres that wins outright when no threshold is given.
DEFAULT_WIN_FRACTION = 0.5

_CANVAS = 500.0
_HOME_RADIUS = 175.0
_GAP_RADIUS = 75.0


@dataclass(frozen=True)
class Board:
    """One map, one set of seats, one victory condition, as plain data.

    Frozen: `_centre_set` and `fingerprint` are derived once at construction,
    and the fingerprint keys the order-set cache, so a board edited in place
    would serve stale legal moves. Build a new Board instead.
    """
    adjacency: Dict[str, List[str]]
    supply_centers: List[str]
    home_centers: Dict[Player, List[str]]      # seat -> where its units start
    win_centers: int
    max_turns: int = DEFAULT_MAX_TURNS
    # True: build only on owned, empty home centres (standard Diplomacy).
    # False: build on any owned, empty centre.
    home_builds: bool = True
    # territory -> (x, y), for the dashboard
    positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Territory permutations carrying the board onto itself. Empty means
    # symmetry is not claimed.
    symmetries: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        put = object.__setattr__
        put(self, "home_centers",
            {Player(p): list(t) for p, t in self.home_centers.items()})
        put(self, "positions", {t: tuple(xy) for t, xy in self.positions.items()})
        put(self, "_centre_set", set(self.supply_centers))
        # Identifies the map for caches keyed on more than unit placement.
        put(self, "fingerprint", hash(tuple(sorted(
            (t, tuple(n)) for t, n in self.adjacency.items()))))
        _check(self)

    @property
    def players(self) -> List[Player]:
        """The seats, in play order."""
        return list(self.home_centers)

    @property
    def territories(self) -> List[str]:
        return list(self.adjacency)

    def to_dict(self) -> dict:
        return {
            "adjacency": {t: list(n) for t, n in self.adjacency.items()},
            "supply_centers": list(self.supply_centers),
            "home_centers": {p.value: list(t) for p, t in self.home_centers.items()},
            "win_centers": self.win_centers,
            "max_turns": self.max_turns,
            "home_builds": self.home_builds,
            "positions": {t: list(xy) for t, xy in self.positions.items()},
            "symmetries": [dict(s) for s in self.symmetries],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Board":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _check(b: Board) -> None:
    """Validate a board at construction. Raises AssertionError."""
    names = set(b.adjacency)
    assert len(names) >= 2, "a board needs at least two territories"
    for t, neighbours in b.adjacency.items():
        assert t not in neighbours, f"{t} is adjacent to itself"
        assert len(set(neighbours)) == len(neighbours), f"{t} lists a neighbour twice"
        for n in neighbours:
            assert n in names, f"{t} -> {n} is not a territory"
            assert t in b.adjacency[n], f"{t} -> {n} is not mutual"

    # A disconnected board is two independent games: seats in one component can
    # never reach the other, so nobody there can be attacked or eliminated.
    # Homes need not be connected to each other -- a split seat is a legitimate
    # setup, and one that cannot support its own units.
    seen_terr, frontier = {next(iter(names))}, [next(iter(names))]
    while frontier:
        for n in b.adjacency[frontier.pop()]:
            if n not in seen_terr:
                seen_terr.add(n)
                frontier.append(n)
    assert seen_terr == names, (
        f"board is not connected: {sorted(names - seen_terr)} unreachable from "
        f"{sorted(seen_terr)[0]}")

    centres = set(b.supply_centers)
    assert centres <= names, f"supply centres off the board: {sorted(centres - names)}"

    seen: Dict[str, Player] = {}
    for p, homes in b.home_centers.items():
        assert homes, f"{p.value} has no home centre to start on"
        for t in homes:
            assert t in names, f"{p.value} starts on {t}, which is not a territory"
            assert t in centres, f"{p.value}'s home {t} is not a supply centre"
            assert t not in seen, f"{t} is a home centre for both {seen[t].value} and {p.value}"
            seen[t] = p
    assert len(b.home_centers) >= 2, "a game needs at least two seats"

    assert 1 <= b.win_centers <= len(centres), (
        f"win_centers={b.win_centers} is unreachable on a board with "
        f"{len(centres)} supply centres")
    start = max(len(homes) for homes in b.home_centers.values())
    assert b.win_centers > start, (
        f"win_centers={b.win_centers} is already held at setup by a player "
        f"starting on {start} centres")
    assert b.max_turns >= 1, "a game needs at least one turn"

    if b.positions:
        missing = names - set(b.positions)
        assert not missing, f"no position for {sorted(missing)}"

    if b.symmetries:
        _check_fairness(b, names, centres)


def _check_fairness(b: Board, names: set, centres: set) -> None:
    """Check the seat positions are interchangeable.

    A statement about the graph, not about outcomes: neighbour sets are
    compared, and passing says nothing about the planner. Not evidence that a
    seat-keyed table is unbiased.
    """
    home_of = {t: p for p, homes in b.home_centers.items() for t in homes}
    moves: List[Dict[Player, Player]] = []
    for i, sigma in enumerate(b.symmetries):
        assert set(sigma) == names, f"symmetry {i} does not cover the board"
        assert set(sigma.values()) == names, f"symmetry {i} is not a bijection"
        for t, image in sigma.items():
            assert {sigma[n] for n in b.adjacency[t]} == set(b.adjacency[image]), (
                f"symmetry {i} does not preserve adjacency at {t}")
            assert (t in centres) == (image in centres), (
                f"symmetry {i} moves {t} onto a square of a different kind")
        permutation: Dict[Player, Player] = {}
        for t, owner in home_of.items():
            target = home_of.get(sigma[t])
            assert target is not None, f"symmetry {i} moves {t} off a home centre"
            assert permutation.setdefault(owner, target) == target, (
                f"symmetry {i} splits {owner.value}'s home centres between players")
        moves.append(permutation)

    # Transitive on players: every seat reachable from any other.
    first = b.players[0]
    orbit, frontier = {first}, [first]
    while frontier:
        p = frontier.pop()
        for permutation in moves:
            q = permutation[p]
            if q not in orbit:
                orbit.add(q)
                frontier.append(q)
    assert orbit == set(b.home_centers), (
        f"seats are not interchangeable: only {sorted(x.value for x in orbit)} "
        f"reachable from {first.value}")


def _polar(angle_index: float, steps: int, radius: float) -> Tuple[float, float]:
    theta = -pi / 2 + 2 * pi * angle_index / steps
    c = _CANVAS / 2
    return (round(c + radius * cos(theta), 1), round(c + radius * sin(theta), 1))


def ring_board(seats: int | Sequence[Player] = DEFAULT_SEATS,
               homes_per_player: int = DEFAULT_HOMES_PER_PLAYER,
               *,
               win_centers: Optional[int] = None,
               win_fraction: float = DEFAULT_WIN_FRACTION,
               max_turns: int = DEFAULT_MAX_TURNS,
               gap_centres: Optional[bool] = None,
               home_builds: bool = True) -> Board:
    """Build a ring map for any seat count.

    Home centres form one ring, `homes_per_player` per seat. Each boundary
    between neighbouring seats gets one middle territory, and those form a ring
    of their own.

    `gap_centres` decides what the middle is made of, and every seat must face
    the same thing or `_check` rejects the board:

        None   alternate centre/corridor when the seat count is even, all
               centres when it is odd (the alternation cannot close otherwise)
        True   every middle territory is a neutral supply centre
        False  every middle territory is a plain corridor

    The default makes the contested pool jump with seat parity, which
    confounds any comparison across table sizes; pass True or False to hold it
    fixed instead.
    """
    # Validate before slicing: list(Player)[:99] truncates silently.
    n = seats if isinstance(seats, int) else len(seats)
    if not MIN_SEATS <= n <= MAX_SEATS:
        raise ValueError(f"a ring board seats {MIN_SEATS}..{MAX_SEATS} players, not {n}")
    players = list(Player)[:seats] if isinstance(seats, int) else list(seats)
    if len(set(players)) != n:
        raise ValueError("the same player cannot hold two seats")
    if homes_per_player < 1:
        raise ValueError("every seat needs at least one home centre")

    if not 0 < win_fraction <= 1:
        raise ValueError(f"win_fraction must be in (0, 1], not {win_fraction}")
    homes = {p: [f"{SEAT_CODES[p]}{j + 1}" for j in range(homes_per_player)]
             for p in players}
    ring = [t for p in players for t in homes[p]]            # home ring, in seat order
    k = len(ring)

    # Gap j sits between seat j's last home and seat j+1's first.
    if gap_centres is None:
        alternating = n % 2 == 0
        gap_is_centre = [(j % 2 == 0) if alternating else True for j in range(n)]
    else:
        gap_is_centre = [bool(gap_centres)] * n
    n_names, c_names = 0, 0
    gaps: List[str] = []
    for j in range(n):
        if gap_is_centre[j]:
            n_names += 1
            gaps.append(f"N{n_names}")
        else:
            c_names += 1
            gaps.append(f"C{c_names}")

    adjacency: Dict[str, List[str]] = {t: [] for t in ring + gaps}

    def link(a: str, b: str) -> None:
        if a != b and b not in adjacency[a]:
            adjacency[a].append(b)
            adjacency[b].append(a)

    for i in range(k):                                        # the home ring
        link(ring[i], ring[(i + 1) % k])
    for j in range(n):                                        # the middle ring
        link(gaps[j], gaps[(j + 1) % n])
        link(gaps[j], ring[(j * homes_per_player + homes_per_player - 1) % k])
        link(gaps[j], ring[(j * homes_per_player + homes_per_player) % k])

    supply_centers = ring + [g for j, g in enumerate(gaps) if gap_is_centre[j]]

    positions = {t: _polar(i, k, _HOME_RADIUS) for i, t in enumerate(ring)}
    positions.update({
        g: _polar(j * homes_per_player + homes_per_player - 0.5, k, _GAP_RADIUS)
        for j, g in enumerate(gaps)
    })

    # Reflection r_j flips the ring about the axis through gap j: seat i lands
    # on seat 2j+1-i, gap g on gap 2j-g. Together they are transitive on seats.
    symmetries = []
    for j in range(n):
        sigma = {}
        for i, p in enumerate(players):
            for h in range(homes_per_player):
                image = players[(2 * j + 1 - i) % n]
                sigma[homes[p][h]] = homes[image][homes_per_player - 1 - h]
        for g in range(n):
            sigma[gaps[g]] = gaps[(2 * j - g) % n]
        symmetries.append(sigma)

    return Board(
        adjacency=adjacency,
        supply_centers=supply_centers,
        home_centers=homes,
        # A share of the centres, floored above what a seat already starts with.
        win_centers=(win_centers if win_centers is not None
                     else max(int(len(supply_centers) * win_fraction + 1e-9),
                              homes_per_player + 1)),
        max_turns=max_turns,
        home_builds=home_builds,
        positions=positions,
        symmetries=symmetries,
    )


# ── the active board ────────────────────────────────────────────────────

_ACTIVE: Board = ring_board()


def active() -> Board:
    return _ACTIVE


def set_board(board: Board) -> Board:
    """Install a board, returning the one it replaced. Prefer `use_board`."""
    global _ACTIVE
    _ACTIVE, old = board, _ACTIVE
    return old


@contextmanager
def use_board(board: Optional[Board]) -> Iterator[Board]:
    """Play on `board` for the block, then restore. `None` is a no-op."""
    if board is None:
        yield _ACTIVE
        return
    old = set_board(board)
    try:
        yield board
    finally:
        set_board(old)


# ── accessors (everything downstream reads the board through these) ──────

def players() -> List[Player]:
    """The seats in this game. Iterate this, not `Player`."""
    return _ACTIVE.players


def home_centers() -> Dict[Player, List[str]]:
    return _ACTIVE.home_centers


def win_centers() -> int:
    return _ACTIVE.win_centers


def max_turns() -> int:
    return _ACTIVE.max_turns


def home_builds() -> bool:
    return _ACTIVE.home_builds


def adjacency() -> Dict[str, List[str]]:
    return _ACTIVE.adjacency


def positions() -> Dict[str, Tuple[float, float]]:
    return _ACTIVE.positions


def get_adjacent(territory: str) -> List[str]:
    return _ACTIVE.adjacency.get(territory, [])


def is_adjacent(t1: str, t2: str) -> bool:
    return t2 in _ACTIVE.adjacency.get(t1, [])


def get_all_territories() -> List[str]:
    return _ACTIVE.territories


def is_supply_center(territory: str) -> bool:
    return territory in _ACTIVE._centre_set


def get_supply_centers() -> List[str]:
    return list(_ACTIVE.supply_centers)
