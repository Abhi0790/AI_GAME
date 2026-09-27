from typing import List

class Territory:
    def __init__(self, name: str, is_supply_center: bool = False):
        self.name = name
        self.is_supply_center = is_supply_center

# 12 territories, 10 supply centres, 2 units per player (deck, slide 3).
#
# Eight home territories form a ring, two per player, both supply centres and
# both occupied at the start. Four territories fill the middle: two neutral
# supply centres (N1 north, N2 south) and two plain corridors (C1 west,
# C2 east).
#
#              R2 ---- B1
#           /     \  /     \
#         R1       N1       B2
#         |      /    \      |
#         C1 ---        --- C2
#         |      \    /      |
#         Y2       N2       G1
#           \     /  \     /
#              Y1 ---- G2
#
# Every player borders exactly two rivals (their ring neighbours), one neutral
# centre and one corridor, so nobody starts with a positional advantage.

TERRITORIES = [
    Territory("R1", is_supply_center=True), Territory("R2", is_supply_center=True),
    Territory("B1", is_supply_center=True), Territory("B2", is_supply_center=True),
    Territory("G1", is_supply_center=True), Territory("G2", is_supply_center=True),
    Territory("Y1", is_supply_center=True), Territory("Y2", is_supply_center=True),
    Territory("N1", is_supply_center=True), Territory("N2", is_supply_center=True),
    Territory("C1"), Territory("C2"),
]

ADJACENCY = {
    # Home ring: R1-R2-B1-B2-G1-G2-Y1-Y2-R1
    "R1": ["R2", "Y2", "C1"],
    "R2": ["R1", "B1", "N1"],
    "B1": ["R2", "B2", "N1"],
    "B2": ["B1", "G1", "C2"],
    "G1": ["B2", "G2", "C2"],
    "G2": ["G1", "Y1", "N2"],
    "Y1": ["G2", "Y2", "N2"],
    "Y2": ["Y1", "R1", "C1"],

    # Middle: neutral centres north/south, corridors west/east
    "N1": ["R2", "B1", "C1", "C2"],
    "N2": ["G2", "Y1", "C1", "C2"],
    "C1": ["R1", "Y2", "N1", "N2"],
    "C2": ["B2", "G1", "N1", "N2"],
}

# Where each player starts: both home centres, one unit on each.
HOME_CENTERS = {
    "Red": ["R1", "R2"],
    "Blue": ["B1", "B2"],
    "Green": ["G1", "G2"],
    "Gold": ["Y1", "Y2"],
}

# Win immediately on this many centres, else most centres after MAX_TURNS.
WIN_CENTERS = 5
MAX_TURNS = 12

def get_adjacent(territory: str) -> List[str]:
    return ADJACENCY.get(territory, [])

def is_adjacent(t1: str, t2: str) -> bool:
    return t2 in ADJACENCY.get(t1, [])

def get_all_territories() -> List[str]:
    return [t.name for t in TERRITORIES]

def is_supply_center(territory: str) -> bool:
    for t in TERRITORIES:
        if t.name == territory:
            return t.is_supply_center
    return False

def get_supply_centers() -> List[str]:
    return [t.name for t in TERRITORIES if t.is_supply_center]


def _check_board():
    """Adjacency must be symmetric or the adjudicator silently allows
    one-way moves. Cheapest possible guard, runs at import."""
    names = set(get_all_territories())
    for t, neighbours in ADJACENCY.items():
        assert t in names, f"{t} adjacency has no territory"
        for n in neighbours:
            assert n in names, f"{t} -> {n} is not a territory"
            assert t in ADJACENCY[n], f"{t} -> {n} is not mutual"
    assert len(names) == 12, f"expected 12 territories, got {len(names)}"
    assert len(get_supply_centers()) == 10, "expected 10 supply centres"

_check_board()
