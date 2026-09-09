from typing import List, Dict, Set

class Territory:
    def __init__(self, name: str, is_supply_center: bool = False):
        self.name = name
        self.is_supply_center = is_supply_center

# A simple 13-territory planar graph for the game
# 6 supply centers.
# 4 home regions, 1 for each player with 3 territories (one being a SC). Wait, initially 4 players, 3 units each. So they must occupy 3 territories each at the start.
# 4 * 3 = 12 territories occupied.
# Let's say 14 territories total. 6 supply centers.
# P1 (Red): R1(SC), R2, R3
# P2 (Blue): B1(SC), B2, B3
# P3 (Green): G1(SC), G2, G3
# P4 (Gold): Y1(SC), Y2, Y3
# Neutral SCs: N1(SC), N2(SC)
# Total = 14 territories, 6 SCs.
# Let's connect them in a circular/planar way.

TERRITORIES = [
    Territory("R1", is_supply_center=True), Territory("R2"), Territory("R3"),
    Territory("B1", is_supply_center=True), Territory("B2"), Territory("B3"),
    Territory("G1", is_supply_center=True), Territory("G2"), Territory("G3"),
    Territory("Y1", is_supply_center=True), Territory("Y2"), Territory("Y3"),
    Territory("N1", is_supply_center=True), Territory("N2", is_supply_center=True)
]

ADJACENCY = {
    "R1": ["R2", "R3", "N1"],
    "R2": ["R1", "R3", "B3"],
    "R3": ["R1", "R2", "Y2"],
    
    "B1": ["B2", "B3", "N1"],
    "B2": ["B1", "B3", "G3"],
    "B3": ["B1", "B2", "R2"],
    
    "G1": ["G2", "G3", "N2"],
    "G2": ["G1", "G3", "Y3"],
    "G3": ["G1", "G2", "B2"],
    
    "Y1": ["Y2", "Y3", "N2"],
    "Y2": ["Y1", "Y3", "R3"],
    "Y3": ["Y1", "Y2", "G2"],
    
    "N1": ["R1", "B1", "N2"],
    "N2": ["G1", "Y1", "N1"]
}

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

