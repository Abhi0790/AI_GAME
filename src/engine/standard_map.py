"""The standard Diplomacy map: 7 powers, 75 provinces, 34 centres, 19 seas."""
from typing import Dict, List, Optional

from src.common.schemas import Player
from src.engine.board import Board, province

# Seat colours follow the board's traditional ones where a colour exists.
POWERS = {
    "Austria": Player.RED, "England": Player.BLUE, "France": Player.TEAL,
    "Germany": Player.ORANGE, "Italy": Player.GREEN, "Russia": Player.PURPLE,
    "Turkey": Player.GOLD,
}

HOMES = {
    "Austria": "Vie Bud Tri", "England": "Lon Edi Lvp", "France": "Par Bre Mar",
    "Germany": "Ber Kie Mun", "Italy": "Rom Ven Nap", "Russia": "Mos War StP Sev",
    "Turkey": "Con Ank Smy",
}
START_FLEETS = "Tri Lon Edi Bre Kie Nap StP/sc Sev Ank"
NEUTRAL_CENTRES = "Nwy Swe Den Hol Bel Spa Por Tun Ser Gre Bul Rum"
STANDARD_WIN = 18

ARMY = """
Cly: Edi Lvp
Edi: Cly Lvp Yor
Lvp: Cly Edi Yor Wal
Yor: Edi Lvp Wal Lon
Wal: Lvp Yor Lon
Lon: Yor Wal
Bre: Pic Par Gas
Pic: Bre Par Bur Bel
Par: Bre Pic Bur Gas
Bur: Pic Par Gas Mar Bel Ruh Mun
Gas: Bre Par Bur Mar Spa
Mar: Gas Bur Pie Spa
Spa: Gas Mar Por
Por: Spa
Bel: Pic Bur Ruh Hol
Hol: Bel Ruh Kie
Kie: Hol Ruh Mun Ber Den
Ber: Kie Mun Sil Pru
Pru: Ber Sil War Lvn
Ruh: Bel Hol Kie Mun Bur
Mun: Bur Ruh Kie Ber Sil Boh Tyr
Sil: Mun Ber Pru War Gal Boh
Den: Kie Swe
Swe: Den Nwy Fin
Nwy: Swe Fin StP
Fin: Swe Nwy StP
StP: Nwy Fin Mos Lvn
Lvn: StP Mos War Pru
Mos: StP Lvn War Ukr Sev
War: Pru Sil Gal Ukr Mos Lvn
Ukr: War Gal Rum Sev Mos
Sev: Ukr Mos Rum Arm
Boh: Mun Sil Gal Vie Tyr
Tyr: Mun Boh Vie Tri Ven Pie
Vie: Boh Gal Bud Tri Tyr
Gal: Boh Sil War Ukr Rum Bud Vie
Bud: Vie Gal Rum Ser Tri
Tri: Tyr Vie Bud Ser Alb Ven
Pie: Mar Tyr Ven Tus
Ven: Pie Tyr Tri Apu Rom Tus
Tus: Pie Ven Rom
Rom: Tus Ven Apu Nap
Apu: Ven Rom Nap
Nap: Rom Apu
Ser: Bud Tri Alb Gre Bul Rum
Alb: Tri Ser Gre
Gre: Alb Ser Bul
Bul: Ser Rum Con Gre
Rum: Bud Gal Ukr Sev Bul Ser
Con: Bul Ank Smy
Ank: Con Smy Arm
Smy: Con Ank Arm Syr
Arm: Ank Smy Syr Sev
Syr: Smy Arm
NAf: Tun
Tun: NAf
"""

FLEET = """
Cly: Edi Lvp NAO NWG
Edi: Cly Yor NTH NWG
Lvp: Cly Wal IRI NAO
Yor: Edi Lon NTH
Wal: Lvp Lon IRI ENG
Lon: Yor Wal NTH ENG
Bre: Pic Gas ENG MAO
Pic: Bre Bel ENG
Gas: Bre Spa/nc MAO
Mar: Spa/sc Pie LYO
Spa/nc: Gas Por MAO
Spa/sc: Mar Por MAO WES LYO
Por: Spa/nc Spa/sc MAO
Bel: Pic Hol ENG NTH
Hol: Bel Kie NTH HEL
Kie: Hol Den Ber HEL BAL
Ber: Kie Pru BAL
Pru: Ber Lvn BAL
Den: Kie Swe HEL NTH SKA BAL
Swe: Den Nwy Fin SKA BAL BOT
Nwy: Swe StP/nc NTH NWG SKA BAR
Fin: Swe StP/sc BOT
StP/nc: Nwy BAR
StP/sc: Fin Lvn BOT
Lvn: StP/sc Pru BAL BOT
Sev: Rum Arm BLA
Tri: Ven Alb ADR
Pie: Mar Tus LYO
Ven: Tri Apu ADR
Tus: Pie Rom LYO TYS
Rom: Tus Nap TYS
Apu: Ven Nap ADR ION
Nap: Rom Apu ION TYS
Alb: Tri Gre ADR ION
Gre: Alb Bul/sc ION AEG
Bul/ec: Rum Con BLA
Bul/sc: Gre Con AEG
Rum: Sev Bul/ec BLA
Con: Bul/ec Bul/sc Ank Smy BLA AEG
Ank: Con Arm BLA
Smy: Con Syr AEG EAS
Arm: Ank Sev BLA
Syr: Smy EAS
NAf: Tun MAO WES
Tun: NAf WES TYS ION
NAO: Cly Lvp NWG IRI MAO
NWG: NAO Cly Edi NTH Nwy BAR
BAR: NWG Nwy StP/nc
NTH: NWG Edi Yor Lon ENG Bel Hol HEL Den SKA Nwy
SKA: NTH Nwy Swe Den
HEL: NTH Den Kie Hol
BAL: Den Swe BOT Lvn Pru Ber Kie
BOT: Swe Fin StP/sc Lvn BAL
ENG: IRI Wal Lon NTH Bel Pic Bre MAO
IRI: NAO Lvp Wal ENG MAO
MAO: NAO IRI ENG Bre Gas Spa/nc Spa/sc Por NAf WES
WES: MAO Spa/sc LYO TYS Tun NAf
LYO: Spa/sc Mar Pie Tus TYS WES
TYS: LYO Tus Rom Nap ION Tun WES
ION: TYS Nap Apu ADR Alb Gre AEG EAS Tun
ADR: Ven Tri Alb ION Apu
AEG: Gre Bul/sc Con Smy EAS ION
EAS: AEG Smy Syr ION
BLA: Rum Sev Arm Ank Con Bul/ec
"""

# On a 100 x 100 grid, scaled to the dashboard's 500 x 500.
POSITIONS = """
NAO 8 22  NWG 40 8  BAR 68 3  Cly 24 21  Edi 28 24  Lvp 24 29  Yor 29 31  Wal 22 36
Lon 29 37  IRI 14 35  ENG 22 43  NTH 36 27  SKA 46 23  HEL 39 33  Den 45 29  Nwy 46 14
Swe 54 18  Fin 63 12  StP 74 15  BOT 58 24  BAL 53 31  Lvn 65 28  Mos 79 30  War 62 40
Pru 56 35  Ber 48 37  Kie 43 39  Hol 37 40  Bel 32 44  Ruh 39 45  Mun 44 50  Sil 53 43
Bur 35 52  Pic 29 47  Par 28 53  Bre 19 50  Gas 21 60  Mar 32 62  MAO 5 56  Spa 14 70
Por 4 72  WES 22 80  NAf 14 89  Tun 34 90  LYO 30 72  Pie 39 59  Tus 41 66  Rom 45 72
Nap 50 79  Apu 53 73  Ven 45 61  Tyr 47 55  Boh 51 49  Vie 55 53  Gal 63 48  Ukr 71 43
Sev 82 45  Bud 60 57  Tri 52 63  ADR 50 67  TYS 39 79  ION 55 87  Alb 58 72  Ser 62 65
Gre 63 78  Rum 71 59  Bul 68 68  BLA 80 61  Con 74 74  AEG 67 84  Ank 83 70  Smy 78 80
Arm 91 69  Syr 89 84  EAS 78 91
"""


def _graph(text: str) -> Dict[str, List[str]]:
    return {k.strip(): v.split() for k, v in
            (line.split(":") for line in text.strip().splitlines())}


def standard_board(max_turns: int = 20, win_centers: int = STANDARD_WIN,
                   home_builds: bool = True, sieges: bool = False,
                   fortified_homes: bool = False) -> Board:
    """The standard map; 18 of 34 centres wins outright."""
    army, fleet = _graph(ARMY), _graph(FLEET)
    adjacency: Dict[str, List[str]] = {}
    for graph in (army, fleet):
        for loc, neighbours in graph.items():
            row = adjacency.setdefault(province(loc), [])
            row += [n for n in dict.fromkeys(map(province, neighbours)) if n not in row]
    homes = {POWERS[k]: v.split() for k, v in HOMES.items()}
    cells = POSITIONS.split()
    return Board(
        adjacency=adjacency,
        supply_centers=[t for ts in homes.values() for t in ts] + NEUTRAL_CENTRES.split(),
        home_centers=homes,
        win_centers=win_centers,
        max_turns=max_turns,
        home_builds=home_builds,
        sieges=sieges,
        fortresses=[t for ts in homes.values() for t in ts] if fortified_homes else [],
        positions={cells[i]: (float(cells[i + 1]) * 5, float(cells[i + 2]) * 5)
                   for i in range(0, len(cells), 3)},
        army_adjacency=army,
        fleet_adjacency=fleet,
        start_fleets=START_FLEETS.split(),
    )


def power_of(player: Player) -> Optional[str]:
    return next((k for k, v in POWERS.items() if v == player), None)
