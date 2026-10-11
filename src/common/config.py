"""One object that says how a game is set up.

The board, the agents, and `knobs`: overrides for UPPERCASE module constants
by name, for one game.
"""
import argparse
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.common.schemas import Player
from src.engine.board import (
    Board, ring_board, DEFAULT_SEATS, DEFAULT_HOMES_PER_PLAYER, DEFAULT_MAX_TURNS,
    DEFAULT_WIN_FRACTION,
)
from src.engine.standard_map import standard_board, STANDARD_WIN

# Games per experiment unless --games says otherwise; a multiple of 12 seats
# every persona order equally, and fewer is too noisy to compare settings.
DEFAULT_GAMES = 60
SEARCHES = ["expectimax", "mcts", "regret"]

# Personas are dealt round the table in this order, rotated one seat per seed.
PERSONA_ORDER: List[str] = ["Opportunist", "Honest", "Paranoid", "Vengeful"]

# Each block of four seeds uses the next of the three distinct four-seat orders.
TABLE_ORDERS: List[List[str]] = [
    PERSONA_ORDER,
    ["Opportunist", "Honest", "Vengeful", "Paranoid"],
    ["Opportunist", "Paranoid", "Honest", "Vengeful"],
]


def seating_for(seed: int, seats: Optional[List[Player]] = None) -> Dict[Player, str]:
    """Deal the personas round the table, offset by the seed modulo the persona count."""
    seats = list(seats) if seats is not None else list(Player)[:DEFAULT_SEATS]
    n = len(PERSONA_ORDER)
    order = TABLE_ORDERS[(seed // n) % len(TABLE_ORDERS)]
    k = seed % n
    return {seats[i]: order[(i + k) % n]
            for i in range(len(seats))}


@dataclass
class GameConfig:
    seed: int = 0

    # ── the board (`board`, a Board.to_dict(), overrides `map`) ─────────
    map: str = "ring"                       # "ring" | "standard" (7 powers, fleets)
    n_seats: int = DEFAULT_SEATS
    homes_per_player: int = DEFAULT_HOMES_PER_PLAYER
    win_centers: Optional[int] = None       # None -> win_fraction of the board's centres
    win_fraction: float = DEFAULT_WIN_FRACTION
    max_turns: int = DEFAULT_MAX_TURNS
    home_builds: bool = True                # False: build on any owned centre
    outer_rings: int = 0                    # rings of neutral centres outside the homes
    fortified_homes: bool = False           # home centres defend at +1
    sieges: bool = False                    # centres fall only to a unit there since spring
    draws: bool = False                     # survivors may end the game by unanimous draw
    board: Optional[dict] = None

    # ── the agents ──────────────────────────────────────────────────────
    # None -> personas dealt round the table, rotated by seed.
    seating: Optional[Dict[Player, str]] = None
    search: str = "expectimax"          # "expectimax" | "mcts" | "regret"
    node_budget: int = 1500
    depth: int = 2
    # Per-seat PlannerConfig overrides, e.g. {Player.RED: {"search": "mcts"}}.
    # The figures put one seat on a variant and measure its win rate.
    seat_planner: Dict[Player, Dict[str, Any]] = field(default_factory=dict)
    # Research ablation: replace one AI seat's P(keeps) with a fixed forecast.
    # Evidence updates and reputation penalties continue normally.
    seat_keep_forecast: Dict[Player, float] = field(default_factory=dict)
    # Research runs can isolate stochastic draws by turn, stage, and seat.
    event_keyed_rng: bool = False
    negotiation_rounds: int = 3
    # Multiplies every persona's reputation_cost; the sweep's third axis.
    rep_cost_scale: float = 1.0
    human_seat: Optional[Player] = None
    chaos_seat: Optional[Player] = None
    chaos_mode: Optional[str] = None
    # Change of leadership: {"seat": "Red", "turn": 5, "persona": "Opportunist"}
    # switches that seat's persona at the start of that turn.
    regime_change: Optional[Dict[str, Any]] = None
    # Reputation carried in from earlier games: observer persona -> subject
    # persona -> record name -> [alpha, beta] (harness.reputation_of builds it).
    reputation: Optional[Dict[str, Any]] = None
    # Start this game with no history even when `reputation` is given.
    erase_reputation: bool = False
    # This game's reputation carries into the next, so breaking a promise
    # keeps its full reputation cost to the end instead of decaying to zero.
    reputation_carries: bool = False
    # UPPERCASE module constants to override for this game, by bare name,
    # e.g. {"LAMBDA_INCENTIVE": 0.6, "CENTRE_VALUE": 1.0}.
    knobs: Dict[str, float] = field(default_factory=dict)

    def make_board(self) -> Board:
        if self.board is not None:
            return Board.from_dict(self.board)
        if self.map == "standard":
            return standard_board(max_turns=self.max_turns,
                                  win_centers=self.win_centers or STANDARD_WIN,
                                  home_builds=self.home_builds, sieges=self.sieges,
                                  fortified_homes=self.fortified_homes)
        if self.map != "ring":
            raise ValueError(f"unknown map {self.map!r}: ring or standard")
        return ring_board(self.n_seats, self.homes_per_player,
                          win_centers=self.win_centers, win_fraction=self.win_fraction,
                          max_turns=self.max_turns,
                          home_builds=self.home_builds,
                          outer_rings=self.outer_rings,
                          fortified_homes=self.fortified_homes,
                          sieges=self.sieges)

    def seats(self) -> Dict[Player, str]:
        """Seat -> persona, with the seats taken from the board."""
        board_players = self.make_board().players
        if self.seating:
            missing = [p for p in board_players if p not in self.seating]
            if missing:
                raise ValueError(
                    f"no persona for {', '.join(p.value for p in missing)}")
            return {p: self.seating[p] for p in board_players}
        return seating_for(self.seed, board_players)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seating"] = {p.value: n for p, n in self.seats().items()}
        d["seat_planner"] = {p.value: o for p, o in self.seat_planner.items()}
        d["seat_keep_forecast"] = {p.value: v for p, v in self.seat_keep_forecast.items()}
        for k in ("human_seat", "chaos_seat"):
            d[k] = d[k].value if d[k] else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GameConfig":
        d = dict(d)
        if d.get("seating"):
            d["seating"] = {Player(k): v for k, v in d["seating"].items()}
        if d.get("seat_planner"):
            d["seat_planner"] = {Player(k): v for k, v in d["seat_planner"].items()}
        if d.get("seat_keep_forecast"):
            d["seat_keep_forecast"] = {
                Player(k): v for k, v in d["seat_keep_forecast"].items()}
        for k in ("human_seat", "chaos_seat"):
            if d.get(k):
                d[k] = Player(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── CLI ─────────────────────────────────────────────────────────────────

def add_game_args(parser: argparse.ArgumentParser, budget: int = 1500) -> None:
    """Add the board and agent settings as CLI flags."""
    g = parser.add_argument_group("game")
    g.add_argument("--standard", action="store_true",
                   help="play the standard 7-power map with fleets and convoys; "
                        "--seats, --homes and --outer-rings apply to the ring only")
    g.add_argument("--seats", type=int, default=DEFAULT_SEATS,
                   help=f"players at the table (default {DEFAULT_SEATS})")
    g.add_argument("--homes", type=int, default=DEFAULT_HOMES_PER_PLAYER,
                   help="home centres per player, one unit starts on each "
                        f"(default {DEFAULT_HOMES_PER_PLAYER})")
    g.add_argument("--win-centers", type=int, default=None,
                   help="centres that win outright (default: --win-fraction of the board)")
    g.add_argument("--win-fraction", type=float, default=DEFAULT_WIN_FRACTION,
                   help=f"share of the board's centres that wins when --win-centers is not "
                        f"given (default {DEFAULT_WIN_FRACTION})")
    g.add_argument("--turns", type=int, default=DEFAULT_MAX_TURNS,
                   help=f"turn limit (default {DEFAULT_MAX_TURNS})")
    g.add_argument("--outer-rings", type=int, default=0,
                   help="rings of neutral supply centres outside the home ring, "
                        "one centre per home centre each (default 0)")
    g.add_argument("--build-anywhere", action="store_true",
                   help="build on any owned centre instead of only on home centres "
                        "(standard Diplomacy, the default)")
    g.add_argument("--forts", action="store_true",
                   help="home centres defend at +1, so taking one needs two supports")
    g.add_argument("--sieges", action="store_true",
                   help="a centre changes hands only if the unit was on it since spring")
    g.add_argument("--draws", action="store_true",
                   help="survivors may end the game early by a unanimous draw vote")
    g.add_argument("--search", choices=SEARCHES, default="expectimax",
                   help="how agents choose orders (default expectimax)")
    g.add_argument("--budget", type=int, default=budget,
                   help=f"adjudications per decision (default {budget})")
    g.add_argument("--knob", action="append", default=[], metavar="NAME=VALUE",
                   help="override a tuning constant, e.g. ORDER_NOISE=0.1; repeatable")


def game_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """`GameConfig(**game_kwargs(args), ...)` for the flags above."""
    return {"map": "standard" if args.standard else "ring",
            "n_seats": args.seats, "homes_per_player": args.homes,
            "win_centers": args.win_centers, "win_fraction": args.win_fraction,
            "max_turns": args.turns,
            "home_builds": not args.build_anywhere,
            "outer_rings": args.outer_rings,
            "fortified_homes": args.forts,
            "sieges": args.sieges,
            "draws": args.draws,
            "search": args.search,
            "node_budget": args.budget,
            "knobs": {k: float(v) for k, v in (kv.split("=", 1) for kv in args.knob)}}
