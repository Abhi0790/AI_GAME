"""One object that says how a game is set up.

Every script, the sweep and the dashboard build games from a GameConfig, so a
result can always be reproduced from the config it prints.

Three kinds of setting live here:

  * the *board* -- how many seats, how much land, what wins, how long. These
    build a `Board` (see `src.engine.board`), which the harness installs for
    the duration of the game.
  * the *agents* -- who sits where, which search each one runs.
  * the *knobs* -- evaluation weights, trust constants, negotiation
    thresholds. Those live as UPPERCASE module globals next to the code that
    reads them; `knobs` overrides any of them by name for one game (see
    `src/harness.py`), so nothing here needs to know which module a knob is in.
"""
import argparse
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.common.schemas import Player
from src.engine.board import (
    Board, ring_board, DEFAULT_SEATS, DEFAULT_HOMES_PER_PLAYER, DEFAULT_MAX_TURNS,
    DEFAULT_WIN_FRACTION,
)

# Personas are dealt round the table in this order, rotated one seat per seed
# so no persona is ever measured on a single colour. With more seats than
# personas the order simply repeats.
PERSONA_ORDER: List[str] = ["Opportunist", "Honest", "Paranoid", "Vengeful"]

# Rotation alone never changes who sits next to whom, so every persona kept the
# same neighbours in every game. Each block of four seeds uses the next of the
# three distinct orders round a four-seat ring; twelve seeds cover them all.
TABLE_ORDERS: List[List[str]] = [
    PERSONA_ORDER,
    ["Opportunist", "Honest", "Vengeful", "Paranoid"],
    ["Opportunist", "Paranoid", "Honest", "Vengeful"],
]


def seating_for(seed: int, seats: Optional[List[Player]] = None) -> Dict[Player, str]:
    """Deal the personas round the table, offset by the seed.

    The offset is taken modulo the number of *personas*, not the number of
    seats. Those are the same thing at four seats and are not at five or six,
    where `seed % len(seats)` leaves every seat permanently twice as likely to
    hold two of the four personas -- an imbalance no number of seeds averages
    out, which would quietly bias every persona table at those sizes. Modulo
    the persona count, any four consecutive seeds give each seat each persona
    exactly once, whatever the seat count.
    """
    seats = list(seats) if seats is not None else list(Player)[:DEFAULT_SEATS]
    n = len(PERSONA_ORDER)
    order = TABLE_ORDERS[(seed // n) % len(TABLE_ORDERS)]
    k = seed % n
    return {seats[i]: order[(i + k) % n]
            for i in range(len(seats))}


@dataclass
class GameConfig:
    seed: int = 0

    # ── the board ───────────────────────────────────────────────────────
    # A ring map is generated from these. `board` overrides all four with an
    # explicit `Board.to_dict()`, for a map that is not a ring at all.
    n_seats: int = DEFAULT_SEATS
    homes_per_player: int = DEFAULT_HOMES_PER_PLAYER
    win_centers: Optional[int] = None       # None -> win_fraction of the board's centres
    win_fraction: float = DEFAULT_WIN_FRACTION
    max_turns: int = DEFAULT_MAX_TURNS
    home_builds: bool = True                # False: build on any owned centre
    board: Optional[dict] = None

    # ── the agents ──────────────────────────────────────────────────────
    # None -> personas dealt round the table, rotated by seed.
    seating: Optional[Dict[Player, str]] = None
    search: str = "expectiminimax"          # "expectiminimax" | "mcts"
    node_budget: int = 1500
    depth: int = 2
    # Per-seat PlannerConfig overrides, e.g. {Player.RED: {"search": "mcts"}}.
    # The figures put one seat on a variant and measure its win rate.
    seat_planner: Dict[Player, Dict[str, Any]] = field(default_factory=dict)
    negotiation_rounds: int = 3
    # Multiplies every persona's reputation_cost; the sweep's third axis.
    rep_cost_scale: float = 1.0
    human_seat: Optional[Player] = None
    chaos_seat: Optional[Player] = None
    chaos_mode: Optional[str] = None
    # UPPERCASE module constants to override for this game, by bare name,
    # e.g. {"LAMBDA_INCENTIVE": 0.6, "CENTRE_VALUE": 1.0}.
    knobs: Dict[str, float] = field(default_factory=dict)

    def make_board(self) -> Board:
        if self.board is not None:
            return Board.from_dict(self.board)
        return ring_board(self.n_seats, self.homes_per_player,
                          win_centers=self.win_centers, win_fraction=self.win_fraction,
                          max_turns=self.max_turns,
                          home_builds=self.home_builds)

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
        for k in ("human_seat", "chaos_seat"):
            if d.get(k):
                d[k] = Player(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── CLI ─────────────────────────────────────────────────────────────────

def add_board_args(parser: argparse.ArgumentParser) -> None:
    """Add the board settings as CLI flags."""
    g = parser.add_argument_group("board")
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
    g.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                   help=f"turn limit (default {DEFAULT_MAX_TURNS})")
    g.add_argument("--build-anywhere", action="store_true",
                   help="build on any owned centre instead of only on home centres "
                        "(standard Diplomacy, the default)")


def board_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """`GameConfig(**board_kwargs(args), ...)` for the flags above."""
    return {"n_seats": args.seats, "homes_per_player": args.homes,
            "win_centers": args.win_centers, "win_fraction": args.win_fraction,
            "max_turns": args.max_turns,
            "home_builds": not args.build_anywhere}
