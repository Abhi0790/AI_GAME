"""
Evaluation Metrics — quantitative measures for analysing game outcomes.

Each function operates on a game history (list of turn records produced by
GameRunner) and returns structured data.
"""

from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from src.common.schemas import Player, GameState, CommitmentType


# ── Per-game metrics ─────────────────────────────────────────────────────

def supply_center_timeline(history: list) -> Dict[Player, List[int]]:
    """Return a dict mapping each player to a list of supply-center counts,
    one entry per turn."""
    timeline: Dict[Player, List[int]] = {p: [] for p in Player}

    for step in history:
        state: GameState = step[0]
        counts = {p: 0 for p in Player}
        for owner in state.supply_centers.values():
            if owner:
                counts[owner] += 1
        for p in Player:
            timeline[p].append(counts[p])

    return timeline


def betrayal_events(history: list) -> List[Dict[str, Any]]:
    """Extract every betrayal from the game history."""
    events = []
    for turn_idx, step in enumerate(history):
        _state, _orders, outcomes, _log, _traces = step
        for o in outcomes:
            if not o.kept:
                events.append({
                    "turn": turn_idx + 1,
                    "commitment_type": o.commitment.commitment_type.value,
                    "players": [p.value for p in o.commitment.players],
                    "broken_by": [p.value for p in o.broken_by],
                })
    return events


def betrayal_rate_per_player(history: list) -> Dict[Player, float]:
    """Fraction of commitments each player broke out of commitments they
    participated in."""
    participated: Dict[Player, int] = defaultdict(int)
    broken: Dict[Player, int] = defaultdict(int)

    for step in history:
        _s, _o, outcomes, _l, _t = step
        for o in outcomes:
            for p in o.commitment.players:
                participated[p] += 1
                if p in o.broken_by:
                    broken[p] += 1

    rates: Dict[Player, float] = {}
    for p in Player:
        if participated[p] > 0:
            rates[p] = broken[p] / participated[p]
        else:
            rates[p] = 0.0
    return rates


def alliance_durations(history: list) -> List[Dict[str, Any]]:
    """Return a list of alliance records with their actual duration
    (turns before broken or expiry)."""
    # Track when commitments first appear and when they break
    commitment_first_seen: Dict[str, int] = {}
    commitment_broken_turn: Dict[str, Optional[int]] = {}
    commitment_info: Dict[str, Any] = {}

    for turn_idx, step in enumerate(history):
        _s, _o, outcomes, _l, _t = step
        for o in outcomes:
            cid = o.commitment.id
            if cid not in commitment_first_seen:
                commitment_first_seen[cid] = turn_idx + 1
                commitment_info[cid] = {
                    "type": o.commitment.commitment_type.value,
                    "players": [p.value for p in o.commitment.players],
                    "valid_until": o.commitment.valid_until_turn,
                }
            if not o.kept and cid not in commitment_broken_turn:
                commitment_broken_turn[cid] = turn_idx + 1

    records = []
    for cid, info in commitment_info.items():
        started = commitment_first_seen[cid]
        if cid in commitment_broken_turn:
            ended = commitment_broken_turn[cid]
            duration = ended - started
            status = "broken"
        else:
            ended = info["valid_until"]
            duration = max(0, ended - started)
            status = "kept"
        records.append({**info, "started": started, "ended": ended,
                        "duration": duration, "status": status})
    return records


def final_scores(state: GameState) -> Dict[Player, int]:
    """Return supply-center counts from the final game state."""
    counts = {p: 0 for p in Player}
    for owner in state.supply_centers.values():
        if owner:
            counts[owner] += 1
    return counts


# ── Cross-game (tournament) metrics ──────────────────────────────────────

def win_rates(game_results: List[Dict[Player, int]]) -> Dict[Player, float]:
    """Given a list of per-game score dicts, return win rates."""
    wins: Dict[Player, int] = {p: 0 for p in Player}
    for scores in game_results:
        mx = max(scores.values())
        winners = [p for p, s in scores.items() if s == mx]
        for w in winners:
            wins[w] += 1
    total = max(1, len(game_results))
    return {p: wins[p] / total for p in Player}
