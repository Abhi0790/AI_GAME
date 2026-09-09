"""
Replay System — save and load complete game histories as JSON.

A replay file contains the full sequence of game states, orders, commitment
outcomes, and decision traces so that a game can be reviewed after the fact.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.common.schemas import (
    GameState, Order, OrderType, Player, Unit, Commitment,
    CommitmentType, UnitType,
)


def _serialise_state(state: GameState) -> dict:
    return {
        "turn": state.turn,
        "units": [
            {"player": u.player.value, "territory": u.territory}
            for u in state.units
        ],
        "supply_centers": {
            k: (v.value if v else None)
            for k, v in state.supply_centers.items()
        },
        "territory_owners": {
            k: (v.value if v else None)
            for k, v in state.territory_owners.items()
        },
    }


def _serialise_orders(orders: List[Order]) -> list:
    return [
        {
            "player": o.player.value,
            "unit_territory": o.unit_territory,
            "order_type": o.order_type.value,
            "target": o.target,
            "supported_from": o.supported_from,
        }
        for o in orders
    ]


def _serialise_outcomes(outcomes: list) -> list:
    result = []
    for o in outcomes:
        result.append({
            "commitment_id": o.commitment.id,
            "commitment_type": o.commitment.commitment_type.value,
            "players": [p.value for p in o.commitment.players],
            "kept": o.kept,
            "broken_by": [p.value for p in o.broken_by],
        })
    return result


def _serialise_traces(traces: dict) -> dict:
    out = {}
    for player, trace in traces.items():
        if trace is None:
            out[player.value] = None
            continue
        out[player.value] = {
            "expected_value": trace.expected_value,
            "penalty": trace.penalty,
            "explanation": trace.explanation,
            "commitment_broken": trace.commitment_broken,
        }
    return out


def save_replay(
    history: list,
    final_state: GameState,
    directory: str = "replays",
    filename: Optional[str] = None,
) -> str:
    """Save a full game replay to a JSON file.

    Args:
        history: list of (state, orders, outcomes, log, traces) tuples
        final_state: the final GameState after all turns
        directory: directory to save into
        filename: optional filename; defaults to timestamp-based name

    Returns:
        The path of the saved replay file.
    """
    os.makedirs(directory, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"replay_{ts}.json"

    turns_data = []
    for step in history:
        state, orders, outcomes, log, traces = step
        turns_data.append({
            "state": _serialise_state(state),
            "orders": _serialise_orders(orders),
            "outcomes": _serialise_outcomes(outcomes),
            "log": log.events,
            "traces": _serialise_traces(traces),
        })

    replay = {
        "version": 1,
        "turns": len(turns_data),
        "final_state": _serialise_state(final_state),
        "history": turns_data,
    }

    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        json.dump(replay, f, indent=2)

    return path


def load_replay(path: str) -> dict:
    """Load a replay file and return the raw dict."""
    with open(path) as f:
        return json.load(f)


def reconstruct_states(replay: dict) -> List[GameState]:
    """Reconstruct GameState objects from a loaded replay."""
    states = []
    for turn_data in replay["history"]:
        sd = turn_data["state"]
        units = [
            Unit(player=Player(u["player"]), territory=u["territory"])
            for u in sd["units"]
        ]
        supply_centers = {
            k: (Player(v) if v else None)
            for k, v in sd["supply_centers"].items()
        }
        territory_owners = {
            k: (Player(v) if v else None)
            for k, v in sd["territory_owners"].items()
        }
        states.append(GameState(
            turn=sd["turn"],
            units=units,
            supply_centers=supply_centers,
            territory_owners=territory_owners,
        ))

    # Also reconstruct final state
    fsd = replay["final_state"]
    units = [
        Unit(player=Player(u["player"]), territory=u["territory"])
        for u in fsd["units"]
    ]
    supply_centers = {
        k: (Player(v) if v else None)
        for k, v in fsd["supply_centers"].items()
    }
    territory_owners = {
        k: (Player(v) if v else None)
        for k, v in fsd["territory_owners"].items()
    }
    states.append(GameState(
        turn=fsd["turn"],
        units=units,
        supply_centers=supply_centers,
        territory_owners=territory_owners,
    ))

    return states
