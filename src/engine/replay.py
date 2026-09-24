"""
Replay System — save and load complete game histories as JSON.

A replay file contains the full sequence of game states, orders, commitment
outcomes, and decision traces so that a game can be reviewed after the fact.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.engine.board import active
from src.common.schemas import (
    GameState, Order, OrderType, Player, Unit, Commitment,
    CommitmentType, UnitType, describe_message,
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


def _serialise_commitment(c) -> dict:
    return {
        "id": c.id,
        "commitment_type": c.commitment_type.value,
        "players": [p.value for p in c.players],
        "valid_until_turn": c.valid_until_turn,
        "target_territory": c.target_territory,
        "supported_from": c.supported_from,
        "dmz_territories": c.dmz_territories,
        "private": c.private,
        "repay_turn": c.repay_turn,
    }


def _serialise_outcomes(outcomes: list) -> list:
    result = []
    for o in outcomes:
        result.append({
            "commitment_id": o.commitment.id,
            "commitment_type": o.commitment.commitment_type.value,
            "players": [p.value for p in o.commitment.players],
            "kept": o.kept,
            "broken_by": [p.value for p in o.broken_by],
            # A replay that records only the verdict cannot say whether the
            # broken promise was public, or bound three people, which is most
            # of what makes the verdict interesting.
            "private": o.commitment.private,
            "pact": len(o.commitment.players) > 2,
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


def _serialise_messages(messages: list) -> list:
    return [
        {
            "id": m.id,
            "sender": m.sender.value,
            "receiver": m.receiver.value if m.receiver else None,
            "message_type": m.message_type.value,
            "commitment_type": m.commitment_type.value if m.commitment_type else None,
            "turns": m.turns,
            "target_territory": m.target_territory,
            "supported_from": m.supported_from,
            "dmz_territories": m.dmz_territories,
            "reference_id": m.reference_id,
            "condition": m.condition,
            "action": m.action,
            "text": describe_message(m),
        }
        for m in messages
    ]


def _serialise_beliefs(beliefs: list) -> list:
    return [
        {
            "observer": b.observer.value,
            "subject": b.subject.value,
            "commitment_type": b.commitment_type.value,
            "alpha": b.alpha,
            "beta": b.beta_param,
            "reliability": b.expected_reliability,
            "reliability_toward": b.reliability_toward_observer,
        }
        for b in beliefs
    ]


def save_replay(
    history: list,
    final_state: GameState,
    directory: str = "replays",
    filename: Optional[str] = None,
    board: Optional[dict] = None,
) -> str:
    """Save a full game replay to a JSON file.

    Args:
        history: list of (state, orders, outcomes, log, traces) tuples
        final_state: the final GameState after all turns
        directory: directory to save into
        filename: optional filename; defaults to timestamp-based name
        board: the board played on, as `Board.to_dict()`; defaults to active

    Returns:
        The path of the saved replay file.
    """
    os.makedirs(directory, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"replay_{ts}.json"

    turns_data = []
    for step in history:
        turns_data.append({
            "state": _serialise_state(step.state),
            "orders": _serialise_orders(step.orders),
            "outcomes": _serialise_outcomes(step.outcomes),
            "log": step.log.events,
            "traces": _serialise_traces(step.traces),
            # Everything the negotiation and trust panels replay from. A
            # replay that drops the messages cannot show why a deal existed.
            "messages": _serialise_messages(step.messages),
            "beliefs": _serialise_beliefs(step.beliefs),
            # The deals live at the time, not just the ones graded. Without
            # these a replay cannot show what was on the table during a turn
            # nobody broke anything.
            "commitments": [_serialise_commitment(c) for c in step.commitments],
            "nodes": {p.value: n for p, n in (step.nodes or {}).items()},
            "total_nodes": {p.value: n for p, n in (step.total_nodes or {}).items()},
        })

    replay = {
        # 2: turn records carry messages, beliefs, live deals, search cost.
        # 3: the board is saved with the game.
        "version": 3,
        "board": board if board is not None else active().to_dict(),
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


def reconstruct_commitments(replay: dict) -> List[List[Commitment]]:
    """The deals live during each turn, rebuilt as Commitment objects."""
    out = []
    for turn_data in replay["history"]:
        out.append([
            Commitment(
                id=c["id"],
                commitment_type=CommitmentType(c["commitment_type"]),
                players=[Player(p) for p in c["players"]],
                valid_until_turn=c["valid_until_turn"],
                target_territory=c.get("target_territory"),
                supported_from=c.get("supported_from"),
                dmz_territories=c.get("dmz_territories"),
                private=c.get("private", False),
                repay_turn=c.get("repay_turn"),
            )
            for c in turn_data.get("commitments", [])
        ])
    return out


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
