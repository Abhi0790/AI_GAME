"""
Web UI — FastAPI application serving an interactive game dashboard.

Endpoints:
    GET  /              → Dashboard page
    POST /api/game/new  → Start a new headless game, return game ID
    GET  /api/game/{id}/state → Current game state
    POST /api/game/{id}/step  → Advance one turn
    GET  /api/game/{id}/replay → Full history
"""

import os
import uuid
import random
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.common.schemas import Player, OrderType
from src.agents.agent import Agent
from src.engine.runner import GameRunner
from src.engine.replay import save_replay


app = FastAPI(title="Territory Game Dashboard")
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

# ── In-memory game store ────────────────────────────────────────────────
_games: Dict[str, dict] = {}


def _serialise_state(state):
    """Convert a GameState to a JSON-friendly dict."""
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


def _serialise_history_step(step):
    state, orders, outcomes, log, traces = step
    return {
        "state": _serialise_state(state),
        "orders": [
            {
                "player": o.player.value,
                "unit_territory": o.unit_territory,
                "order_type": o.order_type.value,
                "target": o.target,
                "supported_from": o.supported_from,
            }
            for o in orders
        ],
        "outcomes": [
            {
                "commitment_type": o.commitment.commitment_type.value,
                "players": [p.value for p in o.commitment.players],
                "kept": o.kept,
                "broken_by": [p.value for p in o.broken_by],
            }
            for o in outcomes
        ],
        "log": log.events,
        "traces": {
            p.value: {
                "expected_value": t.expected_value,
                "penalty": t.penalty,
                "explanation": t.explanation,
                "commitment_broken": t.commitment_broken,
            } if t else None
            for p, t in traces.items()
        },
    }


# ── Routes ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/game/new")
async def new_game(seed: int = 42):
    """Create a new game. Returns the game ID."""
    game_id = str(uuid.uuid4())[:8]
    random.seed(seed)
    agents = [
        Agent(Player.RED, "Opportunist"),
        Agent(Player.BLUE, "Honest"),
        Agent(Player.GREEN, "Paranoid"),
        Agent(Player.GOLD, "Vengeful"),
    ]
    runner = GameRunner(agents)
    _games[game_id] = {
        "runner": runner,
        "finished": False,
        "current_turn": 1,
    }
    return {"game_id": game_id, "state": _serialise_state(runner.state)}


@app.get("/api/game/{game_id}/state")
async def get_state(game_id: str):
    if game_id not in _games:
        return JSONResponse(status_code=404, content={"error": "Game not found"})
    g = _games[game_id]
    runner = g["runner"]
    return {
        "state": _serialise_state(runner.state),
        "finished": g["finished"],
        "current_turn": g["current_turn"],
        "history_length": len(runner.history),
    }


@app.post("/api/game/{game_id}/step")
async def step_game(game_id: str):
    """Advance the game by one turn."""
    if game_id not in _games:
        return JSONResponse(status_code=404, content={"error": "Game not found"})
    g = _games[game_id]
    if g["finished"]:
        return {"finished": True, "message": "Game already finished"}

    runner = g["runner"]
    t = g["current_turn"]

    # --- Run a single turn (extracted from runner.run) ---
    import uuid as _uuid
    from src.common.schemas import Message, MessageType, CommitmentType

    # Clean expired commitments
    runner.commitments = [c for c in runner.commitments if c.valid_until_turn >= t]

    # Negotiation
    new_messages = []
    for a in runner.agents.values():
        new_messages.extend(a.propose(runner.state))

    for _ in range(runner.negotiation_rounds):
        if not new_messages:
            break
        replies = []
        inbox = {p: [] for p in Player}
        for m in new_messages:
            if m.receiver:
                inbox[m.receiver].append(m)
            else:
                for p in Player:
                    if p != m.sender:
                        inbox[p].append(m)
        for p, a in runner.agents.items():
            r = a.reply(runner.state, inbox[p])
            replies.extend(r)
            for msg in inbox[p]:
                if msg.message_type == MessageType.BROADCAST:
                    a.receive_gossip(msg)
        from src.agents.negotiation.strategy import message_to_commitment
        for rep in replies:
            if rep.message_type == MessageType.ACCEPT:
                orig = next((m for m in new_messages if m.id == rep.reference_id), None)
                if orig:
                    c = message_to_commitment(orig, rep.sender, runner.state.turn)
                    runner.commitments.append(c)
        new_messages = replies

    # Orders
    all_orders = []
    traces = {}
    for p, a in runner.agents.items():
        orders, trace = a.act(runner.state, runner.commitments)
        all_orders.extend(orders)
        traces[p] = trace

    # Resolve
    from src.engine.adjudicator import resolve
    new_state, outcomes, log = resolve(runner.state, all_orders, runner.commitments)

    # Gossip
    for o in outcomes:
        if not o.kept:
            for victim in o.commitment.players:
                if victim in o.broken_by:
                    continue
                for betrayer in o.broken_by:
                    gossip_msg = Message(
                        id=str(_uuid.uuid4()),
                        sender=victim,
                        receiver=None,
                        message_type=MessageType.BROADCAST,
                        broadcast_kind="BETRAYED",
                        broadcast_target=betrayer,
                        commitment_type=o.commitment.commitment_type,
                    )
                    for p, a in runner.agents.items():
                        if p != victim:
                            a.receive_gossip(gossip_msg)

    # Update beliefs & opponent models
    for a in runner.agents.values():
        a.update_beliefs_from_outcomes(new_state, outcomes)
        if hasattr(a, 'observe_orders'):
            a.observe_orders(all_orders)

    step_data = (runner.state, all_orders, outcomes, log, traces)
    runner.history.append(step_data)
    runner.state = new_state

    g["current_turn"] = t + 1
    if t >= runner.max_turns:
        g["finished"] = True

    return {
        "turn": t,
        "state": _serialise_state(new_state),
        "step": _serialise_history_step(step_data),
        "finished": g["finished"],
    }


@app.get("/api/game/{game_id}/replay")
async def get_replay(game_id: str):
    if game_id not in _games:
        return JSONResponse(status_code=404, content={"error": "Game not found"})
    runner = _games[game_id]["runner"]
    return {
        "turns": len(runner.history),
        "finished": _games[game_id]["finished"],
        "history": [_serialise_history_step(s) for s in runner.history],
        "final_state": _serialise_state(runner.state),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
