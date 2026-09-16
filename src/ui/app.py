"""Web UI — FastAPI application serving an interactive game dashboard.

Endpoints:
    GET  /                        Dashboard page
    POST /api/game/new            Start a game; optionally take a seat
    GET  /api/game/{id}/state     Current state, scores, live commitments
    GET  /api/game/{id}/pending   What the human seat has been offered
    POST /api/game/{id}/step      Advance one turn (with the human's input)
    GET  /api/game/{id}/replay    Whole history, for the scrubber
    GET  /api/game/{id}/legal     Legal orders for the human's units
"""

import os
import uuid
import random
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.common.schemas import (
    Player, Order, OrderType, MessageType, describe_message, exchange_leg_due,
)
from src.agents.agent import Agent, HumanAgent
from src.agents.chaos import ChaosAgent
from src.engine.runner import GameRunner
from src.engine.replay import save_replay
from src.engine.orders import generate_orders_for_unit
from src.engine.board import (
    ADJACENCY, get_all_territories, get_supply_centers, WIN_CENTERS, MAX_TURNS,
)
from src.evaluation.metrics import brier_score, reliability_bins


app = FastAPI(title="Territory Game Dashboard")

# The dashboard is a React app built by Vite into web/dist. In the container
# those files sit next to this module; in a source checkout they are two
# levels up. Either way FastAPI serves them, so the page and the API share an
# origin and no CORS or proxy configuration is needed in production.
_HERE = os.path.dirname(__file__)
_DIST_CANDIDATES = [
    os.path.join(_HERE, "dist"),
    os.path.abspath(os.path.join(_HERE, "..", "..", "web", "dist")),
]
DIST = next((d for d in _DIST_CANDIDATES if os.path.isdir(d)), None)

# ── In-memory game store ────────────────────────────────────────────────
_games: Dict[str, dict] = {}

DEFAULT_PERSONAS = {
    Player.RED: "Opportunist",
    Player.BLUE: "Honest",
    Player.GREEN: "Paranoid",
    Player.GOLD: "Vengeful",
}


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


def _serialise_message(m):
    return {
        "id": m.id,
        "sender": m.sender.value,
        "receiver": m.receiver.value if m.receiver else None,
        "type": m.message_type.value,
        "commitment_type": m.commitment_type.value if m.commitment_type else None,
        "turns": m.turns,
        "coalition": [p.value for p in m.coalition] if m.coalition else None,
        "private": m.private,
        "repay_turn": m.repay_turn,
        "target_territory": m.target_territory,
        "supported_from": m.supported_from,
        "dmz_territories": m.dmz_territories,
        "reference_id": m.reference_id,
        "condition": m.condition,
        "action": m.action,
        "broadcast_kind": m.broadcast_kind,
        "broadcast_target": m.broadcast_target.value if m.broadcast_target else None,
        "verdict": m.engine_verdict,
        "truthful": m.truthful,
        "text": describe_message(m),
    }


def _serialise_commitment(c, turn=None):
    return {
        "id": c.id,
        "type": c.commitment_type.value,
        "players": [p.value for p in c.players],
        "valid_until": c.valid_until_turn,
        "target_territory": c.target_territory,
        "supported_from": c.supported_from,
        "dmz_territories": c.dmz_territories,
        "private": c.private,
        "pact": len(c.players) > 2,
        "repay_turn": c.repay_turn,
        "leg_due": exchange_leg_due(c, turn) if turn is not None else None,
    }


def _serialise_history_step(step):
    return {
        "state": _serialise_state(step.state),
        "orders": [
            {
                "player": o.player.value,
                "unit_territory": o.unit_territory,
                "order_type": o.order_type.value,
                "target": o.target,
                "supported_from": o.supported_from,
            }
            for o in step.orders
        ],
        "outcomes": [
            {
                "commitment_type": o.commitment.commitment_type.value,
                "players": [p.value for p in o.commitment.players],
                "kept": o.kept,
                "broken_by": [p.value for p in o.broken_by],
            }
            for o in step.outcomes
        ],
        "log": step.log.events,
        "messages": [_serialise_message(m) for m in step.messages],
        "commitments": [_serialise_commitment(c, step.state.turn)
                        for c in step.commitments],
        "nodes": {p.value: n for p, n in (step.nodes or {}).items()},
        # The trust panel was dead because nothing ever sent it this.
        "beliefs": [
            {
                "observer": b.observer.value,
                "subject": b.subject.value,
                "commitment_type": b.commitment_type.value,
                "alpha": round(b.alpha, 3),
                "beta": round(b.beta_param, 3),
                "reliability": round(b.expected_reliability, 3),
                # "...and does this player keep promises to *me*", which is
                # what the observer actually signs on.
                "reliability_toward": (
                    None if b.reliability_toward_observer is None
                    else round(b.reliability_toward_observer, 3)),
            }
            for b in step.beliefs
        ],
        "traces": {
            p.value: {
                "expected_value": t.expected_value,
                "penalty": t.penalty,
                "explanation": t.explanation,
                "commitment_broken": t.commitment_broken,
                "nodes": getattr(t, "nodes", 0),
                "candidates": getattr(t, "candidates", 0),
                "pruned": getattr(t, "pruned", 0),
                "vengeance": round(getattr(t, "vengeance", 0.0), 3),
                "search": getattr(t, "search", "expectiminimax"),
                "penalty_rows": [
                    {"partner": r[0].value, "vcoop": round(r[1], 3),
                     "delta_p": round(r[2], 3), "horizon": round(r[3], 3),
                     "amount": round(r[4], 3)}
                    for r in getattr(t, "penalty_rows", [])
                ],
                "orders": [
                    {"unit_territory": o.unit_territory,
                     "order_type": o.order_type.value,
                     "target": o.target, "supported_from": o.supported_from}
                    for o in (t.candidate_orders or [])
                ],
            } if t else None
            for p, t in step.traces.items()
        },
    }


def _game_summary(g) -> dict:
    runner = g["runner"]
    counts = {p.value: c for p, c in runner.center_counts().items()}
    return {
        "state": _serialise_state(runner.state),
        "scores": counts,
        "finished": g["finished"],
        "winner": g["winner"],
        "turn": runner.state.turn,
        "max_turns": runner.max_turns,
        "win_centers": WIN_CENTERS,
        "history_length": len(runner.history),
        "commitments": [_serialise_commitment(c, runner.state.turn)
                        for c in runner.commitments],
        "human_seat": g["human_seat"].value if g["human_seat"] else None,
        "personas": {p.value: getattr(a, "persona_name", "?")
                     for p, a in runner.agents.items()},
        "brier": brier_score(runner.calibration),
        "reliability": reliability_bins(runner.calibration, bins=5),
    }


# ── Routes ──────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def read_root():
    if DIST is None:
        return PlainTextResponse(
            "The dashboard has not been built.\n\n"
            "    npm --prefix web install && npm --prefix web run build\n\n"
            "The API is up regardless: try /api/board.",
            status_code=503,
        )
    return FileResponse(os.path.join(DIST, "index.html"))


@app.get("/api/board")
async def board():
    """Static map data, so the page cannot drift out of sync with the engine."""
    return {
        "territories": get_all_territories(),
        "supply_centers": get_supply_centers(),
        "adjacency": ADJACENCY,
        "players": [p.value for p in Player],
        "max_turns": MAX_TURNS,
        "win_centers": WIN_CENTERS,
    }


class NewGame(BaseModel):
    seed: Optional[int] = None
    human_seat: Optional[str] = None      # "Red" etc, or None for spectating
    chaos_seat: Optional[str] = None      # put the chaos agent in a chair
    broadcast: bool = True
    search: str = "expectiminimax"


@app.post("/api/game/new")
async def new_game(cfg: NewGame = NewGame()):
    """Create a new game. Returns the game ID."""
    game_id = str(uuid.uuid4())[:8]
    seed = cfg.seed if cfg.seed is not None else random.randrange(1, 10 ** 6)
    random.seed(seed)

    human = Player(cfg.human_seat) if cfg.human_seat else None
    chaos = Player(cfg.chaos_seat) if cfg.chaos_seat else None

    from src.agents.planner.planner import PlannerConfig
    agents = []
    for p, persona in DEFAULT_PERSONAS.items():
        if p == human:
            agents.append(HumanAgent(p, persona))
        elif p == chaos:
            agents.append(ChaosAgent(p, seed=seed))
        else:
            agents.append(Agent(p, persona, PlannerConfig(search=cfg.search)))

    runner = GameRunner(agents, broadcast_enabled=cfg.broadcast)
    _games[game_id] = {
        "runner": runner, "finished": False, "winner": None,
        "human_seat": human, "seed": seed,
    }
    return {"game_id": game_id, "seed": seed, **_game_summary(_games[game_id])}


def _get(game_id: str):
    return _games.get(game_id)


@app.get("/api/game/{game_id}/state")
async def get_state(game_id: str):
    g = _get(game_id)
    if g is None:
        return JSONResponse(status_code=404, content={"error": "Game not found"})
    return _game_summary(g)


@app.get("/api/game/{game_id}/pending")
async def pending(game_id: str):
    """Proposals waiting on the human seat, plus its legal orders.

    Calling this runs the turn's opening proposals but nothing else, so the
    examiner sees what they have been offered before committing to orders.
    """
    g = _get(game_id)
    if g is None:
        return JSONResponse(status_code=404, content={"error": "Game not found"})
    runner, seat = g["runner"], g["human_seat"]
    if seat is None or g["finished"]:
        return {"proposals": [], "legal": {}}

    messages = runner.begin_turn()
    mine = [m for m in messages if m.receiver == seat
            and m.message_type in (MessageType.PROPOSE, MessageType.THREAT)]
    return {
        "proposals": [_serialise_message(m) for m in mine],
        "legal": _legal_orders(runner, seat),
    }


def _legal_orders(runner, seat: Player) -> Dict[str, List[dict]]:
    """Every order each of the seat's units may legally be given."""
    out: Dict[str, List[dict]] = {}
    for u in runner.state.units:
        if u.player != seat:
            continue
        out[u.territory] = [
            {"order_type": o.order_type.value, "target": o.target,
             "supported_from": o.supported_from,
             "label": _order_label(o)}
            for o in generate_orders_for_unit(runner.state, u)
        ]
    return out


def _order_label(o: Order) -> str:
    if o.order_type == OrderType.HOLD:
        return f"{o.unit_territory} holds"
    if o.order_type == OrderType.MOVE:
        return f"{o.unit_territory} → {o.target}"
    if o.supported_from:
        return f"{o.unit_territory} supports {o.supported_from} → {o.target}"
    return f"{o.unit_territory} supports {o.target} holding"


class HumanTurn(BaseModel):
    orders: List[dict] = []
    decisions: Dict[str, bool] = {}   # message id -> accept?


@app.post("/api/game/{game_id}/step")
async def step_game(game_id: str, turn: HumanTurn = HumanTurn()):
    """Advance the game by one turn."""
    g = _get(game_id)
    if g is None:
        return JSONResponse(status_code=404, content={"error": "Game not found"})
    if g["finished"]:
        return {"finished": True, "message": "Game already finished",
                **_game_summary(g)}

    runner, seat = g["runner"], g["human_seat"]
    if seat is not None:
        agent = runner.agents[seat]
        agent.pending_orders = [
            Order(player=seat, unit_territory=o["unit_territory"],
                  order_type=OrderType(o["order_type"]),
                  target=o.get("target"), supported_from=o.get("supported_from"))
            for o in turn.orders
        ]
        # Decisions arrive keyed by message id; the agent keys them by the
        # deal so a counter-offer inherits the same answer.
        offered = {m.id: m for m in runner.begin_turn()}
        agent.decisions = {
            (m.sender, m.commitment_type): accept
            for mid, accept in turn.decisions.items()
            if (m := offered.get(mid)) is not None
        }
        agent.inbox = []

    played_turn = runner.state.turn
    step_data = runner.step()

    champion = runner.winner()
    if champion:
        g["finished"] = True
        g["winner"] = champion.value
    elif runner.state.turn > runner.max_turns:
        g["finished"] = True
        counts = runner.center_counts()
        g["winner"] = max(counts, key=lambda p: counts[p]).value

    return {
        "turn": played_turn,
        "step": _serialise_history_step(step_data),
        **_game_summary(g),
    }


@app.get("/api/game/{game_id}/replay")
async def get_replay(game_id: str):
    g = _get(game_id)
    if g is None:
        return JSONResponse(status_code=404, content={"error": "Game not found"})
    runner = g["runner"]
    return {
        "turns": len(runner.history),
        "history": [_serialise_history_step(s) for s in runner.history],
        **_game_summary(g),
    }


@app.get("/api/game/{game_id}/inspect/{player}")
async def inspect(game_id: str, player: str, turn: Optional[int] = None):
    """Everything behind one seat's current decision, step by step.

    This is the dashboard's detailed mode: not a summary of what the agent
    did, but the intermediate quantities it did it from — every live deal
    priced three ways, the trust network's two inputs and its output, and the
    candidate order sets that survived pruning with their scores. Re-derived
    on demand from the live state rather than logged every turn, because
    logging all of it for four seats over twelve turns is most of a megabyte
    per game and nobody reads 95% of it.
    """
    g = _get(game_id)
    if g is None:
        return JSONResponse(status_code=404, content={"error": "Game not found"})
    try:
        seat = Player(player)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": f"no such seat {player!r}"})

    runner = g["runner"]
    agent = runner.agents[seat]
    state = runner.state
    if not hasattr(agent, "negotiation"):
        return {"seat": seat.value, "kind": "not an AI seat", "deals": [],
                "beliefs": [], "candidates": []}

    from src.agents.planner.planner import (
        evaluate_state, cooperation_value, penalty_breakdown, breaks_commitment,
        defection_incentive,
    )
    from src.common.schemas import CommitmentType

    ns = agent.negotiation
    ns._live_commitments = runner.commitments

    # ── every live deal, priced the way the agent prices it ──────────────
    deals = []
    for c in runner.commitments:
        if seat not in c.players:
            continue
        others = [p for p in c.players if p != seat]
        v_none, v_kept, v_broken = ns.deal_totals(state, c, others[0], runner.commitments)
        iota = defection_incentive(state, seat, [c])
        rows = []
        for other in others:
            p_keep = agent.trust_model.p_keeps(
                other, c.commitment_type,
                defection_incentive(state, other, [c]), toward=seat)
            rows.append({
                "partner": other.value,
                "p_keeps": round(p_keep, 4),
                "reliability_general": round(
                    agent.trust_model.get_reliability(other, c.commitment_type), 4),
                "reliability_toward_me": round(
                    agent.trust_model.get_reliability(
                        other, c.commitment_type, toward=seat), 4),
                "their_incentive": round(defection_incentive(state, other, [c]), 4),
                "vcoop": round(cooperation_value(state, seat, other), 4),
                "my_delta_p_if_i_break": round(
                    agent.trust_model.reputation_drop(
                        seat, c.commitment_type, iota, toward=other), 4),
            })
        deals.append({
            "commitment": _serialise_commitment(c, state.turn),
            "v_none": round(v_none, 4),
            "v_kept": round(v_kept, 4),
            "v_broken": round(v_broken, 4),
            "expected_value_of_keeping": round(
                sum(r["p_keeps"] for r in rows) / max(1, len(rows)) * v_kept
                + (1 - sum(r["p_keeps"] for r in rows) / max(1, len(rows))) * v_broken,
                4),
            "my_incentive_to_break": round(iota, 4),
            "parties": rows,
        })

    # ── the candidate order sets, scored ─────────────────────────────────
    from src.engine.orders import generate_all_order_sets
    from src.engine.adjudicator import resolve as _resolve

    planner = agent.planner
    planner.nodes = 0
    all_sets = generate_all_order_sets(state, seat)
    kept_sets, pruned = planner.prune(state, all_sets, runner.commitments)
    worlds = ns._worlds(state, next(iter(
        [p for p in Player if p != seat])), None)[1]

    candidates = []
    for cand in kept_sets[:12]:
        after, _o, _l = _resolve(state, cand + worlds, runner.commitments)
        breaks = breaks_commitment(state, seat, cand, runner.commitments)
        rows = penalty_breakdown(state, seat, cand, runner.commitments,
                                 planner.config, agent.trust_model,
                                 defection_incentive(state, seat, runner.commitments))
        candidates.append({
            "orders": [
                {"unit": o.unit_territory, "type": o.order_type.value,
                 "target": o.target, "supported_from": o.supported_from}
                for o in cand
            ],
            "value": round(evaluate_state(after, seat), 4),
            "breaks_a_promise": breaks,
            "penalty": round(sum(r[-1] for r in rows), 4),
            "penalty_rows": [
                {"partner": r[0].value, "vcoop": round(r[1], 4),
                 "delta_p": round(r[2], 4), "horizon": round(r[3], 4),
                 "amount": round(r[4], 4)}
                for r in rows
            ],
        })
    candidates.sort(key=lambda c: c["value"] - c["penalty"], reverse=True)

    # ── the belief table, both layers ────────────────────────────────────
    beliefs = []
    for subject in Player:
        if subject == seat:
            continue
        for c_type in CommitmentType:
            rec = agent.trust_model.get_record(subject, c_type)
            pair = agent.trust_model.get_pair_record(subject, seat, c_type)
            beliefs.append({
                "subject": subject.value,
                "commitment_type": c_type.value,
                "alpha": round(rec.alpha, 3), "beta": round(rec.beta, 3),
                "reliability_general": round(rec.get_expected_value(), 4),
                "pair_alpha": round(pair.alpha, 3), "pair_beta": round(pair.beta, 3),
                "reliability_toward_me": round(
                    agent.trust_model.get_reliability(subject, c_type, toward=seat), 4),
                "pair_weight": round(agent.trust_model._pair_weight(pair), 4),
            })

    return {
        "seat": seat.value,
        "persona": getattr(agent, "persona_name", "?"),
        "turn": state.turn,
        "position_value": round(evaluate_state(state, seat), 4),
        "stance": {p.value: round(v, 4) for p, v in agent.stance().items()},
        "grudges": {p.value: round(v, 3) for p, v in agent.grudges.items()},
        "deterrence": {p.value: round(v, 3) for p, v in agent.deterrence.items()},
        "search": {
            "algorithm": planner.config.search,
            "depth": planner.config.depth,
            "opponent_samples": planner.config.opponent_samples,
            "node_budget": planner.config.node_budget,
            "candidates_total": len(all_sets),
            "candidates_kept": len(kept_sets),
            "pruned": pruned,
            "adjudications_spent_pruning": planner.nodes,
        },
        "deals": deals,
        "candidates": candidates,
        "beliefs": beliefs,
    }


@app.post("/api/game/{game_id}/save")
async def save(game_id: str):
    g = _get(game_id)
    if g is None:
        return JSONResponse(status_code=404, content={"error": "Game not found"})
    runner = g["runner"]
    path = save_replay(runner.history, runner.state)
    return {"path": path}


# Mounted last so it cannot shadow an /api route.
if DIST is not None:
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")),
              name="assets")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
