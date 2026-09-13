# Negotiating agents that keep, and break, their word

A 4-player territory game where software agents form alliances through a formal negotiation language, maintain explicit beliefs about who can be trusted, and plan moves with adversarial search. Betrayal is never scripted. It is what the planner concludes when the reputation cost falls below the gain.

## Architecture

* **Engine:** Headless adjudicator that processes simultaneous turns, verifies commitments, and updates the board map (12 territories, 10 supply centres, 2 units per player). A game ends at turn 12 or the moment someone holds 5 centres.
* **Trust & Belief Model:** Uses Beta distributions to track trustworthiness. Explains changes using a forward-chaining rule base (R0–R3).
* **Planner:** Expectiminimax depth-1 search over order combinations, pricing every broken promise at `Vcoop(partner) x deltaP(keeps) x (turns left / 12)`. No betrayal flag exists: late in the game the horizon term shrinks, and mid-game `Vcoop` collapses once the partner stops being useful.
* **Opponent Model:** Tracks historical move patterns per opponent, estimates persona types, and feeds weighted probability distributions into the planner's sampling.
* **Negotiation:** Closed structured grammar — Alliance, DMZ, Support proposals, and Threat messages.
* **Evaluation:** Quantitative metrics (supply centre control, betrayal rates, alliance durations, win rates) with analysis reports and JSON export.
* **Replay System:** Full game histories saved/loaded as JSON for post-game review.
* **Web Dashboard:** Interactive SVG board with turn-by-turn controls, trust matrix, negotiation log, and decision trace viewer.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Demo

To run the deterministic demonstration of rational betrayal:

```bash
python scripts/demo.py
```

You will see agents forming alliances and finally betraying each other when the reputation cost shrinks, with explicit decision traces explaining the math.

## Running a Headless Simulation

```bash
python scripts/run_headless.py --seed 42
```

## Running the Web Dashboard

```bash
python -m src.ui.app
```

Then open http://localhost:8000 in your browser. Click **New Game** to start, then **Step** through turns or check **Auto** for animated playback.

## Running Analysis

Run multiple games and see aggregated statistics:

```bash
python scripts/run_analysis.py --games 10 --save-replays --export data/
```

Options:
- `--games N` — Number of games to run (default: 5)
- `--seed S` — Starting random seed (default: 0)
- `--save-replays` — Save replay JSON files to `replays/`
- `--export DIR` — Export analysis data to the specified directory

## Running a Tournament

```bash
python scripts/run_tournament.py
```

## Playing (Human vs AI, CLI)

```bash
python scripts/play.py
```

You play as Red against three AI opponents with a matplotlib board visualization.

## Testing

```bash
pytest tests/ -v
```

Test coverage includes:
- **Engine:** move resolution, bouncing, support
- **Commitments:** DMZ/Alliance violation detection
- **Chaos:** invalid/malformed orders
- **Trust:** Beta distribution updates, gossip propagation, all rule types
- **Negotiation:** proposal generation, evaluation, commitment conversion
- **Planner:** state evaluation, commitment penalties, order selection
- **Opponent Model:** profile tracking, persona estimation, weighted sampling
- **Integration:** full 12-turn games, replay save/load, evaluation metrics

## Project Structure

```
├── src/
│   ├── common/schemas.py          # Pydantic models (Player, Order, GameState, etc.)
│   ├── engine/
│   │   ├── board.py               # 12-territory map, 10 centres, adjacency
│   │   ├── orders.py              # Legal order enumeration
│   │   ├── adjudicator.py         # Simultaneous move resolution
│   │   ├── runner.py              # step() — the one turn loop, + gossip
│   │   └── replay.py              # Save/load game replays
│   ├── agents/
│   │   ├── agent.py               # Agent class with 4 personas
│   │   ├── trust/model.py         # Beta-distribution trust tracker
│   │   ├── trust/rules.py         # Forward-chaining rule engine (R0–R3)
│   │   ├── planner/planner.py     # Expectiminimax with reputation cost
│   │   ├── negotiation/strategy.py # Alliance/DMZ/Support/Threat proposals
│   │   ├── negotiation/personas.py # The four personas (parameters, not code)
│   │   └── opponent_model/opponent_model.py
│   ├── evaluation/
│   │   ├── metrics.py             # Quantitative game metrics
│   │   └── analysis.py            # Reports & JSON export
│   └── ui/
│       ├── app.py                 # FastAPI web dashboard
│       ├── visualize.py           # Matplotlib board renderer
│       └── templates/index.html   # Interactive SVG dashboard
├── scripts/
│   ├── demo.py                    # Betrayal demonstration
│   ├── play.py                    # Human-vs-AI CLI
│   ├── run_headless.py            # Headless simulation
│   ├── run_tournament.py          # Multi-game tournament
│   └── run_analysis.py            # Analysis & export
├── tests/                         # Comprehensive test suite
├── replays/                       # Saved game replays
├── data/                          # Exported analysis data
└── experiments/                   # Experiment logs
```
