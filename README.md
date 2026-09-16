# Territory

A 4-player territory game played by software agents. They negotiate through a
closed grammar, keep explicit Beta-distributed beliefs about who honours a
promise, and choose orders with adversarial search that prices the reputation
cost of breaking one. Betrayal is not scripted.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python -m src.ui.app                         # dashboard on http://localhost:8000
python scripts/demo.py                       # betrayal demo, arithmetic printed
python scripts/run_headless.py --seed 42     # one game, no UI
python scripts/play.py --seat Red            # play a seat in the terminal
python scripts/run_analysis.py --games 10    # metrics per persona
python scripts/run_tournament.py             # multi-game tournament
python scripts/make_figures.py --games 8     # report figures -> figures/
python scripts/run_sweep.py --seeds 5        # fit constants -> data/sweep.json
pytest tests/ -q
```

In the dashboard: **New game** to start, **seat** to play one yourself,
**chaos** to seat a deliberately malfunctioning player. The slider replays
turns already played; left/right arrows step, space advances.

## Structure

```
src/
  common/schemas.py               shared types, negotiation grammar + validator
  engine/
    board.py                      12 territories, 10 supply centres, adjacency
    orders.py                     legal order enumeration
    adjudicator.py                simultaneous resolution, commitment grading
    runner.py                     step() - the turn loop
    replay.py                     save/load game histories
  agents/
    agent.py                      Agent, HumanAgent
    chaos.py                      ChaosAgent - six misbehaviour modes
    trust/model.py                Beta records, P(keeps) network
    trust/rules.py                rule base R0-R4
    planner/planner.py            expectiminimax, MCTS, pruning, pricing
    negotiation/strategy.py       deal valuation, proposals, replies
    negotiation/personas.py       four personas
    opponent_model/               per-opponent move statistics
  evaluation/
    metrics.py                    per-persona metrics, calibration
    figures.py                    report figures
    sweep.py                      parameter fitting
    analysis.py                   summaries, JSON export
  ui/
    app.py                        FastAPI dashboard
    templates/index.html          dashboard page
scripts/                          demo, play, headless, analysis, figures, sweep
tests/                            engine, DATC cases, trust, negotiation,
                                  planner, chaos, integration
```

## Rules

12 territories, 10 supply centres, 4 players, 2 units each. Orders are move,
support move, support hold and hold, submitted simultaneously. Attack strength
is 1 plus supports and must strictly exceed the defence; equal strength bounces
and supports can be cut. Odd turns are spring, even turns autumn, when centres
change hands and units are built or removed. A game ends when someone holds 5
centres, or after 12 turns.
