# Territory

A 4-player territory game played by software agents. They negotiate through a
closed grammar, keep explicit Beta-distributed beliefs about who honours a
promise, and choose orders with adversarial search that prices the reputation
cost of breaking one. Betrayal is not scripted.

Deals can be struck privately, bind three players at once, or trade a favour
now for a repayment later. Trust is tracked both as a public reputation and
per relationship. Accusations can be false, and the engine can only settle
the ones about deals it was told about publicly.

## Docker

One Dockerfile, four images, selected with `--target`. They share the Python
dependency stage, so it is built once and cached across all of them.

```bash
docker compose up web                 # dashboard on http://localhost:8000
docker compose run --rm sim           # one headless game
docker compose run --rm test          # the test suite
docker compose run --rm report        # figures and sweep -> ./figures, ./data
```

Or directly:

```bash
docker build -t territory-web .                     # dashboard (default target)
docker build -t territory-sim --target sim .        # headless, no Node in the image
docker run --rm territory-sim scripts/run_headless.py --seed 7
```

The tool images take a script path as their argument (`ENTRYPOINT` is
`python`), so `docker run --rm territory-report scripts/run_sweep.py --seeds 5`
works — not `... python scripts/...`.

## Setup without Docker

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

npm --prefix web install             # the dashboard is a React app
npm --prefix web run build           # FastAPI serves the build from web/dist
```

## Running

```bash
python -m src.ui.app                         # dashboard on http://localhost:8000
npm --prefix web run dev                     # or Vite on :5173, proxying the API
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

**Detailed mode** — click any player's score card. A third column opens
showing the arithmetic behind that seat's next decision, in the order the
agent computes it: its position and grudges, its beliefs split into general
reputation and what it has seen from each player *personally*, every live
promise priced three ways (`V_none`, `V_kept`, `V_broken`) with the trust
network's inputs per party, the search budget and how much of the candidate
set dominance pruning removed, and each surviving order set with its
`value − Vcoop × ΔP × horizon` broken out.

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
    app.py                        FastAPI: API + serves the built dashboard
web/                              React dashboard (Vite)
  src/App.jsx                     shell, game loop, replay scrubbing
  src/components/Board.jsx        the map
  src/components/Panels.jsx       log, talks, deals, trust, why, calibration
  src/components/Inspector.jsx    detailed mode
  src/components/Seat.jsx         the human seat
scripts/                          demo, play, headless, analysis, figures, sweep
tests/                            engine, DATC cases, trust, negotiation,
                                  planner, deals, chaos, integration
Dockerfile                        four images: web, sim, test, report
```

## Rules

12 territories, 10 supply centres, 4 players, 2 units each. Orders are move,
support move, support hold and hold, submitted simultaneously. Attack strength
is 1 plus supports and must strictly exceed the defence; equal strength bounces
and supports can be cut. Odd turns are spring, even turns autumn, when centres
change hands and units are built or removed. A game ends when someone holds 5
centres, or after 12 turns.

Promises come in four kinds — alliance, DMZ, support, and an exchange whose
two legs fall due on different turns. Any of them can be private, and an
alliance can bind three players as a single pact that forms only if everyone
signs and dissolves the moment one member breaks it.
