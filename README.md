# Territory

A 4-player territory game played by software agents. They negotiate through a
closed grammar, keep explicit Beta-distributed beliefs about who honours a
promise, and choose orders with adversarial search that prices the reputation
cost of breaking one. Betrayal is not scripted.

Deals can be struck privately, bind three players at once, or trade a favour
now for a repayment later. Trust is tracked both as a public reputation and
per relationship: a public deal's verdict moves everyone's view of the
players in it, a private deal's only the view of the players who signed it.

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
`python`), so `docker run --rm territory-report scripts/run_sweep.py --games 5`
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
python scripts/run_headless.py               # one game, no UI
python scripts/play.py --seat Red            # play a seat in the terminal
python scripts/run_analysis.py               # metrics per persona, with CIs
python scripts/run_calibration.py            # held-out-policy trust calibration
python scripts/run_trust_ablation.py         # paired trust forecast experiment
python scripts/run_keep_coverage.py          # keeping-candidate experiment
python scripts/run_tournament.py             # multi-game tournament
python scripts/make_figures.py               # report figures -> figures/
python scripts/run_sweep.py                  # sweep the knobs -> data/sweep.json
pytest tests/ -q
```

Every script that plays games takes the same game flags: the board
(`--standard`, `--seats`, `--homes`, `--outer-rings`, `--turns`,
`--win-centers`, `--win-fraction`, `--build-anywhere`, `--forts`, `--sieges`,
`--draws`) and the agents (`--search expectimax|mcts|regret`, `--budget`,
`--knob NAME=VALUE`). Multi-game scripts play `--games 60` by default, a
multiple of the 12-seed seating cycle and about the fewest that separates
settings; the full sweep is 108 cells of that, so pass `--games 20` for a quicker
one.

The trust ablation fits a constant keep forecast on separate training games,
then compares it with the learned forecast in paired test games. It writes a
summary JSON and a compressed per-offer log beside it. Research runs key
random draws by turn, stage, and seat so earlier decisions do not shift every
later planner sample.

The keeping-candidate experiment records whether a legal promise-keeping order
was generated, survived pruning, and was scored. Its reservation option is
experimental and is disabled in ordinary games.

Every run is built from one `GameConfig` (`src/common/config.py`) and played
through `harness.play()`, so a result can be reproduced from the config the
run prints. Personas rotate one seat per seed unless a seating is given, and
any tuning constant can be overridden by name:

```bash
python scripts/run_headless.py --seed 42 --knob FORFEIT_WEIGHT=0
python scripts/run_sweep.py --games 20 --axes LAMBDA_INCENTIVE=0.2,0.6
```

Multi-game scripts (analysis, tournament, figures, sweep) play games in parallel
through `harness.play_many()`. Each game is deterministic in its seed, so the
results match a serial run. The worker count is one per spare core, capped so
the workers fit in half the memory available when the run starts (inside Docker, what the container limit leaves) at
512 MB each; set `GAME_WORKERS=N` to override it, and `GAME_WORKERS=1` to run
serially.

In the dashboard: **New game** opens a form for seed, persona per seat,
search, budget and negotiation rounds, and the effective config
is shown so a screenshot is reproducible. **seat** to play one yourself,
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
  common/config.py                GameConfig - seed, seating, search, knobs
  harness.py                      play(cfg) - the one way a game is built
  engine/
    board.py                      Board + ring_board(n) - map, seats, win condition
    standard_map.py               the 7-power map with fleets, coasts and seas
    orders.py                     legal order enumeration
    adjudicator.py                simultaneous resolution, commitment grading
    runner.py                     step() - the turn loop
    replay.py                     save/load game histories
  agents/
    agent.py                      Agent, HumanAgent
    chaos.py                      ChaosAgent - six misbehaviour modes
    trust/model.py                Beta records, P(keeps) network
    trust/rules.py                rule base R0-R2, R5
    planner/planner.py            expectimax, MCTS, regret matching, pruning, pricing
    planner/regret.py             RM+ solver for one simultaneous-move turn
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

By default: 12 territories, 10 supply centres, 4 players, 2 units each. Orders
are move, support move, support hold and hold, submitted simultaneously. Attack
strength is 1 plus supports and must strictly exceed the defence; equal strength
bounces and supports can be cut, though not by an attack from the supporter's
own side. A dislodged unit retreats to an adjacent empty square that is not
where its attacker came from or a square left empty by a bounce; with nowhere
to go, or when two units retreat to the same square, it disbands. Odd turns
are spring, even turns autumn, when centres change hands and each player
chooses where to build (it may waive a build) and which units to disband.
New units go only on the player's own empty home centres, as in standard
Diplomacy; `--build-anywhere` (or unticking the dashboard's "home builds" box)
allows any owned centre. A game ends when someone holds 6 centres, or after
12 turns. With `--draws` (the dashboard's "draws") the survivors can also end
it early by a unanimous draw vote, shared equally among them; an agent votes
for one after four turns without a centre changing hands, unless it leads alone.

The ring board has armies only: every square is land. `--standard` (the
dashboard's "standard map") plays the real 7-power map instead: 75 provinces,
34 centres, 19 seas, 18 centres to win. Austria is Red, England Blue, France
Teal, Germany Orange, Italy Green, Russia Purple, Turkey Gold. It adds fleets,
the split coasts of Spain, St Petersburg and Bulgaria, and convoys: an army
crosses water when a chain of its convoying fleets reaches the far shore, and
stays put if one of them is dislodged. A convoyed army never fights a head-to-head
and does not cut a support aimed at its own convoying fleet. Builds offer an
army or a fleet on each coastal home centre. Agents only consider moving an
army by sea together with their own fleet ordered to carry it. `--seats`,
`--homes` and `--outer-rings` apply to the ring only.

```bash
python scripts/run_headless.py --standard --turns 20 --budget 300
```

None of those numbers are fixed. Every script takes `--seats` (2–8), `--homes`
(home centres, and so units, per player), `--win-centers`, `--win-fraction` (the
share of the board's centres that wins when `--win-centers` is not given, 0.6 by
default) and `--turns`, and
the dashboard has the same controls; the map is generated to match and is proved
symmetric before it is played on. The default is exactly the board above.

`--outer-rings N` (0–5, the dashboard's "outer rings") adds N rings of neutral
supply centres outside the home ring, one per home centre each, for a larger
map with more to fight over: 4 seats and 2 rings is 26 centres instead of 10.

```bash
python scripts/run_headless.py --seats 6 --outer-rings 2 --turns 20
python scripts/play.py --seat Purple --seats 6
```

`--forts` (the dashboard's "forts") makes every home centre defend at
+1, so taking one needs two supports instead of one. `--knob ORDER_NOISE=0.1`
(the dashboard's "noise") loses each order with that probability and holds the
unit instead; promises are graded on what was executed, so a lost support reads
as a broken one. `TurnRecord.intended` and `.slipped` keep what was meant, and
each outcome's `accidental` lists the breakers whose intended orders kept it.

`--sieges` makes a centre change hands only if the unit was already on it after
spring, so a defender has the autumn to answer; the planner then credits an
occupation in progress at `SIEGE_CREDIT` of a centre, because the capture lies
past its search horizon. `GameConfig(regime_change={"seat": "Red", "turn": 5,
"persona": "Opportunist"})` switches a seat's persona mid-game, keeping its
beliefs and grudges: a change of leadership the others are not told about.

Reputation can carry from one game to the next. `harness.play_series(cfgs)`
plays games in order, each starting from what every persona learned about the
others in the last one, keyed by persona since seats rotate;
`GameConfig(erase_reputation=True)` starts a game with no history. The
dashboard carries the most recent game's reputation into a new one unless
"erase reputation" is ticked. What carries is the trust record per promise
type and a "keeps when tempted" record, which is what separates personas:
everyone keeps 93-97% of promises that cost nothing, but when breaking pays
Honest keeps 0.90 and the Opportunist 0.58. Within one game there is too
little of that evidence to learn from; carried across games, predictions
beat the base rate from the second game on. A game whose reputation carries
forward (every game of a series but the last, and dashboard games) charges a
broken promise its full reputation cost to the end, instead of letting it
fall to zero on the last turn.

For a map that is not a ring, pass `GameConfig(board=...)` an explicit
`Board.to_dict()` — see `tests/test_board_config.py`.

Promises come in four kinds — alliance, DMZ, support, and an exchange whose
two legs fall due on different turns. Any of them can be private, and an
alliance can bind three players as a single pact that forms only if everyone
signs and dissolves the moment one member breaks it. An exchange is the
conditional promise: if the support is not given, the deal dissolves and the
repayment is never owed.

After the deals are struck, each agent may state an intended move to everyone
(`INTENT`). It binds nobody; it is graded on the executed orders, each listener
keeps a candour record of the speaker, and its opponent model expects the
announced move with that probability. Going back on an announcement costs the
speaker `INTENT_COST` times its reputation coefficient, so Honest rarely does.

Agents also keep second-order beliefs: what each other player has seen of
them, replayed through that player's own update rules. An agent does not offer
deals to a player it expects to refuse it, and among the deals it does offer it
ranks by the Nash bargaining product of its own surplus and the partner's,
priced from the partner's seat with its standing in their eyes as P(keeps).
The same product picks a counter-offer, out of every shorter term and every
DMZ one square smaller, and picks a pact's two allies, multiplying all three
members' surpluses; a version anyone would refuse is not offered.

## Search

`--search` picks how each agent chooses orders, at the same adjudication budget:

- `expectimax` (default): depth-2 expectimax against sampled opponents.
- `mcts`: UCT over the same candidates with one random rollout ply.
- `regret`: solves the turn as a simultaneous-move game. RM+ runs over a few
  candidate order sets per seat, opponents scoring positions from their own
  seats and anchored toward the opponent model, and the agent plays a draw
  from its average strategy. The trace reports that mixed strategy and its
  exploitability gap.

At the root of every search the opponent model is level-1: each opponent order
set is weighted by its move-frequency prior times `exp(LEVEL1_BETA x value)`,
the value being what that set is worth from the opponent's own seat.
