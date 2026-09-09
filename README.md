# Negotiating agents that keep, and break, their word

A 4-player territory game where software agents form alliances through a formal negotiation language, maintain explicit beliefs about who can be trusted, and plan moves with adversarial search. Betrayal is never scripted. It is what the planner concludes when the reputation cost falls below the gain.

## Architecture

* **Engine:** Headless adjudicator that processes simultaneous turns, verifies commitments, and updates the board map (14 nodes, 6 supply centers).
* **Trust & Belief Model:** Uses Beta distributions to track trustworthiness. Explains changes using a forward-chaining rule base.
* **Planner:** Uses expectiminimax depth-1 search over possible order combinations, factoring in reputation cost.
* **Negotiation:** Closed structured grammar (`PROPOSE(ALLIANCE, turns)`, etc.).

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

You will see agents forming alliances and finally betraying each other on Turn 12 when the reputation cost shrinks, with explicit decision traces explaining the math.

## Testing

```bash
pytest tests/
```

## Running a Headless Simulation

```bash
python scripts/run_headless.py --seed 42
```
