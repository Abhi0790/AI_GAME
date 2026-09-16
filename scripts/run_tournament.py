import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.schemas import Player
from src.agents.agent import Agent
from src.engine.runner import GameRunner
from src.evaluation.metrics import betrayal_rate_per_persona
import random

def run_tournament(games: int = 5):
    wins = {p: 0 for p in Player}
    by_persona = {}
    briers = []
    personas = {}
    
    for i in range(games):
        random.seed(i)
        agents = [
            Agent(Player.RED, "Opportunist"),
            Agent(Player.BLUE, "Honest"),
            Agent(Player.GREEN, "Paranoid"),
            Agent(Player.GOLD, "Vengeful")
        ]
        runner = GameRunner(agents)
        runner.run()
        personas = runner.personas()
        for name, rate in betrayal_rate_per_persona(runner.history, personas).items():
            by_persona.setdefault(name, []).append(rate)
        score = runner.brier_score()
        if score is not None:
            briers.append(score)

        # Determine winner by max centers
        centers = {p: 0 for p in Player}
        for owner in runner.state.supply_centers.values():
            if owner: centers[owner] += 1
            
        max_c = max(centers.values())
        winners = [p for p, c in centers.items() if c == max_c]
        for w in winners:
            wins[w] += 1 # Share the win
            
    print(f"\nTournament Results ({games} games):")
    for p, w in wins.items():
        print(f"  {p.value:6s} ({personas[p]:12s}): {w} wins  {'#' * (w * 4)}")

    print("\nBetrayal rate by persona, pooled:")
    for name, values in sorted(by_persona.items()):
        avg = sum(values) / len(values)
        print(f"  {name:12s}: {avg:.1%}")

    if briers:
        print(f"\nP(keeps) Brier across the tournament: "
              f"{sum(briers) / len(briers):.3f}  (0.25 = a coin flip)")

if __name__ == "__main__":
    run_tournament()
