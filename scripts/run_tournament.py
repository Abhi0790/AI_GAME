import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.schemas import Player
from src.agents.agent import Agent
from src.engine.runner import GameRunner
import random

def run_tournament(games: int = 5):
    wins = {p: 0 for p in Player}
    
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
        
        # Determine winner by max centers
        centers = {p: 0 for p in Player}
        for owner in runner.state.supply_centers.values():
            if owner: centers[owner] += 1
            
        max_c = max(centers.values())
        winners = [p for p, c in centers.items() if c == max_c]
        for w in winners:
            wins[w] += 1 # Share the win
            
    print("Tournament Results (Wins out of", games, "games):")
    for p, w in wins.items():
        print(f"{p}: {w}")

if __name__ == "__main__":
    run_tournament()
