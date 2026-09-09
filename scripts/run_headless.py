import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.schemas import Player
from src.agents.agent import Agent
from src.engine.runner import GameRunner
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # 4 Agents
    agents = [
        Agent(Player.RED, "Opportunist"),
        Agent(Player.BLUE, "Honest"),
        Agent(Player.GREEN, "Paranoid"),
        Agent(Player.GOLD, "Vengeful")
    ]
    
    runner = GameRunner(agents)
    runner.run()

if __name__ == "__main__":
    main()
