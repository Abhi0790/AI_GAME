import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.schemas import Player, CommitmentType
from src.agents.agent import Agent
from src.engine.runner import GameRunner
import random

def main():
    print("==================================================")
    print("  DIPLOMACY AI DEMO - RATIONAL BETRAYAL SCENARIO  ")
    print("==================================================")
    print("Setting up a 12-turn game with 4 agents.")
    print("Agents:")
    print("  Red:   Opportunist (Rational, will betray if profitable)")
    print("  Blue:  Honest      (High reputation cost)")
    print("  Green: Paranoid    (Low initial trust)")
    print("  Gold:  Vengeful    (Sharp trust reactions)")
    print("--------------------------------------------------")
    
    random.seed(42) # Seed 42 showed a betrayal on turn 12. Let's trace it nicely.
    
    agents = [
        Agent(Player.RED, "Opportunist"),
        Agent(Player.BLUE, "Honest"),
        Agent(Player.GREEN, "Paranoid"),
        Agent(Player.GOLD, "Vengeful")
    ]
    
    runner = GameRunner(agents)
    
    # Run the game but intercept history logging to print detailed traces
    runner.run()
    
    print("==================================================")
    print("  DEMO COMPLETE: ANALYZING TRACES                 ")
    print("==================================================")
    
    # Look for the turn where betrayal happened
    for i, step in enumerate(runner.history):
        state, orders, outcomes, log, traces = (
            step.state, step.orders, step.outcomes, step.log, step.traces)
        broken = [o for o in outcomes if not o.kept]
        if broken:
            print(f"\n--- Turn {i+1}: Betrayal Detected! ---")
            for o in broken:
                print(f"Commitment {o.commitment.commitment_type} between {o.commitment.players} broken by {o.broken_by}")
                for p in o.broken_by:
                    trace = traces[p]
                    if trace:
                        print(f"\nDecision Trace for {p}:")
                        print(f"  Explanation: {trace.explanation}")
                        print(f"  Expected value of orders: {trace.expected_value:.2f}")
                        print(f"  Penalty calculated for breaking: {trace.penalty:.2f}")
                        print(f"  (Since Expected Value > Penalty, the betrayal was executed)")

if __name__ == "__main__":
    main()
