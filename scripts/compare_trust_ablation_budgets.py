"""Compare paired trust-forecast effects across two planner budgets."""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.trust_ablation import compare_budgets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("low", help="lower-budget trust ablation JSON")
    parser.add_argument("high", help="higher-budget trust ablation JSON")
    parser.add_argument("--out", default="data/trust_ablation_budget_interaction.json")
    args = parser.parse_args()
    with open(args.low) as stream:
        low = json.load(stream)
    with open(args.high) as stream:
        high = json.load(stream)
    results = {
        "low_report": args.low,
        "high_report": args.high,
        "low_budget": low["config"]["node_budget"],
        "high_budget": high["config"]["node_budget"],
        "metrics": compare_budgets(low, high),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(results, stream, indent=2)
    for name, row in results["metrics"].items():
        print(f"{name}: high-minus-low trust effect {row['high_minus_low_ci95']}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
