"""Compare the keeping-candidate safeguard across two planner budgets."""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.keep_coverage import compare_budgets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("low")
    parser.add_argument("high")
    parser.add_argument("--out", default="data/keep_coverage_budget_interaction.json")
    args = parser.parse_args()
    with open(args.low) as stream:
        low = json.load(stream)
    with open(args.high) as stream:
        high = json.load(stream)
    report = {
        "low_report": args.low,
        "high_report": args.high,
        "metrics": compare_budgets(low, high),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(report, stream, indent=2)
    for name, row in report["metrics"].items():
        print(f"{name}: high-minus-low safeguard effect "
              f"{row['high_minus_low_ci95']}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
