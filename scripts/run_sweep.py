"""Fit the penalty constants and the trust network, and print the sweep.

    python scripts/run_sweep.py --seeds 6 --out data/sweep.json

Reports, for every combination of LAMBDA_INCENTIVE, EVIDENCE_DECAY and the
reputation-cost multiplier: the betrayal rate of each persona, the spread
between Honest and Opportunist, the deltaP actually reachable at a break, and
the calibration of P(keeps).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

from src.evaluation.sweep import sweep, measure, fit_cpt, baseline_brier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6, help="games per combination")
    ap.add_argument("--out", default="data/sweep.json")
    ap.add_argument("--quick", action="store_true",
                    help="fit the trust network only, skip the penalty grid")
    args = ap.parse_args()
    seeds = list(range(1, args.seeds + 1))

    print(f"Fitting the trust network on {len(seeds)} games...")
    base = measure(seeds)
    calibration = base.pop("_calibration")
    floor, tempted, scale, brier = fit_cpt(calibration)
    print(f"  current      : Brier {base['brier']:.4f}  ECE {base['calibration_error']:.4f}")
    print(f"  base rate    : Brier {baseline_brier(calibration):.4f}")
    print(f"  best CPT     : floor={floor} tempted={tempted} scale={scale}"
          f" -> Brier {brier:.4f}")
    print(f"  deltaP at a break: mean {base['mean_delta_p']:.3f}, max {base['max_delta_p']:.3f}")
    print(f"  betrayal rate: " + ", ".join(
        f"{k} {v:.1%}" for k, v in sorted(base['betrayal_rate'].items())))

    rows = []
    if not args.quick:
        print("\nSweeping lambda x decay x reputation cost "
              f"({len(seeds)} games each, this takes a few minutes)...")
        rows = sweep(seeds)
        header = f"{'lambda':>7} {'decay':>6} {'repx':>5} {'spread':>7} {'meandP':>7} {'brier':>7}"
        print(header)
        print("-" * len(header))
        for r in sorted(rows, key=lambda r: -r["persona_spread"]):
            print(f"{r['lambda']:>7.2f} {r['decay']:>6.2f} {r['rep_cost_scale']:>5.1f} "
                  f"{r['persona_spread']:>7.3f} {(r['mean_delta_p'] or 0):>7.3f} "
                  f"{(r['brier'] or 0):>7.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"fit": {"bn_floor": floor, "bn_tempted": tempted,
                           "bn_incentive_scale": scale, "brier": brier},
                   "baseline": base, "grid": rows}, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
