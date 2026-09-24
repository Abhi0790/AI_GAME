"""Fit the penalty constants and the trust network, and print the sweep.

    python scripts/run_sweep.py --seeds 20 --out data/sweep.json
    python scripts/run_sweep.py --axes LAMBDA_INCENTIVE=0.2,0.6 EVIDENCE_DECAY=0.85

The grid defaults to the knobs in `sweep.default_axes()` and 20 seeds per
cell, because below `metrics.MIN_GAMES` a win rate is 0% or 100% and means
nothing. `--quick` is the opt-in escape hatch that fits the trust network and
skips the grid; it used to be how data/sweep.json ended up with an empty grid.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math

from src.evaluation.metrics import MIN_GAMES, fmt_bounds, fmt_ci
from src.evaluation.sweep import (
    sweep, measure, fit_cpt, baseline_brier, default_axes,
)


def parse_axes(specs):
    """--axes NAME=v1,v2 NAME2=v3 -> {NAME: [v1, v2], NAME2: [v3]}"""
    axes = {}
    for spec in specs:
        name, _, values = spec.partition("=")
        axes[name] = [float(v) for v in values.split(",") if v]
    return axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20, help="games per cell")
    ap.add_argument("--out", default="data/sweep.json")
    ap.add_argument("--axes", nargs="*", default=None,
                    help="override the grid, e.g. LAMBDA_INCENTIVE=0.2,0.6")
    ap.add_argument("--quick", action="store_true",
                    help="fit the trust network only, skip the grid")
    args = ap.parse_args()
    seeds = list(range(1, args.seeds + 1))
    if args.seeds < MIN_GAMES:
        print(f"warning: {args.seeds} seeds per cell is below MIN_GAMES="
              f"{MIN_GAMES}; every number below is tagged not reportable.")

    print(f"Fitting the trust network on {len(seeds)} games...")
    base = measure(seeds)
    calibration = base.pop("_calibration")
    ceiling, bias, temptation, brier = fit_cpt(calibration)

    print(f"  current      : Brier {fmt_ci(*base['brier_ci'], fmt='{:.4f}')}"
          f"  ECE {base['calibration_error']:.4f}")
    print(f"  base rate    : Brier {baseline_brier(calibration):.4f}")
    print(f"  best CPT     : ceiling={ceiling} bias={bias} temptation={temptation}"
          f" -> Brier {brier:.4f} (held out; constrained so a 3-time betrayer is refused)")
    print(f"                 NOT applied. To adopt, pass "
          f'knobs={{"BN_CEILING": {ceiling}, "BN_BIAS": {bias}, '
          f'"BN_TEMPTATION": {temptation}}}')
    print(f"  deltaP at a break: {fmt_ci(*base['delta_p'], fmt='{:.3f}')}"
          f"  max {base['max_delta_p']}")
    print("  betrayal rate by persona:")
    for name, triple in sorted(base["betrayal_rate"].items()):
        print(f"    {name:12s}: {fmt_ci(*triple)}")
    print(f"  win share by persona:")
    for name, quad in sorted(base["wins"].items()):
        print(f"    {name:12s}: {fmt_bounds(*quad)}")
    print(f"  preference reversals: {fmt_ci(*base['reversal_rate'])} of priced "
          f"deals ({base['reversal_rows']} rows) were signed and then worth more "
          f"broken than kept the same turn")

    rows = []
    if not args.quick:
        axes = parse_axes(args.axes) if args.axes else default_axes()
        cells = math.prod(len(v) for v in axes.values())
        print(f"\nSweeping {' x '.join(axes)} — {cells} cells x {len(seeds)} "
              f"seeds = {cells * len(seeds)} games. This takes a while.")
        rows = sweep(seeds, axes)
        names = list(axes)
        header = ("  ".join(f"{n[:9]:>9}" for n in names) +
                  f"  {'spread':>7}  {'mean dP':>24}  {'brier':>7}  {'reversals':>24}")
        print(header)
        print("-" * len(header))
        for r in sorted(rows, key=lambda r: -r["persona_spread"]):
            print("  ".join(f"{r['settings'][n]:>9.2f}" for n in names) +
                  f"  {r['persona_spread']:>7.3f}"
                  f"  {fmt_ci(*r['delta_p'], fmt='{:.3f}'):>24}"
                  f"  {(r['brier'] or 0):>7.3f}"
                  f"  {fmt_ci(*r['reversal_rate']):>28}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"fit": {"bn_ceiling": ceiling, "bn_bias": bias,
                           "bn_temptation": temptation, "brier": brier,
                           "applied": False},
                   "min_games": MIN_GAMES,
                   "baseline": base, "grid": rows}, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
