"""The whole reportable corpus: figures and sweep, at a reportable size.

The old report target ran 8 games against MIN_GAMES=20 and never ran the
sweep, so the artifacts it produced said "not reportable" on every line and
were missing half the evidence.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import MIN_GAMES


def provenance() -> dict:
    """What produced these numbers, so a stale artifact can be spotted."""
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {
        "revision": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "min_games": MIN_GAMES,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=MIN_GAMES)
    ap.add_argument("--games-per-cell", type=int, default=None,
                    help="games behind each search x budget cell (default: --games)")
    ap.add_argument("--seeds", type=int, default=MIN_GAMES,
                    help="games per sweep cell")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--data", default="data")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    if args.games < MIN_GAMES:
        print(f"warning: --games {args.games} is below MIN_GAMES={MIN_GAMES}; "
              f"every figure will be tagged not reportable", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.data, exist_ok=True)
    prov = provenance()
    print(f"revision {prov['revision']} ({prov['branch']})"
          f"{' +dirty' if prov['dirty'] else ''}")

    sweep_path = os.path.join(args.data, "sweep.json")
    steps = [([sys.executable, "scripts/make_figures.py",
               "--games", str(args.games), "--games-per-cell",
               str(args.games_per_cell or args.games), "--out", args.out],
              os.path.join(args.out, "summary.json"))]
    if not args.skip_sweep:
        steps.append(([sys.executable, "scripts/run_sweep.py",
                       "--seeds", str(args.seeds), "--out", sweep_path],
                      sweep_path))

    produced = []
    for cmd, artifact in steps:
        print("+", " ".join(cmd), flush=True)
        if subprocess.call(cmd) != 0:
            return 1
        produced.append(artifact)

    # Only stamp what THIS run produced. Stamping an artifact we skipped would
    # put a current revision on a stale file, which is the exact confusion the
    # stamp exists to prevent.
    for path in produced:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            blob = json.load(f)
        if isinstance(blob, dict):
            blob["provenance"] = prov
            with open(path, "w") as f:
                json.dump(blob, f, indent=2)
            print(f"stamped {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
