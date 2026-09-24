"""Parameter sweep — fit the constants instead of asserting them.

The axes are now just knob names. Anything `src.harness.knob_values()` knows
about, plus any `GameConfig` field, can be an axis, and every cell is one
`GameConfig` that the grid JSON records verbatim — so a cell can be re-run
from the file it was written to.

What each cell reports, all with 95% intervals:

  * the betrayal rate of each persona, and the Honest/Opportunist spread —
    if one constant is carrying the whole persona difference, it shows here;
  * mean deltaP at the moment a deal is broken — the deck's worked example
    assumes ~0.65, and the sweep says what is actually reachable;
  * Brier score and calibration error — whether P(keeps) means anything;
  * the preference-reversal rate, which the penalty constants are supposed
    to be what closes.

The CPT is fitted directly on logged (reliability, incentive, outcome)
triples rather than by grid-searching whole games, because the anchors do not
change what agents do until they change what agents predict.
"""

from typing import Dict, List, Any, Optional, Tuple
import itertools
import time

import numpy as np
from scipy.optimize import minimize

from src.common.config import GameConfig
from src.common.schemas import leaders
from src.harness import play_many, knob_values
from src.agents.trust import model as trust_model_module
from src.evaluation.metrics import (
    betrayal_rate_per_persona, brier_score, calibration_error, vcoop_at_break,
    final_scores, preference_reversals, ci95, mean_ci95,
)

CFG_FIELDS = set(GameConfig.__dataclass_fields__)

# A fitted CPT must still let trust say no: a partner who broke their last
# DISTRUST_BREAKS promises, with nothing tempting them, must fall below
# DISTRUST_P, the P(keep) at which an even-odds deal is refused.
DISTRUST_BREAKS = 3
DISTRUST_P = 0.5


def betrayer_reliability(breaks: int = DISTRUST_BREAKS) -> float:
    """Reliability of a partner who broke `breaks` promises in a row, from a uniform prior."""
    record = trust_model_module.TrustRecord(1.0, 1.0)
    for _ in range(breaks):
        record.update(kept=False)
    return record.get_expected_value()


def allows_distrust(ceiling: float, bias: float, temptation: float) -> bool:
    return trust_model_module.p_keeps_given(
        betrayer_reliability(), 0.0, ceiling, bias, temptation) < DISTRUST_P


def default_axes() -> Dict[str, List[float]]:
    """The grid we actually care about. `PAIR_SHRINKAGE` was swept once
    (108 cells, 2026-09-22) and was flat on every metric, so it is back to
    judgement and out of the grid; `FORFEIT_WEIGHT` only exists once the planner prices a
    forfeited exchange leg, so it is included only when it is there."""
    axes = {
        "LAMBDA_INCENTIVE": [0.2, 0.6, 0.9],
        "EVIDENCE_DECAY": [0.7, 0.85, 1.0],
        "rep_cost_scale": [0.5, 1.0, 2.0],
    }
    if "FORFEIT_WEIGHT" in knob_values():
        axes["FORFEIT_WEIGHT"] = [0.5, 2.0]
    return axes


def cell_config(seed: int, settings: Dict[str, float]) -> GameConfig:
    """One cell's config. A setting is a GameConfig field if it is one, and a
    module knob otherwise — the caller never has to know which."""
    return GameConfig(
        seed=seed,
        knobs={k: v for k, v in settings.items() if k not in CFG_FIELDS},
        **{k: v for k, v in settings.items() if k in CFG_FIELDS},
    )


def _game_result(cfg: GameConfig, runner) -> Dict[str, Any]:
    """Everything `summarise` needs from one game, computed in the worker."""
    personas = runner.personas()
    rows = vcoop_at_break(runner.history, personas)
    for point in runner.calibration + runner.reversals:
        point.game = cfg.seed
    return {
        "brier": brier_score(runner.calibration),
        # None, not 0, for a game with no priced break: it has no deltaP to average.
        "delta_p": sum(r["delta_p"] for r in rows) / len(rows) if rows else None,
        "delta_ps": [r["delta_p"] for r in rows],
        "betrayal": betrayal_rate_per_persona(runner.history, personas),
        "calibration": runner.calibration,
        "reversals": runner.reversals,
        "leaders": [personas[p] for p in leaders(final_scores(runner.state))],
    }


def summarise(seeds: List[int], settings: Dict[str, float],
              games: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce one cell's per-game results to the numbers the report needs, each
    with the sample size and interval behind it."""
    rates: Dict[str, List[float]] = {}
    wins: Dict[str, float] = {}
    calibration: List = []
    reversals: List = []
    for g in games:
        for name, rate in g["betrayal"].items():
            rates.setdefault(name, []).append(rate)
        for name in g["leaders"]:
            wins[name] = wins.get(name, 0) + 1.0 / len(g["leaders"])
        calibration += g["calibration"]
        reversals += g["reversals"]
    delta_ps = [d for g in games for d in g["delta_ps"]]

    n = len(seeds)
    by_persona = {name: mean_ci95(v) for name, v in rates.items()}
    honest = (by_persona.get("Honest") or (0.0,))[0] or 0.0
    opportunist = (by_persona.get("Opportunist") or (0.0,))[0] or 0.0
    graded = [c for c in calibration if c.observed is not None]
    rev = preference_reversals(reversals)

    return {
        # The cell's first config in full (the rest differ only in seed, and
        # the seating rotates with it), so a row can be re-run from the file.
        "config": cell_config(seeds[0] if seeds else 0, settings).to_dict(),
        "settings": dict(settings),
        "seeds": list(seeds),
        "games": n,
        # (mean, half-width, n) throughout, and n is GAMES: rows from one game
        # share its seed, so they are not independent observations.
        "betrayal_rate": by_persona,
        "persona_spread": opportunist - honest,
        "wins": {name: (*ci95(w, n), n) for name, w in wins.items()},
        "delta_p": mean_ci95([g["delta_p"] for g in games]),
        "delta_p_rows": len(delta_ps),
        "max_delta_p": max(delta_ps) if delta_ps else None,
        "brier": brier_score(calibration),
        "brier_ci": mean_ci95([g["brier"] for g in games]),
        "brier_rows": len(graded),
        "calibration_error": calibration_error(calibration),
        "reversal_rate": rev.get("ci") or mean_ci95([]),
        "reversal_rows": rev.get("n") or 0,
        "predictions": len(graded),
        "_calibration": calibration,
    }


def measure(seeds: List[int], settings: Optional[Dict[str, float]] = None
            ) -> Dict[str, Any]:
    """Run one cell and summarise it."""
    settings = settings or {}
    games = play_many((cell_config(s, settings) for s in seeds), _game_result)
    return summarise(seeds, settings, games)


def sweep(seeds: List[int], axes: Optional[Dict[str, List[float]]] = None
          ) -> List[Dict[str, Any]]:
    """Grid over `axes` ({knob_name: [values]}). Every game of every cell goes
    to one worker pool, so a cell smaller than the pool does not idle it."""
    axes = axes or default_axes()
    names = list(axes)
    cells = [dict(zip(names, combo))
             for combo in itertools.product(*(axes[n] for n in names))]
    started = time.time()
    print(f"  {len(cells)} cells x {len(seeds)} seeds = {len(cells) * len(seeds)} games",
          flush=True)
    games = play_many((cell_config(s, cell) for cell in cells for s in seeds),
                      _game_result)
    print(f"  played in {(time.time() - started) / 60:.1f}m", flush=True)
    rows = []
    for i, cell in enumerate(cells):
        row = summarise(seeds, cell, games[i * len(seeds):(i + 1) * len(seeds)])
        row.pop("_calibration", None)
        rows.append(row)
    return rows


def fit_cpt(calibration: List, holdout: float = 0.3, distrust: bool = True
            ) -> Tuple[float, float, float, float]:
    """Refit the Keeps node's conditional table on logged predictions.

    Each logged point carries the reliability and incentive that produced it,
    so the CPT can be re-evaluated on the same evidence without replaying a
    single game. Fits (BN_CEILING, BN_BIAS, BN_TEMPTATION) by least squares --
    the Brier score -- and returns them with the Brier on a held-out split.

    The split is BY GAME so a game's rows cannot straddle the two -- they share
    a seed, a board and the same players. Pass holdout=0 for in-sample.

    Nothing is applied: the fitted values are printed with the knob names to
    set them under, because adopting a CPT changes which promises ever get
    graded, and it has to be judged on the games it produces itself.
    """
    graded = [c for c in calibration if c.observed is not None]
    current = (trust_model_module.BN_CEILING, trust_model_module.BN_BIAS,
               trust_model_module.BN_TEMPTATION)
    if not graded:
        return current + (float("nan"),)

    def rows(cs):
        return np.array([(c.reliability, c.incentive, 1.0 if c.observed else 0.0)
                         for c in cs])

    # Split by game where the corpus says which game a row came from, so the
    # holdout is genuinely unseen rather than a random slice of the same games.
    game = lambda c: getattr(c, "game", None) or getattr(c, "seed", None)
    keys = sorted({game(c) for c in graded} - {None})
    if holdout and len(keys) > 2:
        test_keys = set(keys[:max(1, int(len(keys) * holdout))])
        train = rows([c for c in graded if game(c) not in test_keys])
        test = rows([c for c in graded if game(c) in test_keys])
    elif holdout and len(graded) > 20:
        cut = max(1, int(len(graded) * holdout))
        train, test = rows(graded[cut:]), rows(graded[:cut])
    else:
        train = test = rows(graded)

    p_keeps = np.vectorize(trust_model_module.p_keeps_given)

    def brier(x, pts):
        return float(np.mean((p_keeps(pts[:, 0], pts[:, 1], *x) - pts[:, 2]) ** 2))

    bounds = [(0.0, 1.0), (-20.0, 20.0), (0.0, 1.0)]
    cons = []
    if distrust:
        # Unconstrained, the fit sits near the base rate and no partner can ever be refused.
        rb = betrayer_reliability()
        cons = [{"type": "ineq", "fun": lambda x: DISTRUST_P - 1e-3 - float(
            trust_model_module.p_keeps_given(rb, 0.0, *x))}]
    best = min((minimize(brier, x0, args=(train,), bounds=bounds, constraints=cons,
                         method="SLSQP")
                for x0 in (current, (0.9, -2.0, 0.1), (0.95, 0.0, 0.0))),
               key=lambda o: o.fun)
    x = tuple(round(float(v), 3) for v in best.x)
    if distrust and not allows_distrust(*x):     # rounding can step over the line
        x = (x[0], round(x[1] - 0.01, 3), x[2])
    # The reported number is the held-out one.
    return x + (brier(x, test),)


def baseline_brier(calibration: List) -> Optional[float]:
    """What you score by predicting the corpus base rate every time. Any
    model that cannot beat this is not a model."""
    obs = [1.0 if c.observed else 0.0 for c in calibration if c.observed is not None]
    if not obs:
        return None
    base = sum(obs) / len(obs)
    return sum((base - o) ** 2 for o in obs) / len(obs)
