"""Parameter sweep — fit the constants instead of asserting them.

Three things in the model were numbers somebody picked:

  LAMBDA_INCENTIVE   how much a profitable betrayal is forgiven
  EVIDENCE_DECAY     how fast old evidence stops counting
  reputation_cost    the persona multiplier on the penalty

and two more define the trust network's conditional table (BN_FLOOR,
BN_TEMPTED). This module measures what each setting does, over a fixed set of
seeds so the comparison is like for like:

  * the betrayal-rate spread between Honest and Opportunist — if one constant
    is carrying the whole persona difference, that shows up here;
  * mean deltaP at the moment a deal is broken — the deck's worked example
    assumes ~0.65, and the sweep says what is actually reachable;
  * Brier score and calibration error — whether P(keeps) means anything.

The CPT is fitted directly on logged (reliability, incentive, outcome)
triples rather than by grid-searching whole games, because the anchors do not
change what agents do until they change what agents predict.
"""

from typing import Dict, List, Any, Optional, Tuple
import random
import itertools

from src.common.schemas import Player
from src.agents.agent import Agent
from src.agents.planner.planner import PlannerConfig
from src.agents.negotiation.personas import PERSONAS
from src.engine.runner import GameRunner
from src.agents.trust import model as trust_model_module
from src.evaluation.metrics import (
    betrayal_rate_per_persona, brier_score, calibration_error, vcoop_at_break,
    final_scores,
)

SEATS = [(Player.RED, "Opportunist"), (Player.BLUE, "Honest"),
         (Player.GREEN, "Paranoid"), (Player.GOLD, "Vengeful")]


def play(seed: int, rep_cost_scale: float = 1.0) -> GameRunner:
    random.seed(seed)
    agents = []
    for player, persona in SEATS:
        agent = Agent(player, persona)
        agent.planner.config.reputation_cost_coefficient = (
            PERSONAS[persona].reputation_cost * rep_cost_scale)
        agents.append(agent)
    runner = GameRunner(agents)
    runner.run(verbose=False)
    return runner


def measure(seeds: List[int], rep_cost_scale: float = 1.0) -> Dict[str, Any]:
    """Run the corpus once and reduce it to the numbers the report needs."""
    rates: Dict[str, List[float]] = {}
    delta_ps: List[float] = []
    calibration = []
    wins: Dict[str, int] = {name: 0 for _p, name in SEATS}

    for seed in seeds:
        runner = play(seed, rep_cost_scale)
        personas = runner.personas()
        for name, rate in betrayal_rate_per_persona(runner.history, personas).items():
            rates.setdefault(name, []).append(rate)
        delta_ps += [row["delta_p"] for row in vcoop_at_break(runner.history, personas)]
        calibration += runner.calibration
        scores = final_scores(runner.state)
        winner = max(scores, key=lambda p: scores[p])
        wins[personas[winner]] = wins.get(personas[winner], 0) + 1

    mean = lambda xs: sum(xs) / len(xs) if xs else None
    by_persona = {name: mean(v) for name, v in rates.items()}
    honest = by_persona.get("Honest") or 0.0
    opportunist = by_persona.get("Opportunist") or 0.0

    return {
        "lambda": trust_model_module.LAMBDA_INCENTIVE,
        "decay": trust_model_module.EVIDENCE_DECAY,
        "bn_floor": trust_model_module.BN_FLOOR,
        "bn_tempted": trust_model_module.BN_TEMPTED,
        "bn_incentive_scale": trust_model_module.BN_INCENTIVE_SCALE,
        "rep_cost_scale": rep_cost_scale,
        "betrayal_rate": by_persona,
        "persona_spread": opportunist - honest,
        "mean_delta_p": mean(delta_ps),
        "max_delta_p": max(delta_ps) if delta_ps else None,
        "brier": brier_score(calibration),
        "calibration_error": calibration_error(calibration),
        "wins": wins,
        "predictions": len([c for c in calibration if c.observed is not None]),
        "_calibration": calibration,
    }


def sweep(seeds: List[int],
          lambdas: Optional[List[float]] = None,
          decays: Optional[List[float]] = None,
          rep_scales: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    """Grid over the three penalty constants. Restores the globals after."""
    lambdas = lambdas or [0.2, 0.6, 0.9]
    decays = decays or [0.7, 0.85, 1.0]
    rep_scales = rep_scales or [0.5, 1.0, 2.0]

    original = (trust_model_module.LAMBDA_INCENTIVE, trust_model_module.EVIDENCE_DECAY)
    rows = []
    try:
        for lam, decay, scale in itertools.product(lambdas, decays, rep_scales):
            trust_model_module.set_parameters(lambda_incentive=lam, evidence_decay=decay)
            row = measure(seeds, scale)
            row.pop("_calibration", None)
            rows.append(row)
    finally:
        trust_model_module.set_parameters(lambda_incentive=original[0],
                                          evidence_decay=original[1])
    return rows


def fit_cpt(calibration: List, floors=None, tempteds=None, scales=None
            ) -> Tuple[float, float, float, float]:
    """Refit the Keeps node's conditional table on logged predictions.

    Each logged point carries the reliability and incentive that produced it,
    so the CPT can be re-evaluated on the same evidence without replaying a
    single game. Returns (floor, tempted_row, incentive_scale, brier).
    """
    floors = floors or [round(0.05 * i, 2) for i in range(0, 17)]
    tempteds = tempteds or [round(0.05 * i, 2) for i in range(0, 21)]
    scales = scales or [0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0]
    points = [(c.reliability, c.incentive, 1.0 if c.observed else 0.0)
              for c in calibration if c.observed is not None]
    if not points:
        return (trust_model_module.BN_FLOOR, trust_model_module.BN_TEMPTED,
                trust_model_module.BN_INCENTIVE_SCALE, float("nan"))

    best = None
    for floor, tempted, scale in itertools.product(floors, tempteds, scales):
        total = 0.0
        for r, i, observed in points:
            p = trust_model_module.p_keeps_given(r, i, floor, tempted, scale)
            total += (p - observed) ** 2
        score = total / len(points)
        if best is None or score < best[3]:
            best = (floor, tempted, scale, score)
    return best


def baseline_brier(calibration: List) -> Optional[float]:
    """What you score by predicting the corpus base rate every time. Any
    model that cannot beat this is not a model."""
    obs = [1.0 if c.observed else 0.0 for c in calibration if c.observed is not None]
    if not obs:
        return None
    base = sum(obs) / len(obs)
    return sum((base - o) ** 2 for o in obs) / len(obs)
