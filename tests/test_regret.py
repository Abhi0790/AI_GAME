"""Regret matching converges on games with known equilibria, and the planner plays from it."""
import random

from src.agents.planner.regret import regret_matching, exploitability
from src.common.config import GameConfig
from src.harness import play


def test_matching_pennies_converges_to_half_half():
    payoff = lambda p, j: (1 if j[0] == j[1] else -1) * (1 if p == 0 else -1)
    sig, q, n = regret_matching([2, 2], payoff, 4000, rng=random.Random(1))
    assert n == 4000
    assert abs(sig[0][0] - 0.5) < 0.05 and abs(sig[1][0] - 0.5) < 0.05
    assert exploitability(sig, q) < 0.1


def test_dominated_action_is_dropped():
    # Prisoner's dilemma: defect (1) strictly dominates for both.
    table = {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)}
    sig, _q, _n = regret_matching([2, 2], lambda p, j: table[j][p], 2000,
                                  rng=random.Random(2))
    assert sig[0][1] > 0.95 and sig[1][1] > 0.95


def test_anchor_makes_me_respond_to_the_modelled_opponent():
    # I score 1 for matching player 1, who is indifferent; anchored to action 0,
    # player 1 is sampled as playing 0, so I learn to play 0.
    sig, _q, _n = regret_matching([2, 2], lambda p, j: float(p == 0 and j[0] == j[1]), 500,
                                  anchors=[None, [1.0, 0.0]], anchor_weight=1.0,
                                  rng=random.Random(3))
    assert sig[0][0] > 0.95


def test_budget_stops_the_solve():
    calls = []
    _s, _q, n = regret_matching([3], lambda p, j: calls.append(j) or 0.0, 100,
                                keep_going=lambda: len(calls) < 30)
    assert n == 10


def test_regret_planner_plays_a_mixed_strategy_within_budget():
    runner = play(GameConfig(seed=4, max_turns=2, node_budget=400, search="regret"))
    for step in runner.history:
        for p, t in step.traces.items():
            assert step.nodes[p] <= 400
            if getattr(t, "mixed", None):
                assert abs(sum(pr for _c, pr in t.mixed) - 1) < 1e-9
                assert t.candidate_orders in [c for c, _pr in t.mixed]
                assert t.exploitability >= 0


def test_regret_solve_always_has_a_way_to_keep_its_promises(monkeypatch):
    from src.agents.planner import planner as P
    from src.common.schemas import obligated_parties
    missing = []
    orig = P.Planner.regret_best_orders

    def wrapped(self, state, commitments, trust_model, stance=None):
        orders, trace = orig(self, state, commitments, trust_model, stance)
        due = any(self.player in obligated_parties(c, state.turn) for c in commitments)
        if due and getattr(trace, "mixed", None) and P.keeping_order_set(
                state, self.player, commitments) is not None:
            missing.append(all(P.breaks_commitment(state, self.player, c, commitments)
                               for c, _p in trace.mixed))
        return orders, trace
    monkeypatch.setattr(P.Planner, "regret_best_orders", wrapped)
    for seed in range(3):
        play(GameConfig(seed=seed, max_turns=6, node_budget=600, search="regret"))
    assert missing and not any(missing)
