"""Sampled regret matching (RM+) for one simultaneous-move turn."""
import random
from typing import Callable, List, Optional, Sequence, Tuple


def _strategy(regrets: List[float]) -> List[float]:
    total = sum(regrets)
    n = len(regrets)
    return [r / total for r in regrets] if total > 0 else [1.0 / n] * n


def regret_matching(sizes: Sequence[int], utility: Callable[[int, Tuple[int, ...]], float],
                    iterations: int, anchors: Optional[List[Optional[List[float]]]] = None,
                    anchor_weight: float = 0.0, keep_going: Callable[[], bool] = lambda: True,
                    rng=None
                    ) -> Tuple[List[List[float]], List[List[float]], int]:
    """Approximate a coarse correlated equilibrium of an n-player normal-form game.

    `utility(p, joint)` is player p's payoff when each player i plays action
    joint[i]. Each iteration samples one joint action from the current
    strategies (or, with probability `anchor_weight`, a player's anchor
    distribution) and updates every player's regret for every action against
    it. Returns (average strategies, mean action values, iterations run).
    """
    rng = rng or random
    players = len(sizes)
    regrets = [[0.0] * n for n in sizes]
    average = [[0.0] * n for n in sizes]
    q_sum = [[0.0] * n for n in sizes]
    done = 0
    for t in range(1, iterations + 1):
        if not keep_going():
            break
        strategies = [_strategy(r) for r in regrets]
        joint = []
        for p in range(players):
            dist = strategies[p]
            if anchors and anchors[p] and rng.random() < anchor_weight:
                dist = anchors[p]
            joint.append(rng.choices(range(sizes[p]), weights=dist)[0])
        for p in range(players):
            values = [utility(p, tuple(joint[:p] + [a] + joint[p + 1:]))
                      for a in range(sizes[p])]
            expected = sum(s * v for s, v in zip(strategies[p], values))
            for a, v in enumerate(values):
                regrets[p][a] = max(0.0, regrets[p][a] + v - expected)
                average[p][a] += t * strategies[p][a]   # linear averaging
                q_sum[p][a] += v
        done = t
    norm = [[x / sum(row) if sum(row) else 1.0 / len(row) for x in row] for row in average]
    q = [[x / done if done else 0.0 for x in row] for row in q_sum]
    return norm, q, done


def exploitability(strategies: List[List[float]], values: List[List[float]]) -> float:
    """Sum over players of best-response value minus the average strategy's value."""
    return sum(max(q) - sum(s * v for s, v in zip(sig, q))
               for sig, q in zip(strategies, values))
