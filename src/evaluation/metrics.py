"""
Evaluation Metrics — quantitative measures for analysing game outcomes.

Each function operates on a game history (list of turn records produced by
GameRunner) and returns structured data.
"""

import math
import statistics
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from src.engine.board import players
from src.common.schemas import (
    Player, GameState, CommitmentType, CalibrationPoint, ReversalPoint, leaders,
    obligated_parties,
)


# ── Uncertainty ──────────────────────────────────────────────────────────
# Nothing here used to carry an interval, so a 0% and a 100% measured over
# two games read like facts. Every rate a script prints now goes through
# these, and anything backed by fewer than MIN_GAMES games says so.

MIN_GAMES = 20


def ci95(successes: int, n: int) -> Tuple[float, float, float]:
    """Observed rate and the bounds of its 95% Wilson interval: (rate, lo, hi).

    Wilson rather than normal-approximation because it stays sane at 0/n and
    n/n. The interval is asymmetric about the observed rate -- 0 wins from 20
    is 0% with an upper bound near 16%, not 0% +/- 8% -- so the bounds are
    returned rather than a half-width somebody would centre on the wrong point.
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    z = 1.96
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def clustered_rate_ci95(groups: List[Tuple[int, int]]) -> Tuple[Optional[float], float, int]:
    """(pooled rate, 95% half-width, groups) for (trials, successes) per group.

    The pooled rate with a group-clustered standard error, so the interval is
    centred on the reported rate while counting groups, not rows.
    """
    total = sum(n for n, _ in groups)
    if not total:
        return None, 0.0, len(groups)
    rate = sum(b for _, b in groups) / total
    g = len(groups)
    if g < 2:
        return rate, 0.0, g
    var = g / (g - 1) * sum((b - rate * n) ** 2 for n, b in groups) / total ** 2
    return rate, 1.96 * math.sqrt(var), g


def mean_ci95(values: List[float]) -> Tuple[Optional[float], float, int]:
    """(mean, 95% half-width, n) for a list of measurements."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return None, 0.0, 0
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0, n
    return mean, 1.96 * statistics.stdev(vals) / math.sqrt(n), n


def _tag(n: int) -> str:
    return (f" [n={n} < {MIN_GAMES}: not reportable]" if n < MIN_GAMES
            else f" (n={n})")


def fmt_ci(value: Optional[float], half: float, n: int, fmt: str = "{:.1%}") -> str:
    """'32.0% +/-6.1% (n=418)', or the not-reportable tag below MIN_GAMES.

    The one place a number with an interval is turned into text, so the
    scripts cannot drift apart on how they say "we did not measure enough".
    """
    if value is None:
        return f"n/a (n={n})"
    return f"{fmt.format(value)} +/-{fmt.format(half)}{_tag(n)}"


def fmt_bounds(value: float, lo: float, hi: float, n: int,
               fmt: str = "{:.1%}") -> str:
    """'32.0% [26.1, 38.5] (n=418)' — an asymmetric interval, stated as bounds."""
    return (f"{fmt.format(value)} [{fmt.format(lo)}, {fmt.format(hi)}]"
            f"{_tag(n)}")


def fmt_rate(successes: int, n: int) -> str:
    """A rate out of n trials, with its Wilson interval."""
    return fmt_bounds(*ci95(successes, n), n)


def fmt_mean(values: List[float], fmt: str = "{:.3f}") -> str:
    """A mean of per-game measurements, with its interval."""
    return fmt_ci(*mean_ci95(values), fmt=fmt)


# ── Per-game metrics ─────────────────────────────────────────────────────

def supply_center_timeline(history: list, final_state: GameState = None
                           ) -> Dict[Player, List[int]]:
    """Centres held per player, one entry per turn.

    `step.state` is the position BEFORE that turn's orders, so the last
    autumn's captures never appeared. Pass `final_state` to close the series;
    without it the last turn of the game is still missing from the plot.
    """
    timeline: Dict[Player, List[int]] = {p: [] for p in players()}

    for step in history:
        state: GameState = step.state
        counts = {p: 0 for p in players()}
        for owner in state.supply_centers.values():
            if owner:
                counts[owner] += 1
        for p in players():
            timeline[p].append(counts[p])

    if final_state is not None:
        counts = {p: 0 for p in players()}
        for owner in final_state.supply_centers.values():
            if owner:
                counts[owner] += 1
        for p in players():
            timeline[p].append(counts[p])

    return timeline


def betrayal_events(history: list) -> List[Dict[str, Any]]:
    """Extract every betrayal from the game history."""
    events = []
    for turn_idx, step in enumerate(history):
        for o in step.outcomes:
            if not o.kept:
                events.append({
                    "turn": turn_idx + 1,
                    "commitment_type": o.commitment.commitment_type.value,
                    "players": [p.value for p in o.commitment.players],
                    "broken_by": [p.value for p in o.broken_by],
                })
    return events


def obligations(step) -> Dict[tuple, bool]:
    """{(player, partner): broke} for one turn: whether each player broke
    anything it owed each partner.

    The unit a betrayal rate counts. One hostile move typically breaks an
    alliance, a DMZ and an exchange with the same partner at once, and
    counting per deal made that one decision three betrayals.
    """
    out: Dict[tuple, bool] = {}
    for o in step.outcomes:
        for p in obligated_parties(o.commitment, step.state.turn):
            for q in o.commitment.players:
                if q != p:
                    out[(p, q)] = out.get((p, q), False) or p in o.broken_by
    return out


def betrayal_rate_per_player(history: list) -> Dict[Player, float]:
    """Fraction of (turn, partner) obligations each player broke."""
    participated: Dict[Player, int] = defaultdict(int)
    broken: Dict[Player, int] = defaultdict(int)

    for step in history:
        for (p, _q), broke in obligations(step).items():
            participated[p] += 1
            broken[p] += broke

    rates: Dict[Player, float] = {}
    for p in players():
        if participated[p] > 0:
            rates[p] = broken[p] / participated[p]
        else:
            rates[p] = 0.0
    return rates


def alliance_durations(history: list, kind: str = "Alliance", horizon: int = None
                       ) -> List[Dict[str, Any]]:
    """How long each deal of `kind` lasted before it broke or expired.

    Two things it used to get wrong: it covered every commitment type despite
    its name, and a deal still running when the game ended was recorded as
    having been kept for its full intended term. That is right-censoring, and
    counting it as a completed success overstates how long deals hold. Pass
    `horizon` (the last turn played) to mark those `censored` instead.
    """
    # Track when commitments first appear and when they break
    commitment_first_seen: Dict[str, int] = {}
    commitment_broken_turn: Dict[str, Optional[int]] = {}
    commitment_info: Dict[str, Any] = {}

    for turn_idx, step in enumerate(history):
        for o in step.outcomes:
            # The name says alliances; it used to count every deal type.
            if kind is not None and o.commitment.commitment_type.value != kind:
                continue
            cid = o.commitment.id
            if cid not in commitment_first_seen:
                commitment_first_seen[cid] = turn_idx + 1
                commitment_info[cid] = {
                    "type": o.commitment.commitment_type.value,
                    "players": [p.value for p in o.commitment.players],
                    "valid_until": o.commitment.valid_until_turn,
                }
            if not o.kept and cid not in commitment_broken_turn:
                commitment_broken_turn[cid] = turn_idx + 1

    records = []
    for cid, info in commitment_info.items():
        started = commitment_first_seen[cid]
        if cid in commitment_broken_turn:
            ended = commitment_broken_turn[cid]
            duration = ended - started
            status = "broken"
        else:
            # valid_until_turn is the LAST turn graded, inclusive, so a deal
            # struck and expiring on the same turn lasted one turn, not zero.
            ended = info["valid_until"]
            if horizon is not None and ended > horizon:
                # Still alive when the clock stopped: observed, not completed.
                ended = horizon
                status = "censored"
            else:
                status = "kept"
            duration = max(0, ended - started + 1)
        records.append({**info, "started": started, "ended": ended,
                        "duration": duration, "status": status})
    return records


def final_scores(state: GameState) -> Dict[Player, int]:
    """Return supply-center counts from the final game state."""
    counts = {p: 0 for p in players()}
    for owner in state.supply_centers.values():
        if owner:
            counts[owner] += 1
    return counts


# ── Cross-game (tournament) metrics ──────────────────────────────────────

def win_rates(game_results: List[Dict[Player, int]]) -> Dict[Player, float]:
    """Given a list of per-game score dicts, return win rates."""
    # Seats come from the results, not the currently installed board.
    seats = list(game_results[0]) if game_results else list(players())
    # A shared lead is split, so the rates sum to 1 rather than to the mean
    # number of tied leaders.
    wins: Dict[Player, float] = {p: 0.0 for p in seats}
    for scores in game_results:
        ahead = leaders(scores)
        for w in ahead:
            wins[w] += 1.0 / len(ahead)
    total = max(1, len(game_results))
    return {p: wins[p] / total for p in seats}


# ── Persona-keyed metrics ────────────────────────────────────────────────
#
# Everything above is keyed by seat colour, which is useless for the report:
# "Red betrays more than Blue" says nothing unless you know which persona sat
# in which chair. Every function below takes the seat -> persona map that
# GameRunner.personas() returns.

def betrayal_rate_per_persona(history: list, personas: Dict[Player, str]
                              ) -> Dict[str, float]:
    """Fraction of (turn, partner) obligations each persona broke."""
    participated: Dict[str, int] = defaultdict(int)
    broken: Dict[str, int] = defaultdict(int)

    for step in history:
        for (p, _q), broke in obligations(step).items():
            name = personas.get(p, "Unknown")
            participated[name] += 1
            broken[name] += broke

    return {name: broken[name] / participated[name]
            for name in participated if participated[name]}


def betrayal_rate_by_turn(history: list, personas: Dict[Player, str]
                          ) -> Dict[str, List[Optional[float]]]:
    """Betrayal rate per persona per turn — the first report figure.

    None where a persona had no live commitment that turn, so a gap in the
    line is an honest gap rather than a zero.
    """
    series: Dict[str, List[Optional[float]]] = defaultdict(list)
    for step in history:
        participated: Dict[str, int] = defaultdict(int)
        broken: Dict[str, int] = defaultdict(int)
        for (p, _q), broke in obligations(step).items():
            name = personas.get(p, "Unknown")
            participated[name] += 1
            broken[name] += broke
        for name in set(personas.values()):
            series[name].append(
                broken[name] / participated[name] if participated[name] else None)
    return dict(series)


def vcoop_at_break(history: list, personas: Dict[Player, str]) -> List[Dict[str, Any]]:
    """Every betrayal the planner actually priced, with the Vcoop it was
    walking away from.

    Read off the decision trace, which is the only place the number exists —
    it used to be computed every turn and discarded.
    """
    rows = []
    for turn_idx, step in enumerate(history):
        for player, trace in (step.traces or {}).items():
            for partner, vcoop, delta_p, horizon, amount in getattr(trace, "penalty_rows", []):
                rows.append({
                    "turn": turn_idx + 1,
                    "player": player.value,
                    "persona": personas.get(player, "Unknown"),
                    "partner": partner.value,
                    "vcoop": vcoop,
                    "delta_p": delta_p,
                    "horizon": horizon,
                    "penalty": amount,
                })
    return rows


def forfeit_at_break(history: list) -> List[Dict[str, Any]]:
    """The other half of the price of a break: what an unfinished exchange
    leg costs to walk away from, read off `trace.forfeit_rows`.

    Empty until the planner records them — a break priced without a forfeit is
    not an error, it just means no exchange was outstanding.
    """
    rows = []
    for turn_idx, step in enumerate(history):
        for player, trace in (step.traces or {}).items():
            for key, partner, price, fraction, amount in getattr(trace, "forfeit_rows", []):
                rows.append({
                    "turn": turn_idx + 1,
                    "player": player.value,
                    "commitment": key,
                    "partner": getattr(partner, "value", partner),
                    "price": price,
                    "fraction_remaining": fraction,
                    "forfeit": amount,
                })
    return rows


def adjudications_per_turn(history: list) -> List[int]:
    """Total calls to the adjudicator each turn, across all seats — the
    x-axis of the search-variant comparison."""
    return [sum((step.nodes or {}).values()) for step in history]


def turns_to_coalition(history: list, stable_for: int = 3) -> Optional[int]:
    """The turn a coalition that actually *holds* first exists, or None.

    "An alliance exists" is not a coalition — under expected-value
    negotiation somebody signs one on turn 1 of every game, so that version
    of this metric was a constant. What matters is an alliance that has
    survived `stable_for` consecutive turns without either side breaking it.
    """
    from src.common.schemas import commitment_key

    runs: Dict[Any, int] = defaultdict(int)
    for turn_idx, step in enumerate(history):
        broken = {
            commitment_key(o.commitment) for o in step.outcomes if not o.kept
        }
        live = set()
        for c in (step.commitments or []):
            if c.commitment_type != CommitmentType.ALLIANCE:
                continue
            key = commitment_key(c)
            live.add(key)
            runs[key] = 0 if key in broken else runs[key] + 1
            if runs[key] >= stable_for:
                return turn_idx + 1
        for key in list(runs):
            if key not in live:
                runs[key] = 0
    return None


def preference_reversals(reversals: List[ReversalPoint],
                         personas: Optional[Dict[Player, str]] = None
                         ) -> Dict[str, Any]:
    """How often an agent signed a deal and then, in the same turn, preferred
    to break it.

    A reversal is `signed_gain > 0 and break_advantage > 0`: the negotiator
    paid for the promise and the planner immediately valued walking away from
    it more highly, penalty included. It is not hypocrisy — it is the two
    halves of one agent running on different models of the same partner, and
    it is measurable only because both prices are now written down.
    """
    if not reversals:
        return {"n": 0}
    flipped = [r for r in reversals if r.reversed_]
    by_kind: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    by_persona: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in reversals:
        row = by_kind[r.commitment_type.value]
        row[0] += 1
        row[1] += int(r.reversed_)
        if personas:
            prow = by_persona[personas.get(r.player, "Unknown")]
            prow[0] += 1
            prow[1] += int(r.reversed_)
    # Clustered by game where the rows say which game they came from: a
    # pooled Wilson interval over 2,000 rows from 20 games reported a
    # precision the corpus does not have. Falls back to pooling when nothing
    # stamped them.
    by_game: Dict[Any, List[int]] = defaultdict(lambda: [0, 0])
    for r in reversals:
        if r.game is not None:
            row = by_game[r.game]
            row[0] += 1
            row[1] += int(r.reversed_)
    ci = (clustered_rate_ci95(list(by_game.values()))
          if by_game else (*ci95(len(flipped), len(reversals)), len(reversals)))

    return {
        "n": len(reversals),
        "games": len(by_game) or None,
        "ci": ci,
        "reversed": len(flipped),
        "rate": len(flipped) / len(reversals),
        "mean_gap": (sum(r.break_advantage for r in flipped) / len(flipped)
                     if flipped else 0.0),
        "by_kind": {k: (n, b, b / n) for k, (n, b) in by_kind.items()},
        "by_persona": {k: (n, b, b / n) for k, (n, b) in by_persona.items()},
    }


# ── Calibration ──────────────────────────────────────────────────────────

def brier_score(points: list) -> Optional[float]:
    """Mean squared error of P(keeps) against what happened.

    0 is a perfect forecaster; 0.25 is what you score by saying 50% every
    time. A model above 0.25 is worse than useless and the reliability
    diagram will show which direction it is wrong in.
    """
    pairs = [(c.predicted, 1.0 if c.observed else 0.0)
             for c in points if c.observed is not None]
    if not pairs:
        return None
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def reliability_bins(points: list, bins: int = 10) -> List[Dict[str, Any]]:
    """Reliability diagram data: for each bin of predicted probability, what
    fraction of those promises were actually kept."""
    buckets: Dict[int, List[tuple]] = defaultdict(list)
    for c in points:
        if c.observed is None:
            continue
        idx = min(bins - 1, int(c.predicted * bins))
        buckets[idx].append((c.predicted, 1.0 if c.observed else 0.0))

    # mean_predicted is the mean of the predictions that landed in the bin,
    # not the bin's midpoint: a forecaster that says 1.0 and is right every
    # time should score a calibration error of zero, and with midpoints it
    # scores half a bin width instead.
    return [
        {
            "bin_lower": i / bins,
            "bin_upper": (i + 1) / bins,
            "mean_predicted": sum(p for p, _o in buckets[i]) / len(buckets[i]),
            "observed_rate": sum(o for _p, o in buckets[i]) / len(buckets[i]),
            "count": len(buckets[i]),
        }
        for i in sorted(buckets)
    ]


def calibration_error(points: list, bins: int = 10) -> Optional[float]:
    """Expected calibration error: average |predicted - observed| per bin,
    weighted by how many predictions landed in it."""
    rows = reliability_bins(points, bins)
    n = sum(r["count"] for r in rows)
    if not n:
        return None
    return sum(r["count"] * abs(r["mean_predicted"] - r["observed_rate"])
               for r in rows) / n


# ── The richer deal structures ───────────────────────────────────────────

def deal_mix(history: list) -> Dict[str, Any]:
    """What kinds of promise actually got made, and how they fared.

    Without this the four structures are features nobody can tell apart in
    the results: "betrayal rate 38%" says nothing about whether a private
    exchange is broken more often than a public alliance.
    """
    made: Dict[str, int] = defaultdict(int)
    broken: Dict[str, int] = defaultdict(int)
    seen: set = set()
    broken_ids: set = set()

    def label(c):
        out = [c.commitment_type.value, "private" if c.private else "public"]
        if len(c.players) > 2:
            out.append("pact")
        return out

    for step in history:
        for c in (step.commitments or []):
            if c.id in seen:
                continue
            seen.add(c.id)
            for k in label(c):
                made[k] += 1
        for o in step.outcomes:
            # Counted once per deal, not once per turn it stayed broken: a
            # promise live for three turns and broken in each is one broken
            # promise, and counting the turns pushed the rate above 1.
            if o.kept or o.commitment.id in broken_ids:
                continue
            broken_ids.add(o.commitment.id)
            for k in label(o.commitment):
                broken[k] += 1

    return {
        "made": dict(made),
        "broken": dict(broken),
        "break_rate": {
            k: broken.get(k, 0) / made[k] for k in made if made[k]
        },
    }


def pair_trust_divergence(history: list) -> Dict[str, Any]:
    """How far "keeps promises to me" drifts from "keeps promises".

    Reported two ways on purpose. The mean over every belief row is dominated
    by pairs nothing has happened between yet, whose two numbers are both the
    prior and therefore identical; averaging those in makes the layer look
    dead when it is not. The second figure is the one that answers the
    question: among relationships that have actually diverged, by how much.
    """
    gaps = []
    for step in history:
        for b in (step.beliefs or []):
            if b.reliability_toward_observer is None:
                continue
            gaps.append(abs(b.reliability_toward_observer - b.expected_reliability))
    if not gaps:
        return {"mean_all": None, "mean_where_diverged": None,
                "share_over_0.1": None, "n": 0}

    moved = [g for g in gaps if g > 1e-9]
    return {
        "mean_all": sum(gaps) / len(gaps),
        "mean_where_diverged": (sum(moved) / len(moved)) if moved else 0.0,
        "share_over_0.1": sum(1 for g in gaps if g > 0.1) / len(gaps),
        "n": len(gaps),
    }
