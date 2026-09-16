"""
Evaluation Metrics — quantitative measures for analysing game outcomes.

Each function operates on a game history (list of turn records produced by
GameRunner) and returns structured data.
"""

from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from src.common.schemas import Player, GameState, CommitmentType, CalibrationPoint


# ── Per-game metrics ─────────────────────────────────────────────────────

def supply_center_timeline(history: list) -> Dict[Player, List[int]]:
    """Return a dict mapping each player to a list of supply-center counts,
    one entry per turn."""
    timeline: Dict[Player, List[int]] = {p: [] for p in Player}

    for step in history:
        state: GameState = step.state
        counts = {p: 0 for p in Player}
        for owner in state.supply_centers.values():
            if owner:
                counts[owner] += 1
        for p in Player:
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


def betrayal_rate_per_player(history: list) -> Dict[Player, float]:
    """Fraction of commitments each player broke out of commitments they
    participated in."""
    participated: Dict[Player, int] = defaultdict(int)
    broken: Dict[Player, int] = defaultdict(int)

    for step in history:
        for o in step.outcomes:
            for p in o.commitment.players:
                participated[p] += 1
                if p in o.broken_by:
                    broken[p] += 1

    rates: Dict[Player, float] = {}
    for p in Player:
        if participated[p] > 0:
            rates[p] = broken[p] / participated[p]
        else:
            rates[p] = 0.0
    return rates


def alliance_durations(history: list) -> List[Dict[str, Any]]:
    """Return a list of alliance records with their actual duration
    (turns before broken or expiry)."""
    # Track when commitments first appear and when they break
    commitment_first_seen: Dict[str, int] = {}
    commitment_broken_turn: Dict[str, Optional[int]] = {}
    commitment_info: Dict[str, Any] = {}

    for turn_idx, step in enumerate(history):
        for o in step.outcomes:
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
            ended = info["valid_until"]
            duration = max(0, ended - started)
            status = "kept"
        records.append({**info, "started": started, "ended": ended,
                        "duration": duration, "status": status})
    return records


def final_scores(state: GameState) -> Dict[Player, int]:
    """Return supply-center counts from the final game state."""
    counts = {p: 0 for p in Player}
    for owner in state.supply_centers.values():
        if owner:
            counts[owner] += 1
    return counts


# ── Cross-game (tournament) metrics ──────────────────────────────────────

def win_rates(game_results: List[Dict[Player, int]]) -> Dict[Player, float]:
    """Given a list of per-game score dicts, return win rates."""
    wins: Dict[Player, int] = {p: 0 for p in Player}
    for scores in game_results:
        mx = max(scores.values())
        winners = [p for p, s in scores.items() if s == mx]
        for w in winners:
            wins[w] += 1
    total = max(1, len(game_results))
    return {p: wins[p] / total for p in Player}


# ── Persona-keyed metrics ────────────────────────────────────────────────
#
# Everything above is keyed by seat colour, which is useless for the report:
# "Red betrays more than Blue" says nothing unless you know which persona sat
# in which chair. Every function below takes the seat -> persona map that
# GameRunner.personas() returns.

def betrayal_rate_per_persona(history: list, personas: Dict[Player, str]
                              ) -> Dict[str, float]:
    """Fraction of the commitments each persona was in that it broke."""
    participated: Dict[str, int] = defaultdict(int)
    broken: Dict[str, int] = defaultdict(int)

    for step in history:
        for o in step.outcomes:
            for p in o.commitment.players:
                name = personas.get(p, "Unknown")
                participated[name] += 1
                if p in o.broken_by:
                    broken[name] += 1

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
        for o in step.outcomes:
            for p in o.commitment.players:
                name = personas.get(p, "Unknown")
                participated[name] += 1
                if p in o.broken_by:
                    broken[name] += 1
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

    The control is the same game with broadcasts switched off
    (GameRunner(broadcast_enabled=False)).
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


def accusation_stats(history: list, personas: Dict[Player, str]) -> Dict[str, Any]:
    """Who accused whom, and how often the engine backed them up."""
    total = confirmed = refuted = lies = 0
    by_persona: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"made": 0, "confirmed": 0, "refuted": 0})
    for step in history:
        for m in (step.messages or []):
            if m.broadcast_kind != "BETRAYED":
                continue
            total += 1
            name = personas.get(m.sender, "Unknown")
            by_persona[name]["made"] += 1
            if m.engine_verdict == "CONFIRMED":
                confirmed += 1
                by_persona[name]["confirmed"] += 1
            else:
                refuted += 1
                by_persona[name]["refuted"] += 1
            if m.truthful is False:
                lies += 1
    return {"total": total, "confirmed": confirmed, "refuted": refuted,
            "known_lies": lies, "by_persona": dict(by_persona)}


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


def privacy_and_lying(history: list, personas: Dict[Player, str]) -> Dict[str, Any]:
    """Does hiding a deal actually buy anything?

    The claim the private-deal feature rests on is that an accusation the
    engine cannot settle is worth making. These are the numbers that support
    or sink it: how many accusations were unfalsifiable, how many of those
    were lies, and whether the liars were believed.
    """
    verdicts: Dict[str, int] = defaultdict(int)
    lies_by_persona: Dict[str, int] = defaultdict(int)
    unfalsifiable_lies = refuted_lies = 0

    for step in history:
        for m in (step.messages or []):
            if m.broadcast_kind != "BETRAYED":
                continue
            verdicts[m.engine_verdict or "NONE"] += 1
            if m.truthful is False:
                lies_by_persona[personas.get(m.sender, "Unknown")] += 1
                if m.engine_verdict == "UNVERIFIED":
                    unfalsifiable_lies += 1
                elif m.engine_verdict == "REFUTED":
                    refuted_lies += 1

    total_lies = unfalsifiable_lies + refuted_lies
    return {
        "verdicts": dict(verdicts),
        "lies": total_lies,
        "lies_by_persona": dict(lies_by_persona),
        "unfalsifiable_lies": unfalsifiable_lies,
        "refuted_lies": refuted_lies,
        # 1.0 means every lie went unchallenged, which would say the engine
        # has stopped constraining anyone.
        "lie_success_rate": (unfalsifiable_lies / total_lies) if total_lies else None,
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
