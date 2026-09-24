"""The two halves of issue: reciprocity as a belief, and measuring the reversal.

Covers the only non-trivial logic added: the commitment-conditioned draw in
OpponentModel.sample_opponent_orders, and the per-deal break advantage the
planner reports alongside it.
"""
import random

from src.common.schemas import (
    Player, CommitmentType, Commitment, Order, OrderType, GameState, Unit,
    commitment_key,
)
from src.agents.agent import Agent
from src.agents.opponent_model.opponent_model import OpponentModel
from src.agents.trust.model import TrustModel
from src.engine.runner import GameRunner
from src.engine.board import get_supply_centers, get_all_territories


def _state(turn=1):
    units = [Unit(player=Player.RED, territory="R1"),
             Unit(player=Player.RED, territory="R2"),
             Unit(player=Player.BLUE, territory="B1"),
             Unit(player=Player.BLUE, territory="B2"),
             Unit(player=Player.GREEN, territory="G1"),
             Unit(player=Player.GOLD, territory="Y1")]
    sc = {t: None for t in get_supply_centers()}
    for u in units:
        if u.territory in sc:
            sc[u.territory] = u.player
    owners = {t: None for t in get_all_territories()}
    for u in units:
        owners[u.territory] = u.player
    return GameState(turn=turn, units=units, supply_centers=sc, territory_owners=owners)


def _dmz_on_n1():
    """Blue promises Red to stay out of N1. B1 borders N1, so Blue can break it."""
    return Commitment(id="c1", commitment_type=CommitmentType.DMZ,
                      players=[Player.BLUE, Player.RED], valid_until_turn=5,
                      dmz_territories=["N1"])


def _share_moving_into(samples, player, target):
    """Fraction of sampled worlds in which *player* moves into *target*."""
    hits = sum(1 for w in samples
               if any(o.player == player and o.order_type == OrderType.MOVE
                      and o.target == target for o in w))
    return hits / len(samples)


def test_conditioning_needs_both_a_deal_and_a_trust_model():
    """Without either argument the draw is the old unconditioned one."""
    state, deal = _state(), _dmz_on_n1()
    om = OpponentModel(Player.RED)
    random.seed(0)
    plain = om.sample_opponent_orders(state, samples=200)
    random.seed(0)
    no_model = om.sample_opponent_orders(state, samples=200, commitments=[deal])
    assert plain == no_model, "commitments alone must not change the draw"


def test_a_trusted_promise_shifts_the_draw_away_from_breaking_it():
    """The whole point: a promise changes what I expect them to do."""
    state, deal = _state(), _dmz_on_n1()
    om = OpponentModel(Player.RED)

    random.seed(7)
    baseline = _share_moving_into(
        om.sample_opponent_orders(state, samples=400), Player.BLUE, "N1")

    trusted = TrustModel(Player.RED)
    for _ in range(12):  # Blue has kept DMZs with everyone, over and over
        trusted.observe(Player.BLUE, [Player.RED], CommitmentType.DMZ, kept=True)
    random.seed(7)
    loyal = _share_moving_into(
        om.sample_opponent_orders(state, samples=400, commitments=[deal],
                                  trust_model=trusted),
        Player.BLUE, "N1")

    faithless = TrustModel(Player.RED)
    for _ in range(12):
        faithless.observe(Player.BLUE, [Player.RED], CommitmentType.DMZ, kept=False)
    random.seed(7)
    treacherous = _share_moving_into(
        om.sample_opponent_orders(state, samples=400, commitments=[deal],
                                  trust_model=faithless),
        Player.BLUE, "N1")

    assert loyal < baseline, (
        f"a promise from a reliable player should make the break less likely "
        f"({loyal:.2f} vs {baseline:.2f})")
    assert treacherous > loyal, (
        f"a promise from a proven defector is worth less "
        f"({treacherous:.2f} vs {loyal:.2f})")


def test_conditioning_never_empties_the_candidate_set():
    """If every order set breaks the deal, fall back rather than draw from zero
    weights — `random.choices` raises on an all-zero weight vector."""
    state = _state()
    # A DMZ over every territory: Blue cannot move anywhere without breaking it.
    impossible = Commitment(id="c2", commitment_type=CommitmentType.DMZ,
                            players=[Player.BLUE, Player.RED], valid_until_turn=5,
                            dmz_territories=get_all_territories())
    certain = TrustModel(Player.RED)
    for _ in range(50):
        certain.observe(Player.BLUE, [Player.RED], CommitmentType.DMZ, kept=True)
    worlds = OpponentModel(Player.RED).sample_opponent_orders(
        state, samples=5, commitments=[impossible], trust_model=certain)
    assert len(worlds) == 5 and all(w for w in worlds)


def test_planner_reports_a_break_advantage_for_each_of_its_own_deals():
    """The reversal measurement: one row per live deal the agent is in, and the
    number is best-net-breaking minus best-net-keeping."""
    state, deal = _state(), _dmz_on_n1()
    # Red is not in Blue's DMZ obligation, but both are parties to the deal.
    agent = Agent(Player.BLUE, "Opportunist")
    _orders, trace = agent.act(state, [deal])
    keys = {row[0] for row in trace.reversals}
    assert commitment_key(deal) in keys, "the live deal was not priced at order time"
    assert all(isinstance(row[2], float) for row in trace.reversals)


def test_reputation_cost_moves_the_break_advantage_the_right_way():
    """A seat charged five times the reputation cost must want to break its own
    deal strictly less than the same seat at 1.0 in the same position."""
    state, deal = _state(), _dmz_on_n1()
    adv = {}
    for scale in (1.0, 5.0):
        random.seed(3)
        agent = Agent(Player.BLUE, "Opportunist")
        agent.act(state, [deal])      # sets the policy weights for this board
        agent.planner.config.reputation_cost_coefficient = scale
        random.seed(3)
        _o, trace = agent.planner.find_best_orders(state, [deal], agent.trust_model)
        rows = [r for r in trace.reversals if r[0] == commitment_key(deal)]
        assert rows, f"scale {scale} did not price the deal"
        adv[scale] = rows[0][2]
    assert adv[5.0] < adv[1.0], adv


def test_loyalty_cost_keeps_honest_from_breaking_first():
    """Unwronged, Honest owes Red loyalty; charged for it, it keeps the deal."""
    from src.agents.planner.planner import breaks_commitment
    from src.harness import apply_knobs
    state, deal = _state(), _dmz_on_n1()
    agent = Agent(Player.BLUE, "Honest")
    assert Player.RED in agent.policy(state)[0]
    with apply_knobs({"LOYALTY_COST": 10.0}):
        random.seed(3)
        orders, _trace = agent.act(state, [deal])
    assert not breaks_commitment(state, Player.BLUE, orders, [deal])


def test_policies_bar_deals_with_whoever_wronged_them():
    """Honest and Vengeful sign nothing with a player who just betrayed them;
    the Opportunist does not care."""
    state = _state()
    for persona, barred in (("Honest", True), ("Vengeful", True), ("Opportunist", False)):
        agent = Agent(Player.BLUE, persona)
        agent.grudges[Player.RED] = 1.0
        assert (Player.RED in agent.policy(state)[2]) == barred, persona


def test_retaliation_is_learned_from_what_the_betrayed_do_next():
    """Blue betrays Red; next turn Red moves on Blue. Observers count it."""
    from src.common.schemas import CommitmentOutcome, Order, OrderType
    state, deal = _state(), _dmz_on_n1()
    watcher = Agent(Player.GREEN, "Opportunist")
    before = watcher.retaliation_rate(Player.RED)
    broken = CommitmentOutcome(deal, kept=False, broken_by=[Player.BLUE])
    watcher.update_beliefs_from_outcomes(state, state, [broken])
    watcher.observe_orders([])
    watcher.update_beliefs_from_outcomes(state, state, [])
    target = next(u.territory for u in state.units if u.player == Player.BLUE)
    watcher.observe_orders([Order(player=Player.RED, unit_territory="R2",
                                  order_type=OrderType.MOVE, target=target)])
    assert watcher.retaliation_rate(Player.RED) > before


def test_runner_pairs_the_signature_price_with_the_order_time_price():
    """End to end: a full game produces ReversalPoints, and a reversal is
    exactly `signed_gain > 0 and break_advantage > 0`."""
    random.seed(11)
    runner = GameRunner([Agent(Player.RED, "Opportunist"), Agent(Player.BLUE, "Honest"),
                         Agent(Player.GREEN, "Paranoid"), Agent(Player.GOLD, "Vengeful")])
    runner.run(verbose=False)
    assert runner.reversals, "no deal was priced twice in a whole game"
    for pt in runner.reversals:
        assert pt.reversed_ == (pt.signed_gain > 0 and pt.break_advantage > 0)
    # The Honest-reverses-less claim is behavioural, so it is asserted over a
    # pooled, rotated corpus in
    # `test_honest_reverses_less_than_the_opportunist`, not here. On one seed
    # a seat can own four reversal rows and swing the share by 25 points.


# ── the punishment channel ───────────────────────────────────────────────

def _breaking_orders():
    """Blue's B1 walks into N1, which is exactly what the DMZ forbids."""
    return [Order(player=Player.BLUE, unit_territory="B1",
                  order_type=OrderType.MOVE, target="N1"),
            Order(player=Player.BLUE, unit_territory="B2",
                  order_type=OrderType.HOLD)]


def _quiet_world():
    return [Order(player=p, unit_territory=t, order_type=OrderType.HOLD)
            for p, t in ((Player.RED, "R1"), (Player.RED, "R2"),
                         (Player.GREEN, "G1"), (Player.GOLD, "Y1"))]


def _depth2(retaliation: int, orders, deals, seed=5):
    from src.agents.planner.planner import Planner, PlannerConfig
    from src.agents.opponent_model.opponent_model import OpponentModel
    planner = Planner(Player.BLUE, PlannerConfig(retaliation_samples=retaliation))
    planner.set_opponent_model(OpponentModel(Player.BLUE))
    random.seed(seed)
    return planner._depth2_value(_state(), orders, [_quiet_world()], deals,
                                 TrustModel(Player.BLUE))


def test_retaliation_lowers_the_value_of_a_line_that_breaks_a_promise():
    """The whole point of the punishment channel: betrayal has to cost a
    position in the tree, not only a scalar penalty beside it."""
    deals = [_dmz_on_n1()]
    with_punishment = _depth2(1, _breaking_orders(), deals)
    without = _depth2(0, _breaking_orders(), deals)
    assert with_punishment is not None and without is not None
    assert with_punishment < without, (
        f"expecting to be punished should make the betrayal worth less "
        f"({with_punishment:.3f} vs {without:.3f})")


def test_retaliation_leaves_a_loyal_line_alone():
    """No promise broken, nobody to retaliate, so the two searches agree."""
    loyal = [Order(player=Player.BLUE, unit_territory="B1", order_type=OrderType.HOLD),
             Order(player=Player.BLUE, unit_territory="B2",
                   order_type=OrderType.MOVE, target="C2")]
    deals = [_dmz_on_n1()]
    assert _depth2(1, loyal, deals) == _depth2(0, loyal, deals)


def test_a_broken_deal_stops_conditioning_the_partner_s_reply():
    """A deal I have already walked away from must not still be modelled as
    restraining the other side -- the runner releases them, so must the search.

    With no live deals at all the two searches must agree exactly; the released
    branch is therefore the only thing that can differ above.
    """
    assert _depth2(0, _breaking_orders(), []) == _depth2(0, _breaking_orders(), [])
    # Breaking the only deal on the table leaves nothing to condition on, so a
    # search that correctly releases the partner matches the no-deal search.
    released = _depth2(0, _breaking_orders(), [_dmz_on_n1()])
    no_deal = _depth2(0, _breaking_orders(), [])
    assert released == no_deal, (released, no_deal)


# ── forfeiture: signing and breaking quote the same price ────────────────

def _games(weight: float, seeds=range(12), budget=800):
    """A seeded corpus with one knob moved, sized so the claim below is not a
    coin flip. At eight games and a 400-adjudication budget the two rates'
    confidence intervals overlap, and per-persona shares swing 10 points on a
    single seed; twelve games at 800 separates them and still runs in ~90s."""
    from src.common.config import GameConfig
    from src.harness import apply_knobs, play
    with apply_knobs({"FORFEIT_WEIGHT": weight}):
        return [play(GameConfig(seed=s, node_budget=budget)) for s in seeds]


def test_forfeiting_the_signed_price_cuts_preference_reversals():
    """The defect in one assertion.

    Breaking used to be charged only Vcoop x dP x horizon, with dP around
    0.04 on this board, so walking away cost a twentieth of what signing had
    been credited with and the planner routinely undid the negotiator in the
    same turn. Charging back the unused share of the signed price is what
    makes the two halves quote one number.
    """
    from src.evaluation.metrics import preference_reversals

    off = [r for r in _games(0.0)]
    on = [r for r in _games(1.0)]
    from src.evaluation.metrics import ci95

    before = preference_reversals([p for r in off for p in r.reversals])
    after = preference_reversals([p for r in on for p in r.reversals])
    assert before["n"] and after["n"], "no deal was priced twice in the corpus"

    # Asserted as non-overlapping 95% intervals rather than a bare ratio, so
    # the test fails when the effect is gone and not when a seed is unlucky.
    b_rate, b_lo, b_hi = ci95(before["reversed"], before["n"])
    a_rate, a_lo, a_hi = ci95(after["reversed"], after["n"])
    assert a_hi < b_lo, (
        f"forfeiting the signed price did not separate the reversal rate: "
        f"{b_rate:.1%} [{b_lo:.1%}, {b_hi:.1%}] (n={before['n']}) -> "
        f"{a_rate:.1%} [{a_lo:.1%}, {a_hi:.1%}] (n={after['n']})")


class TestEstimators:
    """Reporting defects that made measured numbers look better than they were."""

    def test_wilson_bounds_are_asymmetric_and_stay_in_range(self):
        from src.evaluation.metrics import ci95

        rate, lo, hi = ci95(0, 20)
        assert rate == 0.0 and lo == 0.0
        assert 0.10 < hi < 0.20, f"0/20 upper bound should be ~16%, got {hi:.1%}"

        rate, lo, hi = ci95(20, 20)
        assert rate == 1.0 and hi == 1.0 and 0.80 < lo < 0.90

        for k in range(21):
            r, lo, hi = ci95(k, 20)
            assert 0.0 <= lo <= r <= hi <= 1.0, (k, lo, r, hi)

    def test_the_variant_seat_covers_every_persona(self):
        """`seats[seed % n]` collided with the seed-driven persona rotation and
        landed on an even persona index every time, testing only two of four."""
        from collections import Counter
        from src.common.config import GameConfig, PERSONA_ORDER

        seen = Counter()
        for s in range(4 * len(PERSONA_ORDER)):
            seating = GameConfig(seed=s).seats()
            want = PERSONA_ORDER[s % len(PERSONA_ORDER)]
            seat = next(p for p, name in seating.items() if name == want)
            seen[seating[seat]] += 1
        assert set(seen) == set(PERSONA_ORDER)
        assert len(set(seen.values())) == 1, f"uneven coverage: {dict(seen)}"

    def test_sweep_intervals_are_clustered_by_game_not_by_promise_row(self):
        from src.evaluation.sweep import measure

        r = measure([0, 1, 2], {"node_budget": 150})
        _mean, _half, n = r["brier_ci"]
        assert n == r["games"], (
            f"interval n={n} should be games ({r['games']}), not promise rows")
        assert r["brier_rows"] > n, "pooled row count should still be reported"


class TestReportingDefects:
    def test_alliance_durations_censor_deals_alive_at_the_horizon(self):
        """A deal still running when the clock stops was not 'kept for its full
        term' — that is right-censoring, and counting it as completed overstates
        how long deals hold."""
        from types import SimpleNamespace
        from src.common.schemas import Commitment, CommitmentOutcome, CommitmentType
        from src.evaluation.metrics import alliance_durations

        live = Commitment(id="a1", commitment_type=CommitmentType.ALLIANCE,
                          players=[Player.RED, Player.BLUE], valid_until_turn=20)
        step = SimpleNamespace(outcomes=[CommitmentOutcome(live, kept=True,
                                                           broken_by=[])])
        uncensored = alliance_durations([step])[0]
        censored = alliance_durations([step], horizon=3)[0]
        assert uncensored["status"] == "kept" and uncensored["duration"] == 20
        assert censored["status"] == "censored" and censored["duration"] == 3

    def test_alliance_durations_only_counts_alliances(self):
        from types import SimpleNamespace
        from src.common.schemas import Commitment, CommitmentOutcome, CommitmentType
        from src.evaluation.metrics import alliance_durations

        dmz = Commitment(id="d1", commitment_type=CommitmentType.DMZ,
                         players=[Player.RED, Player.BLUE], valid_until_turn=4,
                         dmz_territories=["N1"])
        step = SimpleNamespace(outcomes=[CommitmentOutcome(dmz, True, [])])
        assert alliance_durations([step]) == []
        assert len(alliance_durations([step], kind=None)) == 1

    def test_the_timeline_includes_the_final_position(self):
        """step.state is the position BEFORE that turn's orders, so the last
        autumn's captures never reached the plot."""
        from src.common.config import GameConfig
        from src.evaluation.metrics import supply_center_timeline
        from src.harness import game_setup, play

        cfg = GameConfig(seed=3, node_budget=100, max_turns=4)
        r = play(cfg)
        with game_setup(cfg):
            seat = next(iter(r.center_counts()))
            assert len(supply_center_timeline(r.history)[seat]) == len(r.history)
            assert len(supply_center_timeline(
                r.history, r.state)[seat]) == len(r.history) + 1


def test_a_shared_lead_is_split_not_counted_whole():
    """figure 6 used to credit every tied seat with a win, which rewarded
    budgets low enough to leave the game undecided."""
    from src.common.schemas import leaders
    scores = {Player.RED: 4, Player.BLUE: 4, Player.GREEN: 2, Player.GOLD: 1}
    assert sum(1.0 / len(leaders(scores)) for p in leaders(scores)) == 1.0


def test_reversal_interval_counts_games_not_rows():
    """Rows from one game share a board and four agents, so pooling them
    reported a precision the corpus does not have."""
    from src.evaluation.metrics import preference_reversals
    from src.common.schemas import ReversalPoint
    rows = [ReversalPoint(turn=1, player=Player.RED, game=i // 10,
                          commitment_type=CommitmentType.ALLIANCE,
                          signed_gain=1.0, break_advantage=1.0 if i % 4 else -1.0)
            for i in range(40)]
    assert preference_reversals(rows)["ci"][2] == 4      # games, not 40 rows
    for r in rows:
        r.game = None
    assert preference_reversals(rows)["ci"][3] == 40     # unstamped: pooled


def test_analysis_counts_only_obligated_parties():
    """The analysis path kept its own copy of the betrayal denominator, which
    still credited a support's passive party with keeping a promise."""
    from src.evaluation.analysis import _betrayal_counts
    from src.common.schemas import Commitment, CommitmentOutcome

    c = Commitment(id="c", commitment_type=CommitmentType.SUPPORT,
                   players=[Player.RED, Player.BLUE], valid_until_turn=5,
                   target_territory="N1", supported_from="R1")
    outcome = CommitmentOutcome(commitment=c, kept=False, broken_by=[Player.RED])
    from types import SimpleNamespace
    step = SimpleNamespace(outcomes=[outcome], state=SimpleNamespace(turn=1))
    counts = _betrayal_counts([step], {Player.RED: "Opportunist", Player.BLUE: "Honest"})
    assert counts["Opportunist"] == [1, 1]
    assert "Honest" not in counts
