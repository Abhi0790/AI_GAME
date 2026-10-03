"""Promise-keeping search coverage and the optional candidate safeguard."""

from src.agents.planner.planner import (
    Planner, PlannerConfig, breaks_commitment, keeping_order_set,
)
from src.common.schemas import (
    Commitment, CommitmentType, GameState, Player, Unit,
)
from src.engine import orders as order_module
from src.evaluation.keep_coverage import compare_budgets, run


def _two_support_state():
    return GameState(
        turn=1,
        units=[Unit(player=Player.RED, territory="R1"),
               Unit(player=Player.RED, territory="R2"),
               Unit(player=Player.BLUE, territory="B1"),
               Unit(player=Player.GOLD, territory="Y2")],
        supply_centers={"R1": Player.RED, "R2": Player.RED,
                        "B1": Player.BLUE, "Y2": Player.GOLD},
        territory_owners={"R1": Player.RED, "R2": Player.RED,
                          "B1": Player.BLUE, "Y2": Player.GOLD},
    )


def _two_supports():
    return [
        Commitment(id="one", commitment_type=CommitmentType.SUPPORT,
                   players=[Player.RED, Player.BLUE], valid_until_turn=1,
                   target_territory="B1"),
        Commitment(id="two", commitment_type=CommitmentType.SUPPORT,
                   players=[Player.RED, Player.GOLD], valid_until_turn=1,
                   target_territory="Y2"),
    ]


def test_constructs_joint_support_set_that_sampler_misses(monkeypatch):
    state, commitments = _two_support_state(), _two_supports()
    keeping = keeping_order_set(state, Player.RED, commitments)
    assert keeping is not None
    assert not breaks_commitment(state, Player.RED, keeping, commitments)

    monkeypatch.setattr(order_module, "MAX_ORDER_SETS", 1)
    order_module._SETS_CACHE.clear()
    try:
        planner = Planner(Player.RED, PlannerConfig(
            node_budget=400, trace_coverage=True, reserve_keep_candidate=True))
        candidates, _, coverage = planner._prepare_candidates(state, commitments)
        assert coverage["keepable"] is True
        assert coverage["generated_keep"] is False
        assert coverage["pruned_keep"] is True
        assert not breaks_commitment(state, Player.RED, candidates[0], commitments)
    finally:
        order_module._SETS_CACHE.clear()


def test_conflicting_supports_cannot_all_be_kept():
    state, commitments = _two_support_state(), _two_supports()
    state.units = [u for u in state.units if not (u.player == Player.RED
                                                  and u.territory == "R1")]
    assert keeping_order_set(state, Player.RED, commitments) is None


def test_paired_report_accounts_for_each_broken_obligation():
    report = run([41000, 41001], {"max_turns": 3}, 100, workers=1)
    assert len(report["games"]) == 2
    for arm in ("baseline", "reserved"):
        broken = sum(pair[arm]["broken_obligations"] for pair in report["games"])
        assert sum(report["break_categories"][arm].values()) == broken
        assert all(pair[arm]["seed"] in (41000, 41001)
                   for pair in report["games"])


def test_budget_comparison_uses_matched_game_effects():
    def row(value):
        return {name: value for name in (
            "win_share", "final_centers", "due_obligations",
            "broken_obligations", "betrayal_rate", "missed_keep_turns")}

    low = {"seeds": [7], "config": {"node_budget": 400},
           "games": [{"baseline": row(2), "reserved": row(1)}]}
    high = {"seeds": [7], "config": {"node_budget": 1500},
            "games": [{"baseline": row(2), "reserved": row(3)}]}
    assert compare_budgets(low, high)["win_share"]["high_minus_low_ci95"][0] == 2
