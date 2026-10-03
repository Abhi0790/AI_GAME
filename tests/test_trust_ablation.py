import pytest
import random

from src.agents.trust.model import ConstantForecastTrustModel, TrustModel
from src.common.config import GameConfig
from src.common.schemas import CommitmentType, Player
from src.evaluation.trust_ablation import compare, compare_budgets, run, variant_seat
from src.harness import build_agents, build_runner, game_setup, play


def test_constant_forecast_keeps_reputation_pricing():
    learned = TrustModel(Player.RED)
    constant = ConstantForecastTrustModel(Player.RED, 0.8)
    assert constant.p_keeps(Player.BLUE, CommitmentType.ALLIANCE, 1.0) == 0.8
    assert constant.reputation_drop(Player.RED, CommitmentType.ALLIANCE) == pytest.approx(
        learned.reputation_drop(Player.RED, CommitmentType.ALLIANCE))
    for model in (learned, constant):
        model.observe(Player.RED, [Player.BLUE], CommitmentType.ALLIANCE, kept=False)
    assert constant.p_keeps(Player.RED, CommitmentType.ALLIANCE) == 0.8
    assert constant.reputation_drop(Player.RED, CommitmentType.ALLIANCE) == pytest.approx(
        learned.reputation_drop(Player.RED, CommitmentType.ALLIANCE))
    assert constant.snapshot() and learned.snapshot()


def test_control_is_confined_to_one_seat_and_round_trips():
    cfg = GameConfig(seed=3, seat_keep_forecast={Player.RED: 0.75})
    assert GameConfig.from_dict(cfg.to_dict()).seat_keep_forecast == cfg.seat_keep_forecast
    with game_setup(cfg):
        agents = {a.player: a for a in build_agents(cfg)}
    assert isinstance(agents[Player.RED].trust_model, ConstantForecastTrustModel)
    assert type(agents[Player.BLUE].trust_model) is TrustModel


def test_variant_schedule_covers_every_seat_and_persona():
    from collections import Counter

    seats = GameConfig().make_board().players
    assert set(Counter(variant_seat(seed, seats) for seed in range(24)).values()) == {6}
    assert set(Counter(GameConfig(seed=seed).seats()[variant_seat(seed, seats)]
                       for seed in range(24)).values()) == {6}


def test_comparison_uses_paired_differences():
    learned = [{"seed": i, "win_share": a} for i, a in enumerate((1.0, 0.0))]
    constant = [{"seed": i, "win_share": b} for i, b in enumerate((0.0, 1.0))]
    for rows in (learned, constant):
        for row in rows:
            row.update(final_centers=2, deals=1, betrayal_rate=None,
                       mean_signed_gain=None, brier=0.2,
                       offers_received=2, priced_offers=1, early_rejections=1,
                       accepted_offers=1, countered_offers=0, rejected_offers=1,
                       planner_nodes_per_turn=10, total_nodes_per_turn=20)
    result = compare(learned, constant)
    assert result["win_share"]["learned_minus_constant_ci95"][0] == 0.0
    assert result["betrayal_rate"]["paired_games"] == 0
    with pytest.raises(ValueError, match="same seeds"):
        compare(learned, list(reversed(constant)))


def test_small_paired_run_has_disjoint_training_and_test_games():
    report = run([100, 101], [0, 1], {"max_turns": 3},
                 node_budget=100, workers=1)
    assert 0 <= report["constant_keep_forecast"] <= 1
    assert report["training_predictions"] > 0
    assert len(report["games"]) == 2
    assert report["offer_summary"]["learned"]["received"] == sum(
        pair["learned"]["offers_received"] for pair in report["games"])
    assert all(pair["learned"]["seat"] == pair["constant"]["seat"]
               for pair in report["games"])
    assert report["config"]["event_keyed_rng"] is True
    for pair in report["games"]:
        for game in pair.values():
            assert game["offers_received"] == len(game["opportunities"])
            assert game["offers_received"] == (
                game["accepted_offers"] + game["countered_offers"]
                + game["rejected_offers"])
            assert all(row["forecast"] is None or 0 <= row["forecast"] <= 1
                       for row in game["opportunities"])
            assert all(not row["outcomes"] or row["signed"]
                       for row in game["opportunities"])
    with pytest.raises(ValueError, match="disjoint"):
        run([0, 1], [1, 2], {"max_turns": 3}, node_budget=100, workers=1)


def test_event_keyed_draws_are_stable_and_do_not_advance_shared_rng():
    cfg = GameConfig(seed=17, event_keyed_rng=True)
    with game_setup(cfg):
        runner = build_runner(cfg)
        before = random.getstate()
        with runner._event_rng(("act", 1, "Red")):
            red_first = random.random()
        assert random.getstate() == before
        with runner._event_rng(("act", 1, "Blue")):
            blue = random.random()
        with runner._event_rng(("act", 1, "Red")):
            red_again = random.random()
    assert red_first == red_again
    assert red_first != blue


def test_event_keyed_game_replays_orders_despite_random_message_ids():
    cfg = GameConfig(seed=8, max_turns=3, node_budget=100,
                     event_keyed_rng=True)
    first, second = play(cfg), play(cfg)
    assert [[o.model_dump() for o in step.orders] for step in first.history] == [
        [o.model_dump() for o in step.orders] for step in second.history]


def test_budget_interaction_uses_the_same_game_in_both_budgets():
    def game(seed, seat, win):
        return {"seed": seed, "seat": seat, "persona": "Honest",
                "win_share": win, "final_centers": 2, "deals": 1,
                "offers_received": 3, "priced_offers": 2,
                "accepted_offers": 1, "betrayal_rate": None, "brier": 0.2}

    low = {"games": [{"learned": game(1, "Red", 0),
                      "constant": game(1, "Red", 1)}]}
    high = {"games": [{"learned": game(1, "Red", 1),
                       "constant": game(1, "Red", 0)}]}
    assert compare_budgets(low, high)["win_share"]["high_minus_low_ci95"][0] == 2
    high["games"][0]["learned"]["seed"] = 2
    with pytest.raises(ValueError, match="same seeds"):
        compare_budgets(low, high)
