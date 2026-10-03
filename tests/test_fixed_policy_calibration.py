from types import SimpleNamespace

from src.common.config import GameConfig
from src.evaluation.fixed_policy import (
    FixedPolicyAgent, crossfit_baselines, play_fixed, summarise,
)


def test_base_rate_is_fitted_on_other_games():
    games = [[SimpleNamespace(observed=True), SimpleNamespace(observed=False)],
             [SimpleNamespace(observed=False)]]
    assert crossfit_baselines(games) == [0.0, 0.5]


def test_fixed_policy_actions_do_not_depend_on_forecasts(monkeypatch):
    cfg = GameConfig(seed=4, max_turns=3)
    original = play_fixed(cfg)
    monkeypatch.setattr(FixedPolicyAgent, "predict_keep_parts",
                        lambda self, state, commitment, subject: (0.5, 0.0, 0.01))
    changed_forecasts = play_fixed(cfg)
    assert [[str(o) for o in step.orders] for step in original.history] == [
        [str(o) for o in step.orders] for step in changed_forecasts.history]
    assert [[c.id for c in step.commitments] for step in original.history] == [
        [c.id for c in step.commitments] for step in changed_forecasts.history]
    assert all(p.predicted == 0.01 for p in changed_forecasts.calibration)


def test_fixed_policy_report_has_paired_game_rows():
    games = [play_fixed(GameConfig(seed=s, max_turns=3)).calibration for s in (1, 2)]
    report = summarise(games, [1, 2])
    assert report["games"] == 2
    assert report["predictions"] > 0
    assert len(report["by_game"]) == 2
    assert report["baseline_brier"] is not None
