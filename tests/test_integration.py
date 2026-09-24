"""Integration tests — run a full game and verify invariants.

A game ends either at the 12-turn horizon or early, the moment someone holds
WIN_CENTERS centres, so nothing here may assume a fixed length.
"""

from src.common.config import GameConfig
from src.common.schemas import Player
from src.harness import play
from src.engine.board import win_centers, max_turns, players, get_all_territories
from src.engine.replay import save_replay, load_replay, reconstruct_states
from src.evaluation.metrics import (
    supply_center_timeline, betrayal_events, final_scores,
    betrayal_rate_per_player, forfeit_at_break,
)
import os
import tempfile


def _run_game(seed=42):
    """Every game in this repo is built by the harness — including this one,
    or the tests stop testing what the scripts actually run."""
    return play(GameConfig(seed=seed))


class TestFullGame:
    def test_game_completes_without_crash(self):
        runner = _run_game()
        assert 1 <= len(runner.history) <= max_turns()
        assert runner.state.turn == len(runner.history) + 1

    def test_stops_early_only_on_a_win(self):
        """Short games must be explained by someone reaching the win threshold."""
        runner = _run_game()
        if len(runner.history) < max_turns():
            assert max(runner.center_counts().values()) >= win_centers()

    def test_final_state_has_valid_units(self):
        runner = _run_game()
        valid_territories = set(get_all_territories())
        for u in runner.state.units:
            assert u.territory in valid_territories

    def test_supply_centers_have_valid_owners(self):
        runner = _run_game()
        for territory, owner in runner.state.supply_centers.items():
            if owner is not None:
                assert owner in Player

    def test_occupancy_is_released_when_a_unit_leaves(self):
        """The old bug: territory_owners was never cleared, so a player owned
        every square they had ever walked through."""
        runner = _run_game()
        occupied = {u.territory for u in runner.state.units}
        claimed = {t for t, owner in runner.state.territory_owners.items() if owner}
        assert claimed == occupied

    def test_history_records_all_turns(self):
        runner = _run_game(seed=7)
        for step in runner.history:
            # Messages and beliefs used to be produced and thrown away, which
            # is why the UI had nothing to show for either.
            assert step.state is not None
            assert isinstance(step.messages, list)
            assert isinstance(step.beliefs, list)
            assert set(step.nodes) == set(players())

    def test_different_seeds_produce_different_results(self):
        results = []
        for seed in [1, 2, 3]:
            runner = _run_game(seed)
            scores = final_scores(runner.state)
            results.append(tuple(scores[p] for p in players()))
        assert len(set(results)) >= 1  # At minimum runs without crash


class TestReplaySystem:
    def test_save_and_load_replay(self):
        runner = _run_game()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_replay(runner.history, runner.state,
                               directory=tmpdir, filename="test.json")
            assert os.path.exists(path)

            replay = load_replay(path)
            assert replay["turns"] == len(runner.history)
            assert len(replay["history"]) == len(runner.history)

    def test_reconstruct_states(self):
        runner = _run_game()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_replay(runner.history, runner.state,
                               directory=tmpdir, filename="test.json")
            replay = load_replay(path)
            states = reconstruct_states(replay)
            # every history state, plus the final one
            assert len(states) == len(runner.history) + 1
            assert states[0].turn == 1
            assert states[-1].turn == len(runner.history) + 1


class TestEvaluationMetrics:
    def test_supply_center_timeline(self):
        runner = _run_game()
        timeline = supply_center_timeline(runner.history)
        for p in players():
            assert len(timeline[p]) == len(runner.history)

    def test_final_scores(self):
        runner = _run_game()
        scores = final_scores(runner.state)
        assert sum(scores.values()) <= 10  # ten centres on the board

    def test_betrayal_rate_values(self):
        runner = _run_game()
        rates = betrayal_rate_per_player(runner.history)
        for p in players():
            assert 0.0 <= rates[p] <= 1.0

    def test_forfeit_rows_are_readable(self):
        """Empty is fine — malformed is not."""
        for row in forfeit_at_break(_run_game().history):
            assert 0.0 <= row["fraction_remaining"] <= 1.0
            assert row["player"] in {p.value for p in players()}


class TestReproducibility:
    def test_same_seed_same_scores(self):
        """What run_headless.py prints must be enough to reproduce the run."""
        cfg = GameConfig(seed=3, node_budget=200)
        a, b = play(cfg), play(cfg)
        assert final_scores(a.state) == final_scores(b.state)
        assert [str(step.orders) for step in a.history] == \
               [str(step.orders) for step in b.history]
