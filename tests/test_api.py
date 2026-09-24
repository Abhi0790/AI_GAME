"""The dashboard's API, end to end through TestClient.

A game is built from a full GameConfig, so the one thing worth asserting is
that what goes in comes back out — a screenshot of the dashboard is only
reproducible if /state echoes the config it was started from.
"""
from fastapi.testclient import TestClient

from src.ui import app as app_module

client = TestClient(app_module.app)

CONFIG = {
    "seed": 7,
    "seating": {"Red": "Honest", "Blue": "Opportunist",
                "Green": "Vengeful", "Gold": "Paranoid"},
    "search": "expectiminimax",
    "node_budget": 200,          # keeps the whole file well under a minute
    "negotiation_rounds": 2,
    "knobs": {"LAMBDA_INCENTIVE": 0.6},
}


def test_config_round_trips_and_two_steps_replay():
    r = client.post("/api/game/new", json=CONFIG)
    assert r.status_code == 200
    gid = r.json()["game_id"]

    state = client.get(f"/api/game/{gid}/state").json()
    cfg = state["config"]
    for key, value in CONFIG.items():
        assert cfg[key] == value, key
    # Effective knobs = the process defaults with this game's overrides on top.
    assert state["knobs"]["LAMBDA_INCENTIVE"] == 0.6
    assert "CENTRE_VALUE" in state["knobs"]
    assert state["personas"]["Red"] == "Honest"

    # No human seat, so nothing is pending, but the route must still answer.
    assert client.get(f"/api/game/{gid}/pending").json() == {"proposals": [], "legal": {}}

    for _ in range(2):
        assert client.post(f"/api/game/{gid}/step", json={}).status_code == 200

    replay = client.get(f"/api/game/{gid}/replay").json()
    assert replay["turns"] == 2
    assert len(replay["history"]) == 2


def test_unknown_knob_is_rejected():
    r = client.post("/api/game/new", json={"knobs": {"NOT_A_KNOB": 1.0}})
    assert r.status_code == 400
    body = r.json()
    assert "NOT_A_KNOB" in body["error"]
    assert "CENTRE_VALUE" in body["known"]


def test_second_concurrent_step_is_refused():
    gid = client.post("/api/game/new", json=CONFIG).json()["game_id"]
    lock = app_module._games[gid]["lock"]
    lock.acquire()
    try:
        r = client.post(f"/api/game/{gid}/step", json={})
        assert r.status_code == 409
    finally:
        lock.release()
    assert client.post(f"/api/game/{gid}/step", json={}).status_code == 200


def test_knobs_endpoint_lists_personas():
    body = client.get("/api/knobs").json()
    assert "CENTRE_VALUE" in body["knobs"]
    assert set(body["personas"]) == {"Honest", "Opportunist", "Vengeful", "Paranoid"}


def test_a_six_seat_game_is_created_drawn_and_inspected():
    """A non-default size must start and return everything needed to draw it."""
    r = client.post("/api/game/new", json={**CONFIG, "seating": None,
                                           "n_seats": 6, "max_turns": 4})
    assert r.status_code == 200, r.json()
    game = r.json()
    gid = game["game_id"]

    board = game["board"]
    assert len(board["players"]) == 6
    assert len(board["territories"]) == 18
    assert set(board["positions"]) == set(board["territories"])
    assert game["max_turns"] == 4 and game["win_centers"] == board["win_centers"]

    assert client.post(f"/api/game/{gid}/step").status_code == 200
    assert client.get(f"/api/game/{gid}/inspect/Orange").status_code == 200
    # Pink is on the roster but not in this game.
    assert client.get(f"/api/game/{gid}/inspect/Pink").status_code == 404


def test_an_impossible_board_is_a_400_not_a_500():
    for bad in ({"n_seats": 99}, {"n_seats": 1}, {"win_centers": 999}):
        r = client.post("/api/game/new", json={**CONFIG, "seating": None, **bad})
        assert r.status_code == 400, (bad, r.status_code)
        assert "board" in r.json()["error"]


def test_saved_replays_can_be_listed_and_loaded():
    """A replay carries its own board, so a six-seat game saved yesterday is
    drawable today without knowing what map it was played on."""
    import os
    from src.common.config import GameConfig
    from src.engine.replay import save_replay
    from src.harness import build_runner, game_setup

    cfg = GameConfig(seed=9, n_seats=6, max_turns=2, node_budget=100)
    with game_setup(cfg):
        runner = build_runner(cfg)
        runner.run(verbose=False)
        name = "test_six_seat.json"
        save_replay(runner.history, runner.state, board=cfg.make_board().to_dict(),
                    filename=name)
    try:
        listed = client.get("/api/replays").json()["replays"]
        assert any(r["name"] == name and r["seats"] == 6 for r in listed)

        d = client.get(f"/api/replays/{name}").json()
        assert len(d["board"]["players"]) == 6
        assert set(d["board"]["positions"]) == set(d["board"]["territories"])
        assert len(d["history"]) == d["turns"] and d["final_state"]
    finally:
        os.remove(os.path.join("replays", name))


def test_a_replay_name_cannot_escape_the_replay_directory():
    for bad in ("../secrets.json", "..%2Fsecrets.json", "sub/dir.json", "nope.txt"):
        assert client.get(f"/api/replays/{bad}").status_code in (400, 404)


def test_private_deals_do_not_reach_a_seat_that_is_not_a_party():
    """A private deal is unverifiable to outsiders by design; shipping it to
    the client defeats that at the UI layer even though the engine honours it."""
    gid = client.post("/api/game/new", json={
        "seed": 2, "human_seat": "Red", "node_budget": 300, "max_turns": 8,
    }).json()["game_id"]
    for _ in range(6):
        r = client.post(f"/api/game/{gid}/step",
                        json={"orders": [], "decisions": {}}).json()
        if r.get("finished"):
            break
        for com in r.get("commitments", []):
            assert not com["private"] or "Red" in com["players"]
    for com in client.get(f"/api/game/{gid}/state").json()["commitments"]:
        assert not com["private"] or "Red" in com["players"]


def test_a_malformed_order_is_a_400_not_a_crash():
    gid = client.post("/api/game/new", json={
        "seed": 3, "human_seat": "Red", "node_budget": 150, "max_turns": 4,
    }).json()["game_id"]
    for bad in ({"unit_territory": "R1", "order_type": "Teleport"},
                {"order_type": "Hold"},
                {"unit_territory": "R1"}):
        r = client.post(f"/api/game/{gid}/step",
                        json={"orders": [bad], "decisions": {}})
        assert r.status_code == 400, bad
        assert "malformed order" in r.json()["error"]


def test_inspecting_a_game_does_not_change_how_it_plays():
    """Detailed mode re-derives a decision, which means running the planner.
    That samples opponent worlds, so without a guard, opening the panel moves
    the stream the real game draws from."""
    def play_through(inspect):
        gid = client.post("/api/game/new", json={
            "seed": 5, "node_budget": 300, "max_turns": 6}).json()["game_id"]
        client.post(f"/api/game/{gid}/step")
        if inspect:
            for seat in ("Red", "Blue", "Green"):
                client.get(f"/api/game/{gid}/inspect/{seat}")
        for _ in range(4):
            client.post(f"/api/game/{gid}/step")
        return client.get(f"/api/game/{gid}/state").json()["scores"]

    assert play_through(False) == play_through(True)


def test_two_games_on_different_boards_do_not_contaminate_each_other():
    """The board and the knobs are process-global while a game runs."""
    import threading

    a = client.post("/api/game/new", json={
        "seed": 1, "n_seats": 4, "node_budget": 200, "max_turns": 6}).json()
    b = client.post("/api/game/new", json={
        "seed": 1, "n_seats": 7, "node_budget": 200, "max_turns": 6,
        "knobs": {"CENTRE_VALUE": 3.0}}).json()

    def step(gid):
        for _ in range(3):
            assert client.post(f"/api/game/{gid}/step").status_code in (200, 409)

    ta = threading.Thread(target=step, args=(a["game_id"],))
    tb = threading.Thread(target=step, args=(b["game_id"],))
    ta.start(); tb.start(); ta.join(); tb.join()

    sa = client.get(f"/api/game/{a['game_id']}/state").json()
    sb = client.get(f"/api/game/{b['game_id']}/state").json()
    assert len(sa["scores"]) == 4 and len(sb["scores"]) == 7
    assert sa["knobs"]["CENTRE_VALUE"] == 1.0
    assert sb["knobs"]["CENTRE_VALUE"] == 3.0


def test_interleaved_games_replay_their_own_seed():
    """Games share one process; stepping one must not change another."""
    cfg = {**CONFIG, "max_turns": 3}

    def orders(gid):
        return [sorted(map(str, h.get("orders") or []))
                for h in client.get(f"/api/game/{gid}/replay").json()["history"]]

    alone = client.post("/api/game/new", json=cfg).json()["game_id"]
    for _ in range(3):
        client.post(f"/api/game/{alone}/step")
    a = client.post("/api/game/new", json=cfg).json()["game_id"]
    b = client.post("/api/game/new", json=cfg).json()["game_id"]
    for _ in range(3):
        client.post(f"/api/game/{a}/step")
        client.get(f"/api/game/{b}/inspect/Red")
        client.post(f"/api/game/{b}/step")
    assert orders(a) == orders(alone) == orders(b)
