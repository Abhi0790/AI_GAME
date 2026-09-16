import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "./api.js";
import { COLOR } from "./constants.js";
import Board from "./components/Board.jsx";
import Inspector from "./components/Inspector.jsx";
import Seat, { collectOrders } from "./components/Seat.jsx";
import {
  LogPanel, TalksPanel, DealsPanel, TrustPanel, WhyPanel, CalibPanel,
} from "./components/Panels.jsx";

const TABS = [
  ["log", "Log"], ["talks", "Talks"], ["deals", "Deals"],
  ["trust", "Trust"], ["why", "Why"], ["calib", "Calib"],
];

export default function App() {
  const [board, setBoard] = useState(null);
  const [game, setGame] = useState(null);
  const [history, setHistory] = useState([]);
  const [viewing, setViewing] = useState(-1);      // index into history
  const [tab, setTab] = useState("log");
  const [running, setRunning] = useState(false);
  const [detail, setDetail] = useState(null);      // seat name, or null
  const [pending, setPending] = useState(null);
  const [choices, setChoices] = useState({});
  const [orders, setOrders] = useState({});
  const [config, setConfig] = useState({ seat: "", chaos: "", broadcast: true });
  const [error, setError] = useState(null);
  const cancel = useRef(false);

  useEffect(() => { api.getBoard().then(setBoard).catch((e) => setError(String(e))); }, []);

  const live = viewing >= history.length - 1;
  const step = history[viewing];

  // history[i].state is the board *before* turn i's orders, so what that turn
  // produced is the next record's state — or the live one for the last turn.
  const shown = useMemo(() => {
    if (!step) return game?.state ?? null;
    return history[viewing + 1]?.state ?? game?.state ?? step.state;
  }, [step, history, viewing, game]);

  const scores = useMemo(() => {
    const counts = {};
    for (const owner of Object.values(shown?.supply_centers ?? {})) {
      if (owner) counts[owner] = (counts[owner] ?? 0) + 1;
    }
    return counts;
  }, [shown]);

  const refreshSeat = useCallback(async (g) => {
    if (!g?.human_seat || g.finished) { setPending(null); return; }
    try {
      const p = await api.getPending(g.id);
      setPending(p);
      setChoices({});
      setOrders({});
    } catch (e) { setError(String(e)); }
  }, []);

  const newGame = useCallback(async () => {
    cancel.current = true;
    setRunning(false);
    try {
      const d = await api.newGame({
        human_seat: config.seat || null,
        chaos_seat: config.chaos || null,
        broadcast: config.broadcast,
      });
      const g = { ...d, id: d.game_id };
      setGame(g);
      setHistory([]);
      setViewing(-1);
      setError(null);
      await refreshSeat(g);
    } catch (e) { setError(String(e)); }
  }, [config, refreshSeat]);

  const step1 = useCallback(async () => {
    if (!game || game.finished) return true;
    try {
      const payload = game.human_seat
        ? { orders: collectOrders(pending, orders), decisions: choices }
        : { orders: [], decisions: {} };
      const d = await api.stepGame(game.id, payload);
      const g = { ...game, ...d };
      setGame(g);
      if (d.step) {
        setHistory((h) => {
          const next = [...h, d.step];
          setViewing(next.length - 1);
          return next;
        });
      }
      await refreshSeat(g);
      return g.finished;
    } catch (e) { setError(String(e)); return true; }
  }, [game, pending, orders, choices, refreshSeat]);

  const runAll = useCallback(async () => {
    cancel.current = false;
    setRunning(true);
    try {
      for (let i = 0; i < 40; i++) {
        if (cancel.current) break;
        if (await step1()) break;
        await new Promise((r) => setTimeout(r, 240));
      }
    } finally { setRunning(false); }
  }, [step1]);

  // Keyboard: arrows scrub, space advances. Ignored while typing in a control.
  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target?.tagName;
      if (tag === "SELECT" || tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.key === "ArrowLeft" && viewing > 0) setViewing(viewing - 1);
      if (e.key === "ArrowRight" && viewing < history.length - 1) setViewing(viewing + 1);
      if (e.key === " " && game && !game.finished && !running) {
        e.preventDefault();
        step1();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [viewing, history.length, game, running, step1]);

  const status = !game ? "no game"
    : game.finished ? (game.winner ? `${game.winner} wins` : "finished")
    : "playing";

  return (
    <div className="app">
      <header>
        <h1>Territory <span>· negotiating agents that keep, and break, their word</span></h1>
        <span className={`pill ${!game ? "" : game.finished ? "done" : "live"}`}>{status}</span>
        {game && (
          <span className="pill">
            turn {Math.min(game.turn, game.max_turns)}/{game.max_turns} ·{" "}
            {game.win_centers} centres to win
          </span>
        )}
        {detail && <span className="pill on">detailed mode</span>}
        <div className="spacer" />
        <div className="ctl">
          <label htmlFor="seat">seat</label>
          <select id="seat" value={config.seat}
                  onChange={(e) => setConfig({ ...config, seat: e.target.value })}>
            <option value="">spectate</option>
            {(board?.players ?? []).map((p) => (
              <option key={p} value={p}>play {p}</option>
            ))}
          </select>
          <label htmlFor="chaos">chaos</label>
          <select id="chaos" value={config.chaos}
                  onChange={(e) => setConfig({ ...config, chaos: e.target.value })}>
            <option value="">none</option>
            {(board?.players ?? []).map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
          <label>
            <input type="checkbox" checked={config.broadcast}
                   onChange={(e) => setConfig({ ...config, broadcast: e.target.checked })} />
            {" "}broadcasts
          </label>
        </div>
        <div className="ctl">
          <button className="primary" onClick={newGame}>New game</button>
          <button disabled={!game || game.finished || running} onClick={step1}>Step</button>
          <button disabled={!game || game.finished || running} onClick={runAll}>Run all</button>
          {running && <button onClick={() => { cancel.current = true; }}>Stop</button>}
          <button disabled={!game || !history.length}
                  onClick={() => api.saveReplay(game.id).catch((e) => setError(String(e)))}>
            Save replay
          </button>
        </div>
      </header>

      {error && (
        <div className="line bad" style={{ padding: "6px 18px" }}>
          <span>{error}</span>
        </div>
      )}

      <main className={detail ? "detailed" : ""}>
        <section className="stage">
          <Board board={board} state={shown} step={step} />

          <div className="scores">
            {(board?.players ?? []).map((p) => (
              <button
                key={p}
                className={`score${detail === p ? " sel" : ""}`}
                onClick={() => setDetail(detail === p ? null : p)}
                title={`Open ${p}'s internals`}
              >
                <div className="who">
                  <span className="dot" style={{ background: COLOR[p] }} />
                  {p}
                </div>
                <div className="n">{scores[p] ?? 0}</div>
                <div className="sub">
                  {game?.personas?.[p] ?? ""}
                  {game?.human_seat === p ? " · you" : ""}
                </div>
                <div className="bar">
                  <i style={{
                    width: `${Math.min(100, ((scores[p] ?? 0) / (board?.win_centers ?? 5)) * 100)}%`,
                    background: COLOR[p],
                  }} />
                </div>
              </button>
            ))}
          </div>

          <div className="scrub">
            <span className="lbl">
              turn {history.length ? viewing + 1 : "—"} / {history.length || "—"}
            </span>
            <input type="range" min={0} max={Math.max(0, history.length - 1)}
                   value={Math.max(0, viewing)} disabled={history.length < 2}
                   onChange={(e) => setViewing(Number(e.target.value))}
                   aria-label="Replay scrubber" />
            <button disabled={live} onClick={() => setViewing(history.length - 1)}>Live</button>
          </div>

          {game?.human_seat && !game.finished && (
            <Seat seat={game.human_seat} turn={game.turn} pending={pending}
                  choices={choices} setChoices={setChoices}
                  orders={orders} setOrders={setOrders} />
          )}
        </section>

        <aside>
          <div className="tabs" role="tablist">
            {TABS.map(([id, label]) => (
              <button key={id} role="tab" aria-selected={tab === id}
                      onClick={() => setTab(id)}>
                {label}
              </button>
            ))}
          </div>
          <div className="panels">
            {!history.length && <div className="empty">Step the game to populate this.</div>}
            {!!history.length && tab === "log" && <LogPanel history={history} upto={viewing} />}
            {!!history.length && tab === "talks" && <TalksPanel history={history} upto={viewing} />}
            {!!history.length && tab === "deals" && <DealsPanel history={history} upto={viewing} />}
            {!!history.length && tab === "trust" && (
              <TrustPanel history={history} upto={viewing} players={board?.players ?? []} />
            )}
            {!!history.length && tab === "why" && <WhyPanel history={history} upto={viewing} />}
            {!!history.length && tab === "calib" && <CalibPanel game={game} history={history} />}
          </div>
        </aside>

        {detail && game && (
          <Inspector gameId={game.id} seat={detail} turn={game.turn}
                     onClose={() => setDetail(null)} />
        )}
      </main>
    </div>
  );
}
