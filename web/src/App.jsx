import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "./api.js";
import { COLOR } from "./constants.js";
import Board from "./components/Board.jsx";
import Inspector from "./components/Inspector.jsx";
import Seat, { collectOrders } from "./components/Seat.jsx";
import ThinkPanel from "./components/Think.jsx";
import Setup from "./components/Setup.jsx";
import CentresChart from "./components/Chart.jsx";
import {
  LogPanel, TalksPanel, DealsPanel, TrustPanel, WhyPanel, CalibPanel, TurnSummary,
} from "./components/Panels.jsx";

const TABS = [
  ["log", "Log", "Everything the engine did, newest first"],
  ["talks", "Talks", "Every proposal, counter and reply this turn"],
  ["deals", "Deals", "Promises in force this turn"],
  ["trust", "Trust", "How reliable each player believes the others are"],
  ["why", "Why", "Why each agent chose its orders, and what breaking a deal cost it"],
  ["calib", "Calibration", "Whether the predicted chance a deal is kept matches what happened"],
];
// Verbose mode adds one tab: the same turn, replayed per agent.
const THINK = ["think", "Think", "The turn replayed step by step for each agent"];

// The config a screenshot has to be reproducible from, in one line.
const summarise = (c) => [
  `seed ${c.seed}`,
  Object.entries(c.seating ?? {}).map(([p, n]) => `${p}:${n}`).join(" "),
  `${c.search} ${c.node_budget}n`,
  `${c.negotiation_rounds} rounds`,
  c.human_seat && `you:${c.human_seat}`,
  c.chaos_seat && `chaos:${c.chaos_seat}`,
  c.rep_cost_scale !== 1 && `rep×${c.rep_cost_scale}`,
  ...Object.entries(c.knobs ?? {}).map(([k, v]) => `${k}=${v}`),
].filter(Boolean).join(" · ");

export default function App() {
  const [defaults, setDefaults] = useState(null);   // /api/board: the map with no game yet
  const [replays, setReplays] = useState([]);
  const [replay, setReplay] = useState(null);      // a loaded saved replay, or null
  const [game, setGame] = useState(null);
  const [history, setHistory] = useState([]);
  const [viewing, setViewing] = useState(-1);      // index into history
  const [tab, setTab] = useState("log");
  const [verbose, setVerbose] = useState(false);
  const [running, setRunning] = useState(false);
  const [detail, setDetail] = useState(null);      // seat name, or null
  const [pending, setPending] = useState(null);
  const [choices, setChoices] = useState({});
  const [orders, setOrders] = useState({});
  const [config, setConfig] = useState({
    seat: "", chaos: "", seed: "", search: "expectimax",
    node_budget: 1500, negotiation_rounds: 3, seating: {},
    n_seats: 4, homes_per_player: 2, outer_rings: 0, max_turns: 12, standard_map: false, win_centers: "", win_fraction: 0.6, home_builds: true, fortified_homes: false, sieges: false, draws: false, erase_reputation: false, order_noise: 0,
  });
  const [error, setError] = useState(null);
  const [setupOpen, setSetupOpen] = useState(false);
  const cancel = useRef(false);

  useEffect(() => { api.getBoard().then(setDefaults).catch((e) => setError(String(e))); }, []);
  useEffect(() => {
    api.listReplays().then((d) => setReplays(d.replays.filter((r) => r.playable)))
      .catch(() => setReplays([]));
  }, []);

  // A saved replay stands in for a game: it carries its own board and history.
  const openReplay = useCallback(async (name) => {
    if (!name) { setReplay(null); setHistory([]); setViewing(-1); return; }
    cancel.current = true;
    setRunning(false);
    try {
      const d = await api.loadReplay(name);
      setReplay(d);
      setGame(null);
      setPending(null);
      setHistory(d.history);
      setViewing(0);
      setError(null);
    } catch (e) { setError(String(e)); }
  }, []);

  // ?replay=name.json&turn=N opens that saved replay at that turn, so it can be linked to.
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const name = q.get("replay"), turn = Number(q.get("turn"));
    if (name) openReplay(name).then(() => turn > 0 && setViewing(turn - 1));
  }, [openReplay]);

  // A game carries its own board; defaults cover the persona list and the
  // moment before the first game exists.
  const board = useMemo(
    () => (defaults || game?.board || replay?.board
      ? { ...defaults, ...(game?.board ?? replay?.board ?? {}) } : null),
    [defaults, game, replay]);

  // Seats the next game will have, which is what the setup controls act on.
  const setupSeats = useMemo(
    () => (defaults?.seat_colours ?? []).slice(0, config.standard_map ? 7 : Number(config.n_seats) || 0),
    [defaults, config.n_seats, config.standard_map]);

  const live = viewing >= history.length - 1;
  const step = history[viewing];

  // history[i].state is the board *before* turn i's orders, so what that turn
  // produced is the next record's state — or the live one for the last turn.
  const shown = useMemo(() => {
    if (!step) return game?.state ?? null;
    return history[viewing + 1]?.state ?? game?.state
      ?? replay?.final_state ?? step.state;
  }, [step, history, viewing, game, replay]);

  const count = (s) => {
    const counts = {};
    for (const owner of Object.values(s?.supply_centers ?? {})) {
      if (owner) counts[owner] = (counts[owner] ?? 0) + 1;
    }
    return counts;
  };
  const scores = useMemo(() => count(shown), [shown]);
  const before = useMemo(() => (step ? count(step.state) : null), [step]);

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
      // A seat left on "auto" means the whole seating is left to the seed's
      // rotation; the selects are refilled from what the server chose.
      const full = setupSeats.length && setupSeats.every((p) => config.seating[p]);
      const d = await api.newGame({
        seed: config.seed === "" ? null : Number(config.seed),
        seating: full ? config.seating : null,
        search: config.search,
        node_budget: Number(config.node_budget),
        negotiation_rounds: Number(config.negotiation_rounds),
        human_seat: config.seat || null,
        chaos_seat: config.chaos || null,
        map: config.standard_map ? "standard" : "ring",
        n_seats: Number(config.n_seats),
        homes_per_player: String(config.homes_per_player).includes(",")
          ? String(config.homes_per_player).split(",").map(Number)
          : Number(config.homes_per_player),
        outer_rings: Number(config.outer_rings),
        max_turns: Number(config.max_turns),
        win_centers: config.win_centers === "" ? null : Number(config.win_centers),
        win_fraction: Number(config.win_fraction),
        home_builds: config.home_builds,
        fortified_homes: config.fortified_homes,
        sieges: config.sieges,
        draws: config.draws,
        erase_reputation: config.erase_reputation,
        knobs: Number(config.order_noise) > 0 ? { ORDER_NOISE: Number(config.order_noise) } : {},
      });
      const g = { ...d, id: d.game_id };
      setConfig((c) => ({ ...c, seed: String(d.config.seed), seating: d.config.seating }));
      setGame(g);
      setHistory([]);
      setViewing(-1);
      setDetail(null);
      setError(null);
      await refreshSeat(g);
    } catch (e) { setError(String(e)); }
  }, [config, setupSeats, refreshSeat]);

  const step1 = useCallback(async () => {
    if (!game || game.finished) return true;
    try {
      const payload = game.human_seat
        ? { orders: collectOrders(pending, orders), decisions: choices, draw: !!choices.__draw__ }
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
      if (setupOpen || tag === "SELECT" || tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.key === "ArrowLeft" && viewing > 0) setViewing(viewing - 1);
      if (e.key === "ArrowRight" && viewing < history.length - 1) setViewing(viewing + 1);
      if (e.key === " " && game && !game.finished && !running) {
        e.preventDefault();
        step1();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [viewing, history.length, game, running, step1, setupOpen]);

  const status = !game ? "no game"
    : game.finished ? (game.winner ? `${game.winner} wins` : "finished")
    : "playing";

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><span className="logo" />Territory</div>
        {game && (
          <div className="meta" title={summarise(game.config)}>
            <span className={`status ${game.finished ? "done" : "live"}`}>{status}</span>
            <span>Turn <b>{Math.min(game.turn, game.max_turns)}</b> / {game.max_turns}</span>
            <span className="sep" />
            <span><b>{game.win_centers}</b> centres to win</span>
            <span className="sep" />
            <span className="seed">seed {game.config.seed}</span>
          </div>
        )}
        {replay && !game && <div className="meta"><span className="status">replay</span></div>}
        <div className="spacer" />
        <select className="quiet" aria-label="Replay" value={replay?.name ?? ""}
                onChange={(e) => openReplay(e.target.value)}>
          <option value="">Live game</option>
          {replays.map((r) => (
            <option key={r.name} value={r.name}>
              {r.name.replace(/\.json$/, "")} · {r.seats} seats · {r.turns}t
            </option>
          ))}
        </select>
        <label className="toggle" title="Replay each turn the way every agent went through it">
          <input type="checkbox" role="switch" checked={verbose}
                 onChange={(e) => {
                   setVerbose(e.target.checked);
                   setTab(e.target.checked ? "think" : "log");
                 }} />
          <span>Agent replay</span>
        </label>
        <div className="group">
          <button disabled={!game || game.finished || running} onClick={step1}
                  title="Play one turn (space)">Step</button>
          {running
            ? <button onClick={() => { cancel.current = true; }}>Stop</button>
            : <button disabled={!game || game.finished} onClick={runAll}>Run all</button>}
        </div>
        <button className="ghost" disabled={!game || !history.length}
                onClick={() => api.saveReplay(game.id).catch((e) => setError(String(e)))}>
          Save
        </button>
        <button className="primary" onClick={() => setSetupOpen(true)}>New game</button>
      </header>
      <Setup open={setupOpen} onClose={() => setSetupOpen(false)}
             onStart={() => { setSetupOpen(false); newGame(); }}
             config={config} setConfig={setConfig} defaults={defaults} seats={setupSeats} />

      {error && (
        <div className="banner" role="alert">
          <span>{error}</span>
          <button className="ghost" onClick={() => setError(null)} aria-label="Dismiss">✕</button>
        </div>
      )}

      <main className={[detail && "detailed", verbose && "verbose"].filter(Boolean).join(" ")}>
        <section className="stage">
          {!game && !replay && (
            <div className="intro">
              <h2>Four agents negotiate, promise and sometimes betray each other.</h2>
              <p>Step through turns, scrub back with ← →, and open any player to see what it
                believed, priced and chose.</p>
              <div className="intro-actions">
                <button className="primary" onClick={newGame}>Quick start</button>
                <button onClick={() => setSetupOpen(true)}>Configure…</button>
              </div>
            </div>
          )}
          <Board board={board} state={shown} step={step} />

          <div className="scores">
            {(board?.players ?? []).map((p) => (
              <button
                key={p}
                className={`score${detail === p ? " sel" : ""}`}
                onClick={() => setDetail(detail === p ? null : p)}
                disabled={Boolean(game?.human_seat && !game.finished && p !== game.human_seat)}
                title={game?.human_seat && !game.finished && p !== game.human_seat
                  ? "Other seats' internals are available after the game"
                  : `Open ${p}'s internals`}
              >
                <div className="who">
                  <span className="dot" style={{ background: COLOR[p] }} />
                  {board?.seat_names?.[p] ?? p}
                  {board?.seat_names?.[p] && <small>{p}</small>}
                </div>
                <div className="n">
                  {scores[p] ?? 0}
                  {before && (scores[p] ?? 0) !== (before[p] ?? 0) && (
                    <span className={(scores[p] ?? 0) > (before[p] ?? 0) ? "up" : "down"}>
                      {(scores[p] ?? 0) > (before[p] ?? 0) ? "+" : ""}
                      {(scores[p] ?? 0) - (before[p] ?? 0)}
                    </span>
                  )}
                </div>
                <div className="sub">
                  {game?.personas?.[p] ?? ""}
                  {game?.human_seat === p ? " · you" : ""}
                </div>
                <div className="bar" title={`${scores[p] ?? 0} of ${board?.win_centers ?? 5} needed to win outright`}>
                  <i style={{
                    width: `${Math.min(100, ((scores[p] ?? 0) / (board?.win_centers ?? 5)) * 100)}%`,
                    background: COLOR[p],
                  }} />
                </div>
              </button>
            ))}
          </div>

          <CentresChart
            states={history.length ? [...history.map((h) => h.state), game?.state ?? replay?.final_state] : null}
            players={board?.players} names={board?.seat_names} winAt={board?.win_centers}
            viewing={viewing} onPick={setViewing} />

          <div className="scrub">
            <span className="lbl">
              Turn {history.length ? viewing + 1 : "—"} of {history.length || "—"}
            </span>
            <input type="range" min={0} max={Math.max(0, history.length - 1)}
                   value={Math.max(0, viewing)} disabled={history.length < 2}
                   onChange={(e) => setViewing(Number(e.target.value))}
                   aria-label="Replay scrubber" />
            <button disabled={live} onClick={() => setViewing(history.length - 1)}>Live</button>
          </div>

          <TurnSummary history={history} viewing={viewing} shown={shown} />

          {game?.human_seat && !game.finished && (
            <Seat seat={game.human_seat} turn={game.turn} pending={pending}
                  choices={choices} setChoices={setChoices}
                  orders={orders} setOrders={setOrders}
                  draws={game.config?.draws} />
          )}
        </section>

        <aside>
          <div className="tabs" role="tablist">
            {(verbose ? [THINK, ...TABS] : TABS).map(([id, label, hint]) => (
              <button key={id} role="tab" aria-selected={tab === id} title={hint}
                      onClick={() => setTab(id)}>
                {label}
              </button>
            ))}
          </div>
          <div className="panels">
            {!history.length && <div className="empty">Step the game to populate this.</div>}
            {!!history.length && tab === "think" && (
              <ThinkPanel history={history} upto={viewing}
                          players={board?.players ?? []} personas={game?.personas} />
            )}
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
