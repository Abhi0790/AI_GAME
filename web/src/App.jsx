import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "./api.js";
import { COLOR } from "./constants.js";
import Board from "./components/Board.jsx";
import Inspector from "./components/Inspector.jsx";
import Seat, { collectOrders } from "./components/Seat.jsx";
import ThinkPanel from "./components/Think.jsx";
import {
  LogPanel, TalksPanel, DealsPanel, TrustPanel, WhyPanel, CalibPanel,
} from "./components/Panels.jsx";

const TABS = [
  ["log", "Log"], ["talks", "Talks"], ["deals", "Deals"],
  ["trust", "Trust"], ["why", "Why"], ["calib", "Calib"],
];
// Verbose mode adds one tab: the same turn, replayed per agent.
const THINK = ["think", "Think"];

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
    seat: "", chaos: "", seed: "", search: "expectiminimax",
    node_budget: 1500, negotiation_rounds: 3, seating: {},
    n_seats: 4, max_turns: 12, win_centers: "", win_fraction: 0.5, home_builds: true,
  });
  const [error, setError] = useState(null);
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

  // A game carries its own board; defaults cover the persona list and the
  // moment before the first game exists.
  const board = useMemo(
    () => (defaults || game?.board || replay?.board
      ? { ...defaults, ...(game?.board ?? replay?.board ?? {}) } : null),
    [defaults, game, replay]);

  // Seats the next game will have, which is what the setup controls act on.
  const setupSeats = useMemo(
    () => (defaults?.seat_colours ?? []).slice(0, Number(config.n_seats) || 0),
    [defaults, config.n_seats]);

  const live = viewing >= history.length - 1;
  const step = history[viewing];

  // history[i].state is the board *before* turn i's orders, so what that turn
  // produced is the next record's state — or the live one for the last turn.
  const shown = useMemo(() => {
    if (!step) return game?.state ?? null;
    return history[viewing + 1]?.state ?? game?.state
      ?? replay?.final_state ?? step.state;
  }, [step, history, viewing, game, replay]);

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
        n_seats: Number(config.n_seats),
        max_turns: Number(config.max_turns),
        win_centers: config.win_centers === "" ? null : Number(config.win_centers),
        win_fraction: Number(config.win_fraction),
        home_builds: config.home_builds,
      });
      const g = { ...d, id: d.game_id };
      setConfig((c) => ({ ...c, seed: String(d.config.seed), seating: d.config.seating }));
      setGame(g);
      setHistory([]);
      setViewing(-1);
      setError(null);
      await refreshSeat(g);
    } catch (e) { setError(String(e)); }
  }, [config, setupSeats, refreshSeat]);

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
        {game && <span className="pill" title={JSON.stringify(game.config)}>{summarise(game.config)}</span>}
        {detail && <span className="pill on">detailed mode</span>}
        {verbose && <span className="pill on">verbose</span>}
        <div className="spacer" />
        <div className="ctl">
          <label htmlFor="replay">replay</label>
          <select id="replay" value={replay?.name ?? ""}
                  onChange={(e) => openReplay(e.target.value)}>
            <option value="">live game</option>
            {replays.map((r) => (
              <option key={r.name} value={r.name}>
                {r.name.replace(/\.json$/, "")} · {r.seats} seats · {r.turns}t
              </option>
            ))}
          </select>
          <label htmlFor="seed">seed</label>
          <input id="seed" type="number" className="sm" placeholder="random"
                 value={config.seed}
                 onChange={(e) => setConfig({ ...config, seed: e.target.value })} />
          <label htmlFor="seats">seats</label>
          <input id="seats" type="number" className="sm" min="2"
                 max={(defaults?.seat_colours ?? []).length || 8}
                 value={config.n_seats}
                 onChange={(e) => setConfig({ ...config, n_seats: e.target.value })} />
          <label htmlFor="max_turns">turns</label>
          <input id="max_turns" type="number" className="sm" min="1"
                 value={config.max_turns}
                 onChange={(e) => setConfig({ ...config, max_turns: e.target.value })} />
          <label htmlFor="win_centers" title="Centres needed to win outright. Blank uses the win share of the board.">win</label>
          <input id="win_centers" type="number" className="sm" min="1"
                 placeholder="auto" value={config.win_centers}
                 onChange={(e) => setConfig({ ...config, win_centers: e.target.value })} />
          <label htmlFor="win_fraction" title="Share of the board's centres that wins when no threshold is given.">win share</label>
          <input id="win_fraction" type="number" className="sm" min="0.05" max="1" step="0.05"
                 value={config.win_fraction}
                 onChange={(e) => setConfig({ ...config, win_fraction: e.target.value })} />
          <label htmlFor="home_builds" title="Build new units only on your own home centres, as in standard Diplomacy. Off: build on any centre you own.">home builds</label>
          <input id="home_builds" type="checkbox" checked={config.home_builds}
                 onChange={(e) => setConfig({ ...config, home_builds: e.target.checked })} />
          {setupSeats.map((p) => (
            <React.Fragment key={p}>
              <label htmlFor={`persona-${p}`}>{p}</label>
              <select id={`persona-${p}`} value={config.seating[p] ?? ""}
                      onChange={(e) => setConfig({
                        ...config,
                        seating: { ...config.seating, [p]: e.target.value },
                      })}>
                <option value="">auto</option>
                {(board?.personas ?? []).map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </React.Fragment>
          ))}
        </div>
        <div className="ctl">
          <label htmlFor="search">search</label>
          <select id="search" value={config.search}
                  onChange={(e) => setConfig({ ...config, search: e.target.value })}>
            <option value="expectiminimax">expectiminimax</option>
            <option value="mcts">mcts</option>
          </select>
          <label htmlFor="budget">nodes</label>
          <input id="budget" type="number" className="sm" min={50} step={50}
                 value={config.node_budget}
                 onChange={(e) => setConfig({ ...config, node_budget: e.target.value })} />
          <label htmlFor="rounds">rounds</label>
          <input id="rounds" type="number" className="sm" min={1} max={6}
                 value={config.negotiation_rounds}
                 onChange={(e) => setConfig({ ...config, negotiation_rounds: e.target.value })} />
        </div>
        <div className="ctl">
          <label htmlFor="seat">seat</label>
          <select id="seat" value={config.seat}
                  onChange={(e) => setConfig({ ...config, seat: e.target.value })}>
            <option value="">spectate</option>
            {setupSeats.map((p) => (
              <option key={p} value={p}>play {p}</option>
            ))}
          </select>
          <label htmlFor="chaos">chaos</label>
          <select id="chaos" value={config.chaos}
                  onChange={(e) => setConfig({ ...config, chaos: e.target.value })}>
            <option value="">none</option>
            {setupSeats.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
          <label title="Replay each turn the way every agent went through it">
            <input type="checkbox" checked={verbose}
                   onChange={(e) => {
                     setVerbose(e.target.checked);
                     setTab(e.target.checked ? "think" : "log");
                   }} />
            {" "}verbose
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

      <main className={[detail && "detailed", verbose && "verbose"].filter(Boolean).join(" ")}>
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
                <div className="bar" title={`${scores[p] ?? 0} of ${board?.win_centers ?? 5} needed to win outright`}>
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
            {(verbose ? [THINK, ...TABS] : TABS).map(([id, label]) => (
              <button key={id} role="tab" aria-selected={tab === id}
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
