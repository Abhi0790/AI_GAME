import React, { useEffect, useRef } from "react";
import { COLOR } from "../constants.js";

/** New-game form in a modal dialog. Edits `config`; `onStart` makes the game. */
export default function Setup({ open, onClose, onStart, config, setConfig, defaults, seats }) {
  const ref = useRef(null);
  useEffect(() => {
    const d = ref.current;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  const set = (k) => (e) =>
    setConfig({ ...config, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value });

  const num = ({ k, label, hint, ...rest }) => (
    <label className="field" title={hint} key={k}>
      <span>{label}</span>
      <input type="number" value={config[k]} onChange={set(k)} {...rest} />
    </label>
  );
  const toggle = ({ k, label, hint }) => (
    <label className="toggle" title={hint} key={k}>
      <input type="checkbox" role="switch" checked={config[k]} onChange={set(k)} />
      <span>{label}</span>
    </label>
  );

  return (
    <dialog ref={ref} className="setup" onClose={onClose}
            onClick={(e) => e.target === ref.current && onClose()}>
      <form method="dialog" onSubmit={(e) => { e.preventDefault(); onStart(); }}>
        <div className="dlg-head">
          <h2>New game</h2>
          <p>Everything is optional. Leave the seed blank for a random game.</p>
        </div>

        <div className="dlg-body">
          <section>
            <h3>Game</h3>
            <div className="fields">
              {num({ k: "seed", label: "Seed", placeholder: "random" })}
              {num({ k: "max_turns", label: "Turns", min: "1" })}
              {num({ k: "win_centers", label: "Centres to win", placeholder: "auto", min: "1", hint: "Blank uses the win share of the board." })}
              {num({ k: "win_fraction", label: "Win share", min: "0.05", max: "1", step: "0.05", hint: "Share of the board's centres that wins when no threshold is given." })}
            </div>
          </section>

          <section>
            <h3>Board</h3>
            <div className="fields">
              {num({ k: "n_seats", label: "Seats", min: "2", max: (defaults?.seat_colours ?? []).length || 8 })}
              {num({ k: "homes_per_player", label: "Homes per seat", min: "1", hint: "Home centres per seat; one unit starts on each." })}
              {num({ k: "outer_rings", label: "Outer rings", min: "0", max: "5", hint: "Rings of neutral centres outside the home ring." })}
              {num({ k: "order_noise", label: "Order noise", min: "0", max: "1", step: "0.05", hint: "Chance each order is lost and the unit holds instead." })}
            </div>
          </section>

          <section>
            <h3>Rules</h3>
            <div className="toggles">
              {toggle({ k: "standard_map", label: "Standard map", hint: "The 7-power Diplomacy map with fleets, coasts and convoys; seats, homes and rings are ignored." })}
              {toggle({ k: "home_builds", label: "Home builds", hint: "Build only on your own home centres, as in Diplomacy." })}
              {toggle({ k: "fortified_homes", label: "Fortified homes", hint: "Home centres defend at +1, so taking one needs two supports." })}
              {toggle({ k: "sieges", label: "Sieges", hint: "A centre changes hands only if the unit was already on it after spring." })}
              {toggle({ k: "draws", label: "Draws", hint: "Survivors may end a stalled game by a unanimous draw vote." })}
              {toggle({ k: "erase_reputation", label: "Erase reputation", hint: "Start with no memory of earlier games." })}
            </div>
          </section>

          <section>
            <h3>Seats</h3>
            <div className="fields">
              {seats.map((p) => (
                <label className="field" key={p}>
                  <span><i className="dot" style={{ background: COLOR[p] }} />{p}</span>
                  <select value={config.seating[p] ?? ""}
                          onChange={(e) => setConfig({
                            ...config, seating: { ...config.seating, [p]: e.target.value },
                          })}>
                    <option value="">Auto</option>
                    {(defaults?.personas ?? []).map((n) => <option key={n} value={n}>{n}</option>)}
                  </select>
                </label>
              ))}
              <label className="field">
                <span>You play</span>
                <select value={config.seat} onChange={set("seat")}>
                  <option value="">Spectate</option>
                  {seats.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </label>
              <label className="field" title="A seat that plays random legal orders.">
                <span>Chaos seat</span>
                <select value={config.chaos} onChange={set("chaos")}>
                  <option value="">None</option>
                  {seats.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </label>
            </div>
          </section>

          <section>
            <h3>Agents</h3>
            <div className="fields">
              <label className="field">
                <span>Search</span>
                <select value={config.search} onChange={set("search")}>
                  <option value="expectimax">Expectimax</option>
                  <option value="mcts">MCTS</option>
                  <option value="regret">Regret matching</option>
                </select>
              </label>
              {num({ k: "node_budget", label: "Node budget", min: 50, step: 50 })}
              {num({ k: "negotiation_rounds", label: "Negotiation rounds", min: 1, max: 6 })}
            </div>
          </section>
        </div>

        <div className="dlg-foot">
          <button type="button" className="ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="primary">Start game</button>
        </div>
      </form>
    </dialog>
  );
}
