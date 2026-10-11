import React from "react";
import { COLOR } from "../constants.js";

/** The map: one node per territory, arrows for the turn's orders.
 *
 * `state` is the position being displayed, which during a replay is not the
 * live one. `step` supplies the orders drawn over it, in each player's
 * colour: solid when the move went through, dashed when it bounced.
 */
export default function Board({ board, state, step, onPick, selected }) {
  if (!board?.positions) return <div className="board" />;

  const POS = board.positions;
  const names = Object.keys(POS);
  const seas = new Set(board.seas ?? []);
  const centres = new Set(board.supply_centers);
  const NAME = board.seat_names ?? {};
  const code = (p) => (NAME[p] ? NAME[p][0] : board.seat_codes?.[p] ?? "");
  const unitAt = {};
  const fleetAt = {};
  for (const u of state?.units ?? []) {
    unitAt[u.territory] = u.player;
    fleetAt[u.territory] = u.unit_type === "Fleet";
  }

  // Nodes are sized by how close neighbours sit, so a dense map stays legible.
  const edges = [];
  const seen = new Set();
  let shortest = Infinity;
  for (const [a, ns] of Object.entries(board.adjacency)) {
    for (const b of ns) {
      const key = [a, b].sort().join("|");
      if (seen.has(key) || !POS[a] || !POS[b]) continue;
      seen.add(key);
      shortest = Math.min(shortest, Math.hypot(POS[a].x - POS[b].x, POS[a].y - POS[b].y));
      edges.push(
        <line key={key} className={`edge${seas.has(a) || seas.has(b) ? " wet" : ""}`}
              x1={POS[a].x} y1={POS[a].y} x2={POS[b].x} y2={POS[b].y} />
      );
    }
  }
  const R = Math.max(9, Math.min(25, 0.36 * shortest));
  const k = R / 25;
  const xs = names.map((t) => POS[t].x), ys = names.map((t) => POS[t].y);
  const pad = R * 1.6;
  const box = [Math.min(...xs) - pad, Math.min(...ys) - pad,
               Math.max(...xs) - Math.min(...xs) + 2 * pad,
               Math.max(...ys) - Math.min(...ys) + 2 * pad];

  const log = step?.log ?? [];
  const succeeded = new Set(log.filter((e) => e.startsWith("Move succeeds: "))
    .map((e) => e.slice(15).split(" ->")[0]));
  const arrow = (key, from, to, cls, colour) => {
    const a = POS[from], b = POS[to];
    if (!a || !b) return null;
    const dx = b.x - a.x, dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1, s = R * 1.12;
    return (
      <line key={key} className={`arrow ${cls}`} stroke={colour}
            markerEnd={`url(#ah-${colour.replace("#", "")})`}
            x1={a.x + (dx / len) * s} y1={a.y + (dy / len) * s}
            x2={b.x - (dx / len) * s} y2={b.y - (dy / len) * s} />
    );
  };
  const arrows = (step?.orders ?? []).map((o, i) => {
    const c = COLOR[o.player] ?? "#888";
    const key = `${o.player}-${o.unit_territory}-${i}`;
    if (o.order_type === "Move")
      return arrow(key, o.unit_territory, o.target, succeeded.has(o.unit_territory) ? "ok" : "no", c);
    if (o.order_type === "Support") return arrow(key, o.unit_territory, o.target, "sup", c);
    if (o.order_type === "Convoy") return arrow(key, o.unit_territory, o.target, "cvy", c);
    return null;
  });
  const retreats = log.filter((e) => e.startsWith("Retreat: ")).map((e, i) => {
    const [from, to] = e.slice(9).split(" -> ");
    return arrow(`r${i}`, from, to.split("/")[0], "ret", "#9aa0aa");
  });
  const colours = [...new Set([...Object.values(COLOR), "#888", "#9aa0aa"])];

  return (
    <div className="board">
      <svg viewBox={box.join(" ")} role="img" aria-label="Game board">
        <defs>
          {colours.map((c) => (
            <marker key={c} id={`ah-${c.replace("#", "")}`} markerWidth="7" markerHeight="6"
                    refX="6.5" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0 0 L7 3 L0 6 Z" fill={c} />
            </marker>
          ))}
        </defs>
        <g>{edges}</g>
        <g>
          {names.map((name) => {
            const p = POS[name];
            const sea = seas.has(name);
            const isSc = centres.has(name);
            const owner = isSc
              ? state?.supply_centers?.[name]
              : state?.territory_owners?.[name];
            const r = (sea ? 0.7 : isSc ? 1 : 0.8) * R;
            const occupied = !!unitAt[name];
            const fill = occupied ? COLOR[unitAt[name]]
              : owner ? COLOR[owner] : sea ? "var(--sea)" : "var(--panel-2)";
            return (
              <g key={name}
                 onClick={onPick ? () => onPick(name) : undefined}
                 style={onPick ? { cursor: "pointer" } : undefined}>
                <circle
                  cx={p.x} cy={p.y} r={r}
                  className={`node${isSc ? " sc" : ""}${sea ? " sea" : ""}`}
                  fill={fill}
                  fillOpacity={occupied ? 1 : owner ? 0.45 : 1}
                  stroke={
                    selected === name ? "var(--accent)"
                      : isSc ? (owner ? COLOR[owner] : "var(--line-2)") : "var(--bg)"
                  }
                  strokeWidth={(selected === name ? 3 : isSc ? 2.5 : 1.5) * Math.max(k, 0.6)}
                >
                  <title>
                    {name}{sea ? " (sea)" : isSc ? " (supply centre)" : ""}
                    {owner ? ` — ${NAME[owner] ?? owner}` : ""}
                    {occupied ? `, ${NAME[unitAt[name]] ?? unitAt[name]} ${fleetAt[name] ? "fleet" : "army"}` : ""}
                  </title>
                </circle>
                <text className={`nlabel${sea ? " sea" : ""}`} x={p.x}
                      y={occupied ? p.y - 4.5 * k : p.y}
                      style={{ fontSize: `${Math.max(11 * k, 8)}px` }}>
                  {name}
                </text>
                {occupied && (
                  <text className="nunit" x={p.x} y={p.y + 8.5 * k} fill="var(--bg)"
                        style={{ fontSize: `${Math.max(11 * k, 8)}px` }}>
                    {fleetAt[name] ? "◆" : "▲"}{code(unitAt[name])}
                  </text>
                )}
                {!occupied && owner && !sea && (
                  <text className="nowner" x={p.x} y={p.y + 9 * k} fill="var(--bg)"
                        style={{ fontSize: `${Math.max(9 * k, 7)}px` }}>
                    {code(owner)}
                  </text>
                )}
              </g>
            );
          })}
        </g>
        <g>{arrows}{retreats}</g>
      </svg>
      <div className="legend">
        <span><i className="ln ok" />move, in the player's colour</span>
        <span><i className="ln no" />bounced</span>
        <span><i className="ln sup" />support</span>
        {seas.size > 0 && <span><i className="ln cvy" />convoy</span>}
        <span><i className="ln ret" />retreat</span>
        <span><i className="sw sc" />supply centre (larger)</span>
        <span><i className="sw faded" />owned, empty</span>
        <span>▲ army</span>
        {seas.size > 0 && <span>◆ fleet</span>}
        {seas.size > 0 && <span><i className="sw sea" />sea</span>}
      </div>
    </div>
  );
}
