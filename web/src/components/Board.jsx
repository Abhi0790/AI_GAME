import React from "react";
import { COLOR, VIEWBOX } from "../constants.js";

/** The map: one node per territory, arrows for the turn's orders.
 *
 * `state` is the position being displayed, which during a replay is not the
 * live one. `step` supplies the orders drawn over it; a move is green when
 * the log says it succeeded and dashed red when it bounced.
 */
export default function Board({ board, state, step, onPick, selected }) {
  if (!board?.positions) return <div className="board" />;

  const POS = board.positions;
  const CODE = board.seat_codes ?? {};
  // Shrink nodes as the board grows, or they overlap.
  const scale = Math.min(1, Math.sqrt(12 / board.territories.length));
  const centres = new Set(board.supply_centers);
  const unitAt = {};
  for (const u of state?.units ?? []) unitAt[u.territory] = u.player;

  const edges = [];
  const seen = new Set();
  for (const [a, ns] of Object.entries(board.adjacency)) {
    for (const b of ns) {
      const key = [a, b].sort().join("|");
      if (seen.has(key)) continue;
      seen.add(key);
      edges.push(
        <line key={key} className="edge"
              x1={POS[a].x} y1={POS[a].y} x2={POS[b].x} y2={POS[b].y} />
      );
    }
  }

  const log = (step?.log ?? []).join("\n");
  const arrows = (step?.orders ?? []).flatMap((o, i) => {
    const a = POS[o.unit_territory];
    const b = POS[o.target];
    if (!a || !b) return [];
    if (o.order_type !== "Move" && o.order_type !== "Support") return [];
    const dx = b.x - a.x, dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1, s = 28 * scale;
    const isSupport = o.order_type === "Support";
    const ok = log.includes(`Move succeeds: ${o.unit_territory} ->`);
    return [
      <line
        key={`${o.player}-${o.unit_territory}-${i}`}
        className={`arrow ${isSupport ? "sup" : ok ? "ok" : "no"}`}
        markerEnd={`url(#${isSupport ? "ah-sup" : ok ? "ah-ok" : "ah-no"})`}
        x1={a.x + (dx / len) * s} y1={a.y + (dy / len) * s}
        x2={b.x - (dx / len) * s} y2={b.y - (dy / len) * s}
      />,
    ];
  });

  return (
    <div className="board">
      <svg viewBox={`0 0 ${VIEWBOX} ${VIEWBOX}`} role="img" aria-label="Game board">
        <defs>
          <marker id="ah-ok" markerWidth="7" markerHeight="6" refX="6.5" refY="3" orient="auto">
            <path d="M0 0 L7 3 L0 6 Z" fill="#5fb98b" />
          </marker>
          <marker id="ah-no" markerWidth="7" markerHeight="6" refX="6.5" refY="3" orient="auto">
            <path d="M0 0 L7 3 L0 6 Z" fill="#d4736f" />
          </marker>
          <marker id="ah-sup" markerWidth="6" markerHeight="5" refX="5.5" refY="2.5" orient="auto">
            <path d="M0 0 L6 2.5 L0 5 Z" fill="#6ea8fe" />
          </marker>
        </defs>
        <g>{edges}</g>
        <g>{arrows}</g>
        <g>
          {board.territories.map((name) => {
            const p = POS[name];
            if (!p) return null;
            const isSc = centres.has(name);
            const owner = isSc
              ? state?.supply_centers?.[name]
              : state?.territory_owners?.[name];
            const r = (isSc ? 25 : 20) * scale;
            const occupied = !!unitAt[name];
            return (
              <g key={name}
                 onClick={onPick ? () => onPick(name) : undefined}
                 style={onPick ? { cursor: "pointer" } : undefined}>
                <circle
                  cx={p.x} cy={p.y} r={r}
                  className={`node${isSc ? " sc" : ""}`}
                  fill={owner ? COLOR[owner] : "#1a1f27"}
                  fillOpacity={owner ? (occupied ? 1 : 0.45) : 1}
                  stroke={
                    selected === name ? "#6ea8fe"
                      : isSc ? (owner ? COLOR[owner] : "#2a313b") : "#0b0d10"
                  }
                  strokeWidth={selected === name ? 3 : isSc ? 2.5 : 1.5}
                >
                  <title>
                    {name}{isSc ? " (supply centre)" : ""}
                    {owner ? ` — ${owner}` : " — unowned"}
                    {occupied ? `, ${unitAt[name]} unit` : ""}
                  </title>
                </circle>
                <text className="nlabel" x={p.x} y={occupied ? p.y - 5 * scale : p.y}
                      style={{ fontSize: `${11 * scale}px` }}>
                  {name}
                </text>
                {occupied && (
                  <text className="nunit" x={p.x} y={p.y + 8 * scale} fill="#0b0d10"
                        style={{ fontSize: `${11 * scale}px` }}>
                    ▲{CODE[unitAt[name]] ?? ""}
                  </text>
                )}
                {!occupied && owner && (
                  <text className="nowner" x={p.x} y={p.y + 9 * scale} fill="#0b0d10"
                        style={{ fontSize: `${9 * scale}px` }}>
                    {CODE[owner] ?? ""}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}
