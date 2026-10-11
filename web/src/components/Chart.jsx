import React, { useRef, useState } from "react";
import { COLOR } from "../constants.js";

const W = 1000, H = 170, L = 28, RIGHT = 96, T = 14, B = 22;

/** Centres per player over the game, one line each; click a turn to view it. */
export default function CentresChart({ states, players, names = {}, winAt, viewing, onPick }) {
  const svg = useRef(null);
  const [hover, setHover] = useState(null);
  if (!states || states.length < 2 || !players?.length) return null;

  const series = players.map((p) => ({
    p, values: states.map((s) => Object.values(s?.supply_centers ?? {}).filter((o) => o === p).length),
  }));
  const top = Math.max(winAt ?? 0, ...series.flatMap((s) => s.values), 1);
  const n = states.length - 1;
  const x = (i) => L + (i / n) * (W - L - RIGHT);
  const y = (v) => T + (1 - v / top) * (H - T - B);
  const step = top > 12 ? 5 : top > 6 ? 2 : 1;
  const grid = [];
  for (let v = 0; v <= top; v += step) grid.push(v);
  const label = (p) => names[p] ?? p;
  // Turn i's result is state i + 1, so the cursor sits one to the right of the step.
  const at = hover ?? (viewing >= 0 ? viewing + 1 : n);

  const turnAt = (e) => {
    const pt = svg.current.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const { x: sx } = pt.matrixTransform(svg.current.getScreenCTM().inverse());
    return Math.max(0, Math.min(n, Math.round(((sx - L) / (W - L - RIGHT)) * n)));
  };

  // End labels only while they fit; past four lines the legend carries identity.
  const ends = series.length <= 4
    ? [...series].sort((a, b) => b.values[n] - a.values[n]).reduce((acc, s) => {
        const prev = acc[acc.length - 1];
        const want = y(s.values[n]);
        acc.push({ ...s, ly: prev ? Math.max(want, prev.ly + 11) : want });
        return acc;
      }, [])
    : [];
  const ranked = [...series].sort((a, b) => b.values[at] - a.values[at]);

  return (
    <div className="chart">
      <div className="chart-head">
        <span className="chart-title">Centres by turn</span>
        <span className="chart-legend">
          {series.map(({ p }) => (
            <span key={p}><i style={{ background: COLOR[p] }} />{label(p)}</span>
          ))}
        </span>
      </div>
      <svg ref={svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Supply centres held by each player after every turn"
           onMouseMove={(e) => setHover(turnAt(e))} onMouseLeave={() => setHover(null)}
           onClick={(e) => { const i = turnAt(e); if (i > 0) onPick?.(i - 1); }}>
        {grid.map((v) => (
          <g key={v}>
            <line className="grid" x1={L} x2={W - RIGHT} y1={y(v)} y2={y(v)} />
            <text className="tick" x={L - 6} y={y(v)}>{v}</text>
          </g>
        ))}
        {winAt && winAt <= top && (
          <g>
            <line className="win" x1={L} x2={W - RIGHT} y1={y(winAt)} y2={y(winAt)} />
            <text className="tick win-label" x={L + 4} y={y(winAt) - 8}>wins at {winAt}</text>
          </g>
        )}
        {[0, n].map((i) => (
          <text key={i} className="tick x" x={x(i)} y={H - 6} style={{ textAnchor: i === 0 ? "start" : "end" }}>{i === 0 ? "start" : `turn ${n}`}</text>
        ))}
        <line className="cursor" x1={x(at)} x2={x(at)} y1={T} y2={H - B} />
        {series.map(({ p, values }) => {
          const points = values.map((v, i) => `${x(i)},${y(v)}`).join(" ");
          return (
            <g key={p}>
              <polyline className="halo" points={points} />
              <polyline className="series" stroke={COLOR[p]} points={points} />
            </g>
          );
        })}
        {series.map(({ p, values }) => (
          <circle key={p} cx={x(at)} cy={y(values[at])} r={4} fill={COLOR[p]} className="dot" />
        ))}
        {ends.map(({ p, values, ly }) => (
          <text key={p} className="end" x={W - RIGHT + 6} y={ly}>{label(p)} {values[n]}</text>
        ))}
      </svg>
      {hover !== null && (
        <div className="chart-tip" style={{ left: `${(x(at) / W) * 100}%` }}>
          <b>{at === 0 ? "Start" : `After turn ${at}`}</b>
          {ranked.map(({ p, values }) => (
            <span key={p}><i style={{ background: COLOR[p] }} />{label(p)}<em>{values[at]}</em></span>
          ))}
        </div>
      )}
      <details className="chart-table">
        <summary>Table</summary>
        <table>
          <thead><tr><th>Turn</th>{series.map(({ p }) => <th key={p}>{label(p)}</th>)}</tr></thead>
          <tbody>
            {states.map((_, i) => (
              <tr key={i}><td>{i === 0 ? "start" : i}</td>
                {series.map(({ p, values }) => <td key={p}>{values[i]}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
