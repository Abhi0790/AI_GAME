import React from "react";
import { COLOR, fmt, pct } from "../constants.js";

const Empty = ({ children }) => <div className="empty">{children}</div>;

const Dot = ({ player }) => (
  <span className="dot" style={{ background: COLOR[player] ?? "var(--dim)" }} />
);

/** Deal tags that carry meaning rather than decoration. */
function DealTags({ c }) {
  return (
    <>
      {c.pact && <span className="tag info">pact · {c.players.length}</span>}
      {c.private && <span className="tag lie">private</span>}
      {c.repay_turn && (
        <span className="tag">
          {c.leg_due === "repay" ? "repayment due" : `repays turn ${c.repay_turn}`}
        </span>
      )}
    </>
  );
}

/* ── Log ──────────────────────────────────────────────────────────── */

export function LogPanel({ history, upto }) {
  const steps = history.slice(0, upto + 1);
  if (!steps.length) return <Empty>Nothing yet.</Empty>;

  const rows = [];
  for (let i = steps.length - 1; i >= 0; i--) {
    const s = steps[i], t = i + 1;
    for (const o of s.outcomes ?? []) {
      if (o.kept) continue;
      rows.push(
        <div className="line bad" key={`b${i}-${rows.length}`}>
          <span className="t">T{t}</span>
          <span>
            ✕ {o.commitment_type} between {o.players.join(" & ")} broken by{" "}
            <b>{o.broken_by.join(", ")}</b>
          </span>
        </div>
      );
    }
    for (const e of [...(s.log ?? [])].reverse()) {
      let cls = "line mut";
      if (/Commitment created/.test(e)) cls = "line good";
      else if (/Pact dissolved/.test(e)) cls = "line bad";
      else if (/BROADCAST|GOSSIP/.test(e)) cls = "line warn";
      else if (/Dislodged|Removal/.test(e)) cls = "line bad";
      else if (/Move succeeds|Build/.test(e)) cls = "line";
      rows.push(
        <div className={cls} key={`l${i}-${rows.length}`}>
          <span className="t">T{t}</span>
          <span>{e}</span>
        </div>
      );
    }
  }
  return <>{rows}</>;
}

/* ── Talks ────────────────────────────────────────────────────────── */

export function TalksPanel({ history, upto }) {
  const steps = history.slice(0, upto + 1);
  if (!steps.some((s) => (s.messages ?? []).length))
    return <Empty>No messages yet.</Empty>;

  const cards = [];
  for (let i = steps.length - 1; i >= 0; i--) {
    const s = steps[i];
    if (!(s.messages ?? []).length) continue;
    const accepted = new Set(
      s.messages.filter((m) => m.type === "Accept").map((m) => m.reference_id)
    );
    cards.push(
      <div className="card" key={i}>
        <h4>
          Turn {i + 1}
          <span className="tag">{s.messages.length} messages</span>
        </h4>
        {s.messages.map((m, j) => {
          let cls = "line mut", note = null;
          if (m.type === "Accept") cls = "line good";
          else if (m.type === "Counter") cls = "line warn";
          else if (m.type === "Threat") cls = "line bad";
          else if (m.type === "Broadcast") {
            // Three verdicts now: proved, disproved, and unfalsifiable —
            // the last is where lying actually lives.
            cls = m.verdict === "CONFIRMED" ? "line warn" : "line bad";
            note =
              m.verdict === "CONFIRMED" ? <span className="tag kept">confirmed</span>
              : m.verdict === "REFUTED" ? <span className="tag broke">refuted</span>
              : <span className="tag lie">unverifiable{m.truthful === false ? " · lie" : ""}</span>;
          } else if (accepted.has(m.id)) {
            note = <span className="tag kept">accepted</span>;
          }
          return (
            <div className={cls} key={j}>
              <span className="t" style={{ color: COLOR[m.sender] }}>■</span>
              <span className="mono">
                {m.text}{" "}
                {m.coalition && <span className="tag info">pact · {m.coalition.length}</span>}{" "}
                {m.private && <span className="tag lie">private</span>} {note}
              </span>
            </div>
          );
        })}
      </div>
    );
  }
  return <>{cards}</>;
}

/* ── Deals ────────────────────────────────────────────────────────── */

export function DealsPanel({ history, upto }) {
  const step = history[upto];
  const deals = step?.commitments ?? [];
  if (!deals.length) return <Empty>No live promises this turn.</Empty>;
  return (
    <>
      {deals.map((c) => (
        <div className="card" key={c.id}>
          <h4>
            {c.type}
            {c.players.map((p) => <Dot key={p} player={p} />)}
            <span style={{ color: "var(--muted)", fontWeight: 400 }}>
              {c.players.join(" & ")}
            </span>
            <DealTags c={c} />
          </h4>
          <p>
            runs to turn {c.valid_until}
            {c.dmz_territories?.length ? ` · off-limits: ${c.dmz_territories.join(", ")}` : ""}
            {c.target_territory
              ? ` · ${c.supported_from ? `${c.supported_from} → ` : "hold "}${c.target_territory}`
              : ""}
          </p>
        </div>
      ))}
    </>
  );
}

/* ── Trust ────────────────────────────────────────────────────────── */

export function TrustPanel({ history, upto, players }) {
  const step = history[upto];
  const beliefs = step?.beliefs ?? [];
  if (!beliefs.length) return <Empty>No beliefs recorded yet.</Empty>;

  const types = [...new Set(beliefs.map((b) => b.commitment_type))].sort();
  const by = {};
  for (const b of beliefs) {
    ((by[b.observer] ??= {})[b.subject] ??= {})[b.commitment_type] = b;
  }

  const colour = (v) =>
    v > 0.66 ? "var(--good)" : v < 0.4 ? "var(--bad)" : "var(--warn)";

  return (
    <>
      {players.filter((o) => by[o]).map((observer) => (
        <div className="card" key={observer}>
          <h4><Dot player={observer} />{observer} believes</h4>
          <table>
            <thead>
              <tr>
                <th>of</th>
                {types.map((t) => <th key={t}>{t}</th>)}
              </tr>
            </thead>
            <tbody>
              {players.map((subject) => {
                const row = by[observer][subject];
                if (!row) return null;
                return (
                  <tr key={subject}>
                    <td>{subject === observer ? <b>self</b> : subject}</td>
                    {types.map((t) => {
                      const b = row[t];
                      if (!b) return <td key={t} style={{ color: "var(--dim)" }}>—</td>;
                      const toward = b.reliability_toward;
                      // Two numbers when they differ: the player's general
                      // reputation, and what this observer specifically has
                      // seen them do. The gap is the interesting part.
                      const split =
                        toward !== null && Math.abs(toward - b.reliability) > 0.02;
                      return (
                        <td key={t} className="num"
                            title={`Beta(${b.alpha}, ${b.beta})`}>
                          <span style={{ color: colour(b.reliability) }}>
                            {fmt(b.reliability)}
                          </span>
                          {split && (
                            <span style={{ color: colour(toward), opacity: 0.85 }}>
                              {" / "}{fmt(toward)}
                            </span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ))}
      <p className="empty">
        Beta posterior mean per promise type. Where two numbers appear, the
        second is that player's record <i>with this observer specifically</i>.
      </p>
    </>
  );
}

/* ── Why ──────────────────────────────────────────────────────────── */

export function WhyPanel({ history, upto }) {
  const step = history[upto];
  if (!step) return <Empty>No turn selected.</Empty>;
  const entries = Object.entries(step.traces ?? {}).filter(([, t]) => t);
  if (!entries.length) return <Empty>No traces for this turn.</Empty>;

  return (
    <>
      {entries.map(([p, t]) => (
        <div className="card" key={p}>
          <h4>
            <Dot player={p} />{p}
            {t.commitment_broken && <span className="tag broke">broke a promise</span>}
            {!!t.vengeance && <span className="tag lie">revenge {fmt(t.vengeance)}</span>}
          </h4>
          <p>{t.explanation}</p>
          {!!t.penalty_rows?.length && (
            <table>
              <thead>
                <tr>
                  <th>partner</th><th>Vcoop</th><th>ΔP</th><th>horizon</th><th>cost</th>
                </tr>
              </thead>
              <tbody>
                {t.penalty_rows.map((r, i) => (
                  <tr key={i}>
                    <td>{r.partner}</td>
                    <td className="num">{fmt(r.vcoop)}</td>
                    <td className="num">{fmt(r.delta_p, 3)}</td>
                    <td className="num">{fmt(r.horizon)}</td>
                    <td className="num">{fmt(r.amount, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="mono" style={{ marginTop: 6 }}>
            orders:{" "}
            {(t.orders ?? [])
              .map((o) =>
                o.order_type === "Hold" ? `${o.unit_territory} H`
                  : o.order_type === "Move" ? `${o.unit_territory}→${o.target}`
                  : `${o.unit_territory} S ${o.supported_from ? o.supported_from + "→" : ""}${o.target}`
              )
              .join(", ") || "—"}
          </p>
          <p>
            {t.search} · {t.nodes} adjudications · {t.candidates} candidates kept,{" "}
            {t.pruned} pruned by dominance
          </p>
        </div>
      ))}
    </>
  );
}

/* ── Calibration ──────────────────────────────────────────────────── */

export function CalibPanel({ game, history }) {
  const bins = game?.reliability ?? [];
  if (!bins.length)
    return <Empty>Nothing graded yet — a promise has to be made and then tested.</Empty>;

  const last = history[history.length - 1];
  return (
    <>
      <div className="card">
        <h4>Is P(keeps) worth anything?</h4>
        {game.brier != null && (
          <p>
            Brier {fmt(game.brier, 3)} — 0 is perfect, 0.25 is what you score by
            saying 50% every time.
          </p>
        )}
        <table>
          <thead>
            <tr><th>predicted</th><th>kept</th><th>n</th></tr>
          </thead>
          <tbody>
            {bins.map((b, i) => {
              const gap = Math.abs(b.mean_predicted - b.observed_rate);
              const c = gap < 0.1 ? "var(--good)" : gap < 0.25 ? "var(--warn)" : "var(--bad)";
              return (
                <tr key={i}>
                  <td className="num">
                    {pct(b.bin_lower)}–{pct(b.bin_upper)}
                  </td>
                  <td className="num" style={{ color: c }}>{pct(b.observed_rate)}</td>
                  <td className="num">{b.count}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {last && !!Object.keys(last.nodes ?? {}).length && (
        <div className="card">
          <h4>Search cost, last turn</h4>
          <table>
            <thead><tr><th>seat</th><th>adjudications</th></tr></thead>
            <tbody>
              {Object.entries(last.nodes).map(([p, n]) => (
                <tr key={p}><td>{p}</td><td className="num">{n}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
