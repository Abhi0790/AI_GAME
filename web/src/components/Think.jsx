import React, { useState } from "react";
import { COLOR, fmt, pct, orderText } from "../constants.js";

/** Verbose mode — one turn, replayed as each agent went through it.
 *
 * The other tabs slice a turn by category: every message here, every deal
 * there. This one slices it by *agent*, in the order the turn loop actually
 * runs (talk, price, search, commit, get graded), so a single seat's turn
 * reads top to bottom. Nothing is fetched: every number below is already in
 * the turn record, it was just never laid out this way.
 */

const Dot = ({ p }) => (
  <span className="dot" style={{ background: COLOR[p] ?? "var(--dim)" }} />
);

const Phase = ({ n, title, muted, children }) => (
  <div className={`step${muted ? " muted" : ""}`}>
    <h5>{n} · {title}</h5>
    {children}
  </div>
);

const Nothing = ({ children }) => (
  <p style={{ color: "var(--dim)", fontSize: 11 }}>{children}</p>
);

/** The expected-value line an agent answered a proposal with. */
function Verdict({ r }) {
  if (!r) return null;
  if (r.gain !== undefined) {
    return (
      <div className="formula">
        worth <b>{fmt(r.gain)}</b> more to me honoured than not · best{" "}
        {r.considered === 1 ? "of the one deal" : `of ${r.considered} deals`} I
        priced against this seat · floor {fmt(r.floor)}
      </div>
    );
  }
  if (r.ev === undefined) return null;
  const take = r.ev > r.v_none;
  return (
    <div className="formula">
      P(keep) <b>{fmt(r.p_keep, 3)}</b> × {fmt(r.v_kept)} + {fmt(1 - r.p_keep, 3)} ×{" "}
      <span className="r">{fmt(r.v_broken)}</span> = <b>{fmt(r.ev)}</b>{" "}
      {take ? ">" : "≤"} {fmt(r.v_none)} without it
      {r.ev_countered !== undefined && (
        <div>
          shorter version: <b>{fmt(r.ev_countered)}</b> &gt;{" "}
          {fmt(r.v_none_countered)} — so counter rather than walk
        </div>
      )}
    </div>
  );
}

export default function ThinkPanel({ history, upto, players, personas }) {
  const [only, setOnly] = useState(null);
  const step = history[upto];
  if (!step) return <div className="empty">No turn selected.</div>;

  const msgs = step.messages ?? [];
  const repliesTo = {};
  for (const m of msgs) {
    if (m.reference_id) (repliesTo[m.reference_id] ??= []).push(m);
  }
  const seats = players.filter((p) => step.traces?.[p]);

  return (
    <>
      <div className="chip-row" style={{ marginBottom: 10 }}>
        <button className={only ? "" : "primary"} onClick={() => setOnly(null)}>
          all seats
        </button>
        {seats.map((p) => (
          <button key={p} className={only === p ? "primary" : ""}
                  onClick={() => setOnly(only === p ? null : p)}>
            {p}
          </button>
        ))}
      </div>
      {seats.filter((p) => !only || p === only).map((p) => (
        <Agent key={p} seat={p} persona={personas?.[p]} step={step}
               turn={upto + 1} msgs={msgs} repliesTo={repliesTo} />
      ))}
    </>
  );
}

function Agent({ seat, persona, step, turn, msgs, repliesTo }) {
  const t = step.traces[seat];
  const mine = (m) => m.sender === seat;
  const toMe = (m) =>
    m.receiver === seat || (!m.receiver && m.sender !== seat);

  const heard = msgs.filter((m) => toMe(m) &&
    ["Propose", "Counter", "Threat"].includes(m.type));
  const said = msgs.filter((m) => mine(m) && m.type === "Propose");
  const answers = Object.fromEntries(
    msgs.filter((m) => mine(m) && m.reference_id).map((m) => [m.reference_id, m])
  );
  const deals = (step.commitments ?? []).filter((c) => c.players.includes(seat));
  const graded = (step.outcomes ?? []).filter((o) => o.players.includes(seat));
  const net = (t?.expected_value ?? 0) - (t?.penalty ?? 0);
  const keptShare = t?.candidates + t?.pruned
    ? t.candidates / (t.candidates + t.pruned) : 0;

  return (
    <div className="think">
      <div className="think-head">
        <Dot p={seat} />
        <b>{seat}</b>
        <span style={{ color: "var(--muted)" }}>{persona}</span>
        <div className="spacer" />
        {t?.commitment_broken && <span className="tag broke">broke a promise</span>}
        {!!t?.vengeance && <span className="tag lie">revenge {fmt(t.vengeance)}</span>}
        <span className="pill">turn {turn}</span>
      </div>

      <Phase n="1" title={`Heard — ${heard.length} sentence${heard.length === 1 ? "" : "s"}`}
             muted={!heard.length}>
        {!heard.length && <Nothing>Nobody said anything to this seat.</Nothing>}
        {heard.map((m, i) => {
          const a = answers[m.id];
          return (
            <div className="card" key={i}>
              <h4>
                <Dot p={m.sender} />
                <span className="mono" style={{ fontWeight: 400 }}>{m.text}</span>
                {m.coalition && <span className="tag info">pact · {m.coalition.length}</span>}
                {m.private && <span className="tag lie">private</span>}
                <div className="spacer" />
                {a
                  ? <span className={`tag ${a.type === "Accept" ? "kept"
                      : a.type === "Counter" ? "info" : "broke"}`}>{a.type.toLowerCase()}</span>
                  : <span className="tag">{m.type === "Threat" ? "priced, not answered" : "no answer"}</span>}
              </h4>
              {m.type === "Threat" && (
                <p>
                  Believed in proportion to how often this sender has actually
                  retaliated before; the planner then charges more to attack them.
                </p>
              )}
              <Verdict r={a?.rationale} />
            </div>
          );
        })}
      </Phase>

      <Phase n="2" title={`Offered — ${said.length}`} muted={!said.length}>
        {!said.length && <Nothing>Offered nothing: no deal cleared the floor.</Nothing>}
        {said.map((m, i) => {
          const back = (repliesTo[m.id] ?? []).filter((r) => r.sender !== seat);
          return (
            <div className="card" key={i}>
              <h4>
                <span className="mono" style={{ fontWeight: 400 }}>{m.text}</span>
                {m.coalition && <span className="tag info">pact · {m.coalition.length}</span>}
                {m.private && <span className="tag lie">private</span>}
                <div className="spacer" />
                {back.length
                  ? back.map((r, j) => (
                      <span key={j} className={`tag ${r.type === "Accept" ? "kept"
                        : r.type === "Counter" ? "info" : "broke"}`}>
                        {r.sender} {r.type.toLowerCase()}s
                      </span>
                    ))
                  : <span className="tag">no reply</span>}
              </h4>
              <Verdict r={m.rationale} />
            </div>
          );
        })}
      </Phase>

      <Phase n="3" title={`On the table — ${deals.length} live promise${deals.length === 1 ? "" : "s"}`}
             muted={!deals.length}>
        {!deals.length && <Nothing>Bound by nothing this turn, so nothing to break.</Nothing>}
        {deals.map((c) => (
          <div className="line" key={c.id}>
            <span className="t">→</span>
            <span className="mono">
              {c.type} with {c.players.filter((q) => q !== seat).join(" & ")} to turn{" "}
              {c.valid_until}
              {c.dmz_territories?.length ? ` · off ${c.dmz_territories.join(", ")}` : ""}
              {c.pact && <span className="tag info">pact</span>}
              {c.private && <span className="tag lie">private</span>}
              {c.leg_due === "repay" && <span className="tag">repayment due</span>}
            </span>
          </div>
        ))}
      </Phase>

      <Phase n="4" title="Searched">
        <dl className="kv">
          <dt>algorithm</dt><dd>{t?.search}</dd>
          <dt>adjudications</dt><dd>{t?.nodes ?? step.nodes?.[seat] ?? 0}</dd>
        </dl>
        <div className="meter" style={{ marginTop: 6 }}>
          <span className="mono" style={{ color: "var(--muted)" }}>
            {t?.candidates}/{(t?.candidates ?? 0) + (t?.pruned ?? 0)}
          </span>
          <div className="track"><i style={{ width: `${Math.max(2, keptShare * 100)}%` }} /></div>
          <span className="mono" style={{ color: "var(--muted)" }}>
            {pct(keptShare)} survived pruning
          </span>
        </div>
      </Phase>

      <Phase n="5" title="Chose">
        <p className="mono" style={{ color: "var(--text)" }}>
          {(t?.orders ?? []).map(orderText).join(", ") || "—"}
        </p>
        <div className="formula">
          board {fmt(t?.expected_value)}
          {t?.penalty > 0 ? <> − <span className="r">{fmt(t.penalty, 3)}</span> reputation</> : null}
          {" = "}<b>{fmt(net)}</b>
        </div>
        <p>{t?.explanation}</p>
        {!!t?.penalty_rows?.length && (
          <table>
            <thead>
              <tr><th>paid to</th><th>Vcoop</th><th>ΔP</th><th>horizon</th><th>cost</th></tr>
            </thead>
            <tbody>
              {t.penalty_rows.map((r, i) => (
                <tr key={i}>
                  <td><Dot p={r.partner} />{r.partner}</td>
                  <td className="num">{fmt(r.vcoop)}</td>
                  <td className="num">{fmt(r.delta_p, 3)}</td>
                  <td className="num">{fmt(r.horizon)}</td>
                  <td className="num">{fmt(r.amount, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {!!t?.forfeit_rows?.length && (
          <table>
            <thead>
              <tr><th>forfeits</th><th>partner</th><th>signed at</th><th>left</th><th>cost</th></tr>
            </thead>
            <tbody>
              {t.forfeit_rows.map((r, i) => (
                <tr key={i}>
                  <td>{r.commitment}</td>
                  <td>{r.partner}</td>
                  <td className="num">{fmt(r.signed_price, 3)}</td>
                  <td className="num">{fmt(r.fraction_remaining)}</td>
                  <td className="num">{fmt(r.amount, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Phase>

      <Phase n="6" title="Graded" muted={!graded.length}>
        {!graded.length && <Nothing>Nothing of this seat's was tested.</Nothing>}
        {graded.map((o, i) => (
          <div className={`line ${o.kept ? "good" : "bad"}`} key={i}>
            <span className="t">{o.kept ? "✓" : "✕"}</span>
            <span>
              {o.commitment_type} with {o.players.filter((q) => q !== seat).join(" & ")}{" "}
              {o.kept ? "kept" : `broken by ${o.broken_by.join(", ")}`}
            </span>
          </div>
        ))}
      </Phase>
    </div>
  );
}
