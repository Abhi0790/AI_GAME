import React, { useEffect, useState } from "react";
import { inspect } from "../api.js";
import { COLOR, fmt, pct } from "../constants.js";

/** Detailed mode: the arithmetic behind one seat's next decision.
 *
 * Reads like a breakpoint rather than a report — the quantities in the order
 * the agent computes them, each shown with the formula that consumes it, so
 * a number that looks wrong can be traced to the step that produced it.
 */
export default function Inspector({ gameId, seat, turn, onClose }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [openDeal, setOpenDeal] = useState(0);

  useEffect(() => {
    if (!gameId || !seat) return;
    let live = true;
    setData(null);
    setError(null);
    inspect(gameId, seat)
      .then((d) => live && setData(d))
      .catch((e) => live && setError(String(e)));
    return () => { live = false; };
  }, [gameId, seat, turn]);

  return (
    <div className="detail">
      <div className="detail-head">
        <span className="dot" style={{ background: COLOR[seat] }} />
        <h2>{seat} — inside the decision</h2>
        <div className="spacer" />
        {data && <span className="pill">turn {data.turn}</span>}
        <button className="ghost" onClick={onClose} aria-label="Close detailed mode">✕</button>
      </div>
      <div className="detail-body">
        {error && <div className="empty">Could not read the agent: {error}</div>}
        {!data && !error && <div className="empty">Reading the agent…</div>}
        {data && (
          <>
            <Position data={data} />
            <Beliefs data={data} />
            <Deals data={data} open={openDeal} setOpen={setOpenDeal} />
            <Search data={data} />
            <Candidates data={data} />
          </>
        )}
      </div>
    </div>
  );
}

/* ── 1. where the seat stands ─────────────────────────────────────── */

function Position({ data }) {
  const stance = Object.entries(data.stance ?? {});
  return (
    <div className="step">
      <h5>1 · Position</h5>
      <dl className="kv">
        <dt>persona</dt><dd>{data.persona}</dd>
        <dt>board value</dt><dd>{fmt(data.position_value)}</dd>
      </dl>
      <div className="formula">
        value = <b>centres</b> + 0.3·units + 0.2·threatened − 0.2·under&nbsp;threat + 0.01·mobility
      </div>
      {!!stance.length && (
        <>
          <div className="formula">
            stance = vengeance·<span className="r">grudge</span> −
            0.4·<span className="g">their&nbsp;threat</span>
            {"  "}· positive means this seat will <b>pay</b> board value to hurt them
          </div>
          <div className="chip-row">
            {stance.map(([p, v]) => (
              <span key={p} className="tag" style={{ color: v > 0 ? "var(--bad)" : "var(--good)" }}>
                {p} {v > 0 ? "+" : ""}{fmt(v)}
              </span>
            ))}
          </div>
        </>
      )}
      {!stance.length && <p style={{ color: "var(--dim)", fontSize: 11 }}>
        No grudges and no threats believed — nobody is worth paying to hurt.
      </p>}
    </div>
  );
}

/* ── 2. what it believes, both layers ─────────────────────────────── */

function Beliefs({ data }) {
  const rows = data.beliefs ?? [];
  if (!rows.length) return null;
  return (
    <div className="step">
      <h5>2 · Beliefs — reputation vs. this relationship</h5>
      <div className="formula">
        reliability = <b>w</b>·pair + (1−<b>w</b>)·general,{"  "}
        w = evidence / (evidence + 2)
      </div>
      <table>
        <thead>
          <tr><th>of</th><th>type</th><th>general</th><th>to me</th><th>w</th></tr>
        </thead>
        <tbody>
          {rows.filter((r) => r.pair_weight > 0 || r.reliability_general !== 0.5)
               .map((r, i) => (
            <tr key={i}>
              <td><span className="dot" style={{ background: COLOR[r.subject] }} /> {r.subject}</td>
              <td style={{ color: "var(--muted)" }}>{r.commitment_type}</td>
              <td className="num" title={`Beta(${r.alpha}, ${r.beta})`}>
                {fmt(r.reliability_general)}
              </td>
              <td className="num" title={`Beta(${r.pair_alpha}, ${r.pair_beta})`}
                  style={{
                    color:
                      r.reliability_toward_me > r.reliability_general + 0.02 ? "var(--good)"
                      : r.reliability_toward_me < r.reliability_general - 0.02 ? "var(--bad)"
                      : undefined,
                  }}>
                {fmt(r.reliability_toward_me)}
              </td>
              <td className="num" style={{ color: "var(--dim)" }}>{fmt(r.pair_weight)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── 3. every live deal, priced three ways ────────────────────────── */

function Deals({ data, open, setOpen }) {
  const deals = data.deals ?? [];
  if (!deals.length)
    return (
      <div className="step muted">
        <h5>3 · Live promises</h5>
        <p style={{ color: "var(--dim)", fontSize: 11 }}>None. Nothing to keep or break.</p>
      </div>
    );

  return (
    <div className="step">
      <h5>3 · Live promises — {deals.length} on the table</h5>
      <div className="formula">
        keep when <b>P(keep)·V_kept + (1−P(keep))·V_broken</b> &gt; V_none
      </div>
      {deals.map((d, i) => {
        const c = d.commitment;
        const isOpen = open === i;
        const ev = d.expected_value_of_keeping;
        const worth = ev > d.v_none;
        return (
          <div className="card" key={c.id + i}>
            <h4 onClick={() => setOpen(isOpen ? -1 : i)} style={{ cursor: "pointer" }}>
              <span style={{ color: "var(--dim)" }}>{isOpen ? "▾" : "▸"}</span>
              {c.type}
              <span style={{ color: "var(--muted)", fontWeight: 400 }}>
                {c.players.filter((p) => p !== data.seat).join(" & ")}
              </span>
              {c.pact && <span className="tag info">pact</span>}
              {c.private && <span className="tag lie">private</span>}
              {c.leg_due === "repay" && <span className="tag">repayment due</span>}
              <div className="spacer" />
              <span className={`tag ${worth ? "kept" : "broke"}`}>
                EV {fmt(ev)} {worth ? ">" : "≤"} {fmt(d.v_none)}
              </span>
            </h4>
            {isOpen && (
              <>
                <dl className="kv" style={{ marginTop: 6 }}>
                  <dt>V_none — no deal at all</dt><dd>{fmt(d.v_none)}</dd>
                  <dt>V_kept — they honour it</dt><dd style={{ color: "var(--good)" }}>{fmt(d.v_kept)}</dd>
                  <dt>V_broken — they walk away</dt><dd style={{ color: "var(--bad)" }}>{fmt(d.v_broken)}</dd>
                  <dt>my incentive to break (ι)</dt><dd>{fmt(d.my_incentive_to_break, 3)}</dd>
                </dl>
                <table style={{ marginTop: 8 }}>
                  <thead>
                    <tr>
                      <th>party</th><th>P(keeps)</th><th>rep</th><th>to me</th>
                      <th>their ι</th><th>Vcoop</th><th>my ΔP</th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.parties.map((r, j) => (
                      <tr key={j}>
                        <td><span className="dot" style={{ background: COLOR[r.partner] }} /> {r.partner}</td>
                        <td className="num"><b>{fmt(r.p_keeps, 3)}</b></td>
                        <td className="num">{fmt(r.reliability_general)}</td>
                        <td className="num">{fmt(r.reliability_toward_me)}</td>
                        <td className="num">{fmt(r.their_incentive, 3)}</td>
                        <td className="num">{fmt(r.vcoop)}</td>
                        <td className="num">{fmt(r.my_delta_p_if_i_break, 3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="formula">
                  P(keeps) = CPT(reliability, ι) — a reliable partner with a lot
                  on offer is <b>not</b> a safe bet, which a Beta mean alone cannot say
                </div>
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ── 4. the search budget ─────────────────────────────────────────── */

function Search({ data }) {
  const s = data.search ?? {};
  const keptShare = s.candidates_total ? s.candidates_kept / s.candidates_total : 0;
  return (
    <div className="step">
      <h5>4 · Search</h5>
      <dl className="kv">
        <dt>algorithm</dt><dd>{s.algorithm} · depth {s.depth}</dd>
        <dt>sampled worlds</dt><dd>{s.opponent_samples}</dd>
        <dt>budget</dt><dd>{s.node_budget} adjudications</dd>
        <dt>pruning cost</dt><dd>{s.adjudications_spent_pruning}</dd>
      </dl>
      <div className="meter" style={{ marginTop: 6 }}>
        <span className="mono" style={{ color: "var(--muted)" }}>
          {s.candidates_kept}/{s.candidates_total}
        </span>
        <div className="track"><i style={{ width: `${Math.max(2, keptShare * 100)}%` }} /></div>
        <span className="mono" style={{ color: "var(--muted)" }}>{pct(keptShare)} kept</span>
      </div>
      <div className="formula">
        Pareto dominance: a candidate is dropped when another does at least as
        well both when everyone <b>holds</b> and when everyone <b>attacks</b>.
        Loyal and treacherous sets are pruned separately, so an option is never
        discarded merely for keeping a promise.
      </div>
    </div>
  );
}

/* ── 5. the candidates it is choosing between ─────────────────────── */

function Candidates({ data }) {
  const rows = data.candidates ?? [];
  if (!rows.length) return null;
  const best = Math.max(...rows.map((r) => r.value - r.penalty));
  const scale = Math.max(...rows.map((r) => r.value)) || 1;

  return (
    <div className="step">
      <h5>5 · Candidate orders — net = value − reputation cost</h5>
      <div className="win" aria-hidden="true">
        {rows.map((r, i) => (
          <i key={i}
             className={r.value - r.penalty === best ? "best" : r.breaks_a_promise ? "breaks" : ""}
             style={{ height: `${Math.max(6, (r.value / scale) * 100)}%` }} />
        ))}
      </div>
      {rows.map((r, i) => {
        const net = r.value - r.penalty;
        const isBest = net === best;
        return (
          <div className="card" key={i}
               style={isBest ? { borderColor: "var(--accent)" } : undefined}>
            <h4>
              {isBest && <span className="tag info">chosen</span>}
              {r.breaks_a_promise && <span className="tag broke">breaks a promise</span>}
              <div className="spacer" />
              <span className="num">net {fmt(net)}</span>
            </h4>
            <p className="mono">
              {r.orders
                .map((o) =>
                  o.type === "Hold" ? `${o.unit} H`
                    : o.type === "Move" ? `${o.unit}→${o.target}`
                    : `${o.unit} S ${o.supported_from ? o.supported_from + "→" : ""}${o.target}`
                )
                .join(", ")}
            </p>
            <div className="formula">
              {fmt(r.value)}
              {r.penalty > 0 ? <> − <span className="r">{fmt(r.penalty, 3)}</span></> : null}
              {" = "}<b>{fmt(net)}</b>
              {r.penalty_rows.map((pr, j) => (
                <div key={j} style={{ paddingLeft: 12 }}>
                  Vcoop({pr.partner}) {fmt(pr.vcoop)} × ΔP {fmt(pr.delta_p, 3)} × horizon{" "}
                  {fmt(pr.horizon)} = {fmt(pr.amount, 3)}
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
