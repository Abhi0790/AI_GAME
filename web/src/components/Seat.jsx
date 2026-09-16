import React from "react";
import { COLOR } from "../constants.js";

/** The examiner's own seat: what they have been offered, and their orders.
 *
 * Orders are chosen from the legal set the engine supplies rather than typed,
 * so a human seat cannot submit something the adjudicator would throw away.
 */
export default function Seat({ seat, turn, pending, choices, setChoices,
                               orders, setOrders }) {
  const proposals = pending?.proposals ?? [];
  const legal = pending?.legal ?? {};

  return (
    <div className="seat">
      <h3>
        Your seat — <span style={{ color: COLOR[seat] }}>{seat}</span> · turn {turn}
      </h3>
      <div className="body">
        {!proposals.length && (
          <div className="empty">Nobody has offered you anything.</div>
        )}
        {proposals.map((m) => {
          const answer = choices[m.id];
          return (
            <div className="offer" key={m.id}>
              <span className="dot" style={{ background: COLOR[m.sender] }} />
              <span className="t">{m.text}</span>
              {m.coalition && <span className="tag info">pact · {m.coalition.length}</span>}
              {m.private && <span className="tag lie">private</span>}
              {m.type === "Threat" ? (
                <span className="tag lie">threat</span>
              ) : (
                <>
                  <button className={answer === true ? "yes" : ""}
                          onClick={() => setChoices({ ...choices, [m.id]: true })}>
                    Accept
                  </button>
                  <button className={answer === false ? "no" : ""}
                          onClick={() => setChoices({ ...choices, [m.id]: false })}>
                    Reject
                  </button>
                </>
              )}
            </div>
          );
        })}

        {Object.entries(legal).map(([territory, options]) => (
          <div className="orow" key={territory}>
            <span className="u">{territory}</span>
            <select
              value={orders[territory] ?? 0}
              onChange={(e) =>
                setOrders({ ...orders, [territory]: Number(e.target.value) })
              }
            >
              {options.map((o, i) => (
                <option key={i} value={i}>{o.label}</option>
              ))}
            </select>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Turn the selected indices back into the order objects the API expects. */
export function collectOrders(pending, orders) {
  const legal = pending?.legal ?? {};
  return Object.entries(legal).map(([territory, options]) => {
    const o = options[orders[territory] ?? 0];
    return {
      unit_territory: territory,
      order_type: o.order_type,
      target: o.target,
      supported_from: o.supported_from,
    };
  });
}
