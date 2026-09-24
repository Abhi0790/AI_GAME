import random
from typing import List, Dict, Optional

from src.common.schemas import (
    GameState, Order, Player, Commitment, Message, MessageType,
    Unit, CalibrationPoint, ReversalPoint, TurnRecord,
)
from src.common.schemas import (
    message_to_commitment, commitment_key, invalid_proposal_reason, leaders,
    obligated_parties,
)
from src.engine import adjudicator
from src.engine.adjudicator import resolve
from src.engine.orders import _SETS_CACHE
from src.engine.board import (
    get_all_territories, get_supply_centers, home_centers, win_centers, max_turns,
    players,
)


# Everything the turn loop calls on every agent, unconditionally. Anything
# optional stays behind a hasattr check in step() instead of going in here.
AGENT_METHODS = ("propose", "reply", "act", "update_beliefs_from_outcomes",
                 "beliefs", "predict_keep")

# Chance per turn that a private deal becomes known to each seat outside it.
# 0 keeps secrets forever; 1 reproduces the old behaviour, where every agent
# was handed every deal regardless of who signed it.
PRIVATE_LEAK_RATE = 0.0


class GameRunner:
    def __init__(self, agents: List, on_turn_resolved=None,
                 negotiation_rounds: int = 3):
        for a in agents:
            missing = [m for m in AGENT_METHODS if not callable(getattr(a, m, None))]
            if missing or not hasattr(a, "player"):
                raise TypeError(
                    f"{type(a).__name__} is not a playable agent: missing "
                    f"{', '.join(missing + ([] if hasattr(a, 'player') else ['player']))}")
        self.agents = {a.player: a for a in agents}
        # The agent list has to fill exactly the seats the board defines.
        if set(self.agents) != set(players()):
            raise ValueError(
                f"agents fill seats {sorted(p.value for p in self.agents)}, but the "
                f"board seats {sorted(p.value for p in players())}")
        # Bounded memory between games; the cache is pure, so this is not needed for correctness.
        _SETS_CACHE.clear()
        self.commitments: List[Commitment] = []
        self.history: List[TurnRecord] = []
        # (predicted P(keeps), what actually happened) pairs. The reliability
        # diagram and the Brier score are computed from nothing else.
        self.calibration: List[CalibrationPoint] = []
        # (what a deal was worth at signature, what breaking it was worth at
        # order time) for the same agent in the same turn. Both positive is a
        # preference reversal.
        self.reversals: List[ReversalPoint] = []
        # Adjudications spent per turn, per player — the x-axis of the
        # search-variant figure.
        self.nodes_per_turn: List[Dict[Player, int]] = []
        # Initial state: one unit on each of a seat's home centres.
        units = [
            Unit(player=p, territory=t)
            for p, homes in home_centers().items() for t in homes
        ]

        supply_centers = {t: None for t in get_supply_centers()}
        for p, homes in home_centers().items():
            for t in homes:
                supply_centers[t] = p

        # Ownership of a plain territory is just occupancy.
        territory_owners = {t: None for t in get_all_territories()}
        for u in units:
            territory_owners[u.territory] = u.player

        self.state = GameState(turn=1, units=units, supply_centers=supply_centers, territory_owners=territory_owners)
        # Snapshotted so the runner keeps the board it was built on.
        # commitment id -> seats outside it that have learned of it.
        self._leaked: Dict[str, set] = {}
        self.max_turns = max_turns()
        self.win_centers = win_centers()
        self.on_turn_resolved = on_turn_resolved
        self.negotiation_rounds = negotiation_rounds
        self._opening: Optional[List[Message]] = None
        # proposal id -> members who have said yes so far. A pact needs all
        # of them before it binds anybody.
        self._pact_signatures: Dict[str, set] = {}
        # Adjudications each seat has caused this turn, budgeted or not.
        self._work: Dict[Player, int] = {p: 0 for p in self.agents}

    def _charge(self, player: Player, fn, *args):
        """Call an agent method, charging the adjudications it causes to that seat."""
        before = adjudicator.resolve_calls
        try:
            return fn(*args)
        finally:
            self._work[player] += adjudicator.resolve_calls - before

    # ── negotiation ─────────────────────────────────────────────────────
    def begin_turn(self) -> List[Message]:
        """Generate this turn's opening proposals without playing the turn.

        The web UI calls this to show a human seat what it has been offered
        before asking for orders; `step()` then reuses exactly these messages
        instead of generating a second, different set.
        """
        if self._opening is None:
            self._opening = [m for p, a in self.agents.items()
                             for m in self._charge(p, a.propose, self.state, self.visible_to(p))]
        return self._opening

    def _negotiate(self, log_lines: List[str]) -> List[Message]:
        """Run the negotiation rounds and return every sentence spoken.

        Rounds 2 and 3 are no longer no-ops: a proposal that fails an agent's
        expected-value test can come back as a COUNTER, which the other side
        then answers.
        """
        new_messages = self.begin_turn()
        self._opening = None
        spoken: List[Message] = list(new_messages)

        for _ in range(self.negotiation_rounds):
            if not new_messages:
                break

            inbox: Dict[Player, List[Message]] = {p: [] for p in self.agents}
            for m in new_messages:
                if m.receiver:
                    inbox[m.receiver].append(m)
                else:
                    for p in self.agents:
                        if p != m.sender:
                            inbox[p].append(m)

            replies: List[Message] = []
            for p, a in self.agents.items():
                replies.extend(self._charge(p, a.reply, self.state, inbox[p], self.visible_to(p)))

            by_id = {m.id: m for m in new_messages}
            for rep in replies:
                if rep.message_type != MessageType.ACCEPT:
                    continue
                orig = by_id.get(rep.reference_id) if rep.reference_id else None
                if orig is None:
                    continue  # accepting a proposal that does not exist
                if orig.coalition:
                    if rep.sender not in orig.coalition:
                        continue  # not a member of the pact being formed
                elif orig.receiver is not None and orig.receiver != rep.sender:
                    continue  # accepting a deal that was offered to someone else
                bad = invalid_proposal_reason(orig, self.max_turns)
                if bad:
                    log_lines.append(f"Proposal rejected by the engine: {bad}")
                    continue

                if orig.coalition:
                    # A pact is one promise, not a bundle of bilateral ones:
                    # it comes into force only once every named member has
                    # said yes, and until then it binds nobody.
                    signed = self._pact_signatures.setdefault(orig.id, {orig.sender})
                    signed.add(rep.sender)
                    if not set(orig.coalition) <= signed:
                        continue

                c = message_to_commitment(orig, rep.sender, self.state.turn)
                existing = next(
                    (e for e in self.commitments
                     if commitment_key(e) == commitment_key(c)), None)
                if existing:
                    # Same deal proposed again: renew it, do not stack
                    # a second copy that the planner would price twice.
                    existing.valid_until_turn = max(
                        existing.valid_until_turn, c.valid_until_turn)
                    # An Exchange renewed on its old repayment date would owe
                    # the repay leg for every remaining turn and never ask for
                    # the support leg again — a bargain quietly turned into a
                    # standing DMZ. Renewing the terms renews the date.
                    if c.repay_turn is not None:
                        existing.repay_turn = c.repay_turn
                        existing.valid_until_turn = max(
                            existing.valid_until_turn, c.repay_turn)
                    # Publicity only travels one way. Once a promise has been
                    # made in the open it cannot be walked back into the dark,
                    # so re-proposing it privately does not hide it again.
                    existing.private = existing.private and c.private
                    self._notify_signed(existing)
                else:
                    self.commitments.append(c)
                    self._notify_signed(c)
                    # Players in a fixed order, so the same pair always reads
                    # the same way: "Blue & Red" and "Red & Blue" are one
                    # relationship and printing both makes distinct deals look
                    # like duplicates.
                    pair = " & ".join(sorted(p.value for p in c.players))
                    detail = (f" on {', '.join(c.dmz_territories)}"
                              if c.dmz_territories else "")
                    tags = "".join([
                        " [pact]" if len(c.players) > 2 else "",
                        " [private]" if c.private else "",
                        f" [repay turn {c.repay_turn}]" if c.repay_turn else "",
                    ])
                    log_lines.append(
                        f"Commitment created: {c.commitment_type.value} between "
                        f"{pair}{detail} until turn {c.valid_until_turn}{tags}")

            spoken.extend(replies)
            new_messages = replies

        return spoken

    def _notify_signed(self, c: Commitment) -> None:
        for p in c.players:
            agent = self.agents.get(p)
            if hasattr(agent, "deal_signed"):
                agent.deal_signed(c, self.state.turn)

    # ── who knows what ──────────────────────────────────────────────────
    def knows(self, player: Player, c: Commitment) -> bool:
        """A deal is visible to its parties, to everyone if public, and to
        whoever it has leaked to."""
        return (not c.private or player in c.players
                or player in self._leaked.get(c.id, ()))

    def visible_to(self, player: Player) -> List[Commitment]:
        return [c for c in self.commitments if self.knows(player, c)]

    def _spread_secrets(self) -> None:
        """Each turn a secret may get out, one seat at a time."""
        if not PRIVATE_LEAK_RATE:
            return
        for c in self.commitments:
            if not c.private:
                continue
            known = self._leaked.setdefault(c.id, set())
            for p in self.agents:
                if p in c.players or p in known:
                    continue
                if random.random() < PRIVATE_LEAK_RATE:
                    known.add(p)

    # ── calibration ─────────────────────────────────────────────────────
    def _open_predictions(self) -> List[CalibrationPoint]:
        """Every live deal, priced by every agent in it, before the turn."""
        points = []
        for c in self.commitments:
            # Only a party that owes something this turn can keep or break it.
            owing = obligated_parties(c, self.state.turn)
            for observer in c.players:
                agent = self.agents.get(observer)
                if agent is None:
                    continue
                for subject in owing:
                    if subject == observer:
                        continue
                    if hasattr(agent, "predict_keep_parts"):
                        r, i, pred = self._charge(observer, agent.predict_keep_parts,
                                                  self.state, c, subject)
                    else:
                        r, i, pred = 0.5, 0.0, self._charge(observer, agent.predict_keep,
                                                            self.state, c, subject)
                    points.append(CalibrationPoint(
                        turn=self.state.turn, observer=observer, subject=subject,
                        commitment_type=c.commitment_type, commitment_id=c.id,
                        predicted=pred, reliability=r, incentive=i,
                    ))
        return points

    @staticmethod
    def _close_predictions(points: List[CalibrationPoint], outcomes: List):
        # Keyed on the deal, not on (player, type): two live DMZs with the
        # same partner are two separate promises, and breaking one must not
        # mark the prediction about the other as wrong.
        broken = {(o.commitment.id, p) for o in outcomes if not o.kept
                  for p in o.broken_by}
        graded = {(o.commitment.id, p) for o in outcomes
                  for p in o.commitment.players}
        closed = []
        for pt in points:
            if (pt.commitment_id, pt.subject) not in graded:
                continue
            pt.observed = (pt.commitment_id, pt.subject) not in broken
            closed.append(pt)
        return closed

    # ── the turn ────────────────────────────────────────────────────────
    def step(self, verbose: bool = False) -> TurnRecord:
        """Run one full turn: negotiate, plan, resolve, update beliefs.

        Single source of truth for a turn — the CLI and the web UI both call it.
        """
        t = self.state.turn
        say = print if verbose else (lambda *a, **k: None)
        say(f"--- Turn {t} ---")

        # 1. Drop expired commitments; let grudges and fear cool off.
        #    valid_until_turn is the last turn graded, inclusive.
        self.commitments = [c for c in self.commitments if c.valid_until_turn >= t]
        for a in self.agents.values():
            if hasattr(a, "decay_stance"):
                a.decay_stance()

        # 2. Negotiation rounds
        # Said once, at the end, from log.events -- these lines are inserted
        # at the front of it below, so the order is the same and the CLI no
        # longer prints every negotiation line twice.
        negotiation_lines: List[str] = []
        messages = self._negotiate(negotiation_lines)

        # 3. What every agent predicts about the deals now on the table
        predictions = self._open_predictions()

        self._spread_secrets()

        # 4. Agents choose orders
        all_orders: List[Order] = []
        traces: Dict[Player, object] = {}
        nodes: Dict[Player, int] = {}
        for p, a in self.agents.items():
            orders, trace = self._charge(p, a.act, self.state, self.visible_to(p))
            all_orders.extend(orders)
            traces[p] = trace
            nodes[p] = getattr(getattr(a, "planner", None), "nodes", 0)
            self.reversals.extend(self._reversals(p, a, trace))

        # 5. Engine resolves and grades commitments
        new_state, outcomes, log = resolve(self.state, all_orders, self.commitments)
        for line in negotiation_lines:
            log.events.insert(0, line)
        if self.on_turn_resolved:
            self.on_turn_resolved(self.state, all_orders, log)

        for o in outcomes:
            if not o.kept:
                say(f"Commitment BROKEN by {o.broken_by}: {o.commitment.commitment_type}")

        self.calibration.extend(self._close_predictions(predictions, outcomes))

        # What was on the table *during* this turn, captured before anything
        # is torn down. A pact formed and broken in the same turn really did
        # exist, and a record written after the dissolution below would show
        # it being broken without ever showing it being made.
        live_this_turn = list(self.commitments)

        # A deal somebody walked out of does not keep binding the rest: a
        # broken promise ends, for every party. Left standing, a two-way deal
        # was "broken" again every turn of the war that followed.
        dissolved = {o.commitment.id for o in outcomes if not o.kept}
        for cid in dissolved:
            broken = next(o for o in outcomes if o.commitment.id == cid)
            if len(broken.commitment.players) < 3:
                continue
            log.events.append(
                f"Pact dissolved: {broken.commitment.commitment_type.value} between "
                f"{' & '.join(sorted(p.value for p in broken.commitment.players))} "
                f"broken by {', '.join(sorted(p.value for p in broken.broken_by))}; "
                f"the others are released")
        self.commitments = [c for c in self.commitments if c.id not in dissolved]

        for l in log.events:
            say(l)

        # 7. Beliefs and opponent models update on what they witnessed
        for p, a in self.agents.items():
            a.update_beliefs_from_outcomes(
                self.state, new_state,
                [o for o in outcomes if self.knows(p, o.commitment)])
            if hasattr(a, 'observe_orders'):
                a.observe_orders(all_orders)

        beliefs = [b for a in self.agents.values() for b in a.beliefs()]

        record = TurnRecord(
            state=self.state, orders=all_orders, outcomes=outcomes, log=log,
            traces=traces, messages=messages, beliefs=beliefs,
            commitments=live_this_turn, nodes=nodes, total_nodes=dict(self._work),
        )
        self._work = {p: 0 for p in self.agents}
        self.history.append(record)
        self.nodes_per_turn.append(nodes)
        self.state = new_state
        return record

    def _reversals(self, player: Player, agent, trace) -> List[ReversalPoint]:
        """Pair each deal this agent signed with what its planner then did.

        The negotiator and the planner price the same promise minutes apart in
        the same turn, against different models of the partner. Only deals the
        agent actually put a price on are paired; a deal it inherited without
        pricing has no signature figure to compare against.
        """
        negotiation = getattr(agent, "negotiation", None)
        prices = getattr(negotiation, "deal_prices", None)
        if not prices:
            return []
        signed_turn = negotiation.deal_signed_turn
        return [
            ReversalPoint(turn=self.state.turn, player=player, commitment_type=c_type,
                          signed_gain=prices[key], break_advantage=advantage)
            for key, c_type, advantage in getattr(trace, "reversals", [])
            if key in prices and signed_turn.get(key) == self.state.turn
        ]

    def center_counts(self):
        counts = {p: 0 for p in self.agents}
        for owner in self.state.supply_centers.values():
            if owner:
                counts[owner] += 1
        return counts

    def leaders(self):
        """Every seat tied for the most centres."""
        return leaders(self.center_counts())

    def winner(self):
        """The one seat that has won outright, else None.

        A threshold reached by two seats at once is a shared lead, not a win.
        """
        counts = self.center_counts()
        ahead = leaders(counts)
        if len(ahead) == 1 and counts[ahead[0]] >= self.win_centers:
            return ahead[0]
        return None

    def run(self, verbose: bool = True):
        while self.state.turn <= self.max_turns:
            self.step(verbose=verbose)
            champion = self.winner()
            if champion:
                if verbose:
                    print(f"Game Over: {champion.value} reached {self.win_centers} centres "
                          f"on turn {self.state.turn - 1}")
                break
        else:
            if verbose:
                print("Game Over: horizon reached")

        counts = self.center_counts()
        if verbose:
            print("Final Centers:", {p.value: c for p, c in counts.items()})
            ahead = leaders(counts)
            if not self.winner():
                print("Leading: " + ", ".join(sorted(p.value for p in ahead)))
        return self.winner()

    # ── evaluation helpers ──────────────────────────────────────────────
    def personas(self) -> Dict[Player, str]:
        """Seat -> persona name, so metrics can be keyed by persona instead of
        by colour. Without this no persona comparison is computable at all."""
        return {p: getattr(a, "persona_name", "Unknown") for p, a in self.agents.items()}

    def brier_score(self) -> Optional[float]:
        """Mean squared error of P(keeps) against what happened. 0 is perfect,
        0.25 is what you get by always saying 50%."""
        pairs = [(c.predicted, 1.0 if c.observed else 0.0)
                 for c in self.calibration if c.observed is not None]
        if not pairs:
            return None
        return sum((p - o) ** 2 for p, o in pairs) / len(pairs)
