from typing import List, Dict, Optional

from src.common.schemas import (
    GameState, Order, Player, Commitment, Message, MessageType,
    Unit, CalibrationPoint, TurnRecord, describe_message,
)
from src.common.schemas import (
    message_to_commitment, commitment_key, invalid_proposal_reason,
)
from src.engine.adjudicator import resolve
from src.engine.board import (
    get_all_territories, get_supply_centers, HOME_CENTERS, WIN_CENTERS, MAX_TURNS,
)


class GameRunner:
    def __init__(self, agents: List, on_turn_resolved=None, broadcast_enabled: bool = True):
        self.agents = {a.player: a for a in agents}
        self.commitments: List[Commitment] = []
        self.history: List[TurnRecord] = []
        # (predicted P(keeps), what actually happened) pairs. The reliability
        # diagram and the Brier score are computed from nothing else.
        self.calibration: List[CalibrationPoint] = []
        # Adjudications spent per turn, per player — the x-axis of the
        # search-variant figure.
        self.nodes_per_turn: List[Dict[Player, int]] = []
        # The gossip channel can be switched off, which is the control
        # condition for the turns-to-coalition figure.
        self.broadcast_enabled = broadcast_enabled

        # Initial state: two units per player, one on each home centre.
        units = [
            Unit(player=Player(p), territory=t)
            for p, homes in HOME_CENTERS.items() for t in homes
        ]

        supply_centers = {t: None for t in get_supply_centers()}
        for p, homes in HOME_CENTERS.items():
            for t in homes:
                supply_centers[t] = Player(p)

        # Ownership of a plain territory is just occupancy.
        territory_owners = {t: None for t in get_all_territories()}
        for u in units:
            territory_owners[u.territory] = u.player

        self.state = GameState(turn=1, units=units, supply_centers=supply_centers, territory_owners=territory_owners)
        self.max_turns = MAX_TURNS
        self.on_turn_resolved = on_turn_resolved
        self.negotiation_rounds = 3
        self._opening: Optional[List[Message]] = None
        # proposal id -> members who have said yes so far. A pact needs all
        # of them before it binds anybody.
        self._pact_signatures: Dict[str, set] = {}

    # ── negotiation ─────────────────────────────────────────────────────
    def begin_turn(self) -> List[Message]:
        """Generate this turn's opening proposals without playing the turn.

        The web UI calls this to show a human seat what it has been offered
        before asking for orders; `step()` then reuses exactly these messages
        instead of generating a second, different set.
        """
        if self._opening is None:
            self._opening = [m for a in self.agents.values()
                             for m in a.propose(self.state, self.commitments)]
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

            inbox: Dict[Player, List[Message]] = {p: [] for p in Player}
            for m in new_messages:
                if m.receiver:
                    inbox[m.receiver].append(m)
                else:
                    for p in Player:
                        if p != m.sender:
                            inbox[p].append(m)

            replies: List[Message] = []
            for p, a in self.agents.items():
                replies.extend(a.reply(self.state, inbox[p], self.commitments))

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
                    # Publicity only travels one way. Once a promise has been
                    # made in the open it cannot be walked back into the dark,
                    # so re-proposing it privately does not hide it again.
                    existing.private = existing.private and c.private
                else:
                    self.commitments.append(c)
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

    # ── accusations ─────────────────────────────────────────────────────
    def verify_accusation(self, msg: Message, outcomes: List) -> str:
        """The engine's verdict on a claim that somebody betrayed somebody.

        Agents decide what to say, so a claim can be false, and the engine is
        the only thing that saw every order. But it can only settle a claim
        about a deal it was told about *publicly*:

          CONFIRMED  a public deal between these two was broken this turn
          REFUTED    a public deal between these two was kept this turn
          UNVERIFIED no public deal of that kind exists between them, so the
                     claim is either about a private deal or invented, and
                     nothing the engine knows separates the two

        That third verdict is what makes lying a decision rather than a
        mistake: a liar who picks a partner they have no public deal with
        cannot be refuted, and third parties have only the accuser's own
        reputation to go on.
        """
        relevant = [
            o for o in outcomes
            if not o.commitment.private
            and msg.sender in o.commitment.players
            and msg.broadcast_target in o.commitment.players
            and (not msg.commitment_type
                 or msg.commitment_type == o.commitment.commitment_type)
        ]
        if not relevant:
            return "UNVERIFIED"
        if any(msg.broadcast_target in o.broken_by for o in relevant):
            return "CONFIRMED"
        return "REFUTED"

    # ── calibration ─────────────────────────────────────────────────────
    def _open_predictions(self) -> List[CalibrationPoint]:
        """Every live deal, priced by every agent in it, before the turn."""
        points = []
        for c in self.commitments:
            for observer in c.players:
                agent = self.agents.get(observer)
                if agent is None:
                    continue
                for subject in c.players:
                    if subject == observer:
                        continue
                    if hasattr(agent, "predict_keep_parts"):
                        r, i, pred = agent.predict_keep_parts(self.state, c, subject)
                    else:
                        r, i, pred = 0.5, 0.0, agent.predict_keep(self.state, c, subject)
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
        """Run one full turn: negotiate, plan, resolve, gossip, update beliefs.

        Single source of truth for a turn — the CLI and the web UI both call it.
        """
        t = self.state.turn
        say = print if verbose else (lambda *a, **k: None)
        say(f"--- Turn {t} ---")

        # 1. Drop expired commitments; let grudges and fear cool off.
        self.commitments = [c for c in self.commitments if c.valid_until_turn >= t]
        for a in self.agents.values():
            if hasattr(a, "decay_stance"):
                a.decay_stance()

        # 2. Negotiation rounds
        negotiation_lines: List[str] = []
        messages = self._negotiate(negotiation_lines)
        for line in negotiation_lines:
            say(line)

        # 3. What every agent predicts about the deals now on the table
        predictions = self._open_predictions()

        # 4. Agents choose orders
        all_orders: List[Order] = []
        traces: Dict[Player, object] = {}
        nodes: Dict[Player, int] = {}
        for p, a in self.agents.items():
            orders, trace = a.act(self.state, self.commitments)
            all_orders.extend(orders)
            traces[p] = trace
            nodes[p] = getattr(getattr(a, "planner", None), "nodes", 0)

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

        # A pact that somebody walked out of does not keep binding the rest.
        # Releasing the loyal members is the whole difference between a
        # three-way promise and three separate ones.
        dissolved = {
            o.commitment.id for o in outcomes
            if not o.kept and len(o.commitment.players) > 2
        }
        for cid in dissolved:
            broken = next(o for o in outcomes if o.commitment.id == cid)
            log.events.append(
                f"Pact dissolved: {broken.commitment.commitment_type.value} between "
                f"{' & '.join(sorted(p.value for p in broken.commitment.players))} "
                f"broken by {', '.join(sorted(p.value for p in broken.broken_by))}; "
                f"the others are released")
        self.commitments = [c for c in self.commitments if c.id not in dissolved]

        # 6. Agents decide whether to accuse anyone; the engine settles it
        broadcasts: List[Message] = []
        if self.broadcast_enabled:
            for p, a in self.agents.items():
                for msg in a.consider_broadcast(new_state, outcomes):
                    msg.engine_verdict = self.verify_accusation(msg, outcomes)
                    broadcasts.append(msg)
                    log.events.append(
                        f"{describe_message(msg)}"
                        f"{' (LIE)' if msg.truthful is False else ''}")
                    say(f"[GOSSIP] {describe_message(msg)}")

            for msg in broadcasts:
                for p, a in self.agents.items():
                    if p != msg.sender:
                        a.receive_gossip(msg)

        messages.extend(broadcasts)

        for l in log.events:
            say(l)

        # 7. Beliefs and opponent models update on what they witnessed
        for a in self.agents.values():
            a.update_beliefs_from_outcomes(self.state, new_state, outcomes)
            if hasattr(a, 'observe_orders'):
                a.observe_orders(all_orders)

        beliefs = [b for a in self.agents.values() for b in a.beliefs()]

        record = TurnRecord(
            state=self.state, orders=all_orders, outcomes=outcomes, log=log,
            traces=traces, messages=messages, beliefs=beliefs,
            commitments=list(self.commitments), nodes=nodes,
        )
        self.history.append(record)
        self.nodes_per_turn.append(nodes)
        self.state = new_state
        return record

    def center_counts(self):
        counts = {p: 0 for p in Player}
        for owner in self.state.supply_centers.values():
            if owner:
                counts[owner] += 1
        return counts

    def winner(self):
        """Whoever holds WIN_CENTERS centres, else None while the game runs."""
        counts = self.center_counts()
        leader = max(counts, key=counts.get)
        return leader if counts[leader] >= WIN_CENTERS else None

    def run(self, verbose: bool = True):
        while self.state.turn <= self.max_turns:
            self.step(verbose=verbose)
            champion = self.winner()
            if champion:
                if verbose:
                    print(f"Game Over: {champion.value} reached {WIN_CENTERS} centres "
                          f"on turn {self.state.turn - 1}")
                break
        else:
            if verbose:
                print("Game Over: horizon reached")

        counts = self.center_counts()
        if verbose:
            print("Final Centers:", {p.value: c for p, c in counts.items()})
        return self.winner() or max(counts, key=counts.get)

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
