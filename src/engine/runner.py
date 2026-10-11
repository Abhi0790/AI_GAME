import random
import hashlib
import zlib
from contextlib import contextmanager
from typing import List, Dict, Optional

from src.common.schemas import (
    GameState, Order, OrderType, Player, Commitment, CommitmentType, Message, MessageType,
    CalibrationPoint, ReversalPoint, TurnRecord,
)
from src.common.schemas import (
    message_to_commitment, commitment_key, invalid_proposal_reason, leaders,
    obligated_parties, mirrors, exchange_leg_due, carried_out,
)
from src.engine import adjudicator
from src.engine.adjudicator import resolve, verify_commitments
from src.engine.orders import _SETS_CACHE, generate_orders_for_unit
from src.engine.board import (
    get_all_territories, get_supply_centers, home_centers, win_centers, max_turns,
    players, active,
)


# Everything the turn loop calls on every agent, unconditionally. Anything
# optional stays behind a hasattr check in step() instead of going in here.
AGENT_METHODS = ("propose", "reply", "act", "update_beliefs_from_outcomes",
                 "beliefs", "predict_keep")

# Chance per turn that a private deal becomes known to each seat outside it.
PRIVATE_LEAK_RATE = 0.0

# Chance that a unit's order is lost and it holds instead. Commitments are
# graded on what was executed, so a slip can look exactly like a betrayal.
ORDER_NOISE = 0.0


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
        # (worth at signature, worth of breaking at order time) per agent and turn.
        self.reversals: List[ReversalPoint] = []
        # Adjudications spent per turn, per player — the x-axis of the
        # search-variant figure.
        self.nodes_per_turn: List[Dict[Player, int]] = []
        # Initial state: one unit on each of a seat's home centres.
        units = active().starting_units()

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
        # Turns in a row without a centre changing hands, and who shared a draw.
        self._stalled = 0
        self.drawn: Optional[List[Player]] = None

    @contextmanager
    def _event_rng(self, key):
        cfg = getattr(self, "config", None)
        if cfg is None or not cfg.event_keyed_rng:
            yield
            return
        old = random.getstate()
        material = f"{cfg.seed}:{key!r}".encode("utf-8")
        random.seed(int.from_bytes(hashlib.sha256(material).digest()[:16], "big"))
        try:
            yield
        finally:
            random.setstate(old)

    def _charge(self, player: Player, fn, *args, event_key=None):
        """Call an agent method, charging the adjudications it causes to that seat."""
        before = adjudicator.resolve_calls
        try:
            with self._event_rng(event_key):
                return fn(*args)
        finally:
            self._work[player] += adjudicator.resolve_calls - before

    # ── negotiation ─────────────────────────────────────────────────────
    def begin_turn(self) -> List[Message]:
        """This turn's opening proposals, generated once and reused by `step()`."""
        if self._opening is None:
            self._opening = [m for p, a in self.agents.items()
                             for m in self._charge(p, a.propose, self.state, self.visible_to(p),
                                                   event_key=("propose", self.state.turn, p.value))]
        return self._opening

    def _negotiate(self, log_lines: List[str]) -> List[Message]:
        """Run the negotiation rounds and return every sentence spoken."""
        new_messages = self.begin_turn()
        self._opening = None
        spoken: List[Message] = list(new_messages)

        for round_index in range(self.negotiation_rounds):
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
                replies.extend(self._charge(
                    p, a.reply, self.state, inbox[p], self.visible_to(p),
                    event_key=("reply", self.state.turn, round_index, p.value)))

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
                    # A pact binds nobody until every member has said yes.
                    signed = self._pact_signatures.setdefault(orig.id, {orig.sender})
                    signed.add(rep.sender)
                    if not set(orig.coalition) <= signed:
                        continue

                c = message_to_commitment(orig, rep.sender, self.state.turn)
                mirror = next((e for e in self.commitments if mirrors(e, c)), None)
                if mirror:
                    log_lines.append(
                        f"Deal dropped: {c.players[0].value} supporting "
                        f"{c.players[1].value} into {c.target_territory} mirrors a "
                        f"live deal the other way; both would support and neither move")
                    continue
                stuck = next((q for q in c.players if not self._keepable(q, c)), None)
                if stuck:
                    # No order set could keep it alongside the player's live promises.
                    log_lines.append(
                        f"Deal dropped: {stuck.value} cannot keep a "
                        f"{c.commitment_type.value} deal alongside its live promises")
                    continue
                existing = next(
                    (e for e in self.commitments
                     if commitment_key(e) == commitment_key(c)), None)
                if existing:
                    # Same deal proposed again: renew it, do not stack
                    # a second copy that the planner would price twice.
                    existing.valid_until_turn = max(
                        existing.valid_until_turn, c.valid_until_turn)
                    # Renewing an Exchange renews its repayment date.
                    if c.repay_turn is not None:
                        existing.repay_turn = c.repay_turn
                        existing.valid_until_turn = max(
                            existing.valid_until_turn, c.repay_turn)
                    # Once public, a renewed deal stays public.
                    existing.private = existing.private and c.private
                    self._notify_signed(existing)
                else:
                    self.commitments.append(c)
                    self._notify_signed(c)
                    # Players sorted, so a pair always prints the same way.
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

    def _announce(self) -> List[Message]:
        """Once the deals are struck, each seat may state an intended move;
        everyone hears it through `reply`, which answers nothing."""
        said = [m for p, a in self.agents.items() if hasattr(a, "announce")
                for m in self._charge(p, a.announce, self.state, self.visible_to(p),
                                      event_key=("announce", self.state.turn, p.value))]
        if said:
            for p, a in self.agents.items():
                heard = [m for m in said if m.sender != p and m.receiver in (None, p)]
                if heard:
                    self._charge(p, a.reply, self.state, heard, self.visible_to(p),
                                 event_key=("hear", self.state.turn, p.value))
        return said

    def _keepable(self, player: Player, c: Commitment) -> bool:
        """Whether some order set keeps c and every live promise of player's (unless
        those were unkeepable already), by matching required supports to units."""
        live = [e for e in self.commitments
                if player in e.players and commitment_key(e) != commitment_key(c)]

        def ok(deals):
            required = set()
            for deal in deals:
                if deal.commitment_type == CommitmentType.SUPPORT and deal.players[0] == player:
                    required.add((deal.supported_from, deal.target_territory))
                elif deal.commitment_type == CommitmentType.EXCHANGE:
                    if exchange_leg_due(deal, self.state.turn) == "give" and deal.players[0] == player:
                        required.add((deal.supported_from, deal.target_territory))
            if not required:
                return True

            capable = {}
            for u in self.state.units:
                if u.player != player:
                    continue
                for order in generate_orders_for_unit(self.state, u):
                    if order.order_type == OrderType.SUPPORT:
                        key = (order.supported_from, order.target)
                        if key in required:
                            capable.setdefault(key, []).append(u.territory)

            assigned = {}

            def assign(need, seen):
                for territory in capable.get(need, []):
                    if territory in seen:
                        continue
                    seen.add(territory)
                    if territory not in assigned or assign(assigned[territory], seen):
                        assigned[territory] = need
                        return True
                return False

            return all(assign(need, set()) for need in sorted(required, key=str))
        # Blame the new deal only: promises the board has already made
        # unkeepable (a supporter dislodged) must not freeze out every deal.
        return ok(live + [c]) or not ok(live)

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
                with self._event_rng(("leak", self.state.turn, commitment_key(c), p.value)):
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
                        r, i, pred = self._charge(
                            observer, agent.predict_keep_parts, self.state, c, subject,
                            event_key=("predict", self.state.turn, commitment_key(c),
                                       observer.value, subject.value))
                    else:
                        r, i, pred = 0.5, 0.0, self._charge(
                            observer, agent.predict_keep, self.state, c, subject,
                            event_key=("predict", self.state.turn, commitment_key(c),
                                       observer.value, subject.value))
                    points.append(CalibrationPoint(
                        turn=self.state.turn, observer=observer, subject=subject,
                        commitment_type=c.commitment_type, commitment_id=c.id,
                        predicted=pred, reliability=r, incentive=i,
                    ))
        return points

    @staticmethod
    def _close_predictions(points: List[CalibrationPoint], outcomes: List):
        # Keyed on the deal, so two live deals with one partner are graded apart.
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

        # 2. Negotiation rounds (their log lines go at the front of log.events).
        negotiation_lines: List[str] = []
        self._regime_change(negotiation_lines)
        messages = self._negotiate(negotiation_lines)
        messages += self._announce()

        # 3. What every agent predicts about the deals now on the table
        predictions = self._open_predictions()

        self._spread_secrets()

        # 4. Agents choose orders
        all_orders: List[Order] = []
        traces: Dict[Player, object] = {}
        nodes: Dict[Player, int] = {}
        for p, a in self.agents.items():
            orders, trace = self._charge(
                p, a.act, self.state, self.visible_to(p),
                event_key=("act", self.state.turn, p.value))
            all_orders.extend(orders)
            traces[p] = trace
            nodes[p] = getattr(getattr(a, "planner", None), "nodes", 0)
            self.reversals.extend(self._reversals(p, a, trace))

        intended = all_orders
        all_orders, slipped = self._slip(all_orders)

        # 5. Engine resolves and grades commitments
        new_state, outcomes, log = resolve(self.state, all_orders, self.commitments,
                                           chooser=self._choose)
        self._annotate(outcomes, intended if slipped else None, predictions)
        for line in negotiation_lines:
            log.events.insert(0, line)
        for o in slipped:
            log.events.append(f"Order lost: {o.unit_territory} held instead of "
                              f"{o.order_type.value} {o.target or ''}".rstrip())
        if self.on_turn_resolved:
            self.on_turn_resolved(self.state, all_orders, log)

        for o in outcomes:
            if not o.kept:
                say(f"Commitment BROKEN by {o.broken_by}: {o.commitment.commitment_type}")

        self.calibration.extend(self._close_predictions(predictions, outcomes))

        # What was on the table during this turn, before broken deals dissolve.
        live_this_turn = list(self.commitments)

        # A broken deal ends for every party.
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

        # Stated intentions, graded on what was executed, like promises.
        intents = [(m, carried_out(m, all_orders)) for m in messages
                   if m.message_type == MessageType.INTENT]
        for m, done in intents:
            log.events.append(f"Intent {'carried out' if done else 'not carried out'}: "
                              f"{m.sender.value} {m.supported_from}->{m.target_territory}")

        # 7. Beliefs and opponent models update on what they witnessed
        for p, a in self.agents.items():
            if hasattr(a, "observe_intents"):
                a.observe_intents([(m, done) for m, done in intents
                                   if m.sender != p and m.receiver in (None, p)])
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
            intended=intended, slipped=slipped, intents=intents,
        )
        self._work = {p: 0 for p in self.agents}
        self.history.append(record)
        self.nodes_per_turn.append(nodes)
        self._stalled = (self._stalled + 1
                         if new_state.supply_centers == self.state.supply_centers else 0)
        self.state = new_state
        self._vote_draw(log)
        return record

    def survivors(self) -> List[Player]:
        """Seats with a unit or a centre left."""
        return [p for p in self.agents
                if any(u.player == p for u in self.state.units)
                or p in self.state.supply_centers.values()]

    def _vote_draw(self, log) -> None:
        """With draws on, every survivor votes; a unanimous yes ends the game
        shared among them (DIAS). Agents without `vote_draw` vote no."""
        if not getattr(getattr(self, "config", None), "draws", False) or self.winner():
            return
        alive = self.survivors()
        yes = [p for p in alive if hasattr(self.agents[p], "vote_draw") and self._charge(
            p, self.agents[p].vote_draw, self.state, self._stalled,
            event_key=("draw", self.state.turn, p.value))]
        if yes and len(yes) == len(alive):
            self.drawn = sorted(alive, key=lambda p: p.value)
            log.events.append(f"Draw agreed by {', '.join(p.value for p in self.drawn)}")
        elif yes:
            log.events.append(f"Draw proposed by {', '.join(sorted(p.value for p in yes))}; "
                              f"refused by {', '.join(sorted(p.value for p in alive if p not in yes))}")

    def result(self) -> dict:
        """Who won outright, or who shared the game and whether by a draw."""
        champion = self.winner()
        return {"winner": champion.value if champion else None,
                "drawn": bool(self.drawn),
                "leaders": sorted(p.value for p in self.leaders())}

    def over(self) -> bool:
        """Won outright, drawn, or past the turn limit."""
        return bool(self.winner() or self.drawn or self.state.turn > self.max_turns)

    def _choose(self, player: Player, kind: str, options: List[str], count: int,
                state: GameState, turn: int) -> List[str]:
        """Ask the seat for its retreat, build or disband; agents without
        `choose` get the engine's default."""
        agent = self.agents[player]
        if not hasattr(agent, "choose"):
            return adjudicator.default_choice(player, kind, options, count, state, turn)
        return self._charge(player, agent.choose, kind, options, count, state,
                            event_key=(kind, turn, player.value))

    def _regime_change(self, log_lines: List[str]):
        """Swap a seat's persona on the turn GameConfig.regime_change names."""
        rc = getattr(getattr(self, "config", None), "regime_change", None)
        if rc and rc["turn"] == self.state.turn:
            self.agents[Player(rc["seat"])].set_persona(rc["persona"], self.config.rep_cost_scale)
            log_lines.append(f"Change of leadership: {rc['seat']} is now {rc['persona']}")

    def _annotate(self, outcomes, intended: Optional[List[Order]], predictions):
        """Mark each outcome's accidental breakers (whose intended orders kept
        the deal) and the parties its members judged tempted."""
        if intended is not None:
            meant = {m.commitment.id: set(m.broken_by)
                     for m in verify_commitments(self.state, self.commitments, intended)}
            for o in outcomes:
                o.accidental = [p for p in o.broken_by if p not in meant[o.commitment.id]]
        tempted = {(pt.commitment_id, pt.subject) for pt in predictions if pt.incentive > 0}
        for o in outcomes:
            o.tempted = [p for p in o.commitment.players if (o.commitment.id, p) in tempted]

    def _slip(self, orders: List[Order]):
        """Replace each non-hold order with a hold at rate ORDER_NOISE, drawn from
        crc32 so the game's random stream is untouched."""
        if ORDER_NOISE <= 0:
            return orders, []
        seed = getattr(getattr(self, "config", None), "seed", 0)
        executed, slipped = [], []
        for o in orders:
            draw = zlib.crc32(f"{seed}|{self.state.turn}|{o.unit_territory}".encode()) / 2**32
            if o.order_type != OrderType.HOLD and draw < ORDER_NOISE:
                slipped.append(o)
                o = Order(player=o.player, unit_territory=o.unit_territory,
                          order_type=OrderType.HOLD)
            executed.append(o)
        return executed, slipped

    def _reversals(self, player: Player, agent, trace) -> List[ReversalPoint]:
        """Pair each deal this agent priced and signed this turn with its planner's
        break advantage."""
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
        """Who shares the result: the draw's survivors, else every seat tied
        for the most centres."""
        return list(self.drawn) if self.drawn else leaders(self.center_counts())

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
            if self.drawn:
                if verbose:
                    print(f"Game Over: draw among {', '.join(p.value for p in self.drawn)}")
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
