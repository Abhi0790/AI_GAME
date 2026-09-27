from typing import List, Dict
import uuid

from src.common.schemas import (
    GameState, Order, Player, Commitment, Message, MessageType,
    CommitmentType, Unit,
)
from src.common.schemas import message_to_commitment, commitment_key
from src.engine.adjudicator import resolve, verify_commitments
from src.engine.board import (
    get_all_territories, get_supply_centers, is_supply_center,
    HOME_CENTERS, WIN_CENTERS, MAX_TURNS, TERRITORIES,
)


class GameRunner:
    def __init__(self, agents: List, on_turn_resolved=None):
        self.agents = {a.player: a for a in agents}
        self.commitments: List[Commitment] = []
        self.history = []
        
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
        
    def step(self, verbose: bool = False):
        """Run one full turn: negotiate, plan, resolve, gossip, update beliefs.

        Returns the history step (state, orders, outcomes, log, traces).
        Single source of truth for a turn — the CLI and the web UI both call it.
        """
        t = self.state.turn
        say = print if verbose else (lambda *a, **k: None)
        say(f"--- Turn {t} ---")

        # 1. Drop expired commitments
        self.commitments = [c for c in self.commitments if c.valid_until_turn >= t]

        # 2. Negotiation rounds
        new_messages = []
        for a in self.agents.values():
            new_messages.extend(a.propose(self.state))

        for _ in range(self.negotiation_rounds):
            if not new_messages:
                break

            inbox = {p: [] for p in Player}
            for m in new_messages:
                if m.receiver:
                    inbox[m.receiver].append(m)
                else:
                    for p in Player:
                        if p != m.sender:
                            inbox[p].append(m)

            replies = []
            for p, a in self.agents.items():
                replies.extend(a.reply(self.state, inbox[p]))
                for msg in inbox[p]:
                    if msg.message_type == MessageType.BROADCAST:
                        a.receive_gossip(msg)

            for rep in replies:
                if rep.message_type == MessageType.ACCEPT:
                    orig = next((m for m in new_messages if m.id == rep.reference_id), None)
                    if orig:
                        c = message_to_commitment(orig, rep.sender, self.state.turn)
                        existing = next(
                            (e for e in self.commitments
                             if commitment_key(e) == commitment_key(c)), None)
                        if existing:
                            # Same deal proposed again: renew it, do not stack
                            # a second copy that the planner would price twice.
                            existing.valid_until_turn = max(
                                existing.valid_until_turn, c.valid_until_turn)
                        else:
                            self.commitments.append(c)
                            say(f"Commitment created: {c.commitment_type} between {c.players}")

            new_messages = replies

        # 3. Agents choose orders
        all_orders = []
        traces = {}
        for p, a in self.agents.items():
            orders, trace = a.act(self.state, self.commitments)
            all_orders.extend(orders)
            traces[p] = trace

        # 4. Engine resolves and grades commitments
        new_state, outcomes, log = resolve(self.state, all_orders, self.commitments)
        if self.on_turn_resolved:
            self.on_turn_resolved(self.state, all_orders, log)

        for o in outcomes:
            if not o.kept:
                say(f"Commitment BROKEN by {o.broken_by}: {o.commitment.commitment_type}")

        # 5. Victims broadcast BETRAYED so the gossip rules fire elsewhere
        for o in outcomes:
            if o.kept:
                continue
            for victim in o.commitment.players:
                if victim in o.broken_by:
                    continue
                for betrayer in o.broken_by:
                    gossip_msg = Message(
                        id=str(uuid.uuid4()),
                        sender=victim,
                        receiver=None,
                        message_type=MessageType.BROADCAST,
                        broadcast_kind="BETRAYED",
                        broadcast_target=betrayer,
                        commitment_type=o.commitment.commitment_type,
                    )
                    for p, a in self.agents.items():
                        if p != victim:
                            a.receive_gossip(gossip_msg)
                    say(f"[GOSSIP] {victim.value} broadcasts: "
                        f"{betrayer.value} broke {o.commitment.commitment_type.value}")

        for l in log.events:
            say(l)

        # 6. Beliefs and opponent models update on what the engine published
        for a in self.agents.values():
            a.update_beliefs_from_outcomes(self.state, new_state, outcomes)
            if hasattr(a, 'observe_orders'):
                a.observe_orders(all_orders)

        step_data = (self.state, all_orders, outcomes, log, traces)
        self.history.append(step_data)
        self.state = new_state
        return step_data

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

    def run(self):
        while self.state.turn <= self.max_turns:
            self.step(verbose=True)
            champion = self.winner()
            if champion:
                print(f"Game Over: {champion.value} reached {WIN_CENTERS} centres "
                      f"on turn {self.state.turn - 1}")
                break
        else:
            print("Game Over: horizon reached")

        counts = self.center_counts()
        print("Final Centers:", {p.value: c for p, c in counts.items()})
        return self.winner() or max(counts, key=counts.get)
