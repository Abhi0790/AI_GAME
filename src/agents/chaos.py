"""A seat that behaves badly on purpose.

The robustness requirement is not "the adjudicator ignores two junk orders",
it is "a real game survives a player who does everything wrong at once". This
agent takes a seat in GameRunner exactly like any other and, every turn,
picks one of the ways a participant can misbehave:

  * orders for units it does not own, or for territories that do not exist
  * two contradictory orders for the same unit in the same turn
  * moves to non-adjacent territories, supports for nobody
  * silence — no orders at all, no replies at all
  * accepting a proposal that was never made
  * flooding the table with malformed, out-of-grammar messages

Nothing here is random noise for its own sake: each mode is a thing a buggy
or hostile client could actually send.
"""

from typing import List, Dict, Optional, Tuple
import random
import uuid

from src.common.schemas import (
    Player, GameState, Order, OrderType, Commitment, Message, MessageType,
    CommitmentType, DecisionTrace, BeliefSnapshot,
)
from src.engine.board import get_all_territories, get_adjacent, players

MODES = ["garbage_orders", "contradictory", "silent", "phantom_accept",
         "spam_proposals", "out_of_grammar"]


class ChaosAgent:
    """Deliberately malformed input, from inside a real game."""

    def __init__(self, player: Player, mode: Optional[str] = None, seed: Optional[int] = None):
        self.player = player
        self.mode = mode                      # None = a different mode each turn
        self.persona_name = "Chaos"
        self.rng = random.Random(seed)
        self.planner = None
        self.trust_traces: List = []
        self.grudges: Dict[Player, float] = {}

    def _mode_for_turn(self, turn: int) -> str:
        return self.mode or MODES[turn % len(MODES)]

    # ── the normal agent interface, abused ──────────────────────────────
    def propose(self, state: GameState, commitments=None) -> List[Message]:
        mode = self._mode_for_turn(state.turn)
        if mode == "silent":
            return []
        if mode == "spam_proposals":
            return [self._junk_proposal(state) for _ in range(12)]
        if mode == "out_of_grammar":
            # A PROPOSE with no commitment type, a DMZ over nothing, a support
            # of a unit that does not exist: all legal Message objects, none
            # of them a sentence the grammar defines.
            return [
                Message(id=str(uuid.uuid4()), sender=self.player,
                        receiver=self._other(), message_type=MessageType.PROPOSE),
                Message(id=str(uuid.uuid4()), sender=self.player,
                        receiver=self._other(), message_type=MessageType.PROPOSE,
                        commitment_type=CommitmentType.DMZ, dmz_territories=[]),
                Message(id=str(uuid.uuid4()), sender=self.player,
                        receiver=self._other(), message_type=MessageType.PROPOSE,
                        commitment_type=CommitmentType.SUPPORT,
                        target_territory="NOWHERE", supported_from="ALSO_NOWHERE"),
            ]
        return [self._junk_proposal(state)]

    def reply(self, state: GameState, incoming: List[Message], commitments=None) -> List[Message]:
        mode = self._mode_for_turn(state.turn)
        if mode == "silent":
            return []
        if mode == "phantom_accept":
            # Accept something nobody proposed, and accept a real proposal
            # twice for good measure.
            out = [Message(id=str(uuid.uuid4()), sender=self.player,
                           receiver=self._other(), message_type=MessageType.ACCEPT,
                           reference_id=str(uuid.uuid4()))]
            for m in incoming[:1]:
                out += [Message(id=str(uuid.uuid4()), sender=self.player,
                                receiver=m.sender, message_type=MessageType.ACCEPT,
                                reference_id=m.id) for _ in range(2)]
            return out
        return [
            Message(id=str(uuid.uuid4()), sender=self.player, receiver=m.sender,
                    message_type=self.rng.choice(
                        [MessageType.ACCEPT, MessageType.REJECT, MessageType.COUNTER]),
                    reference_id=m.id)
            for m in incoming
        ]

    def act(self, state: GameState, active_commitments: List[Commitment]
            ) -> Tuple[List[Order], DecisionTrace]:
        mode = self._mode_for_turn(state.turn)
        mine = [u.territory for u in state.units if u.player == self.player]
        territories = get_all_territories()
        orders: List[Order] = []

        if mode == "silent":
            pass
        elif mode == "contradictory" and mine:
            # Same unit, two incompatible orders, in the same turn.
            t = mine[0]
            far = [x for x in territories if x != t and x not in get_adjacent(t)]
            orders = [
                Order(player=self.player, unit_territory=t, order_type=OrderType.HOLD),
                Order(player=self.player, unit_territory=t, order_type=OrderType.MOVE,
                      target=self.rng.choice(get_adjacent(t) or territories)),
                Order(player=self.player, unit_territory=t, order_type=OrderType.MOVE,
                      target=self.rng.choice(far or territories)),
            ]
        else:
            orders = [
                # A unit that does not exist, on a territory that does not exist
                Order(player=self.player, unit_territory="VOID",
                      order_type=OrderType.MOVE, target="ALSO_VOID"),
                # Somebody else's unit
                Order(player=self.player,
                      unit_territory=self.rng.choice(territories),
                      order_type=OrderType.MOVE, target=self.rng.choice(territories)),
                # A support for nothing
                Order(player=self.player,
                      unit_territory=mine[0] if mine else "VOID",
                      order_type=OrderType.SUPPORT, target=None),
            ]
            if mine:
                # ... and one long-distance teleport
                orders.append(Order(player=self.player, unit_territory=mine[0],
                                    order_type=OrderType.MOVE,
                                    target=self.rng.choice(territories)))

        return orders, DecisionTrace(orders, 0.0, 0.0, f"Chaos seat: {mode}")

    # ── everything downstream still has to work ─────────────────────────
    def update_beliefs_from_outcomes(self, prev_state, new_state, outcomes):
        pass

    def observe_orders(self, orders):
        pass

    def predict_keep(self, state, commitment, subject) -> float:
        return 0.5

    def beliefs(self) -> List[BeliefSnapshot]:
        return []

    def stance(self) -> Dict[Player, float]:
        return {}

    def decay_stance(self):
        pass

    # ── helpers ─────────────────────────────────────────────────────────
    def _other(self) -> Player:
        return self.rng.choice([p for p in players() if p != self.player])

    def _junk_proposal(self, state: GameState) -> Message:
        return Message(
            id=str(uuid.uuid4()), sender=self.player, receiver=self._other(),
            message_type=MessageType.PROPOSE,
            commitment_type=self.rng.choice(list(CommitmentType)),
            turns=self.rng.choice([-5, 0, 999]),
            target_territory=self.rng.choice(get_all_territories() + ["NOWHERE"]),
            dmz_territories=["NOWHERE", "VOID"],
        )
