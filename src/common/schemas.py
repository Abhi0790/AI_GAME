from enum import Enum
from typing import List, Dict, Optional, Tuple, Any, Set, NamedTuple
from pydantic import BaseModel, Field

class Player(str, Enum):
    RED = "Red"
    BLUE = "Blue"
    GREEN = "Green"
    GOLD = "Gold"

class UnitType(str, Enum):
    ARMY = "Army"

class Unit(BaseModel):
    player: Player
    unit_type: UnitType = UnitType.ARMY
    territory: str

class OrderType(str, Enum):
    HOLD = "Hold"
    MOVE = "Move"
    SUPPORT = "Support"

class Order(BaseModel):
    player: Player
    unit_territory: str
    order_type: OrderType
    target: Optional[str] = None # For move and support
    supported_from: Optional[str] = None # For support, the origin of the unit being supported
    
    def __hash__(self):
        return hash((self.player, self.unit_territory, self.order_type, self.target, self.supported_from))
    
    def __eq__(self, other):
        if not isinstance(other, Order): return False
        return (self.player, self.unit_territory, self.order_type, self.target, self.supported_from) == \
               (other.player, other.unit_territory, other.order_type, other.target, other.supported_from)

class GameState(BaseModel):
    turn: int
    units: List[Unit]
    supply_centers: Dict[str, Optional[Player]] # territory -> owner
    territory_owners: Dict[str, Optional[Player]]
    
    def get_unit_at(self, territory: str) -> Optional[Unit]:
        for u in self.units:
            if u.territory == territory:
                return u
        return None

class CommitmentType(str, Enum):
    ALLIANCE = "Alliance"
    SUPPORT = "Support"
    DMZ = "DMZ"

class Commitment(BaseModel):
    id: str
    commitment_type: CommitmentType
    players: List[Player] # Players involved
    valid_until_turn: int
    
    # Specifics depending on type
    target_territory: Optional[str] = None # For Support
    supported_from: Optional[str] = None # For Support
    dmz_territories: Optional[List[str]] = None # For DMZ

class MessageType(str, Enum):
    PROPOSE = "Propose"
    ACCEPT = "Accept"
    REJECT = "Reject"
    COUNTER = "Counter"
    THREAT = "Threat"
    BROADCAST = "Broadcast"

class Message(BaseModel):
    id: str
    sender: Player
    receiver: Optional[Player] = None # None for broadcast
    message_type: MessageType
    
    # Payload
    commitment_type: Optional[CommitmentType] = None
    turns: Optional[int] = None # For alliance
    target_territory: Optional[str] = None
    supported_from: Optional[str] = None
    dmz_territories: Optional[List[str]] = None
    
    reference_id: Optional[str] = None # For Accept, Reject, Counter
    condition: Optional[str] = None # For Threat
    action: Optional[str] = None # For Threat
    broadcast_kind: Optional[str] = None # e.g. "BETRAYED"
    broadcast_target: Optional[Player] = None

    # An accusation is a claim, not a fact. The engine grades it against what
    # it actually adjudicated and stamps a verdict before anyone acts on it,
    # so a liar can be caught and a truthful accuser can be believed.
    engine_verdict: Optional[str] = None # "CONFIRMED" | "REFUTED"
    truthful: Optional[bool] = None      # ground truth, for evaluation only

class BeliefSnapshot(BaseModel):
    observer: Player
    subject: Player
    commitment_type: CommitmentType
    alpha: float
    beta_param: float # Avoid clash with beta function
    expected_reliability: float


# ── Traces & outcomes (shared: every component reads these, none owns them) ──

class CommitmentOutcome:
    """Engine's verdict on one commitment after a turn's orders."""
    def __init__(self, commitment: Commitment, kept: bool, broken_by: List[Player]):
        self.commitment = commitment
        self.kept = kept
        self.broken_by = broken_by

class DecisionTrace:
    """Planner's one-sentence account of why an order set was chosen."""
    def __init__(self, candidate_orders: List[Order], expected_value: float, penalty: float,
                 explanation: str, commitment_broken: bool = False,
                 penalty_rows: Optional[List[tuple]] = None, nodes: int = 0,
                 vengeance: float = 0.0, search: str = "expectiminimax",
                 candidates: int = 0, pruned: int = 0):
        self.candidate_orders = candidate_orders
        self.expected_value = expected_value
        self.penalty = penalty
        self.explanation = explanation
        self.commitment_broken = commitment_broken
        # (partner, vcoop, delta_p, horizon, amount) per broken deal. The
        # Vcoop-at-break figure reads this; without it the number is computed
        # and thrown away every turn.
        self.penalty_rows = penalty_rows or []
        self.nodes = nodes            # adjudications spent on this decision
        self.vengeance = vengeance    # value paid purely to punish a betrayer
        self.search = search
        self.candidates = candidates  # order sets after pruning
        self.pruned = pruned          # order sets discarded by dominance

class TrustTrace:
    """Trust model's account of which rule fired and on what inputs."""
    def __init__(self, rule_name: str, facts: str, old_alpha: float, old_beta: float, new_alpha: float, new_beta: float, explanation: str):
        self.rule_name = rule_name
        self.facts = facts
        self.old_alpha = old_alpha
        self.old_beta = old_beta
        self.new_alpha = new_alpha
        self.new_beta = new_beta
        self.explanation = explanation


def commitment_key(c: "Commitment"):
    """Identity of a deal, ignoring who asked. Red-Blue and Blue-Red are the
    same alliance, and both sides propose it every turn, so without this the
    same promise is counted (and priced) several times over."""
    return (
        c.commitment_type,
        frozenset(c.players),
        c.target_territory,
        c.supported_from,
        tuple(sorted(c.dmz_territories or [])),
    )


def invalid_proposal_reason(msg: Message, max_turns: int = 12) -> Optional[str]:
    """Why this message is not a sentence of the grammar, or None if it is.

    Every accepted deal goes through here before it becomes a commitment the
    engine will grade. A malformed or hostile client can otherwise get an
    unenforceable commitment onto the table — a DMZ over territories that do
    not exist, an alliance lasting 999 turns, a SUPPORT with no type at all —
    and the planner will dutifully price it.
    """
    from src.engine.board import get_all_territories, is_adjacent

    if msg.message_type not in (MessageType.PROPOSE, MessageType.COUNTER):
        return "not a proposal"
    if msg.commitment_type is None:
        return "no commitment type"
    if msg.turns is not None and not (1 <= msg.turns <= max_turns):
        return f"duration {msg.turns} out of range"

    territories = set(get_all_territories())
    if msg.commitment_type == CommitmentType.DMZ:
        if not msg.dmz_territories:
            return "DMZ over no territories"
        unknown = [t for t in msg.dmz_territories if t not in territories]
        if unknown:
            return f"DMZ over unknown territories {unknown}"
    if msg.commitment_type == CommitmentType.SUPPORT:
        if msg.target_territory not in territories:
            return f"support of unknown territory {msg.target_territory!r}"
        if msg.supported_from is not None:
            if msg.supported_from not in territories:
                return f"support from unknown territory {msg.supported_from!r}"
            if not is_adjacent(msg.supported_from, msg.target_territory):
                return "supported move is not between adjacent territories"
    if msg.receiver == msg.sender:
        return "proposal addressed to its own sender"
    return None


def message_to_commitment(msg: Message, accepted_by: Player, current_turn: int) -> Commitment:
    """Translate an accepted proposal into an engine-checkable commitment."""
    return Commitment(
        id=msg.id,
        commitment_type=msg.commitment_type,
        players=[msg.sender, accepted_by],
        valid_until_turn=current_turn + (msg.turns if msg.turns else 1),
        target_territory=msg.target_territory,
        supported_from=msg.supported_from,
        dmz_territories=msg.dmz_territories,
    )


# ── Negotiation grammar helpers ──────────────────────────────────────────

def is_support_hold(msg_or_commitment) -> bool:
    """A SUPPORT deal with no origin is a promise to support a *hold*.

    The grammar distinguishes "support B's move into X" (supported_from set)
    from "support B holding X" (supported_from empty); the engine already
    grades both, only the proposer never generated the second one.
    """
    return (
        getattr(msg_or_commitment, "commitment_type", None) == CommitmentType.SUPPORT
        and not getattr(msg_or_commitment, "supported_from", None)
    )


def describe_message(m: "Message") -> str:
    """One line of the closed grammar, for the log and the UI."""
    who = f"{m.sender.value}->{m.receiver.value}" if m.receiver else f"{m.sender.value}->all"
    t = m.message_type.value.upper()
    if m.message_type in (MessageType.PROPOSE, MessageType.COUNTER):
        body = m.commitment_type.value if m.commitment_type else "?"
        if m.commitment_type == CommitmentType.DMZ:
            body += f"({','.join(m.dmz_territories or [])})"
        elif m.commitment_type == CommitmentType.SUPPORT:
            body += (f"(hold {m.target_territory})" if is_support_hold(m)
                     else f"({m.supported_from}->{m.target_territory})")
        return f"{who} {t} {body} for {m.turns or 1}t"
    if m.message_type == MessageType.THREAT:
        return f"{who} THREAT if {m.condition} then {m.action}"
    if m.message_type == MessageType.BROADCAST:
        target = m.broadcast_target.value if m.broadcast_target else "?"
        verdict = f" [{m.engine_verdict}]" if m.engine_verdict else ""
        return f"{who} BROADCAST {m.broadcast_kind} {target}{verdict}"
    return f"{who} {t}"


class TurnRecord(NamedTuple):
    """One turn of history.

    Was a bare 5-tuple; messages and beliefs were produced every turn and
    thrown away, which is why the UI had no negotiation panel and a dead
    trust tab. Named so new fields can be added without breaking unpacking.
    """
    state: GameState
    orders: List[Order]
    outcomes: List[CommitmentOutcome]
    log: Any
    traces: Dict[Player, Any]
    # Empty tuples, not [] and {}: a NamedTuple's defaults are shared by every
    # instance that omits them, and a mutable one would be appended to in place.
    messages: List[Message] = ()
    beliefs: List[BeliefSnapshot] = ()
    commitments: List[Commitment] = ()
    nodes: Dict[Player, int] = ()


class CalibrationPoint(BaseModel):
    """(what the trust model predicted, what the engine graded).

    Recorded when a commitment is formed, closed when it is graded. Without
    these pairs there is no way to tell whether P(keeps) means anything.
    """
    turn: int
    observer: Player
    subject: Player
    commitment_type: CommitmentType
    commitment_id: str = ""
    predicted: float
    observed: Optional[bool] = None
    # The two network inputs, kept so the CPT can be refitted on logged data
    # instead of guessed at (issue #16).
    reliability: float = 0.5
    incentive: float = 0.0
