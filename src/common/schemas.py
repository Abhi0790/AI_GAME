from enum import Enum
from typing import List, Dict, Optional, Tuple, Any, Set
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
    def __init__(self, candidate_orders: List[Order], expected_value: float, penalty: float, explanation: str, commitment_broken: bool = False):
        self.candidate_orders = candidate_orders
        self.expected_value = expected_value
        self.penalty = penalty
        self.explanation = explanation
        self.commitment_broken = commitment_broken

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
