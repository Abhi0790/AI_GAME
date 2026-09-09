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
