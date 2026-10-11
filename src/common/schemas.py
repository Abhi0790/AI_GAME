from enum import Enum
from typing import List, Dict, Optional, Tuple, Any, NamedTuple
from pydantic import BaseModel

class Player(str, Enum):
    """Colours a seat can take; iterate board.players() for the seats in play."""
    RED = "Red"
    BLUE = "Blue"
    GREEN = "Green"
    GOLD = "Gold"
    PURPLE = "Purple"
    ORANGE = "Orange"
    TEAL = "Teal"
    PINK = "Pink"

class UnitType(str, Enum):
    ARMY = "Army"
    FLEET = "Fleet"

class Unit(BaseModel):
    player: Player
    unit_type: UnitType = UnitType.ARMY
    territory: str
    coast: Optional[str] = None  # a fleet's coast in a province with two

    @property
    def location(self) -> str:
        return f"{self.territory}/{self.coast}" if self.coast else self.territory

class OrderType(str, Enum):
    HOLD = "Hold"
    MOVE = "Move"
    SUPPORT = "Support"
    # A fleet at sea carries the army at supported_from to target.
    CONVOY = "Convoy"

class Order(BaseModel):
    player: Player
    unit_territory: str
    order_type: OrderType
    target: Optional[str] = None # For move, support and convoy
    supported_from: Optional[str] = None # For support and convoy, the origin of the unit being helped
    coast: Optional[str] = None # For a fleet moving into a province with two coasts

    def _key(self):
        return (self.player, self.unit_territory, self.order_type, self.target,
                self.supported_from, self.coast)

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        return isinstance(other, Order) and self._key() == other._key()

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
    # One side supports now; the other repays later by staying out of somewhere.
    EXCHANGE = "Exchange"

class Commitment(BaseModel):
    """A promise the engine will grade, up to and including `valid_until_turn`."""
    id: str
    commitment_type: CommitmentType
    players: List[Player] # Players involved
    valid_until_turn: int

    # Specifics depending on type
    target_territory: Optional[str] = None # For Support / the Exchange's give leg
    supported_from: Optional[str] = None # For Support / the Exchange's give leg
    dmz_territories: Optional[List[str]] = None # For DMZ / the Exchange's repay leg

    # Graded like any other deal, but the verdict is not published.
    private: bool = False

    # Which turn the Exchange's repayment leg falls due. Before it, only the
    # giver can break the deal; on it, only the payer can.
    repay_turn: Optional[int] = None

class MessageType(str, Enum):
    PROPOSE = "Propose"
    ACCEPT = "Accept"
    REJECT = "Reject"
    COUNTER = "Counter"
    THREAT = "Threat"
    # Cheap talk: "my unit at supported_from moves to target_territory".
    # Binds nobody; graded against the executed orders as candour.
    INTENT = "Intent"

class Message(BaseModel):
    id: str
    sender: Player
    receiver: Optional[Player] = None
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

    # Every member a pact binds, sender included; it forms only when all accept.
    coalition: Optional[List[Player]] = None
    private: bool = False
    repay_turn: Optional[int] = None

    # Who takes players[0] (supporter / giver); None means the sender.
    obligated: Optional[Player] = None

    # The sender's arithmetic, for the UI only.
    rationale: Optional[Dict[str, float]] = None

class BeliefSnapshot(BaseModel):
    observer: Player
    subject: Player
    commitment_type: CommitmentType
    alpha: float
    beta_param: float # Avoid clash with beta function
    expected_reliability: float          # the subject's general reputation
    # ...and the same belief narrowed to "does this player keep promises to
    # *me*", which is the number the observer actually signs on.
    reliability_toward_observer: Optional[float] = None


# ── Traces & outcomes (shared: every component reads these, none owns them) ──

class CommitmentOutcome:
    """Engine's verdict on one commitment after a turn's orders."""
    def __init__(self, commitment: Commitment, kept: bool, broken_by: List[Player],
                 accidental: Optional[List[Player]] = None):
        self.commitment = commitment
        self.kept = kept
        self.broken_by = broken_by
        # Breakers whose intended orders kept the deal: a lost order broke it.
        self.accidental = accidental or []
        # Obligated parties some member judged tempted to break, before orders.
        self.tempted: List[Player] = []

class DecisionTrace:
    """Planner's one-sentence account of why an order set was chosen."""
    def __init__(self, candidate_orders: List[Order], expected_value: float, penalty: float,
                 explanation: str, commitment_broken: bool = False,
                 penalty_rows: Optional[List[tuple]] = None, nodes: int = 0,
                 vengeance: float = 0.0, search: str = "expectimax",
                 candidates: int = 0, pruned: int = 0,
                 reversals: Optional[List[tuple]] = None,
                 forfeit_rows: Optional[List[tuple]] = None,
                 coverage: Optional[Dict[str, Any]] = None):
        self.candidate_orders = candidate_orders
        self.expected_value = expected_value
        self.penalty = penalty
        self.explanation = explanation
        self.commitment_broken = commitment_broken
        # (partner, vcoop, delta_p, horizon, amount) per broken deal.
        self.penalty_rows = penalty_rows or []
        self.nodes = nodes            # adjudications spent on this decision
        self.vengeance = vengeance    # value paid purely to punish a betrayer
        self.search = search
        self.candidates = candidates  # order sets after pruning
        self.pruned = pruned          # order sets discarded by dominance
        self.coverage = coverage or {}  # optional research diagnostics
        # (commitment_key, type, best breaking net - best keeping net) per live deal.
        self.reversals = reversals or []
        # (commitment_key, partner, signed_price, fraction_remaining, amount) per
        # broken deal; `penalty` includes them.
        self.forfeit_rows = forfeit_rows or []

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


def leaders(counts: Dict["Player", int]) -> List["Player"]:
    """Every player tied for the most centres."""
    if not counts:
        return []
    top = max(counts.values())
    return [p for p, c in counts.items() if c == top]


def exchange_leg_due(c: "Commitment", turn: int) -> Optional[str]:
    """Which half of an Exchange is owed this turn: "give" (players[0], at
    signing), "repay" (players[1], on repay_turn) or neither."""
    if c.commitment_type != CommitmentType.EXCHANGE:
        return None
    if c.repay_turn is not None and turn >= c.repay_turn:
        return "repay"
    return "give"


def obligated_parties(c: "Commitment", turn: int) -> List["Player"]:
    """Who owes something on this deal this turn: every member of an alliance or
    DMZ, the supporter of a SUPPORT, one side per EXCHANGE leg."""
    if c.commitment_type == CommitmentType.SUPPORT:
        return c.players[:1]
    if c.commitment_type == CommitmentType.EXCHANGE:
        i = 0 if exchange_leg_due(c, turn) == "give" else 1
        return c.players[i:i + 1] or c.players[:1]
    return list(c.players)


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
    # Privacy is not part of the identity: public and private versions are one promise.


def invalid_proposal_reason(msg: Message, max_turns: Optional[int] = None) -> Optional[str]:
    """Why this message is not a sentence of the grammar, or None if it is."""
    from src.engine.board import get_all_territories, is_adjacent
    from src.engine.board import max_turns as board_max_turns

    if max_turns is None:
        max_turns = board_max_turns()

    if msg.message_type not in (MessageType.PROPOSE, MessageType.COUNTER):
        return "not a proposal"
    if msg.commitment_type is None:
        return "no commitment type"
    # Durations are inclusive (see Commitment): k turns struck on turn t run
    # to t+k-1, so max_turns turns from turn 1 ends exactly on the last turn.
    if msg.turns is not None and not (1 <= msg.turns <= max_turns):
        return f"duration {msg.turns} out of range"

    territories = set(get_all_territories())
    if msg.commitment_type == CommitmentType.DMZ:
        if not msg.dmz_territories:
            return "DMZ over no territories"
        unknown = [t for t in msg.dmz_territories if t not in territories]
        if unknown:
            return f"DMZ over unknown territories {unknown}"
    if msg.commitment_type in (CommitmentType.SUPPORT, CommitmentType.EXCHANGE):
        if msg.target_territory not in territories:
            return f"support of unknown territory {msg.target_territory!r}"
        if msg.supported_from is not None:
            if msg.supported_from not in territories:
                return f"support from unknown territory {msg.supported_from!r}"
            if not is_adjacent(msg.supported_from, msg.target_territory):
                return "supported move is not between adjacent territories"
    if msg.commitment_type == CommitmentType.EXCHANGE:
        # Both legs have to be real, or half the bargain is unenforceable.
        if not msg.dmz_territories:
            return "exchange with nothing owed in return"
        unknown = [t for t in msg.dmz_territories if t not in territories]
        if unknown:
            return f"exchange repayment over unknown territories {unknown}"
        if msg.repay_turn is not None and msg.repay_turn > max_turns:
            return f"repayment falls due on turn {msg.repay_turn}, after the game ends"
    if msg.receiver == msg.sender:
        return "proposal addressed to its own sender"
    if msg.coalition is not None:
        if len(set(msg.coalition)) < 3:
            return "a pact needs at least three members"
        if msg.sender not in msg.coalition:
            return "pact proposed by somebody outside it"
        if msg.receiver is not None and msg.receiver not in msg.coalition:
            return "pact offered to somebody outside it"
    return None


def message_to_commitment(msg: Message, accepted_by: Player, current_turn: int) -> Commitment:
    """Translate an accepted proposal into an engine-checkable commitment."""
    if msg.coalition:
        players = list(dict.fromkeys(msg.coalition))
    else:
        players = [msg.sender, accepted_by]
        # A directional deal names its obligated side, so countering it does
        # not hand the duty to whoever happened to speak last.
        if msg.obligated is not None and msg.obligated in players:
            players = [msg.obligated] + [p for p in players if p != msg.obligated]

    turns = msg.turns if msg.turns else 1
    # Graded on turns t .. t+k-1.
    valid_until = current_turn + turns - 1

    repay_turn = None
    if msg.commitment_type == CommitmentType.EXCHANGE:
        repay_turn = msg.repay_turn if msg.repay_turn is not None else current_turn + 1
        # Both legs must fall inside the deal's life.
        valid_until = max(valid_until, repay_turn)

    return Commitment(
        id=msg.id,
        commitment_type=msg.commitment_type,
        players=players,
        valid_until_turn=valid_until,
        target_territory=msg.target_territory,
        supported_from=msg.supported_from,
        dmz_territories=msg.dmz_territories,
        private=msg.private,
        repay_turn=repay_turn,
    )


# ── Negotiation grammar helpers ──────────────────────────────────────────

def is_support_hold(msg_or_commitment) -> bool:
    """A SUPPORT deal with no origin is a promise to support a hold."""
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
    if m.message_type == MessageType.INTENT:
        return f"{who} INTENT {m.supported_from}->{m.target_territory}"
    return f"{who} {t}"


def carried_out(msg: "Message", orders: List["Order"]) -> bool:
    """Did the sender's executed orders move the announced unit where it said?"""
    return any(o.player == msg.sender and o.unit_territory == msg.supported_from
               and o.order_type == OrderType.MOVE and o.target == msg.target_territory
               for o in orders)


class TurnRecord(NamedTuple):
    """One turn of history."""
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
    nodes: Dict[Player, int] = ()            # adjudications the search budget counted
    total_nodes: Dict[Player, int] = ()      # every adjudication the seat caused, negotiation included
    intended: List[Order] = ()               # orders as submitted, before ORDER_NOISE
    slipped: List[Order] = ()                # submitted orders that were lost and became holds
    intents: List[Tuple["Message", bool]] = ()  # each stated intention, and whether it was carried out


class ReversalPoint(BaseModel):
    """One deal priced twice in a turn by one agent: `signed_gain` at signature,
    `break_advantage` at order time. Both positive is a preference reversal."""
    turn: int
    player: Player
    commitment_type: CommitmentType
    signed_gain: float
    break_advantage: float
    # Which game this row came from, for clustered intervals.
    game: Optional[int] = None

    @property
    def reversed_(self) -> bool:
        return self.signed_gain > 0 and self.break_advantage > 0


class CalibrationPoint(BaseModel):
    """(what the trust model predicted, what the engine graded)."""
    turn: int
    observer: Player
    subject: Player
    commitment_type: CommitmentType
    commitment_id: str = ""
    predicted: float
    observed: Optional[bool] = None
    # Which game this row came from, for clustering.
    game: Optional[int] = None
    # The two network inputs, kept so the CPT can be refitted on logged data.
    reliability: float = 0.5
    incentive: float = 0.0


def mirrors(a: "Commitment", b: "Commitment") -> bool:
    """Two support legs into the same square, each supporter the other's
    beneficiary. Honouring both leaves both units supporting and nobody moving."""
    legs = (CommitmentType.SUPPORT, CommitmentType.EXCHANGE)
    return (a.commitment_type in legs and b.commitment_type in legs
            and len(a.players) == len(b.players) == 2
            and a.supported_from is not None and b.supported_from is not None
            and a.target_territory == b.target_territory
            and a.players[0] == b.players[1] and a.players[1] == b.players[0])
