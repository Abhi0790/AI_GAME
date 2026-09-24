from enum import Enum
from typing import List, Dict, Optional, Tuple, Any, Set, NamedTuple
from pydantic import BaseModel, Field

class Player(str, Enum):
    """The roster of colours a seat can take, not the seats in play.

    Which are seated is a property of the board: iterate
    `src.engine.board.players()`, not this enum.
    """
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
    # A trade across turns: one side gives support now, the other pays it back
    # later by staying out of somewhere. The two legs are deliberately of
    # unequal value and fall due on different turns, which is what makes it a
    # bargain rather than a matching of like for like -- and what creates the
    # obvious defection: take the support, then do not pay.
    EXCHANGE = "Exchange"

class Commitment(BaseModel):
    """A promise the engine will grade.

    `valid_until_turn` is the LAST turn on which the deal is graded,
    inclusive. A deal for k turns struck on turn t has
    `valid_until_turn = t + k - 1`, and is therefore graded exactly k times.
    """
    id: str
    commitment_type: CommitmentType
    players: List[Player] # Players involved
    valid_until_turn: int

    # Specifics depending on type
    target_territory: Optional[str] = None # For Support / the Exchange's give leg
    supported_from: Optional[str] = None # For Support / the Exchange's give leg
    dmz_territories: Optional[List[str]] = None # For DMZ / the Exchange's repay leg

    # A private deal is graded by the engine exactly like any other, but the
    # engine does not *publish* the verdict, so nobody outside it can see the
    # outcome.
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

    # Every member a multi-party pact is meant to bind, sender included. The
    # deal only comes into force once all of them have accepted, so a pact is
    # not a bundle of bilateral promises -- it is one promise that either
    # forms or does not.
    coalition: Optional[List[Player]] = None
    private: bool = False
    repay_turn: Optional[int] = None

    # Who takes players[0] — the supporter on a SUPPORT, the giver on an
    # EXCHANGE. Direction was implied by who spoke, so a COUNTER swapped it and
    # graded the wrong side. None means the sender, which is the usual case.
    obligated: Optional[Player] = None

    # The arithmetic the sender did before saying this, so the UI can show
    # the decision rather than only its verdict. Priced at the moment of
    # speaking; nothing reads it back, so it is free to be incomplete.
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
                 candidates: int = 0, pruned: int = 0,
                 reversals: Optional[List[tuple]] = None,
                 forfeit_rows: Optional[List[tuple]] = None):
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
        # (commitment_key, break advantage) per live deal I am in: how much
        # better my best treacherous order set scores than my best loyal one,
        # *after* the reputation penalty. Positive means that at order time I
        # would rather break this deal than keep it. Paired with what I paid
        # for the deal at signature, it measures whether my negotiator and my
        # planner agree about the same promise in the same turn.
        self.reversals = reversals or []
        # (commitment_key, partner, signed_price, fraction_remaining, amount)
        # per broken deal: the share of the price I credited this deal with at
        # signature that breaking it now forfeits. Signing and breaking are
        # then priced with the same number, so the two cannot disagree on an
        # unchanged board. `penalty` includes the sum of these.
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
    """Every player tied for the most centres.

    Ties are the normal case, not an edge case: with the default threshold
    unreachable, 80% of games end shared. Taking `max()` and moving on hands
    every one of them to whichever seat comes first in `Player`.
    """
    if not counts:
        return []
    top = max(counts.values())
    return [p for p, c in counts.items() if c == top]


def exchange_leg_due(c: "Commitment", turn: int) -> Optional[str]:
    """Which half of an Exchange is owed this turn: "give", "repay" or neither.

    players[0] gives the support on the turn the deal is struck; players[1]
    pays it back on repay_turn by keeping out of the listed territories.
    """
    if c.commitment_type != CommitmentType.EXCHANGE:
        return None
    if c.repay_turn is not None and turn >= c.repay_turn:
        return "repay"
    return "give"


def obligated_parties(c: "Commitment", turn: int) -> List["Player"]:
    """Who actually owes something on this deal, this turn.

    Alliance and DMZ bind every member. A SUPPORT obligates only players[0];
    an EXCHANGE obligates one side per leg. Crediting the other party with a
    kept promise inflates trust and deflates every betrayal denominator.
    """
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
    # Privacy is deliberately NOT part of the identity. A public and a private
    # version of the same promise between the same players are one promise,
    # not two, and treating them as two let the same relationship stack twice
    # and be priced twice. Which of the two wins is handled where deals are
    # renewed: publicity is one-way, because you cannot un-say it.


def invalid_proposal_reason(msg: Message, max_turns: Optional[int] = None) -> Optional[str]:
    """Why this message is not a sentence of the grammar, or None if it is.

    Every accepted deal goes through here before it becomes a commitment the
    engine will grade. A malformed or hostile client can otherwise get an
    unenforceable commitment onto the table — a DMZ over territories that do
    not exist, an alliance lasting 999 turns, a SUPPORT with no type at all —
    and the planner will dutifully price it.
    """
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
    """Translate an accepted proposal into an engine-checkable commitment.

    For a pact the members come from the coalition rather than from whoever
    happened to answer last, so all of them are bound by the same object.
    """
    if msg.coalition:
        players = list(dict.fromkeys(msg.coalition))
    else:
        players = [msg.sender, accepted_by]
        # A directional deal names its obligated side, so countering it does
        # not hand the duty to whoever happened to speak last.
        if msg.obligated is not None and msg.obligated in players:
            players = [msg.obligated] + [p for p in players if p != msg.obligated]

    turns = msg.turns if msg.turns else 1
    # This turn counts as the first of the k, so the deal is graded on turns
    # t .. t+k-1 and the runner's `valid_until_turn >= turn` expiry leaves it
    # graded exactly k times.
    valid_until = current_turn + turns - 1

    repay_turn = None
    if msg.commitment_type == CommitmentType.EXCHANGE:
        repay_turn = msg.repay_turn if msg.repay_turn is not None else current_turn + 1
        # Both legs have to be gradable or it is a gift, not a bargain: a
        # one-turn exchange (which a COUNTER can produce by halving the term)
        # would otherwise expire before the repayment fell due.
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
    nodes: Dict[Player, int] = ()            # adjudications the search budget counted
    total_nodes: Dict[Player, int] = ()      # every adjudication the seat caused, negotiation included


class ReversalPoint(BaseModel):
    """One deal, priced twice in the same turn by the same agent.

    `signed_gain` is what the negotiator thought the deal was worth when it
    put its name to it (V_kept - V_none over the deal's life). `break_advantage`
    is what the planner thought breaking it was worth a moment later, net of
    the reputation penalty. Both positive is a *preference reversal*: the agent
    bought something and immediately preferred not to have it.

    The two decisions run on different opponent models, which is the thing this
    record exists to measure -- negotiation prices a deal assuming cooperation
    is conditional, planning prices orders against a partner whose behaviour is
    sampled independently of what was promised.
    """
    turn: int
    player: Player
    commitment_type: CommitmentType
    signed_gain: float
    break_advantage: float
    # Which game this row came from, as CalibrationPoint.game. Rows from one
    # game share a board and the same four agents, so an interval that counts
    # them as independent trials is too narrow.
    game: Optional[int] = None

    @property
    def reversed_(self) -> bool:
        return self.signed_gain > 0 and self.break_advantage > 0


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
    # Which game this row came from. Rows share a seed, a board and the same
    # players, so anything that splits or bootstraps has to cluster on it.
    game: Optional[int] = None
    # The two network inputs, kept so the CPT can be refitted on logged data
    # instead of guessed at (issue #16).
    reliability: float = 0.5
    incentive: float = 0.0
