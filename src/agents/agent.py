from typing import List, Dict, Tuple, Optional

from src.common.schemas import (
    Player, GameState, Order, OrderType, Commitment, CommitmentType, Message,
    MessageType, TrustTrace, BeliefSnapshot,
)
from src.engine.board import players, get_adjacent
from src.agents.trust.model import TrustModel
from src.agents.trust.rules import RuleEngine, OutcomeRule, PublicRecordRule
from src.agents.planner.planner import (
    Planner, PlannerConfig, DecisionTrace, evaluate_state, cooperation_value,
    defection_incentive, near_win, CENTRE_VALUE,
)
from src.agents.negotiation.strategy import NegotiationStrategy
from src.agents.negotiation.personas import PERSONAS, PersonaConfig
from src.agents.opponent_model.opponent_model import OpponentModel

# How long a betrayal stays on the books, and how fast a threat stops
# frightening anyone. Both decay, so nobody nurses a grudge forever.
GRUDGE_DECAY = 0.75
THREAT_DECAY = 0.5
THREAT_WEIGHT = 0.4
# The "trust" policy believes a partner whose P(keep) toward it is at least this;
# the same even-odds line fit_cpt holds a three-time betrayer below.
TRUST_LINE = 0.5
# A betrayal anyone could see, decayed like a grudge. Every policy but "none"
# signs nothing with a player whose record is at or above INFAMY_LINE: one
# break this turn, or a habit of them.
INFAMY_LINE = 1.0


class Agent:
    def __init__(self, player: Player, persona_name: str = "Opportunist",
                 planner_config: Optional[PlannerConfig] = None):
        self.player = player
        self.persona: PersonaConfig = PERSONAS[persona_name]
        self.persona_name = persona_name

        self.trust_model = TrustModel(player, self.persona.prior_alpha, self.persona.prior_beta)
        self.rule_engine = RuleEngine([
            OutcomeRule(), PublicRecordRule(),
        ])

        if planner_config is None:
            planner_config = PlannerConfig()
        planner_config.reputation_cost_coefficient = self.persona.reputation_cost
        planner_config.vengeance = self.persona.vengeance
        self.planner = Planner(player, planner_config)
        self.negotiation = NegotiationStrategy(player, self.planner)
        self.negotiation.privacy_preference = self.persona.privacy
        self.negotiation.bound = self.persona.policy != "none"
        # The planner charges back the unused share of whatever the negotiator
        # signed each deal for, so it needs the prices the negotiator recorded.
        self.planner.negotiation = self.negotiation

        # Opponent model — tracks historical patterns
        self.opponent_model = OpponentModel(player)
        self.planner.set_opponent_model(self.opponent_model)

        self.commitments: List[Commitment] = []
        self.trust_traces: List[TrustTrace] = []
        # Who has wronged me, and how badly. Feeds the planner's stance.
        self.grudges: Dict[Player, float] = {}
        # Who has promised to retaliate against me, and how recently.
        self.deterrence: Dict[Player, float] = {}
        # Who has been seen breaking a promise to anyone, and how recently.
        self.infamy: Dict[Player, float] = {}
        # Who hits back once betrayed: player -> [times betrayed, times they
        # moved on the betrayer the next turn]. Read as Planner.retaliation.
        self.retaliations: Dict[Player, List[int]] = {}
        self._wronged: List[Tuple[Player, Player]] = []    # (victim, betrayer) last turn
        self._wronged_next: List[Tuple[Player, Player]] = []
        self._orders_state: Optional[GameState] = None

    # ── social position the planner acts on ─────────────────────────────
    def stance(self) -> Dict[Player, float]:
        """One number per player: positive means I will pay to hurt them,
        negative means hurting them has been priced up by their threat."""
        out: Dict[Player, float] = {}
        for p in players():
            if p == self.player:
                continue
            w = self.persona.vengeance * self.grudges.get(p, 0.0)
            w -= THREAT_WEIGHT * self.deterrence.get(p, 0.0)
            if w:
                out[p] = w
        return out

    def wronged_by(self, p: Player) -> bool:
        return self.grudges.get(p, 0.0) >= self.persona.forgive_below

    def policy(self, state: GameState) -> Tuple[set, Dict[Player, float], set]:
        """The persona's loyalty policy on this board: (partners owed loyalty,
        charged planner.LOYALTY_COST to break with first; reputation-cost scale
        per partner; partners I sign nothing with).
        Nobody is bound to a player about to win."""
        others = [p for p in players() if p != self.player
                  and any(u.player == p for u in state.units)]
        free = {p for p in others if near_win(state, p)}
        infamous = {p for p in others if self.infamy.get(p, 0.0) >= INFAMY_LINE}
        kind = self.persona.policy
        if kind in ("unconditional", "reciprocal"):
            wronged = {p for p in others if self.wronged_by(p)}
            loyal = set(others) - wronged - free
            if kind == "unconditional":
                return loyal, {}, wronged | infamous
            return loyal, {p: 0.0 for p in wronged}, wronged | infamous
        if kind == "trust":
            doubted = {p for p in others if self.trust_model.p_keeps(
                p, CommitmentType.ALLIANCE, 0.0, toward=self.player) < TRUST_LINE}
            return (set(others) - doubted - free, {p: 0.0 for p in doubted},
                    doubted | infamous)
        # "none": the cost of a break scales with how much of mine they can reach.
        mine = [t for t, o in state.supply_centers.items() if o == self.player]
        weights = {}
        for p in others:
            reach = {a for u in state.units if u.player == p for a in get_adjacent(u.territory)}
            weights[p] = 2.0 * sum(1 for t in mine if t in reach) / max(1, len(mine))
        return set(), weights, set()

    def retaliation_rate(self, p: Player) -> float:
        """P(p hits back the turn after being betrayed), Laplace-smoothed toward
        the planner's prior of one in three."""
        seen, hit = self.retaliations.get(p, (0, 0))
        return (hit + 1) / (seen + 3)

    def decay_stance(self):
        self.grudges = {p: v * GRUDGE_DECAY for p, v in self.grudges.items() if v * GRUDGE_DECAY > 0.05}
        self.infamy = {p: v * GRUDGE_DECAY for p, v in self.infamy.items() if v * GRUDGE_DECAY > 0.05}
        self.deterrence = {p: v * THREAT_DECAY for p, v in self.deterrence.items() if v * THREAT_DECAY > 0.05}

    # ── incentives ──────────────────────────────────────────────────────
    def estimate_incentive(self, prev_state: GameState, new_state: GameState,
                           defector: Player) -> float:
        """How lucrative the break looked, from the board alone.

        The observer cannot see the defector's search, only what the turn won
        them. Positions are already denominated in centres, so the swing over
        one centre is a usable iota in [0, 1].

        ponytail: attributes the whole turn's swing to the betrayal. Good
        enough while one break resolves per turn; split it per commitment if
        multi-break turns become common.
        """
        gain = evaluate_state(new_state, defector) - evaluate_state(prev_state, defector)
        return min(1.0, max(0.0, gain / CENTRE_VALUE))

    def predict_keep(self, state: GameState, commitment: Commitment,
                     subject: Player) -> float:
        """P(this player keeps this deal), through the trust network.

        Logged at the moment a commitment is formed so it can be paired with
        the engine's verdict later — that pair is the only thing that says
        whether these numbers mean anything.
        """
        return self.predict_keep_parts(state, commitment, subject)[2]

    def predict_keep_parts(self, state: GameState, commitment: Commitment,
                           subject: Player) -> Tuple[float, float, float]:
        """(reliability, incentive, P(keeps)) — the network's two inputs and
        its output. Logged together so the CPT can be refitted on real data."""
        incentive = defection_incentive(state, subject, [commitment])
        reliability = self.trust_model.get_reliability(subject, commitment.commitment_type)
        return (reliability, incentive,
                self.trust_model.p_keeps(subject, commitment.commitment_type, incentive))

    # ── learning ────────────────────────────────────────────────────────
    def update_beliefs_from_outcomes(self, prev_state: GameState,
                                     new_state: GameState, outcomes: List):
        """Three tiers of evidence, and they are deliberately different.

          in the deal   first hand. Moves this player's reputation *and* what
                        I have seen them do to me specifically.
          public, not
          in the deal   the engine published it, so I saw it. Reputation only:
                        watching two other players deal says nothing about how
                        either treats me.
          private, not
          in the deal   invisible: a deal I am not in and cannot see.

        The middle tier is what makes the pair records mean anything. Without
        it an agent's general view of a player was built from exactly the same
        evidence as its pair view, so the two could never disagree.
        """
        for o in outcomes:
            for breaker in o.broken_by:
                if breaker != self.player:
                    self.infamy[breaker] = self.infamy.get(breaker, 0.0) + 1.0
            if self.player in o.commitment.players:
                incentive = max(
                    (self.estimate_incentive(prev_state, new_state, p) for p in o.broken_by),
                    default=0.0,
                )
                # The rule engine owns the Beta updates -- it applies them
                # *and* records which rule fired. Calling update_from_outcome
                # here too would count every outcome twice.
                self.trust_traces.extend(self.rule_engine.process(
                    self.trust_model, {'outcome': o, 'turn': prev_state.turn,
                                       'incentive_to_defect': incentive}
                ))
                for breaker in o.broken_by:
                    if breaker != self.player:
                        self.grudges[breaker] = self.grudges.get(breaker, 0.0) + 1.0
            elif not o.commitment.private:
                self.trust_traces.extend(self.rule_engine.process(
                    self.trust_model,
                    {'public_outcome': o, 'turn': prev_state.turn}))

        self.opponent_model.observe_commitment_outcomes(
            [o for o in outcomes if self.player in o.commitment.players])
        # The orders observed next are the ones given in prev_state; the
        # betrayals seen here are answered, or not, in the turn after.
        self._orders_state = prev_state
        self._wronged_next = sorted(
            {(v, b) for o in outcomes for b in o.broken_by
             for v in o.commitment.players if v != b},
            key=lambda vb: (vb[0].value, vb[1].value))

    def deal_signed(self, commitment: Commitment, turn: int):
        """Called by the runner when a deal this seat is in is signed or renewed."""
        self.negotiation.signed(commitment, turn)

    def observe_orders(self, orders: List[Order]):
        """Called by runner after each turn so opponent model can learn.
        Orders are public; who promised what is not."""
        self.opponent_model.observe_orders(orders)
        state = self._orders_state
        for victim, betrayer in self._wronged if state is not None else []:
            theirs = ({u.territory for u in state.units if u.player == betrayer}
                      | {t for t, o in state.supply_centers.items() if o == betrayer})
            hit = any(o.player == victim and o.order_type == OrderType.MOVE
                      and o.target in theirs for o in orders)
            rec = self.retaliations.setdefault(victim, [0, 0])
            rec[0] += 1
            rec[1] += hit
        self._wronged = self._wronged_next

    def receive_threat(self, msg: Message):
        """A threat is not accepted or rejected, it is believed or not.

        Believed in proportion to how much of it they have shown before: an
        agent that has actually retaliated is frightening, one that has only
        ever talked is not.
        """
        credibility = self.opponent_model.retaliation_rate(msg.sender)
        self.deterrence[msg.sender] = self.deterrence.get(msg.sender, 0.0) + credibility

    # ── negotiation ─────────────────────────────────────────────────────
    def propose(self, state: GameState,
                commitments: Optional[List[Commitment]] = None) -> List[Message]:
        self.negotiation._live_commitments = commitments or []
        self.negotiation.refuse = self.policy(state)[2]
        return self.negotiation.generate_proposals(state, self.trust_model)

    def reply(self, state: GameState, incoming: List[Message],
              commitments: Optional[List[Commitment]] = None) -> List[Message]:
        self.negotiation._live_commitments = commitments or []
        self.negotiation.refuse = self.policy(state)[2]
        replies = []
        for msg in incoming:
            if msg.message_type == MessageType.THREAT:
                self.receive_threat(msg)
                continue
            if msg.message_type not in (MessageType.PROPOSE, MessageType.COUNTER):
                continue
            r = self.negotiation.evaluate_proposal(state, msg, self.trust_model)
            if r is not None:
                replies.append(r)
        return replies

    def act(self, state: GameState, active_commitments: List[Commitment]
            ) -> Tuple[List[Order], DecisionTrace]:
        loyal, weights, _refuse = self.policy(state)
        self.planner.loyal_to = loyal
        self.planner.partner_weight = weights
        self.planner.retaliation = {p: self.retaliation_rate(p) for p in players()
                                    if p != self.player}
        return self.planner.find_best_orders(
            state, active_commitments, self.trust_model, self.stance())

    def beliefs(self) -> List[BeliefSnapshot]:
        return self.trust_model.snapshot()


class HumanAgent(Agent):
    """A seat played by a person through the web UI.

    It is a full Agent — same trust model, same opponent model, same beliefs
    panel — with the two decisions a human makes swapped in: which orders to
    give, and which proposals to take. Everything else (learning from
    outcomes) happens exactly as it does for the AI, so
    the trust view and the decision trace stay meaningful with a person in
    the game.
    """

    def __init__(self, player: Player, persona_name: str = "Opportunist"):
        super().__init__(player, persona_name)
        self.pending_orders: List[Order] = []
        # deal identity -> accept?  Keyed on what the deal *is* rather than on
        # the message id, so a counter-offer inherits the answer already given
        # to the original. (sender, type) alone was not enough: a bilateral
        # alliance and a three-way pact from the same player share both, and
        # so do a public and a private version of one, so answering either
        # silently answered the other.
        self.decisions: Dict[tuple, bool] = {}
        self.inbox: List[Message] = []

    @staticmethod
    def decision_key(msg: Message) -> tuple:
        """What makes two offers the same offer, from the answerer's side."""
        return (
            msg.sender,
            msg.commitment_type,
            frozenset(msg.coalition) if msg.coalition else None,
            bool(msg.private),
            msg.target_territory,
            tuple(sorted(msg.dmz_territories or [])),
        )

    def propose(self, state, commitments=None) -> List[Message]:
        return []  # the human answers offers; they do not auto-generate any

    def reply(self, state, incoming, commitments=None) -> List[Message]:
        replies = []
        for msg in incoming:
            if msg.message_type == MessageType.THREAT:
                self.receive_threat(msg)
                self.inbox.append(msg)
                continue
            if msg.message_type not in (MessageType.PROPOSE, MessageType.COUNTER):
                continue
            self.inbox.append(msg)
            choice = self.decisions.get(self.decision_key(msg))
            if choice is None:
                continue  # no answer given: silence is neither accept nor reject
            replies.append(self.negotiation._accept(msg) if choice
                           else self.negotiation._reject(msg))
        return replies

    def act(self, state, active_commitments):
        mine = {u.territory for u in state.units if u.player == self.player}
        orders = [o for o in self.pending_orders
                  if o.player == self.player and o.unit_territory in mine]
        covered = {o.unit_territory for o in orders}
        orders += [Order(player=self.player, unit_territory=t, order_type=OrderType.HOLD)
                   for t in mine - covered]
        self.pending_orders = []
        return orders, DecisionTrace(
            orders, evaluate_state(state, self.player), 0.0,
            "Human seat: orders entered by the examiner.")
