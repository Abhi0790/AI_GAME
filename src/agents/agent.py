from typing import List, Dict, Tuple, Optional
import uuid

from src.common.schemas import (
    Player, GameState, Order, OrderType, Commitment, Message, MessageType,
    CommitmentType, TrustTrace, BeliefSnapshot, CommitmentOutcome,
)
from src.agents.trust.model import TrustModel
from src.agents.trust.rules import (
    RuleEngine, OutcomeRule, GossipRule, FalseAccusationRule,
    FALSE_ACCUSATION_WEIGHT, GOSSIP_TRUST_THRESHOLD,
)
from src.agents.planner.planner import (
    Planner, PlannerConfig, DecisionTrace, evaluate_state, cooperation_value,
    defection_incentive, CENTRE_VALUE,
)
from src.agents.negotiation.strategy import NegotiationStrategy
from src.agents.negotiation.personas import PERSONAS, PersonaConfig
from src.agents.opponent_model.opponent_model import OpponentModel

# How long a betrayal stays on the books, and how fast a threat stops
# frightening anyone. Both decay, so nobody nurses a grudge forever.
GRUDGE_DECAY = 0.75
THREAT_DECAY = 0.5
THREAT_WEIGHT = 0.4


class Agent:
    def __init__(self, player: Player, persona_name: str = "Opportunist",
                 planner_config: Optional[PlannerConfig] = None):
        self.player = player
        self.persona: PersonaConfig = PERSONAS[persona_name]
        self.persona_name = persona_name

        self.trust_model = TrustModel(player, self.persona.prior_alpha, self.persona.prior_beta)
        self.rule_engine = RuleEngine([OutcomeRule(), GossipRule(), FalseAccusationRule()])

        if planner_config is None:
            planner_config = PlannerConfig()
        planner_config.reputation_cost_coefficient = self.persona.reputation_cost
        planner_config.vengeance = self.persona.vengeance
        self.planner = Planner(player, planner_config)
        self.negotiation = NegotiationStrategy(player, self.planner)

        # Opponent model — tracks historical patterns
        self.opponent_model = OpponentModel(player)
        self.planner.set_opponent_model(self.opponent_model)

        self.commitments: List[Commitment] = []
        self.trust_traces: List[TrustTrace] = []
        # Who has wronged me, and how badly. Feeds the planner's stance.
        self.grudges: Dict[Player, float] = {}
        # Who has promised to retaliate against me, and how recently.
        self.deterrence: Dict[Player, float] = {}

    # ── social position the planner acts on ─────────────────────────────
    def stance(self) -> Dict[Player, float]:
        """One number per player: positive means I will pay to hurt them,
        negative means hurting them has been priced up by their threat."""
        out: Dict[Player, float] = {}
        for p in Player:
            if p == self.player:
                continue
            w = self.persona.vengeance * self.grudges.get(p, 0.0)
            w -= THREAT_WEIGHT * self.deterrence.get(p, 0.0)
            if w:
                out[p] = w
        return out

    def decay_stance(self):
        self.grudges = {p: v * GRUDGE_DECAY for p, v in self.grudges.items() if v * GRUDGE_DECAY > 0.05}
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
        """First-hand evidence only.

        An agent grades the deals it was *in*. What happened between two other
        players is hearsay until somebody broadcasts it — which is what makes
        the gossip rules, and lying, mean anything at all. Previously every
        agent was handed every outcome, so no accusation could ever inform
        anyone of anything they did not already know.
        """
        for o in outcomes:
            if self.player not in o.commitment.players:
                continue
            incentive = max(
                (self.estimate_incentive(prev_state, new_state, p) for p in o.broken_by),
                default=0.0,
            )
            # The rule engine owns the Beta updates -- it applies them *and*
            # records which rule fired. Calling update_from_outcome here too
            # would count every outcome twice.
            self.trust_traces.extend(self.rule_engine.process(
                self.trust_model, {'outcome': o, 'incentive_to_defect': incentive}
            ))
            for breaker in o.broken_by:
                if breaker != self.player:
                    self.grudges[breaker] = self.grudges.get(breaker, 0.0) + 1.0

        self.opponent_model.observe_commitment_outcomes(
            [o for o in outcomes if self.player in o.commitment.players])

    def observe_orders(self, orders: List[Order]):
        """Called by runner after each turn so opponent model can learn.
        Orders are public; who promised what is not."""
        self.opponent_model.observe_orders(orders)

    def receive_gossip(self, msg: Message):
        if msg.message_type != MessageType.BROADCAST or msg.broadcast_kind != "BETRAYED":
            return
        facts = {'gossip': {
            'sender': msg.sender,
            'accused': msg.broadcast_target,
            'commitment_type': msg.commitment_type or CommitmentType.ALLIANCE,
            'verdict': msg.engine_verdict,
        }}
        self.trust_traces.extend(self.rule_engine.process(self.trust_model, facts))
        if msg.engine_verdict == "CONFIRMED" and msg.broadcast_target != self.player:
            self.grudges.setdefault(msg.broadcast_target, 0.0)

    def receive_threat(self, msg: Message):
        """A threat is not accepted or rejected, it is believed or not.

        Believed in proportion to how much of it they have shown before: an
        agent that has actually retaliated is frightening, one that has only
        ever talked is not.
        """
        credibility = self.opponent_model.retaliation_rate(msg.sender)
        self.deterrence[msg.sender] = self.deterrence.get(msg.sender, 0.0) + credibility

    # ── accusations: the agent's decision, not the engine's ─────────────
    def consider_broadcast(self, state: GameState, outcomes: List[CommitmentOutcome]
                           ) -> List[Message]:
        """Decide what, if anything, to tell the table.

        Two separate judgements:
          truthful — I was betrayed. Saying so costs nothing and moves other
            players' beliefs, but only if they rate me above the gossip
            threshold and the betrayer is still someone who matters.
          false — nobody betrayed me, but the leader is the problem and a
            story about them would slow everyone else down. The engine will
            refute it and R4 will bill me; a persona with a high `deception`
            discounts that bill, which is exactly the bounded rationality the
            rule exists to punish.
        """
        msgs: List[Message] = []
        my_standing = self.trust_model.general_trust(self.player)

        def accuse(target: Player, c_type: CommitmentType, truthful: bool) -> Message:
            return Message(
                id=str(uuid.uuid4()), sender=self.player, receiver=None,
                message_type=MessageType.BROADCAST, broadcast_kind="BETRAYED",
                broadcast_target=target, commitment_type=c_type, truthful=truthful)

        accused = set()
        for o in outcomes:
            if o.kept or self.player not in o.commitment.players or self.player in o.broken_by:
                continue
            for betrayer in o.broken_by:
                if betrayer in accused:
                    continue
                # Worth saying only if I will be believed and they still
                # matter — shouting about a player with nothing left is noise.
                believable = my_standing > GOSSIP_TRUST_THRESHOLD
                still_matters = any(u.player == betrayer for u in state.units)
                if believable and still_matters:
                    accused.add(betrayer)
                    msgs.append(accuse(betrayer, o.commitment.commitment_type, True))

        if msgs or self.persona.deception <= 0:
            return msgs

        counts = {p: 0 for p in Player}
        for owner in state.supply_centers.values():
            if owner:
                counts[owner] += 1
        leader = max(counts, key=lambda p: counts[p])
        if leader == self.player or counts[leader] <= counts[self.player]:
            return msgs
        if my_standing <= GOSSIP_TRUST_THRESHOLD:
            return msgs

        gain = counts[leader] - counts[self.player]
        perceived_risk = FALSE_ACCUSATION_WEIGHT * (1.0 - self.persona.deception)
        if gain > perceived_risk:
            msgs.append(accuse(leader, CommitmentType.ALLIANCE, False))
        return msgs

    # ── negotiation ─────────────────────────────────────────────────────
    def propose(self, state: GameState,
                commitments: Optional[List[Commitment]] = None) -> List[Message]:
        self.negotiation._live_commitments = commitments or []
        return self.negotiation.generate_proposals(state, self.trust_model)

    def reply(self, state: GameState, incoming: List[Message],
              commitments: Optional[List[Commitment]] = None) -> List[Message]:
        self.negotiation._live_commitments = commitments or []
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
        return self.planner.find_best_orders(
            state, active_commitments, self.trust_model, self.stance())

    def beliefs(self) -> List[BeliefSnapshot]:
        return self.trust_model.snapshot()


class HumanAgent(Agent):
    """A seat played by a person through the web UI.

    It is a full Agent — same trust model, same opponent model, same beliefs
    panel — with the two decisions a human makes swapped in: which orders to
    give, and which proposals to take. Everything else (learning from
    outcomes, being gossiped about) happens exactly as it does for the AI, so
    the trust view and the decision trace stay meaningful with a person in
    the game.
    """

    def __init__(self, player: Player, persona_name: str = "Opportunist"):
        super().__init__(player, persona_name)
        self.pending_orders: List[Order] = []
        # (sender, commitment_type) -> accept?  Keyed on the deal rather than
        # the message id so a counter-offer inherits the same answer.
        self.decisions: Dict[Tuple[Player, CommitmentType], bool] = {}
        self.inbox: List[Message] = []

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
            choice = self.decisions.get((msg.sender, msg.commitment_type))
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

    def consider_broadcast(self, state, outcomes):
        return []  # the human accuses through the UI, not on a heuristic
