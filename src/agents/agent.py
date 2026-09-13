from typing import List, Dict, Tuple
from src.common.schemas import (
    Player, GameState, Order, Commitment, Message, MessageType, CommitmentType,
    TrustTrace,
)
from src.agents.trust.model import TrustModel
from src.agents.trust.rules import RuleEngine, OutcomeRule, GossipRule
from src.agents.planner.planner import (
    Planner, PlannerConfig, DecisionTrace, evaluate_state, CENTRE_VALUE,
)
from src.agents.negotiation.strategy import NegotiationStrategy
from src.agents.negotiation.personas import PERSONAS, PersonaConfig
from src.agents.opponent_model.opponent_model import OpponentModel

class Agent:
    def __init__(self, player: Player, persona_name: str = "Opportunist"):
        self.player = player
        self.persona = PERSONAS[persona_name]
        
        self.trust_model = TrustModel(player, self.persona.prior_alpha, self.persona.prior_beta)
        self.rule_engine = RuleEngine([OutcomeRule(), GossipRule()])
        
        planner_config = PlannerConfig(reputation_cost_coefficient=self.persona.reputation_cost)
        self.planner = Planner(player, planner_config)
        self.negotiation = NegotiationStrategy(player, self.planner)

        # Opponent model — tracks historical patterns
        self.opponent_model = OpponentModel(player)
        self.planner.set_opponent_model(self.opponent_model)
        
        self.commitments: List[Commitment] = []
        self.trust_traces: List[TrustTrace] = []
        
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

    def update_beliefs_from_outcomes(self, prev_state: GameState,
                                     new_state: GameState, outcomes: List):
        for o in outcomes:
            # Each breaker gets scored on their own incentive; a kept
            # commitment does not need one.
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

        self.opponent_model.observe_commitment_outcomes(outcomes)
            
    def observe_orders(self, orders: List[Order]):
        """Called by runner after each turn so opponent model can learn."""
        self.opponent_model.observe_orders(orders)

    def receive_gossip(self, msg: Message):
        if msg.message_type == MessageType.BROADCAST and msg.broadcast_kind == "BETRAYED":
            facts = {'gossip': {'sender': msg.sender, 'accused': msg.broadcast_target, 'commitment_type': msg.commitment_type}}
            self.trust_traces.extend(self.rule_engine.process(self.trust_model, facts))
            
    def propose(self, state: GameState) -> List[Message]:
        return self.negotiation.generate_proposals(state, self.trust_model)
        
    def reply(self, state: GameState, incoming: List[Message]) -> List[Message]:
        replies = []
        for msg in incoming:
            if msg.message_type == MessageType.PROPOSE:
                replies.append(self.negotiation.evaluate_proposal(state, msg, self.trust_model))
        return replies
        
    def act(self, state: GameState, active_commitments: List[Commitment]) -> Tuple[List[Order], DecisionTrace]:
        return self.planner.find_best_orders(state, active_commitments, self.trust_model)
