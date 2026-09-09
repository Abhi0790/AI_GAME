from typing import List, Dict, Tuple
from src.common.schemas import Player, GameState, Order, Commitment, Message, MessageType, CommitmentType
from src.agents.trust.model import TrustModel
from src.agents.trust.rules import RuleEngine, OutcomeRule, GossipRule
from src.agents.planner.planner import Planner, PlannerConfig, DecisionTrace
from src.agents.negotiation.strategy import NegotiationStrategy
from src.agents.opponent_model.opponent_model import OpponentModel

class PersonaConfig:
    def __init__(self, name: str, rep_cost: float, prior_alpha: float, prior_beta: float):
        self.name = name
        self.reputation_cost = rep_cost
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

PERSONAS = {
    "Honest": PersonaConfig("Honest", 100.0, 1.0, 1.0),
    "Opportunist": PersonaConfig("Opportunist", 1.0, 1.0, 1.0),
    "Vengeful": PersonaConfig("Vengeful", 2.0, 1.0, 1.0), # Simplification
    "Paranoid": PersonaConfig("Paranoid", 1.0, 1.0, 3.0),
}

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
        
    def update_beliefs_from_outcomes(self, state: GameState, outcomes: List):
        for o in outcomes:
            self.trust_model.update_from_outcome(o, incentive_to_defect=1.0) # naive incentive
            facts = {'outcome': o, 'incentive_to_defect': 1.0}
            traces = self.rule_engine.process(self.trust_model, facts)
            # Log traces

        # Update opponent model with commitment outcomes
        self.opponent_model.observe_commitment_outcomes(outcomes)
            
    def observe_orders(self, orders: List[Order]):
        """Called by runner after each turn so opponent model can learn."""
        self.opponent_model.observe_orders(orders)

    def receive_gossip(self, msg: Message):
        if msg.message_type == MessageType.BROADCAST and msg.broadcast_kind == "BETRAYED":
            facts = {'gossip': {'sender': msg.sender, 'accused': msg.broadcast_target, 'commitment_type': msg.commitment_type}}
            self.rule_engine.process(self.trust_model, facts)
            
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
