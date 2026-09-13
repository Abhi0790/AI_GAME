import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.schemas import Player, GameState, Order, OrderType, Commitment, Message, MessageType, CommitmentType
from src.agents.agent import Agent
from src.engine.runner import GameRunner
from src.ui.visualize import draw_board, draw_action_phase
from src.agents.planner.planner import DecisionTrace
import uuid

class HumanAgent:
    def __init__(self, player: Player):
        self.player = player
        
    def update_beliefs_from_outcomes(self, prev_state: GameState, new_state: GameState, outcomes: list):
        pass # Human updates their own beliefs
        
    def receive_gossip(self, msg: Message):
        print(f"\n[GOSSIP] {msg.sender} says {msg.broadcast_target} betrayed them!")
        
    def propose(self, state: GameState) -> list:
        draw_board(state)
        print(f"\n=== YOUR TURN (Turn {state.turn}) ===")
        print("Board State:")
        for u in state.units:
            print(f"  {u.player.name} unit at {u.territory}")
        
        proposals = []
        while True:
            ans = input("\nDo you want to propose an alliance to anyone? (y/n): ").strip().lower()
            if ans != 'y': break
            
            target = input("Which player? (Red/Blue/Green/Gold): ").strip().capitalize()
            try:
                target_p = Player(target)
                if target_p == self.player:
                    print("You can't ally with yourself.")
                    continue
                
                proposals.append(Message(
                    id=str(uuid.uuid4()),
                    sender=self.player,
                    receiver=target_p,
                    message_type=MessageType.PROPOSE,
                    commitment_type=CommitmentType.ALLIANCE,
                    turns=3
                ))
                print(f"Proposed alliance to {target_p.name}.")
            except ValueError:
                print("Invalid player.")
        return proposals
        
    def reply(self, state: GameState, incoming: list) -> list:
        replies = []
        for msg in incoming:
            if msg.message_type == MessageType.PROPOSE:
                ans = input(f"\n[NEGOTIATION] {msg.sender.name} proposes a {msg.commitment_type.name} commitment. Accept? (y/n): ").strip().lower()
                if ans == 'y':
                    replies.append(Message(
                        id=str(uuid.uuid4()),
                        sender=self.player,
                        receiver=msg.sender,
                        message_type=MessageType.ACCEPT,
                        reference_id=msg.id
                    ))
                else:
                    replies.append(Message(
                        id=str(uuid.uuid4()),
                        sender=self.player,
                        receiver=msg.sender,
                        message_type=MessageType.REJECT,
                        reference_id=msg.id
                    ))
        return replies
        
    def act(self, state: GameState, active_commitments: list):
        print("\n=== ENTER ORDERS ===")
        my_units = [u for u in state.units if u.player == self.player]
        orders = []
        for u in my_units:
            print(f"\nUnit at {u.territory}:")
            otype = input("  Order (Hold/Move/Support): ").strip().capitalize()
            try:
                order_enum = OrderType(otype)
                if order_enum == OrderType.HOLD:
                    orders.append(Order(player=self.player, unit_territory=u.territory, order_type=order_enum))
                elif order_enum == OrderType.MOVE:
                    target = input("  Target territory: ").strip().upper()
                    orders.append(Order(player=self.player, unit_territory=u.territory, order_type=order_enum, target=target))
                elif order_enum == OrderType.SUPPORT:
                    target = input("  Target territory: ").strip().upper()
                    sup_from = input("  Supporting unit from (leave empty if supporting a hold): ").strip().upper()
                    if not sup_from: sup_from = None
                    orders.append(Order(player=self.player, unit_territory=u.territory, order_type=order_enum, target=target, supported_from=sup_from))
            except ValueError:
                print("  Invalid order. Defaulting to HOLD.")
                orders.append(Order(player=self.player, unit_territory=u.territory, order_type=OrderType.HOLD))
                
        return orders, DecisionTrace(orders, 0, 0, "Human decision", False)

def main():
    print("Welcome to Diplomacy AI (CLI Version)")
    human_player = Player.RED
    
    agents = [
        HumanAgent(human_player),
        Agent(Player.BLUE, "Opportunist"),
        Agent(Player.GREEN, "Honest"),
        Agent(Player.GOLD, "Vengeful")
    ]
    
    runner = GameRunner(agents, on_turn_resolved=draw_action_phase)
    runner.run()
    draw_board(runner.state)
    import matplotlib.pyplot as plt
    plt.show(block=True)

if __name__ == "__main__":
    main()
