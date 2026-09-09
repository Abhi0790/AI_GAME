from typing import List, Dict
import uuid

from src.common.schemas import (
    GameState, Order, Player, Commitment, Message, MessageType,
    CommitmentType, Unit,
)
from src.engine.adjudicator import resolve, verify_commitments
from src.agents.agent import Agent
from src.engine.board import get_all_territories, is_supply_center, TERRITORIES
from src.agents.negotiation.strategy import message_to_commitment


class GameRunner:
    def __init__(self, agents: List, on_turn_resolved=None):
        self.agents = {a.player: a for a in agents}
        self.commitments: List[Commitment] = []
        self.history = []
        
        # Initial State
        units = [
            Unit(player=Player.RED, territory="R1"),
            Unit(player=Player.RED, territory="R2"),
            Unit(player=Player.RED, territory="R3"),
            
            Unit(player=Player.BLUE, territory="B1"),
            Unit(player=Player.BLUE, territory="B2"),
            Unit(player=Player.BLUE, territory="B3"),
            
            Unit(player=Player.GREEN, territory="G1"),
            Unit(player=Player.GREEN, territory="G2"),
            Unit(player=Player.GREEN, territory="G3"),
            
            Unit(player=Player.GOLD, territory="Y1"),
            Unit(player=Player.GOLD, territory="Y2"),
            Unit(player=Player.GOLD, territory="Y3"),
        ]
        
        supply_centers = {
            "R1": Player.RED, "B1": Player.BLUE, "G1": Player.GREEN, "Y1": Player.GOLD,
            "N1": None, "N2": None
        }
        
        territory_owners = {
            "R1": Player.RED, "R2": Player.RED, "R3": Player.RED,
            "B1": Player.BLUE, "B2": Player.BLUE, "B3": Player.BLUE,
            "G1": Player.GREEN, "G2": Player.GREEN, "G3": Player.GREEN,
            "Y1": Player.GOLD, "Y2": Player.GOLD, "Y3": Player.GOLD,
            "N1": None, "N2": None
        }
        
        self.state = GameState(turn=1, units=units, supply_centers=supply_centers, territory_owners=territory_owners)
        self.max_turns = 12
        self.on_turn_resolved = on_turn_resolved
        self.negotiation_rounds = 3
        
    def run(self):
        for t in range(self.state.turn, self.max_turns + 1):
            print(f"--- Turn {t} ---")
            
            # 1. Clean expired commitments
            self.commitments = [c for c in self.commitments if c.valid_until_turn >= t]
            
            # 5. Negotiation
            new_messages = []
            for a in self.agents.values():
                new_messages.extend(a.propose(self.state))
                
            for round_idx in range(self.negotiation_rounds):
                if not new_messages:
                    break
                    
                replies = []
                # Group by receiver
                inbox = {p: [] for p in Player}
                for m in new_messages:
                    if m.receiver:
                        inbox[m.receiver].append(m)
                    else:
                        # Broadcast
                        for p in Player:
                            if p != m.sender: inbox[p].append(m)
                            
                for p, a in self.agents.items():
                    r = a.reply(self.state, inbox[p])
                    replies.extend(r)
                    
                    # Convert accepts to commitments
                    for msg in inbox[p]:
                        if msg.message_type == MessageType.BROADCAST:
                            a.receive_gossip(msg)
                            
                for rep in replies:
                    if rep.message_type == MessageType.ACCEPT:
                        # Find original proposal
                        orig = next((m for m in new_messages if m.id == rep.reference_id), None)
                        if orig:
                            c = message_to_commitment(orig, rep.sender, self.state.turn)
                            self.commitments.append(c)
                            print(f"Commitment created: {c.commitment_type} between {c.players}")
                            
                new_messages = replies
                
            # 7-10. Agents plan and choose orders
            all_orders = []
            traces = {}
            for p, a in self.agents.items():
                orders, trace = a.act(self.state, self.commitments)
                all_orders.extend(orders)
                traces[p] = trace
                
            # 11-13. Resolve
            new_state, outcomes, log = resolve(self.state, all_orders, self.commitments)
            if self.on_turn_resolved:
                self.on_turn_resolved(self.state, all_orders, log)
            
            # Print outcomes
            for o in outcomes:
                if not o.kept:
                    print(f"Commitment BROKEN by {o.broken_by}: {o.commitment.commitment_type}")

            # ── Gossip broadcasts ────────────────────────────────────
            # Agents whose commitments were broken broadcast a BETRAYED
            # message so the gossip trust rules fire for other players.
            for o in outcomes:
                if not o.kept:
                    # Each non-breaking participant broadcasts
                    for victim in o.commitment.players:
                        if victim in o.broken_by:
                            continue
                        for betrayer in o.broken_by:
                            gossip_msg = Message(
                                id=str(uuid.uuid4()),
                                sender=victim,
                                receiver=None,  # broadcast
                                message_type=MessageType.BROADCAST,
                                broadcast_kind="BETRAYED",
                                broadcast_target=betrayer,
                                commitment_type=o.commitment.commitment_type,
                            )
                            # Deliver to all other agents
                            for p, a in self.agents.items():
                                if p != victim:
                                    a.receive_gossip(gossip_msg)
                            print(f"[GOSSIP] {victim.value} broadcasts: "
                                  f"{betrayer.value} broke {o.commitment.commitment_type.value}")
                    
            # Print resolution log
            for l in log.events:
                print(l)
                
            # 14. Beliefs update
            for a in self.agents.values():
                a.update_beliefs_from_outcomes(new_state, outcomes)

            # Feed orders to opponent models
            for a in self.agents.values():
                if hasattr(a, 'observe_orders'):
                    a.observe_orders(all_orders)
                
            self.history.append((self.state, all_orders, outcomes, log, traces))
            self.state = new_state
            
        print("Game Over")
        # Determine winner
        centers = {p: 0 for p in Player}
        for owner in self.state.supply_centers.values():
            if owner: centers[owner] += 1
        print("Final Centers:", centers)
