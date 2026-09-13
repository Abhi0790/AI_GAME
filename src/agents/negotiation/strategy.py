from typing import List, Dict, Optional
import uuid

from src.common.schemas import (
    GameState, Message, MessageType, Player, CommitmentType,
    Commitment, Order, message_to_commitment,
)
from src.agents.planner.planner import Planner, evaluate_state
from src.agents.trust.model import TrustModel
from src.engine.board import get_adjacent, is_supply_center


class NegotiationStrategy:
    def __init__(self, player: Player, planner: Planner):
        self.player = player
        self.planner = planner

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------
    def generate_proposals(
        self, state: GameState, trust_model: TrustModel
    ) -> List[Message]:
        proposals: List[Message] = []

        my_units = [u for u in state.units if u.player == self.player]
        my_territories = {u.territory for u in my_units}

        for p in Player:
            if p == self.player:
                continue

            reliability = trust_model.get_reliability(p, CommitmentType.ALLIANCE)

            # --- Alliance proposals (existing) ---
            if reliability > 0.4:
                proposals.append(Message(
                    id=str(uuid.uuid4()),
                    sender=self.player,
                    receiver=p,
                    message_type=MessageType.PROPOSE,
                    commitment_type=CommitmentType.ALLIANCE,
                    turns=3,
                ))

            # --- DMZ proposals for contested borders ---
            other_units = [u for u in state.units if u.player == p]
            other_territories = {u.territory for u in other_units}

            shared_borders = set()
            for my_t in my_territories:
                for adj in get_adjacent(my_t):
                    if adj in other_territories or (
                        state.territory_owners.get(adj) == p
                    ):
                        shared_borders.add(adj)
            for ot in other_territories:
                for adj in get_adjacent(ot):
                    if adj in my_territories or (
                        state.territory_owners.get(adj) == self.player
                    ):
                        shared_borders.add(adj)

            # Propose DMZ on neutral supply centers at the border
            dmz_candidates = [
                t for t in shared_borders
                if is_supply_center(t)
                and state.territory_owners.get(t) is None
            ]
            if dmz_candidates and reliability > 0.3:
                proposals.append(Message(
                    id=str(uuid.uuid4()),
                    sender=self.player,
                    receiver=p,
                    message_type=MessageType.PROPOSE,
                    commitment_type=CommitmentType.DMZ,
                    turns=2,
                    dmz_territories=dmz_candidates,
                ))

            # --- Support proposals (mutual help against shared enemy) ---
            # If we have a unit adjacent to a territory the other player
            # is also adjacent to, propose supporting them (or vice versa)
            if reliability > 0.5:
                for my_u in my_units:
                    for adj in get_adjacent(my_u.territory):
                        target_owner = state.territory_owners.get(adj)
                        # Target is held by someone else
                        if target_owner and target_owner not in (
                            self.player, p
                        ):
                            # Check if p has a unit that could move there
                            for ou in other_units:
                                if adj in get_adjacent(ou.territory):
                                    proposals.append(Message(
                                        id=str(uuid.uuid4()),
                                        sender=self.player,
                                        receiver=p,
                                        message_type=MessageType.PROPOSE,
                                        commitment_type=CommitmentType.SUPPORT,
                                        turns=1,
                                        target_territory=adj,
                                        supported_from=ou.territory,
                                    ))
                                    break  # one support proposal per target

            # --- Threat messages (deter attacks) ---
            if reliability < 0.35:
                proposals.append(Message(
                    id=str(uuid.uuid4()),
                    sender=self.player,
                    receiver=p,
                    message_type=MessageType.THREAT,
                    condition="attack_my_territory",
                    action="full_retaliation",
                ))

        return proposals

    # ------------------------------------------------------------------
    # Proposal evaluation
    # ------------------------------------------------------------------
    def evaluate_proposal(
        self, state: GameState, msg: Message, trust_model: TrustModel
    ) -> Message:
        """Evaluate an incoming proposal and decide to ACCEPT or REJECT."""
        reliability = trust_model.get_reliability(
            msg.sender, msg.commitment_type or CommitmentType.ALLIANCE
        )

        # --- Alliance ---
        if msg.commitment_type == CommitmentType.ALLIANCE:
            # Consider game position: accept if we have few centres or
            # the sender is stronger (worth allying with)
            my_centres = sum(
                1 for v in state.supply_centers.values()
                if v == self.player
            )
            # Accept if trust is decent or we're in a weak position
            if reliability > 0.4 or my_centres <= 1:
                return self._accept(msg)
            return self._reject(msg)

        # --- DMZ ---
        if msg.commitment_type == CommitmentType.DMZ:
            # Accept DMZ if the territories aren't ones we're planning
            # to invade soon (heuristic: accept if trust is ok)
            if reliability > 0.3:
                return self._accept(msg)
            return self._reject(msg)

        # --- Support ---
        if msg.commitment_type == CommitmentType.SUPPORT:
            if reliability > 0.5:
                return self._accept(msg)
            return self._reject(msg)

        # --- Threat ---
        if msg.message_type == MessageType.THREAT:
            # Threats don't need accept/reject — just acknowledge
            return self._reject(msg)  # We don't capitulate

        # Fallback
        return self._accept(msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _accept(self, msg: Message) -> Message:
        return Message(
            id=str(uuid.uuid4()),
            sender=self.player,
            receiver=msg.sender,
            message_type=MessageType.ACCEPT,
            reference_id=msg.id,
        )

    def _reject(self, msg: Message) -> Message:
        return Message(
            id=str(uuid.uuid4()),
            sender=self.player,
            receiver=msg.sender,
            message_type=MessageType.REJECT,
            reference_id=msg.id,
        )
