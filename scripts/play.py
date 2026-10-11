"""Play a seat from the terminal.

    python scripts/play.py --seat Red
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from src.common.config import GameConfig, add_game_args, game_kwargs, seating_for
from src.harness import build_runner, game_setup
from src.common.schemas import (
    Player, MessageType, CommitmentType, describe_message,
)
from src.agents.agent import HumanAgent
from src.engine.orders import generate_orders_for_unit, order_label
from src.engine.board import players, win_centers

BAR = "─" * 62


def _ask(prompt: str, options: list) -> int:
    """Menu that cannot be answered wrongly: the only inputs are indices."""
    for i, label in enumerate(options):
        print(f"   {i:>2}. {label}")
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return 0
        if raw.isdigit() and 0 <= int(raw) < len(options):
            return int(raw)
        print("   pick one of the numbers above.")


class CLIHuman(HumanAgent):
    """The shared human seat, with a terminal in front of it."""

    def reply(self, state, incoming, commitments=None):
        for msg in incoming:
            if msg.message_type == MessageType.THREAT:
                print(f"\n  ⚠  {describe_message(msg)}")
                self.receive_threat(msg)
                continue
            if msg.message_type == MessageType.INTENT:
                print(f"\n  ➜  {describe_message(msg)}  (candour so far "
                      f"{self.candour_of(msg.sender):.2f})")
                continue
            if msg.message_type not in (MessageType.PROPOSE, MessageType.COUNTER):
                continue
            print(f"\n  ✉  {describe_message(msg)}")
            p = self.trust_model.p_keeps(
                msg.sender, msg.commitment_type or CommitmentType.ALLIANCE, 0.0)
            print(f"     your model says P({msg.sender.value} keeps it) = {p:.2f}")
            take = _ask("     accept? ", ["reject", "accept"]) == 1
            self.decisions[self.decision_key(msg)] = take
        return super().reply(state, incoming, commitments)

    def vote_draw(self, state, stalled):
        print(f"\n  {stalled} turns without a centre changing hands.")
        return _ask("  vote for a draw? ", ["no", "yes"]) == 1

    def act(self, state, active_commitments):
        print(f"\n{BAR}\n  YOUR ORDERS — turn {state.turn}\n{BAR}")
        mine = [u for u in state.units if u.player == self.player]
        if active_commitments:
            print("  live promises:")
            for c in active_commitments:
                if self.player in c.players:
                    other = [p.value for p in c.players if p != self.player]
                    print(f"    {c.commitment_type.value} with {', '.join(other)}"
                          f" until turn {c.valid_until_turn}")
        chosen = []
        for u in mine:
            options = generate_orders_for_unit(state, u)
            print(f"\n  unit at {u.territory}:")
            idx = _ask("  order? ", [order_label(o) for o in options])
            chosen.append(options[idx])
        self.pending_orders = chosen
        return super().act(state, active_commitments)


def show_beliefs(agent):
    print("\n  your beliefs — P(keeps), Beta posterior mean")
    types = list(CommitmentType)
    print("    " + "subject".ljust(9) + "".join(t.value.rjust(11) for t in types))
    for subject in players():
        if subject == agent.player:
            continue
        row = "".join(f"{agent.trust_model.get_reliability(subject, t):>11.2f}"
                      for t in types)
        print("    " + subject.value.ljust(9) + row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seat", default="Red", choices=[p.value for p in Player])
    ap.add_argument("--seed", type=int, default=0)
    add_game_args(ap)
    args = ap.parse_args()

    seat = Player(args.seat)
    # Fixed seating: a human should be told who everyone is once.
    cfg = GameConfig(seed=args.seed, human_seat=seat, **game_kwargs(args))
    cfg.seating = seating_for(0, cfg.make_board().players)
    if seat not in cfg.seating:
        ap.error(f"{seat.value} is not seated in a {args.seats}-player game; "
                 f"pick one of {', '.join(p.value for p in cfg.seating)}")

    with game_setup(cfg):
        runner = build_runner(cfg)
        # The harness seats the shared HumanAgent; swap in the prompting one.
        you = CLIHuman(seat, cfg.seats()[seat])
        runner.agents[seat] = you
        print(f"\n{BAR}\n  You are {seat.value}. First to {win_centers()} supply "
              f"centres wins;\n  otherwise most centres after {runner.max_turns} "
              f"turns.\n{BAR}")
        _play(runner, you, seat)


def _play(runner, you, seat):
    while runner.state.turn <= runner.max_turns:
        record = runner.step()
        counts = runner.center_counts()
        print("\n  centres: " + "  ".join(
            f"{p.value} {c}" for p, c in counts.items()))
        for e in record.log.events:
            print(f"    {e}")
        show_beliefs(you)
        if runner.over():
            break

    champion = runner.winner()
    counts = runner.center_counts()
    ahead = runner.leaders()
    if champion:
        result = f"{champion.value} wins with {counts[champion]} centres."
    elif runner.drawn:
        ahead = runner.drawn
        result = f"Draw agreed: {', '.join(p.value for p in ahead)}."
    else:
        result = (f"Shared lead on {counts[ahead[0]]} centres: "
                  f"{', '.join(sorted(p.value for p in ahead))}.")
    print(f"\n{BAR}\n  {result}"
          f"{'  (that was you)' if seat in ahead else ''}\n{BAR}")
    brier = runner.brier_score()
    if brier is not None:
        print(f"  Everyone's P(keeps) scored a Brier of {brier:.3f} this game "
              f"(0.25 = coin flip).")


if __name__ == "__main__":
    main()
