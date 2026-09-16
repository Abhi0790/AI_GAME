from typing import Dict, List, Tuple, Optional
from src.common.schemas import (
    Player, CommitmentType, Commitment, CommitmentOutcome, BeliefSnapshot,
)

# How much of the reputation hit a profitable betrayal is forgiven.
# w = 1 - lambda * incentive  (slide 6). lambda = 0 makes every break cost the
# same; lambda = 1 makes a maximally profitable break free.
# FITTED by scripts/run_sweep.py over lambda x decay x reputation multiplier,
# 5 games per cell. 0.2 is the best cell on both axes the sweep measures: the
# Honest/Opportunist betrayal spread (0.337, against 0.308 at the original
# 0.6) and calibration (Brier 0.222, against 0.267). Rerun it to see the grid.
#
# The tension this creates: a small lambda forgives a profitable betrayal only
# slightly, so R1 and R2 fire on similar weights and the rule split is more of
# a label than a large numeric difference. lambda = 0.6 costs about 0.03 of
# persona spread and 0.045 of Brier if a demo needs that split to be visible.
LAMBDA_INCENTIVE = 0.2
MIN_BREAK_WEIGHT = 0.1

# Evidence decays before each new observation, so a record never hardens into
# a number one more turn cannot move. Without it a settled Beta(8,1) shifts by
# 0.01 on a betrayal, deltaP vanishes and the planner's reputation term stops
# being a deterrent at all (issue #16). 1.0 restores the old behaviour.
EVIDENCE_DECAY = 0.85
# Evidence is never allowed to decay below the prior's worth of it.
MIN_EVIDENCE = 1e-6

# --- Bayesian network: Reliability -> Keeps <- Incentive (slide 6) ----------
# Three nodes. Reliability is the Beta posterior over "keeps promises of this
# kind". Incentive is how much the board is offering them to break right now.
# Keeps is the child, and this is its conditional table, written as two
# anchors that get interpolated by the reliability estimate:
#
#   P(keeps | r, iota=0) = BN_FLOOR + (1 - BN_FLOOR) * r     (nothing on offer)
#   P(keeps | r, iota=1) = BN_TEMPTED * r                    (everything is)
#
# The gap between the two rows is what makes the prediction a network rather
# than a Beta mean with extra steps.
#
# These are FITTED, not chosen: src/evaluation/sweep.fit_cpt grid-searches
# them against logged (reliability, incentive, kept?) triples, and
# scripts/run_sweep.py reprints the fit and the grid it came from.
BN_FLOOR = 0.2
BN_TEMPTED = 0.35
# Incentive is measured in centres of immediate gain, and a whole centre on
# offer is rare — most live deals sit near zero. Without this the CPT's two
# rows are blended at weights the data never reaches, and the incentive node
# cannot move the answer however clear the evidence is. iota at or above the
# scale counts as maximum temptation.
BN_INCENTIVE_SCALE = 0.1


def set_parameters(lambda_incentive: Optional[float] = None,
                   evidence_decay: Optional[float] = None,
                   bn_floor: Optional[float] = None,
                   bn_tempted: Optional[float] = None,
                   bn_incentive_scale: Optional[float] = None):
    """Rebind the global constants. The parameter sweep drives this."""
    global LAMBDA_INCENTIVE, EVIDENCE_DECAY, BN_FLOOR, BN_TEMPTED, BN_INCENTIVE_SCALE
    if lambda_incentive is not None:
        LAMBDA_INCENTIVE = lambda_incentive
    if evidence_decay is not None:
        EVIDENCE_DECAY = evidence_decay
    if bn_floor is not None:
        BN_FLOOR = bn_floor
    if bn_tempted is not None:
        BN_TEMPTED = bn_tempted
    if bn_incentive_scale is not None:
        BN_INCENTIVE_SCALE = bn_incentive_scale


def break_weight(incentive: float) -> float:
    """w in "broken: beta + w". A gratuitous betrayal costs a full point of
    reputation; a lucrative one costs less, because it explains itself."""
    incentive = min(1.0, max(0.0, incentive))
    return max(MIN_BREAK_WEIGHT, 1.0 - LAMBDA_INCENTIVE * incentive)


def p_keeps_given(reliability: float, incentive: float,
                  floor: Optional[float] = None, tempted_row: Optional[float] = None,
                  scale: Optional[float] = None) -> float:
    """The Keeps node's CPT, interpolated on the incentive.

    Pure function so the sweep and the tests can hit it without a model."""
    floor = BN_FLOOR if floor is None else floor
    tempted_row = BN_TEMPTED if tempted_row is None else tempted_row
    scale = BN_INCENTIVE_SCALE if scale is None else scale

    r = min(1.0, max(0.0, reliability))
    i = min(1.0, max(0.0, incentive / scale if scale > 0 else incentive))
    calm = floor + (1.0 - floor) * r
    tempted = tempted_row * r
    return (1.0 - i) * calm + i * tempted


class TrustRecord:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def get_expected_value(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, kept: bool, discount: float = 1.0):
        # Forget a little, then observe. Keeps the posterior responsive.
        self.alpha = max(MIN_EVIDENCE, self.alpha * EVIDENCE_DECAY)
        self.beta = max(MIN_EVIDENCE, self.beta * EVIDENCE_DECAY)
        if kept:
            self.alpha += discount
        else:
            self.beta += discount


class TrustModel:
    def __init__(self, owner: Player, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.owner = owner
        # target_player -> commitment_type -> TrustRecord
        self.records: Dict[Player, Dict[CommitmentType, TrustRecord]] = {}
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def get_record(self, player: Player, c_type: CommitmentType) -> TrustRecord:
        if player not in self.records:
            self.records[player] = {}
        if c_type not in self.records[player]:
            self.records[player][c_type] = TrustRecord(self.prior_alpha, self.prior_beta)
        return self.records[player][c_type]

    def get_reliability(self, player: Player, c_type: CommitmentType) -> float:
        """The Reliability node alone: the Beta posterior mean."""
        return self.get_record(player, c_type).get_expected_value()

    def p_keeps(self, player: Player, c_type: CommitmentType,
                incentive: float = 0.0) -> float:
        """P(keeps this commitment) — the whole network, not one node.

        Reliability says what they usually do; incentive says what the board
        is offering them this turn. A perfectly reliable partner standing
        next to an undefended centre of mine is not a safe bet, and the Beta
        mean on its own cannot say so.
        """
        return p_keeps_given(self.get_reliability(player, c_type), incentive)

    def update_from_outcome(self, outcome: CommitmentOutcome, incentive_to_defect: float = 0.0):
        """Apply one graded commitment to the Beta records.

        Own record included on purpose: the engine publishes every outcome, so
        every player scores my record off the same public evidence. That record
        is the best estimate I have of how much reputation I stand to lose,
        which is what the planner prices a betrayal against.
        """
        for p in outcome.commitment.players:
            record = self.get_record(p, outcome.commitment.commitment_type)
            if p in outcome.broken_by:
                record.update(kept=False, discount=break_weight(incentive_to_defect))
            else:
                record.update(kept=True, discount=1.0)

    def reputation_drop(self, player: Player, c_type: CommitmentType,
                        incentive: float = 0.0) -> float:
        """Delta-P: how far belief that *player* keeps this kind of promise
        falls if they break it now.

        Measured through the network, at the incentive they are acting under,
        because that is the number an observer will actually revise: the
        posterior moves from Beta(a, b) to Beta(a*d, b*d + w).
        """
        r = self.get_record(player, c_type)
        w = break_weight(incentive)
        a, b = r.alpha * EVIDENCE_DECAY, r.beta * EVIDENCE_DECAY
        before = self.p_keeps(player, c_type, incentive)
        after = p_keeps_given(a / (a + b + w), incentive)
        return max(0.0, before - after)

    def apply_gossip(self, gossip_sender: Player, accused: Player, c_type: CommitmentType):
        # Forward chaining rule base - Rule R3
        # If trust in gossip_sender is high, discount accused trust
        records = self.records.get(gossip_sender, {})
        sender_trust = (sum(r.get_expected_value() for r in records.values()) / len(records)
                        if records else self.prior_alpha / (self.prior_alpha + self.prior_beta))
        if sender_trust > 0.6:
            record = self.get_record(accused, c_type)
            record.alpha *= 0.9 # decrease alpha slightly to reflect suspicion
            record.beta += 0.1

    def general_trust(self, player: Player) -> float:
        """Average reliability across every kind of promise, prior if unseen."""
        records = self.records.get(player, {})
        if not records:
            return self.prior_alpha / (self.prior_alpha + self.prior_beta)
        return sum(r.get_expected_value() for r in records.values()) / len(records)

    def snapshot(self) -> List[BeliefSnapshot]:
        """Every belief this agent holds, flat, for the UI's trust matrix.

        Materialises the prior for pairs no evidence has touched yet: "I have
        never dealt with Gold on a DMZ, and here is what I assume about a
        stranger" is a belief, and a Paranoid seat assuming 0.25 is exactly
        what the panel is there to show.
        """
        return [
            BeliefSnapshot(
                observer=self.owner, subject=subject, commitment_type=c_type,
                alpha=(rec := self.get_record(subject, c_type)).alpha,
                beta_param=rec.beta,
                expected_reliability=rec.get_expected_value(),
            )
            for subject in Player
            for c_type in CommitmentType
        ]
