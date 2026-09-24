import math
from typing import Dict, List, Tuple, Optional
from src.common.schemas import (
    Player, CommitmentType, Commitment, CommitmentOutcome, BeliefSnapshot,
)
from src.engine.board import players

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
# Keeps is the child, and this is its conditional table:
#
#   P(keeps | r, iota) = BN_CEILING * sigmoid(BN_BIAS + BN_SLOPE * r)
#                        * (1 - BN_TEMPTATION * min(1, iota / BN_INCENTIVE_SCALE))
#
# A ceiling (most promises are kept), a drop at low reliability (a proven
# betrayer is refused), and a discount for what is on offer. The old table,
# floor + (1 - floor) * r, could not do the first two at once: once personas
# kept 93% of promises, the floor that still refused a three-time betrayer
# (<= 0.404) put every stranger near 0.7 and scored Brier 0.17 against 0.07 for
# the base rate.
#
# BN_CEILING, BN_BIAS and BN_TEMPTATION are FITTED by src/evaluation/sweep.fit_cpt
# on logged (reliability, incentive, kept?) triples, under the rule that a
# partner who broke three promises in a row falls below even odds. BN_SLOPE is
# judgement: almost no graded promise comes from a partner that unreliable, so
# the data cannot say how steep the drop is.
BN_CEILING = 0.96
BN_BIAS = -1.213
BN_SLOPE = 8.0
BN_TEMPTATION = 0.042
# Incentive is measured in centres of immediate gain, and a whole centre on
# offer is rare — most live deals sit near zero. iota at or above the scale
# counts as maximum temptation.
BN_INCENTIVE_SCALE = 0.1

# How much pair-specific evidence it takes before "X keeps promises to *me*"
# outweighs "X keeps promises". Whether Blue honours deals with Red is a
# different question from whether Blue honours deals, and it is the one that
# decides whether Red should sign; but one observation is not a pattern, so
# the pair record is shrunk toward the general one until it has some weight
# behind it.
PAIR_SHRINKAGE = 2.0


def set_parameters(lambda_incentive: Optional[float] = None,
                   evidence_decay: Optional[float] = None):
    """Rebind the global constants. The parameter sweep drives this."""
    global LAMBDA_INCENTIVE, EVIDENCE_DECAY
    if lambda_incentive is not None:
        LAMBDA_INCENTIVE = lambda_incentive
    if evidence_decay is not None:
        EVIDENCE_DECAY = evidence_decay


def break_weight(incentive: float) -> float:
    """w in "broken: beta + w". A gratuitous betrayal costs a full point of
    reputation; a lucrative one costs less, because it explains itself."""
    incentive = min(1.0, max(0.0, incentive))
    return max(MIN_BREAK_WEIGHT, 1.0 - LAMBDA_INCENTIVE * incentive)


def p_keeps_given(reliability: float, incentive: float,
                  ceiling: Optional[float] = None, bias: Optional[float] = None,
                  temptation: Optional[float] = None) -> float:
    """The Keeps node's CPT.

    Pure function so the sweep and the tests can hit it without a model."""
    ceiling = BN_CEILING if ceiling is None else ceiling
    bias = BN_BIAS if bias is None else bias
    temptation = BN_TEMPTATION if temptation is None else temptation

    r = min(1.0, max(0.0, reliability))
    t = min(1.0, max(0.0, incentive / BN_INCENTIVE_SCALE))
    return ceiling / (1.0 + math.exp(-(bias + BN_SLOPE * r))) * (1.0 - temptation * t)


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
    """Who keeps promises, in general and to whom.

    Two layers of evidence. `records` pools everything observed about a player
    regardless of who they were dealing with — that is their reputation.
    `pair_records` keeps the same thing per (player, counterparty), because
    "Blue honours deals with Red" and "Blue honours deals" are different
    claims and coalitions turn on the first one. Predictions blend the two,
    weighted by how much pair-specific evidence there actually is.
    """

    def __init__(self, owner: Player, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.owner = owner
        # target_player -> commitment_type -> TrustRecord  (reputation)
        self.records: Dict[Player, Dict[CommitmentType, TrustRecord]] = {}
        # (target_player, counterparty) -> commitment_type -> TrustRecord
        self.pair_records: Dict[Tuple[Player, Player], Dict[CommitmentType, TrustRecord]] = {}
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def get_record(self, player: Player, c_type: CommitmentType) -> TrustRecord:
        if player not in self.records:
            self.records[player] = {}
        if c_type not in self.records[player]:
            self.records[player][c_type] = TrustRecord(self.prior_alpha, self.prior_beta)
        return self.records[player][c_type]

    def get_pair_record(self, player: Player, counterparty: Player,
                        c_type: CommitmentType) -> TrustRecord:
        key = (player, counterparty)
        if key not in self.pair_records:
            self.pair_records[key] = {}
        if c_type not in self.pair_records[key]:
            self.pair_records[key][c_type] = TrustRecord(self.prior_alpha, self.prior_beta)
        return self.pair_records[key][c_type]

    def _pair_weight(self, record: TrustRecord) -> float:
        """How much to trust the pair record over the general one."""
        evidence = max(0.0, record.alpha + record.beta
                       - (self.prior_alpha + self.prior_beta))
        return evidence / (evidence + PAIR_SHRINKAGE)

    def get_reliability(self, player: Player, c_type: CommitmentType,
                        toward: Optional[Player] = None) -> float:
        """The Reliability node: the Beta posterior mean.

        With *toward*, the pair record shrunk toward the general one, so a
        player who has been straight with me but treacherous with everybody
        else reads differently depending on who is asking.
        """
        general = self.get_record(player, c_type).get_expected_value()
        if toward is None or toward == player:
            return general
        pair = self.get_pair_record(player, toward, c_type)
        w = self._pair_weight(pair)
        return w * pair.get_expected_value() + (1.0 - w) * general

    def p_keeps(self, player: Player, c_type: CommitmentType,
                incentive: float = 0.0, toward: Optional[Player] = None) -> float:
        """P(keeps this commitment) — the whole network, not one node.

        Reliability says what they usually do; incentive says what the board
        is offering them this turn. A perfectly reliable partner standing
        next to an undefended centre of mine is not a safe bet, and the Beta
        mean on its own cannot say so.
        """
        return p_keeps_given(self.get_reliability(player, c_type, toward), incentive)

    def observe(self, subject: Player, counterparties, c_type: CommitmentType,
                kept: bool, discount: float = 1.0):
        """Record one graded promise against the general and pair records.

        A pact with three members produces one general observation and one per
        counterparty, so betraying two people at once costs twice with each of
        them and once with the table.
        """
        self.get_record(subject, c_type).update(kept=kept, discount=discount)
        for other in counterparties:
            if other != subject:
                self.get_pair_record(subject, other, c_type).update(
                    kept=kept, discount=discount)

    def update_from_outcome(self, outcome: CommitmentOutcome, incentive_to_defect: float = 0.0):
        """Apply one graded commitment to the Beta records.

        Own record included on purpose: the engine publishes every outcome, so
        every player scores my record off the same public evidence. That record
        is the best estimate I have of how much reputation I stand to lose,
        which is what the planner prices a betrayal against.
        """
        players = outcome.commitment.players
        for p in players:
            broke = p in outcome.broken_by
            self.observe(p, players, outcome.commitment.commitment_type,
                         kept=not broke,
                         discount=break_weight(incentive_to_defect) if broke else 1.0)

    def reputation_drop(self, player: Player, c_type: CommitmentType,
                        incentive: float = 0.0,
                        toward: Optional[Player] = None) -> float:
        """Delta-P: how far belief that *player* keeps this kind of promise
        falls if they break it now.

        Measured through the network, at the incentive they are acting under,
        because that is the number an observer will actually revise: the
        posterior moves from Beta(a, b) to Beta(a*d, b*d + w). Both layers
        move. The CPT is flat where reliability is high, so a first lapse with
        a long-trusted partner costs little and a pattern costs a lot.
        """
        w = break_weight(incentive)
        before = self.p_keeps(player, c_type, incentive, toward)

        def broken_mean(rec: TrustRecord) -> float:
            a, b = rec.alpha * EVIDENCE_DECAY, rec.beta * EVIDENCE_DECAY
            return a / (a + b + w)

        general_after = broken_mean(self.get_record(player, c_type))
        if toward is None or toward == player:
            after_reliability = general_after
        else:
            pair = self.get_pair_record(player, toward, c_type)
            pw = self._pair_weight(pair)
            after_reliability = pw * broken_mean(pair) + (1.0 - pw) * general_after

        after = p_keeps_given(after_reliability, incentive)
        return max(0.0, before - after)

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
                reliability_toward_observer=(
                    None if subject == self.owner
                    else self.get_reliability(subject, c_type, toward=self.owner)),
            )
            for subject in players()
            for c_type in CommitmentType
        ]
