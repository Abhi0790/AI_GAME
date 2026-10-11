import math
from typing import Dict, List, Tuple, Optional
from src.common.schemas import (
    Player, CommitmentType, CommitmentOutcome, BeliefSnapshot,
)
from src.engine.board import players

# Break weight w = 1 - lambda * incentive: how much of a profitable betrayal's
# reputation hit is forgiven. Fitted by scripts/run_sweep.py.
LAMBDA_INCENTIVE = 0.2
MIN_BREAK_WEIGHT = 0.1

# Evidence decays before each new observation, so a record never stops moving.
EVIDENCE_DECAY = 0.85
# Evidence is never allowed to decay below the prior's worth of it.
MIN_EVIDENCE = 1e-6

# Bayesian network Reliability -> Keeps <- Incentive, with CPT
#   P(keeps | r, iota) = BN_CEILING * sigmoid(BN_BIAS + BN_SLOPE * r)
#                        * (1 - BN_TEMPTATION * min(1, iota / BN_INCENTIVE_SCALE))
# CEILING, BIAS and TEMPTATION fitted by sweep.fit_cpt; SLOPE is set by hand.
BN_CEILING = 0.96
BN_BIAS = -1.213
BN_SLOPE = 8.0
BN_TEMPTATION = 0.042
# iota (centres of immediate gain) at or above this counts as maximum temptation.
BN_INCENTIVE_SCALE = 0.1
# Prior of the "keeps when tempted" record; 0 for both turns it off.
TEMPTED_PRIOR_ALPHA = 4.0
TEMPTED_PRIOR_BETA = 1.0

# Pseudo-observations shrinking a pair record toward the general one.
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
                  temptation: Optional[float] = None,
                  tempted_keep: Optional[float] = None) -> float:
    """The Keeps node's CPT; *tempted_keep* replaces the constant temptation
    discount with the player's own record."""
    ceiling = BN_CEILING if ceiling is None else ceiling
    bias = BN_BIAS if bias is None else bias
    temptation = BN_TEMPTATION if temptation is None else temptation

    r = min(1.0, max(0.0, reliability))
    t = min(1.0, max(0.0, incentive / BN_INCENTIVE_SCALE))
    resist = 1.0 - temptation if tempted_keep is None else tempted_keep
    return ceiling / (1.0 + math.exp(-(bias + BN_SLOPE * r))) * (1.0 - t * (1.0 - resist))


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
    """Who keeps promises: `records` per player (reputation) and `pair_records`
    per (player, counterparty), blended by how much pair evidence exists."""

    def __init__(self, owner: Player, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.owner = owner
        # target_player -> commitment_type -> TrustRecord  (reputation)
        self.records: Dict[Player, Dict[CommitmentType, TrustRecord]] = {}
        # (target_player, counterparty) -> commitment_type -> TrustRecord
        self.pair_records: Dict[Tuple[Player, Player], Dict[CommitmentType, TrustRecord]] = {}
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        # target_player -> keeps-when-tempted, first-hand only, all promise kinds
        self.tempted: Dict[Player, TrustRecord] = {}

    def tempted_record(self, player: Player) -> TrustRecord:
        if player not in self.tempted:
            self.tempted[player] = TrustRecord(TEMPTED_PRIOR_ALPHA, TEMPTED_PRIOR_BETA)
        return self.tempted[player]

    def tempted_keep(self, player: Player) -> Optional[float]:
        """How often *player* kept a promise when breaking paid; None when the
        record is switched off."""
        if TEMPTED_PRIOR_ALPHA + TEMPTED_PRIOR_BETA <= 0:
            return None
        return self.tempted_record(player).get_expected_value()

    def export_reputation(self, personas: Dict[Player, str]) -> Dict[str, Dict[str, List[float]]]:
        """What I know of each player, keyed by persona so it can follow them
        into another game where they sit in a different seat."""
        out: Dict[str, Dict[str, List[float]]] = {}
        for p, recs in self.records.items():
            for c_type, r in recs.items():
                if p in personas:
                    out.setdefault(personas[p], {})[c_type.value] = [r.alpha, r.beta]
        for p, r in self.tempted.items():
            if p in personas:
                out.setdefault(personas[p], {})["tempted"] = [r.alpha, r.beta]
        return out

    def import_reputation(self, ledger: Dict[str, Dict[str, List[float]]],
                          personas: Dict[Player, str]):
        """Start from what an earlier game ended with."""
        for p, name in personas.items():
            for key, (a, b) in ledger.get(name, {}).items():
                rec = (self.tempted_record(p) if key == "tempted"
                       else self.get_record(p, CommitmentType(key)))
                rec.alpha, rec.beta = a, b

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
        """The Reliability node: the Beta posterior mean, or with *toward* the
        pair record shrunk toward the general one."""
        general = self.get_record(player, c_type).get_expected_value()
        if toward is None or toward == player:
            return general
        pair = self.get_pair_record(player, toward, c_type)
        w = self._pair_weight(pair)
        return w * pair.get_expected_value() + (1.0 - w) * general

    def p_keeps(self, player: Player, c_type: CommitmentType,
                incentive: float = 0.0, toward: Optional[Player] = None) -> float:
        """P(keeps this commitment) from reliability and this turn's incentive."""
        return p_keeps_given(self.get_reliability(player, c_type, toward), incentive,
                             tempted_keep=self.tempted_keep(player))

    def observe(self, subject: Player, counterparties, c_type: CommitmentType,
                kept: bool, discount: float = 1.0):
        """Record one graded promise: once in the general record, once per counterparty."""
        self.get_record(subject, c_type).update(kept=kept, discount=discount)
        for other in counterparties:
            if other != subject:
                self.get_pair_record(subject, other, c_type).update(
                    kept=kept, discount=discount)

    def update_from_outcome(self, outcome: CommitmentOutcome, incentive_to_defect: float = 0.0):
        """Apply one graded commitment to the Beta records, my own included."""
        players = outcome.commitment.players
        for p in players:
            broke = p in outcome.broken_by
            self.observe(p, players, outcome.commitment.commitment_type,
                         kept=not broke,
                         discount=break_weight(incentive_to_defect) if broke else 1.0)

    def reputation_drop(self, player: Player, c_type: CommitmentType,
                        incentive: float = 0.0,
                        toward: Optional[Player] = None) -> float:
        """Delta-P: how far P(*player* keeps this kind of promise) falls if they
        break it now, Beta(a, b) -> Beta(a*d, b*d + w), through the CPT."""
        w = break_weight(incentive)
        # The ablation replaces the forecast, not the price of my own betrayal.
        tempted = self.tempted_keep(player)
        before = p_keeps_given(self.get_reliability(player, c_type, toward), incentive,
                               tempted_keep=tempted)

        def broken_mean(rec: TrustRecord, weight: float = w) -> float:
            a, b = rec.alpha * EVIDENCE_DECAY, rec.beta * EVIDENCE_DECAY
            return a / (a + b + weight)

        general_after = broken_mean(self.get_record(player, c_type))
        if toward is None or toward == player:
            after_reliability = general_after
        else:
            pair = self.get_pair_record(player, toward, c_type)
            pw = self._pair_weight(pair)
            after_reliability = pw * broken_mean(pair) + (1.0 - pw) * general_after

        # A break while tempted is evidence about exactly that, so the
        # tempted record falls too, by a full break.
        if tempted is not None and incentive > 0:
            tempted = broken_mean(self.tempted_record(player), 1.0)
        after = p_keeps_given(after_reliability, incentive, tempted_keep=tempted)
        return max(0.0, before - after)

    def general_trust(self, player: Player) -> float:
        """Average reliability across every kind of promise, prior if unseen."""
        records = self.records.get(player, {})
        if not records:
            return self.prior_alpha / (self.prior_alpha + self.prior_beta)
        return sum(r.get_expected_value() for r in records.values()) / len(records)

    def snapshot(self) -> List[BeliefSnapshot]:
        """Every belief this agent holds, priors included, for the UI's trust matrix."""
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


class ConstantForecastTrustModel(TrustModel):
    """Control arm: one held-out P(keeps), with ordinary belief updates."""

    def __init__(self, owner: Player, probability: float,
                 prior_alpha: float = 1.0, prior_beta: float = 1.0):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("constant keep forecast must be in [0, 1]")
        super().__init__(owner, prior_alpha, prior_beta)
        self.probability = probability

    def p_keeps(self, player: Player, c_type: CommitmentType,
                incentive: float = 0.0, toward: Optional[Player] = None) -> float:
        return self.probability
