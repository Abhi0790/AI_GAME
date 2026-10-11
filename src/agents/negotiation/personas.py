"""Personas are parameter settings of the same code, not different code.

The knobs:

  reputation_cost  multiplies the Vcoop x deltaP x horizon penalty
  prior_alpha/beta the Beta prior every new opponent starts from
  vengeance        how much board value the planner will *pay* to hurt a
                   player who betrayed it (0 = never, 1 = a centre of damage
                   to a betrayer is worth a centre of my own)
  privacy          how readily it keeps a deal off the public record, where
                   neither side can later prove what was agreed
  policy           who it will never break a deal with, and who it will not
                   sign with (see Agent.policy):
                     unconditional  never breaks first; released, and signs
                                    nothing with them, only while wronged,
                                    and forgives quickly
                     reciprocal     never breaks first; free against whoever
                                    wronged it, and signs nothing with them
                                    until the grudge fades
                     trust          loyal to partners it believes, free to
                                    strike first at those it does not, and
                                    signs nothing with them
                     none           breaks whenever it pays; breaking with a
                                    partner who can hit back costs more
  forgive_below    a grudge under this no longer counts as being wronged
"""


class PersonaConfig:
    def __init__(self, name: str, rep_cost: float, prior_alpha: float, prior_beta: float,
                 vengeance: float = 0.0, privacy: float = 0.0,
                 policy: str = "none", forgive_below: float = 0.05):
        self.name = name
        self.reputation_cost = rep_cost
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.vengeance = vengeance
        self.privacy = privacy
        self.policy = policy
        self.forgive_below = forgive_below


PERSONAS = {
    # Pays five times the going rate for its standing.
    "Honest": PersonaConfig("Honest", 5.0, 1.0, 1.0, vengeance=0.0, privacy=0.0,
                            policy="unconditional", forgive_below=0.5),
    # Prices betrayal at face value and takes it when it pays.
    "Opportunist": PersonaConfig("Opportunist", 1.0, 1.0, 1.0, vengeance=0.0,
                                 privacy=0.8, policy="none"),
    # Keeps its word while you keep yours, then spends real value on revenge.
    "Vengeful": PersonaConfig("Vengeful", 2.0, 1.0, 1.0, vengeance=1.5, privacy=0.3,
                              policy="reciprocal", forgive_below=0.05),
    # Assumes the worst of strangers (Beta(1,3)) and is slow to be convinced.
    "Paranoid": PersonaConfig("Paranoid", 1.0, 1.0, 3.0, vengeance=0.3, privacy=0.6,
                              policy="trust"),
}
