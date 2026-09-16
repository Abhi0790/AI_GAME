"""Personas are parameter settings of the same code, not different code.

Four knobs, each one wired to something the planner or the trust model
already does:

  reputation_cost  multiplies the Vcoop x deltaP x horizon penalty
  prior_alpha/beta the Beta prior every new opponent starts from
  vengeance        how much board value the planner will *pay* to hurt a
                   player who betrayed it (0 = never, 1 = a centre of damage
                   to a betrayer is worth a centre of my own)
  deception        willingness to broadcast an accusation it knows is false
"""


class PersonaConfig:
    def __init__(self, name: str, rep_cost: float, prior_alpha: float, prior_beta: float,
                 vengeance: float = 0.0, deception: float = 0.0):
        self.name = name
        self.reputation_cost = rep_cost
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.vengeance = vengeance
        self.deception = deception


PERSONAS = {
    # Pays a reputation cost so large that no board gain ever clears it.
    "Honest": PersonaConfig("Honest", 100.0, 1.0, 1.0, vengeance=0.0, deception=0.0),
    # Prices betrayal at face value and takes it when it pays.
    "Opportunist": PersonaConfig("Opportunist", 1.0, 1.0, 1.0, vengeance=0.0, deception=0.6),
    # Keeps its word while you keep yours, then spends real value on revenge.
    "Vengeful": PersonaConfig("Vengeful", 2.0, 1.0, 1.0, vengeance=1.5, deception=0.2),
    # Assumes the worst of strangers (Beta(1,3)) and is slow to be convinced.
    "Paranoid": PersonaConfig("Paranoid", 1.0, 1.0, 3.0, vengeance=0.3, deception=0.1),
}
