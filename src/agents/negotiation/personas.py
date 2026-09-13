"""Personas are parameter settings of the same code, not different code."""

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
