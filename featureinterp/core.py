from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional
import dacite
import json_repair

from featureinterp.record import FeatureExpressionRecord, SAEIndexId


STRUCTURED_EXPLANATION_PREFIX = "This feature activates for the following list of rules:\n"
STRING_EXPLANATION_PREFIX = "the main thing this neuron does is find "


@dataclass(frozen=True)
class ExplanationComponent:
    """A single component of a feature's behavior explanation."""
    activates_on: str
    """What specific pattern or content the feature activates on"""
    strength: int
    """The strength of the activation, from 0 to 5."""


@dataclass
class StructuredExplanation:
    """A structured explanation of a feature's behavior."""

    components: list[ExplanationComponent]
    """List of different activation rules this feature responds to."""

    def to_json(self) -> str:
        return json.dumps([asdict(c) for c in self.components], indent="\t")

    @classmethod
    def from_json(cls, json_str: str) -> "StructuredExplanation" | None:
        """Create from JSON string format"""
        try:
            components = []
            data = json_repair.loads(json_str)
            for item in data:
                components.append(dacite.from_dict(
                    data_class=ExplanationComponent,
                    data=item
                ))
            return cls(components=components)
        except Exception as e:
            # print(f"Failed to parse explanation JSON: {e}")
            return None


@dataclass
class SequenceSimulation:
    """The result of a simulation of feature expressions on one text sequence."""

    tokens: list[str]
    """The sequence of tokens that was simulated."""
    expected_expressions: list[float]
    """Expected value of the feature expression for each token in the sequence."""

    distribution_values: list[list[float]]
    """
    For each token in the sequence, a list of values from the discrete distribution of activations
    produced from simulation.
    
    When we simulate a feature expression, we produce a
    discrete distribution with values in the arbitrary discretized space, e.g. 10%
    chance of 0, 70% chance of 1, 20% chance of 2. Which we store as distribution_values =
    [0, 1, 2], distribution_probabilities = [0.1, 0.7, 0.2].
    """
    distribution_probabilities: list[list[float]]
    """
    For each token in the sequence, the probability of the corresponding value in
    distribution_values.
    """


@dataclass
class ScoredSequenceSimulation:
    """The result of a simulation of feature expressions on one text sequence."""

    simulation: SequenceSimulation
    """The result of a simulation of feature expressions."""
    true_expressions: list[float]
    """Ground truth expressions on the sequence (not normalized)"""
    record: FeatureExpressionRecord
    """The record that was used to generate this simulation."""

    correlation_score: float
    """
    Correlation coefficient between the expected values of the formatted expressions
    from the simulation and the unnormalized true expressions of the feature.
    """
    rsquared_score: Optional[float] = None
    """R^2 of the simulated expressions."""
    absolute_dev_explained_score: Optional[float] = None
    """
    Score based on absolute difference between real and simulated expressions.
    absolute_dev_explained_score = 1 - mean(abs(real-predicted))/ mean(abs(real))
    """


@dataclass
class ScoredSimulation:
    """Result of scoring a feature expression simulation on multiple sequences."""

    scored_sequence_simulations: list[ScoredSequenceSimulation]
    """ScoredSequenceSimulation for each sequence"""
    correlation_score: Optional[float] = None
    """
    Correlation coefficient between the expected values of the formatted expressions
    from the simulation and the unnormalized true expressions on a dataset created from
    all scored sequence simulations. (Note that this is not equivalent to averaging
    across sequences.)
    """
    rsquared_score: Optional[float] = None
    """R^2 of the simulated expressions."""
    absolute_dev_explained_score: Optional[float] = None
    """
    Score based on absolute difference between real and simulated expressions.
    absolute_dev_explained_score = 1 - mean(abs(real-predicted))/ mean(abs(real)).
    """

    def get_preferred_score(self) -> Optional[float]:
        return self.correlation_score


@dataclass
class ScoredExplanation:
    """Simulator parameters and the results of scoring it on multiple sequences"""

    explanation: StructuredExplanation | str
    """The explanation used for simulation."""

    scored_simulation: ScoredSimulation
    """Result of scoring the feature simulator on multiple sequences."""

    def get_preferred_score(self) -> Optional[float]:
        return self.scored_simulation.get_preferred_score()


@dataclass
class SAEIndexSimulationResults:
    """Simulation results and scores for a feature."""

    sae_index_id: SAEIndexId
    scored_explanations: list[ScoredExplanation]
