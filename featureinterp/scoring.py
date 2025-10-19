from __future__ import annotations

from typing import Sequence

import numpy as np
from featureinterp.formatting import FormattedFeatureExpressionRecord
from featureinterp.record import FeatureExpressionRecord
from featureinterp.core import (
    ScoredSequenceSimulation,
    ScoredSimulation,
)
from featureinterp.simulator import FeatureSimulator


def correlation_score(
    real_expressions: Sequence[float] | np.ndarray,
    predicted_expressions: Sequence[float] | np.ndarray,
) -> float:
    
    if np.var(real_expressions) < 1e-10:
        return float('nan')

    return float(np.corrcoef(real_expressions, predicted_expressions)[0, 1])


def rsquared_score(
    real_expressions: Sequence[float] | np.ndarray,
    predicted_expressions: Sequence[float] | np.ndarray,
) -> float:
    
    if np.var(real_expressions) < 1e-10:
        return float('nan')
    
    return float(
        1
        - np.mean(np.square(np.array(real_expressions) - np.array(predicted_expressions)))
        / np.mean(np.square(np.array(real_expressions)))
    )


def absolute_dev_explained_score(
    real_expressions: Sequence[float] | np.ndarray,
    predicted_expressions: Sequence[float] | np.ndarray,
) -> float:

    if np.var(real_expressions) < 1e-10:
        return float('nan')

    return float(
        1
        - np.mean(np.abs(np.array(real_expressions) - np.array(predicted_expressions)))
        / np.mean(np.abs(np.array(real_expressions)))
    )


def aggregate_scored_sequence_simulations(
    scored_sequence_simulations: list[ScoredSequenceSimulation],
) -> ScoredSimulation:
    """
    Aggregate a list of scored sequence simulations. The logic for doing this is non-trivial for EV
    scores, since we want to calculate the correlation over all activations from all sequences at
    once rather than simply averaging per-sequence correlations.
    """
    all_true: list[float] = []
    all_pred: list[float] = []
    for scored_sequence_simulation in scored_sequence_simulations:
        all_true.extend(scored_sequence_simulation.true_expressions or [])
        all_pred.extend(scored_sequence_simulation.simulation.expected_expressions)
    
    return ScoredSimulation(
        scored_sequence_simulations=scored_sequence_simulations,
        correlation_score=correlation_score(all_true, all_pred),
        rsquared_score=rsquared_score(all_true, all_pred),
        absolute_dev_explained_score=absolute_dev_explained_score(all_true, all_pred),
    )


async def simulate_and_score(
    simulator: FeatureSimulator,
    records: list[FeatureExpressionRecord] | list[FormattedFeatureExpressionRecord],
) -> ScoredSimulation:
    """Score an explanation by how well it predicts activations on the given text."""
    
    simulations = simulator.simulate([r.tokens for r in records])
    scored_sequence_simulations = []
    for simulation, record in zip(simulations, records):
        scored_sequence_simulations.append(ScoredSequenceSimulation(
            simulation=simulation,
            true_expressions=record.expressions,
            record=record,
            correlation_score=correlation_score(
                record.expressions, simulation.expected_expressions
            ),
            rsquared_score=rsquared_score(
                record.expressions, simulation.expected_expressions
            ),
            absolute_dev_explained_score=absolute_dev_explained_score(
                record.expressions, simulation.expected_expressions
            ),
        ))
        
    return aggregate_scored_sequence_simulations(scored_sequence_simulations)
