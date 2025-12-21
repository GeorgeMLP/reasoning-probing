"""
Feature selection utilities for steering experiments.

This module provides methods to select features that are more likely to
represent genuine reasoning vs. shallow token correlations, based on the
methodology described in docs/methodology.md.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FeatureSelectionCriteria:
    """Criteria for selecting features for steering experiments."""
    
    # Reasoning detection thresholds
    min_auc: float = 0.6
    max_p_value: float = 0.01
    min_effect_size: float = 0.3
    
    # Token dependency thresholds
    max_token_concentration: float = 0.5  # Prefer low token dependency
    min_normalized_entropy: float = 0.3   # Prefer high entropy (diverse triggers)
    
    # Combined score weights
    weight_reasoning: float = 0.6  # Weight for reasoning score
    weight_token_independence: float = 0.4  # Weight for token independence


@dataclass 
class SelectedFeature:
    """A feature selected for steering experiments with metadata."""
    feature_index: int
    reasoning_score: float
    token_concentration: float
    normalized_entropy: float
    combined_score: float
    
    # Reasoning metrics
    roc_auc: float
    cohens_d: float
    
    def to_dict(self) -> dict:
        return {
            "feature_index": self.feature_index,
            "reasoning_score": self.reasoning_score,
            "token_concentration": self.token_concentration,
            "normalized_entropy": self.normalized_entropy,
            "combined_score": self.combined_score,
            "roc_auc": self.roc_auc,
            "cohens_d": self.cohens_d,
        }


class FeatureSelector:
    """
    Selects features for steering experiments based on reasoning strength
    and token independence.
    
    The key insight is that not all "reasoning features" are equally good
    for steering experiments. Features with high token dependency are likely
    to capture shallow correlations rather than genuine reasoning. We prefer
    features that:
    
    1. Show strong differential activation on reasoning text (high reasoning score)
    2. Have low token concentration (not dominated by specific tokens)
    3. Have high normalized entropy (respond to diverse token patterns)
    
    Combined Score:
        score = w_r * reasoning_score_norm + w_t * token_independence_score
        
    where:
        - reasoning_score_norm = reasoning_score / max(reasoning_scores)
        - token_independence_score = (1 - token_concentration) * normalized_entropy
    """
    
    def __init__(
        self,
        reasoning_features: list[dict],  # From reasoning_features.json
        token_analysis: dict,  # From token_analysis.json
        criteria: Optional[FeatureSelectionCriteria] = None,
    ):
        """
        Args:
            reasoning_features: List of feature dicts from find_reasoning_features.py
            token_analysis: Token analysis dict from find_reasoning_features.py
            criteria: Selection criteria (uses defaults if None)
        """
        self.reasoning_features = {f['feature_index']: f for f in reasoning_features}
        
        # Index token analysis by feature
        self.token_analysis = {}
        for feat in token_analysis.get('features', []):
            self.token_analysis[feat['feature_index']] = feat
        
        self.criteria = criteria or FeatureSelectionCriteria()
        
        # Compute combined scores
        self._compute_combined_scores()
    
    def _compute_combined_scores(self):
        """Compute combined scores for all features."""
        self.selected_features = []
        
        # Get max reasoning score for normalization
        reasoning_scores = [f['reasoning_score'] for f in self.reasoning_features.values()]
        max_rs = max(reasoning_scores) if reasoning_scores else 1.0
        
        # Compute combined score for each feature with both reasoning and token data
        for feat_idx, rf in self.reasoning_features.items():
            ta = self.token_analysis.get(feat_idx, {})
            
            token_concentration = ta.get('token_concentration', 1.0)  # Default high if no data
            normalized_entropy = ta.get('normalized_entropy', 0.0)  # Default low if no data
            
            # Normalized reasoning score
            rs_norm = rf['reasoning_score'] / max_rs if max_rs > 0 else 0
            
            # Token independence score: high when concentration is low and entropy is high
            token_ind_score = (1 - token_concentration) * (normalized_entropy + 0.1)  # +0.1 to avoid zero
            token_ind_score = min(token_ind_score, 1.0)  # Cap at 1
            
            # Combined score
            combined = (
                self.criteria.weight_reasoning * rs_norm +
                self.criteria.weight_token_independence * token_ind_score
            )
            
            self.selected_features.append(SelectedFeature(
                feature_index=feat_idx,
                reasoning_score=rf['reasoning_score'],
                token_concentration=token_concentration,
                normalized_entropy=normalized_entropy,
                combined_score=combined,
                roc_auc=rf['roc_auc'],
                cohens_d=rf['cohens_d'],
            ))
        
        # Sort by combined score
        self.selected_features.sort(key=lambda x: x.combined_score, reverse=True)
    
    def get_top_features(
        self,
        k: int = 20,
        require_low_token_dependency: bool = True,
    ) -> list[SelectedFeature]:
        """
        Get top k features for steering experiments.
        
        Args:
            k: Number of features to return
            require_low_token_dependency: If True, only include features with
                token_concentration < max_token_concentration
        
        Returns:
            List of SelectedFeature objects
        """
        candidates = self.selected_features
        
        if require_low_token_dependency:
            candidates = [
                f for f in candidates
                if f.token_concentration < self.criteria.max_token_concentration
            ]
        
        return candidates[:k]
    
    def get_features_by_reasoning_only(self, k: int = 20) -> list[SelectedFeature]:
        """Get top k features by reasoning score only (ignoring token dependency)."""
        sorted_by_reasoning = sorted(
            self.selected_features,
            key=lambda x: x.reasoning_score,
            reverse=True
        )
        return sorted_by_reasoning[:k]
    
    def get_features_by_token_independence(self, k: int = 20) -> list[SelectedFeature]:
        """Get features with lowest token dependency among reasoning features."""
        sorted_by_token_ind = sorted(
            self.selected_features,
            key=lambda x: x.token_concentration
        )
        return sorted_by_token_ind[:k]
    
    def compare_selection_strategies(self, k: int = 20) -> dict:
        """
        Compare different feature selection strategies.
        
        Returns statistics about overlap and characteristics of features
        selected by different methods.
        """
        combined = set(f.feature_index for f in self.get_top_features(k, require_low_token_dependency=False))
        combined_filtered = set(f.feature_index for f in self.get_top_features(k, require_low_token_dependency=True))
        reasoning_only = set(f.feature_index for f in self.get_features_by_reasoning_only(k))
        token_ind = set(f.feature_index for f in self.get_features_by_token_independence(k))
        
        return {
            "overlap_combined_vs_reasoning": len(combined & reasoning_only) / k,
            "overlap_combined_vs_token_ind": len(combined & token_ind) / k,
            "overlap_reasoning_vs_token_ind": len(reasoning_only & token_ind) / k,
            "filtered_available": len(combined_filtered),
            "avg_reasoning_score_combined": np.mean([f.reasoning_score for f in self.get_top_features(k, False)]),
            "avg_reasoning_score_reasoning_only": np.mean([f.reasoning_score for f in self.get_features_by_reasoning_only(k)]),
            "avg_token_concentration_combined": np.mean([f.token_concentration for f in self.get_top_features(k, False)]),
            "avg_token_concentration_reasoning_only": np.mean([f.token_concentration for f in self.get_features_by_reasoning_only(k)]),
        }
    
    def summary(self) -> dict:
        """Generate summary statistics about feature selection."""
        all_features = self.selected_features
        
        high_reasoning = [f for f in all_features if f.reasoning_score > 0.5]
        low_token_dep = [f for f in all_features if f.token_concentration < 0.5]
        both = [f for f in all_features if f.reasoning_score > 0.5 and f.token_concentration < 0.5]
        
        return {
            "total_features": len(all_features),
            "high_reasoning_score": len(high_reasoning),
            "low_token_dependency": len(low_token_dep),
            "high_reasoning_and_low_token_dependency": len(both),
            "mean_combined_score": np.mean([f.combined_score for f in all_features]),
            "correlation_reasoning_vs_token_conc": np.corrcoef(
                [f.reasoning_score for f in all_features],
                [f.token_concentration for f in all_features]
            )[0, 1] if len(all_features) > 2 else 0.0,
        }


def load_and_select_features(
    reasoning_features_path: str,
    token_analysis_path: str,
    top_k: int = 20,
    criteria: Optional[FeatureSelectionCriteria] = None,
) -> list[int]:
    """
    Convenience function to load data and select features for steering.
    
    Args:
        reasoning_features_path: Path to reasoning_features.json
        token_analysis_path: Path to token_analysis.json
        top_k: Number of features to select
        criteria: Selection criteria
    
    Returns:
        List of feature indices
    """
    import json
    
    with open(reasoning_features_path) as f:
        rf_data = json.load(f)
    
    with open(token_analysis_path) as f:
        ta_data = json.load(f)
    
    selector = FeatureSelector(
        reasoning_features=rf_data.get('features', []),
        token_analysis=ta_data,
        criteria=criteria,
    )
    
    selected = selector.get_top_features(top_k)
    
    print(f"Feature Selection Summary:")
    summary = selector.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\nSelected {len(selected)} features:")
    for f in selected[:10]:
        print(f"  Feature {f.feature_index}: combined={f.combined_score:.3f}, "
              f"reasoning={f.reasoning_score:.3f}, token_conc={f.token_concentration:.3f}")
    
    return [f.feature_index for f in selected]
