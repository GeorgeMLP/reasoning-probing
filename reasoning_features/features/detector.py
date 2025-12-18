"""
Reasoning feature detection with rigorous statistical metrics.

This module implements multiple statistical tests to identify SAE features
that show differential activation between reasoning and non-reasoning text.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from torch import Tensor
from jaxtyping import Float
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.metrics import roc_auc_score
import warnings

from .collector import FeatureActivations


@dataclass
class FeatureStats:
    """
    Statistical analysis results for a single feature.
    
    Contains multiple metrics to assess whether the feature shows
    differential activation between reasoning and non-reasoning text.
    """
    feature_index: int
    
    # Basic statistics
    mean_reasoning: float  # Mean activation on reasoning samples
    mean_nonreasoning: float  # Mean activation on non-reasoning samples
    std_reasoning: float  # Std on reasoning samples
    std_nonreasoning: float  # Std on non-reasoning samples
    
    # Effect size metrics
    cohens_d: float  # Standardized mean difference
    log_fold_change: float  # log2(mean_reasoning / mean_nonreasoning)
    
    # Statistical tests
    mannwhitney_u: float  # Mann-Whitney U statistic
    mannwhitney_p: float  # Mann-Whitney p-value (two-sided)
    ttest_t: float  # Welch's t-test statistic
    ttest_p: float  # Welch's t-test p-value
    
    # Classification metrics
    roc_auc: float  # ROC-AUC for separating reasoning vs non-reasoning
    
    # Activation frequency
    freq_active_reasoning: float  # Fraction of reasoning samples where feature fires
    freq_active_nonreasoning: float  # Fraction of non-reasoning samples where feature fires
    
    # Composite score (higher = more reasoning-correlated)
    reasoning_score: float = field(default=0.0)
    
    def __post_init__(self):
        """Compute composite reasoning score."""
        # Combine multiple metrics into a single score
        # Prioritize: high AUC, large effect size, low p-value, high activation in reasoning
        
        # Direction-aware metrics (positive = higher in reasoning)
        direction = 1 if self.mean_reasoning > self.mean_nonreasoning else -1
        
        # AUC contribution (0.5 = random, 1.0 = perfect separation)
        auc_contrib = abs(self.roc_auc - 0.5) * 2  # Scale to 0-1
        
        # Effect size contribution (clipped Cohen's d)
        effect_contrib = min(abs(self.cohens_d), 3.0) / 3.0  # Cap at d=3
        
        # P-value contribution (log scale)
        p_contrib = min(-np.log10(self.mannwhitney_p + 1e-300), 50) / 50  # Cap at p=1e-50
        
        # Frequency contribution (prefer features active in reasoning but not in baseline)
        freq_ratio = (self.freq_active_reasoning + 0.01) / (self.freq_active_nonreasoning + 0.01)
        freq_contrib = min(np.log2(freq_ratio + 1) / 5, 1.0) if freq_ratio > 1 else 0
        
        # Weighted combination
        self.reasoning_score = direction * (
            0.3 * auc_contrib +
            0.25 * effect_contrib +
            0.25 * p_contrib +
            0.2 * freq_contrib
        )
    
    def is_reasoning_feature(
        self,
        min_auc: float = 0.6,
        max_p_value: float = 0.01,
        min_effect_size: float = 0.3,
    ) -> bool:
        """Check if this feature qualifies as a reasoning feature."""
        return (
            self.roc_auc >= min_auc and
            self.mannwhitney_p <= max_p_value and
            abs(self.cohens_d) >= min_effect_size and
            self.mean_reasoning > self.mean_nonreasoning
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "feature_index": self.feature_index,
            "mean_reasoning": self.mean_reasoning,
            "mean_nonreasoning": self.mean_nonreasoning,
            "std_reasoning": self.std_reasoning,
            "std_nonreasoning": self.std_nonreasoning,
            "cohens_d": self.cohens_d,
            "log_fold_change": self.log_fold_change,
            "mannwhitney_u": self.mannwhitney_u,
            "mannwhitney_p": self.mannwhitney_p,
            "ttest_t": self.ttest_t,
            "ttest_p": self.ttest_p,
            "roc_auc": self.roc_auc,
            "freq_active_reasoning": self.freq_active_reasoning,
            "freq_active_nonreasoning": self.freq_active_nonreasoning,
            "reasoning_score": self.reasoning_score,
        }


class ReasoningFeatureDetector:
    """
    Detects features that show differential activation between reasoning
    and non-reasoning text using multiple statistical metrics.
    
    ## Metrics Used
    
    1. **Cohen's d (Effect Size)**: Standardized difference between group means.
       - d > 0.2: small effect
       - d > 0.5: medium effect  
       - d > 0.8: large effect
    
    2. **ROC-AUC**: Area under the ROC curve for binary classification.
       - 0.5: random chance
       - 0.7+: acceptable discrimination
       - 0.8+: good discrimination
    
    3. **Mann-Whitney U Test**: Non-parametric test for distribution differences.
       - P < 0.05: significant difference
       - P < 0.01: highly significant
    
    4. **Welch's t-test**: Parametric test (as sanity check).
    
    5. **Activation Frequency**: How often the feature fires in each group.
    
    6. **Composite Reasoning Score**: Weighted combination of all metrics.
    """
    
    def __init__(
        self,
        activations: FeatureActivations,
        aggregation: str = "max",
    ):
        """
        Args:
            activations: Collected feature activations
            aggregation: How to aggregate across sequence dimension
                - "max": Maximum activation per sample (default)
                - "mean": Mean activation per sample
                - "sum": Sum of activations per sample
        """
        self.activations = activations
        self.aggregation = aggregation
        
        # Get aggregated activations
        if aggregation == "max":
            self.agg_acts = activations.get_max_activations()
        elif aggregation == "mean":
            self.agg_acts = activations.get_mean_activations()
        elif aggregation == "sum":
            self.agg_acts = activations.activations.sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Get masks
        self.reasoning_mask = activations.get_reasoning_mask()
        self.n_reasoning = self.reasoning_mask.sum().item()
        self.n_nonreasoning = (~self.reasoning_mask).sum().item()
        
        # Cache for computed stats
        self._feature_stats: Optional[list[FeatureStats]] = None
    
    def compute_feature_stats(self, feature_idx: int) -> FeatureStats:
        """Compute comprehensive statistics for a single feature."""
        acts = self.agg_acts[:, feature_idx].numpy()
        reasoning_acts = acts[self.reasoning_mask.numpy()]
        nonreasoning_acts = acts[~self.reasoning_mask.numpy()]
        
        # Basic statistics
        mean_r = float(np.mean(reasoning_acts))
        mean_nr = float(np.mean(nonreasoning_acts))
        std_r = float(np.std(reasoning_acts))
        std_nr = float(np.std(nonreasoning_acts))
        
        # Cohen's d (pooled standard deviation)
        pooled_std = np.sqrt(
            ((self.n_reasoning - 1) * std_r**2 + (self.n_nonreasoning - 1) * std_nr**2) /
            (self.n_reasoning + self.n_nonreasoning - 2)
        )
        cohens_d = (mean_r - mean_nr) / (pooled_std + 1e-10)
        
        # Log fold change
        log_fc = np.log2((mean_r + 1e-10) / (mean_nr + 1e-10))
        
        # Mann-Whitney U test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                u_stat, u_pval = mannwhitneyu(
                    reasoning_acts, nonreasoning_acts, alternative="two-sided"
                )
            except ValueError:
                u_stat, u_pval = 0.0, 1.0
        
        # Welch's t-test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                t_stat, t_pval = ttest_ind(
                    reasoning_acts, nonreasoning_acts, equal_var=False
                )
            except ValueError:
                t_stat, t_pval = 0.0, 1.0
        
        # ROC-AUC
        try:
            labels = self.reasoning_mask.numpy().astype(int)
            roc_auc = roc_auc_score(labels, acts)
        except ValueError:
            roc_auc = 0.5
        
        # Activation frequency (threshold at 0.01 * max activation)
        threshold = 0.01 * max(acts.max(), 1e-10)
        freq_r = (reasoning_acts > threshold).mean()
        freq_nr = (nonreasoning_acts > threshold).mean()
        
        return FeatureStats(
            feature_index=feature_idx,
            mean_reasoning=mean_r,
            mean_nonreasoning=mean_nr,
            std_reasoning=std_r,
            std_nonreasoning=std_nr,
            cohens_d=cohens_d,
            log_fold_change=log_fc,
            mannwhitney_u=float(u_stat),
            mannwhitney_p=float(u_pval),
            ttest_t=float(t_stat) if not np.isnan(t_stat) else 0.0,
            ttest_p=float(t_pval) if not np.isnan(t_pval) else 1.0,
            roc_auc=roc_auc,
            freq_active_reasoning=float(freq_r),
            freq_active_nonreasoning=float(freq_nr),
        )
    
    def compute_all_stats(self, verbose: bool = True) -> list[FeatureStats]:
        """Compute statistics for all features."""
        if self._feature_stats is not None:
            return self._feature_stats
        
        n_features = self.activations.n_features
        stats = []
        
        iterator = range(n_features)
        if verbose:
            import tqdm
            iterator = tqdm.tqdm(iterator, desc="Computing feature statistics")
        
        for i in iterator:
            stats.append(self.compute_feature_stats(i))
        
        self._feature_stats = stats
        return stats
    
    def get_reasoning_features(
        self,
        min_auc: float = 0.6,
        max_p_value: float = 0.01,
        min_effect_size: float = 0.3,
        top_k: Optional[int] = None,
    ) -> list[FeatureStats]:
        """
        Get features that qualify as reasoning features.
        
        Args:
            min_auc: Minimum ROC-AUC score
            max_p_value: Maximum p-value for Mann-Whitney test
            min_effect_size: Minimum absolute Cohen's d
            top_k: If specified, return only top_k features by reasoning_score
        
        Returns:
            List of FeatureStats for qualifying features, sorted by reasoning_score
        """
        all_stats = self.compute_all_stats()
        
        # Filter by criteria
        reasoning_features = [
            s for s in all_stats
            if s.is_reasoning_feature(min_auc, max_p_value, min_effect_size)
        ]
        
        # Sort by reasoning score (descending)
        reasoning_features.sort(key=lambda x: x.reasoning_score, reverse=True)
        
        if top_k is not None:
            reasoning_features = reasoning_features[:top_k]
        
        return reasoning_features
    
    def get_top_features_by_score(self, top_k: int = 100) -> list[FeatureStats]:
        """Get top K features by reasoning score (regardless of thresholds)."""
        all_stats = self.compute_all_stats()
        sorted_stats = sorted(all_stats, key=lambda x: x.reasoning_score, reverse=True)
        return sorted_stats[:top_k]
    
    def apply_bonferroni_correction(self) -> list[FeatureStats]:
        """Apply Bonferroni correction to p-values."""
        all_stats = self.compute_all_stats()
        n_tests = len(all_stats)
        
        for stat in all_stats:
            stat.mannwhitney_p = min(stat.mannwhitney_p * n_tests, 1.0)
            stat.ttest_p = min(stat.ttest_p * n_tests, 1.0)
        
        return all_stats
    
    def summary(self) -> dict:
        """Generate summary statistics about detected reasoning features."""
        all_stats = self.compute_all_stats()
        reasoning_features = self.get_reasoning_features()
        
        return {
            "total_features": len(all_stats),
            "reasoning_features_count": len(reasoning_features),
            "percentage_reasoning": len(reasoning_features) / len(all_stats) * 100,
            "top_10_features": [s.feature_index for s in reasoning_features[:10]],
            "top_10_scores": [s.reasoning_score for s in reasoning_features[:10]],
            "mean_auc_reasoning_features": np.mean([s.roc_auc for s in reasoning_features]) if reasoning_features else 0,
            "mean_cohens_d_reasoning_features": np.mean([s.cohens_d for s in reasoning_features]) if reasoning_features else 0,
        }

