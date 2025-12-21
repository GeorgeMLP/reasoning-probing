"""
ANOVA dataset construction for disentangling token vs. behavior effects.

This module implements the 2×2 factorial design described in docs/methodology.md.
It creates four conditions by crossing:
- Token Factor: Has top feature tokens vs. No top feature tokens
- Behavior Factor: Is reasoning chain vs. Not reasoning chain

The goal is to measure η²_token and η²_behavior for each feature to determine
whether the feature is dominated by token-level patterns or genuine reasoning.

## Key Changes from Initial Design:
1. Uses actual top tokens from feature analysis (not predefined patterns)
2. Classifies texts based on token presence (doesn't modify them)
3. Splits long reasoning chains into sentences for more samples per quadrant
"""

from dataclasses import dataclass, field
import re
import json
from pathlib import Path
from typing import Optional
from datasets import load_dataset
import numpy as np


@dataclass
class ANOVACondition:
    """Represents one of the four ANOVA conditions."""
    has_feature_tokens: bool
    is_reasoning: bool
    texts: list[str] = field(default_factory=list)
    
    @property
    def name(self) -> str:
        tokens = "has_tokens" if self.has_feature_tokens else "no_tokens"
        behavior = "reasoning" if self.is_reasoning else "nonreasoning"
        return f"{tokens}_{behavior}"
    
    @property
    def quadrant(self) -> str:
        """A, B, C, or D quadrant identifier."""
        if self.is_reasoning and self.has_feature_tokens:
            return "A"  # Reasoning + Has tokens
        elif self.is_reasoning and not self.has_feature_tokens:
            return "B"  # Reasoning + No tokens
        elif not self.is_reasoning and self.has_feature_tokens:
            return "C"  # Non-reasoning + Has tokens
        else:
            return "D"  # Non-reasoning + No tokens


def split_into_sentences(text: str, min_length: int = 50, max_length: int = 500) -> list[str]:
    """
    Split text into chunks of reasonable length.
    
    Args:
        text: Input text to split
        min_length: Minimum chunk length (chars)
        max_length: Maximum chunk length (chars)
    
    Returns:
        List of text chunks within length bounds
    """
    result = []
    
    # First, split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If paragraph is within bounds, use it
        if min_length <= len(para) <= max_length:
            result.append(para)
            continue
        
        # If too short, skip for now (will try to combine later)
        if len(para) < min_length:
            continue
        
        # If too long, split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', para)
        current = ""
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            if len(current) + len(sent) + 1 <= max_length:
                current = (current + " " + sent).strip() if current else sent
            else:
                if len(current) >= min_length:
                    result.append(current)
                current = sent
        
        if len(current) >= min_length:
            result.append(current)
    
    # If we didn't get enough, try splitting by single newlines
    if len(result) < 5 and len(text) > max_length:
        result = []
        lines = text.split('\n')
        current = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                # Paragraph break - save current if long enough
                if len(current) >= min_length:
                    result.append(current)
                current = ""
                continue
            
            if len(current) + len(line) + 1 <= max_length:
                current = (current + " " + line).strip() if current else line
            else:
                if len(current) >= min_length:
                    result.append(current)
                # If this line itself is too long, split by sentence
                if len(line) > max_length:
                    sub_sents = re.split(r'(?<=[.!?])\s+', line)
                    current = ""
                    for s in sub_sents:
                        if len(current) + len(s) + 1 <= max_length:
                            current = (current + " " + s).strip() if current else s
                        else:
                            if len(current) >= min_length:
                                result.append(current)
                            current = s[:max_length]  # Truncate if needed
                else:
                    current = line
        
        if len(current) >= min_length:
            result.append(current)
    
    return result


def text_contains_tokens(text: str, tokens: set[str], threshold: int = 1) -> bool:
    """
    Check if text contains any of the specified tokens.
    
    Args:
        text: Text to check
        tokens: Set of tokens to look for
        threshold: Minimum number of token matches required
    
    Returns:
        True if text contains at least threshold tokens
    """
    text_lower = text.lower()
    matches = 0
    
    for token in tokens:
        # Clean token (remove leading space if present)
        clean_token = token.strip().lower()
        if not clean_token:
            continue
        
        # Check for token presence (word boundary aware)
        # Use simple contains for now, can make more sophisticated
        if clean_token in text_lower:
            matches += 1
            if matches >= threshold:
                return True
    
    return False


def load_top_tokens_for_feature(
    token_analysis_path: Path,
    feature_index: int,
    top_k: int = 10,
) -> set[str]:
    """
    Load top tokens for a specific feature from token analysis JSON.
    
    Args:
        token_analysis_path: Path to token_analysis.json
        feature_index: Index of the feature
        top_k: Number of top tokens to use
    
    Returns:
        Set of top token strings
    """
    with open(token_analysis_path) as f:
        data = json.load(f)
    
    for feature in data.get('features', []):
        if feature.get('feature_index') == feature_index:
            top_tokens = feature.get('top_tokens', [])[:top_k]
            return {t.get('token_str', '') for t in top_tokens}
    
    return set()


def load_all_top_tokens(
    token_analysis_path: Path,
    feature_indices: Optional[list[int]] = None,
    top_k_per_feature: int = 5,
) -> dict[int, set[str]]:
    """
    Load top tokens for multiple features.
    
    Args:
        token_analysis_path: Path to token_analysis.json
        feature_indices: List of feature indices (None = all)
        top_k_per_feature: Number of top tokens per feature
    
    Returns:
        Dict mapping feature_index -> set of top tokens
    """
    with open(token_analysis_path) as f:
        data = json.load(f)
    
    result = {}
    for feature in data.get('features', []):
        feat_idx = feature.get('feature_index')
        if feature_indices is None or feat_idx in feature_indices:
            top_tokens = feature.get('top_tokens', [])[:top_k_per_feature]
            result[feat_idx] = {t.get('token_str', '') for t in top_tokens}
    
    return result


class ANOVADatasetBuilder:
    """
    Builds the 2×2 ANOVA dataset for disentangling token vs. behavior effects.
    
    For each feature, we create four conditions:
    - A: Reasoning text WITH feature's top tokens
    - B: Reasoning text WITHOUT feature's top tokens  
    - C: Non-reasoning text WITH feature's top tokens
    - D: Non-reasoning text WITHOUT feature's top tokens
    
    ## Usage
    
    ```python
    # Load top tokens from existing analysis
    tokens = load_top_tokens_for_feature(
        "results/layer8/token_analysis.json",
        feature_index=42,
        top_k=10
    )
    
    # Build ANOVA dataset
    builder = ANOVADatasetBuilder(feature_tokens=tokens)
    builder.load_source_data(n_samples=500)
    conditions = builder.build_all_conditions()
    
    for cond in conditions:
        print(f"{cond.name}: {len(cond.texts)} samples")
    ```
    """
    
    def __init__(
        self,
        feature_tokens: set[str],
        reasoning_dataset: str = "simplescaling/s1K-1.1",
        nonreasoning_dataset: str = "monology/pile-uncopyrighted",
        reasoning_column: str = "gemini_thinking_trajectory",
        token_match_threshold: int = 1,
    ):
        """
        Args:
            feature_tokens: Set of tokens to use for classification
            reasoning_dataset: HuggingFace dataset for reasoning text
            nonreasoning_dataset: HuggingFace dataset for non-reasoning text
            reasoning_column: Column name for reasoning text
            token_match_threshold: Min tokens needed to classify as "has tokens"
        """
        self.feature_tokens = feature_tokens
        self.reasoning_dataset = reasoning_dataset
        self.nonreasoning_dataset = nonreasoning_dataset
        self.reasoning_column = reasoning_column
        self.token_match_threshold = token_match_threshold
        
        # Storage for split sentences
        self.reasoning_sentences: list[str] = []
        self.nonreasoning_sentences: list[str] = []
    
    def load_source_data(
        self,
        n_reasoning_samples: int = 500,
        n_nonreasoning_samples: int = 1000,
        min_sentence_length: int = 50,
        max_sentence_length: int = 500,
        verbose: bool = True,
    ):
        """
        Load source data and split into sentences.
        
        Args:
            n_reasoning_samples: Max reasoning chains to load (will be split)
            n_nonreasoning_samples: Max non-reasoning samples to load
            min_sentence_length: Minimum sentence length
            max_sentence_length: Maximum sentence length
            verbose: Print progress
        """
        if verbose:
            print(f"Loading reasoning data from {self.reasoning_dataset}...")
        
        # Load reasoning data
        reasoning_ds = load_dataset(self.reasoning_dataset, split="train")
        
        chains_loaded = 0
        for row in reasoning_ds:
            text = row.get(self.reasoning_column, "")
            if not text or len(text) < min_sentence_length:
                continue
            
            # Split into sentences
            sentences = split_into_sentences(
                text, 
                min_length=min_sentence_length,
                max_length=max_sentence_length,
            )
            self.reasoning_sentences.extend(sentences)
            
            chains_loaded += 1
            if chains_loaded >= n_reasoning_samples:
                break
        
        if verbose:
            print(f"  Loaded {chains_loaded} reasoning chains -> {len(self.reasoning_sentences)} sentences")
        
        # Load non-reasoning data
        if verbose:
            print(f"Loading non-reasoning data from {self.nonreasoning_dataset}...")
        
        nonreasoning_ds = load_dataset(
            self.nonreasoning_dataset,
            split="train",
            streaming=True
        )
        
        samples_loaded = 0
        for row in nonreasoning_ds:
            text = row.get("text", "")
            if not text:
                continue
            
            # Split into sentences
            sentences = split_into_sentences(
                text,
                min_length=min_sentence_length,
                max_length=max_sentence_length
            )
            self.nonreasoning_sentences.extend(sentences)
            
            samples_loaded += 1
            if samples_loaded >= n_nonreasoning_samples:
                break
            
            # Early stop if we have enough sentences
            if len(self.nonreasoning_sentences) >= n_nonreasoning_samples * 3:
                break
        
        if verbose:
            print(f"  Loaded {samples_loaded} non-reasoning samples -> {len(self.nonreasoning_sentences)} sentences")
    
    def _classify_sentence(self, text: str) -> bool:
        """Check if sentence contains feature tokens."""
        return text_contains_tokens(
            text, 
            self.feature_tokens, 
            threshold=self.token_match_threshold
        )
    
    def build_condition_a(self) -> ANOVACondition:
        """Quadrant A: Reasoning + Has feature tokens."""
        texts = [s for s in self.reasoning_sentences if self._classify_sentence(s)]
        return ANOVACondition(
            has_feature_tokens=True,
            is_reasoning=True,
            texts=texts,
        )
    
    def build_condition_b(self) -> ANOVACondition:
        """Quadrant B: Reasoning + No feature tokens."""
        texts = [s for s in self.reasoning_sentences if not self._classify_sentence(s)]
        return ANOVACondition(
            has_feature_tokens=False,
            is_reasoning=True,
            texts=texts,
        )
    
    def build_condition_c(self) -> ANOVACondition:
        """Quadrant C: Non-reasoning + Has feature tokens."""
        texts = [s for s in self.nonreasoning_sentences if self._classify_sentence(s)]
        return ANOVACondition(
            has_feature_tokens=True,
            is_reasoning=False,
            texts=texts,
        )
    
    def build_condition_d(self) -> ANOVACondition:
        """Quadrant D: Non-reasoning + No feature tokens."""
        texts = [s for s in self.nonreasoning_sentences if not self._classify_sentence(s)]
        return ANOVACondition(
            has_feature_tokens=False,
            is_reasoning=False,
            texts=texts,
        )
    
    def build_all_conditions(self) -> list[ANOVACondition]:
        """Build all four ANOVA conditions."""
        return [
            self.build_condition_a(),
            self.build_condition_b(),
            self.build_condition_c(),
            self.build_condition_d(),
        ]
    
    def get_balanced_dataset(
        self,
        n_per_condition: int = 100,
        shuffle: bool = True,
        seed: int = 42,
    ) -> dict[str, ANOVACondition]:
        """
        Get a balanced dataset with equal samples per condition.
        
        Args:
            n_per_condition: Target samples per condition
            shuffle: Whether to shuffle before selecting
            seed: Random seed for shuffling
        
        Returns:
            Dictionary mapping condition names to ANOVACondition objects
        """
        conditions = self.build_all_conditions()
        
        if shuffle:
            rng = np.random.RandomState(seed)
        
        balanced = {}
        min_samples = float('inf')
        
        for cond in conditions:
            texts = cond.texts.copy()
            if shuffle:
                rng.shuffle(texts)
            
            n_available = len(texts)
            min_samples = min(min_samples, n_available)
            
            balanced_cond = ANOVACondition(
                has_feature_tokens=cond.has_feature_tokens,
                is_reasoning=cond.is_reasoning,
                texts=texts[:n_per_condition] if n_available >= n_per_condition else texts,
            )
            balanced[cond.name] = balanced_cond
        
        return balanced
    
    def summary(self) -> dict:
        """Get summary statistics about the dataset."""
        conditions = self.build_all_conditions()
        
        return {
            "total_reasoning_sentences": len(self.reasoning_sentences),
            "total_nonreasoning_sentences": len(self.nonreasoning_sentences),
            "n_feature_tokens": len(self.feature_tokens),
            "feature_tokens_sample": list(self.feature_tokens)[:10],
            "conditions": {
                cond.name: {
                    "quadrant": cond.quadrant,
                    "count": len(cond.texts),
                }
                for cond in conditions
            },
            "min_condition_size": min(len(c.texts) for c in conditions),
            "max_condition_size": max(len(c.texts) for c in conditions),
        }


@dataclass
class ANOVAResult:
    """Results from ANOVA analysis for a single feature."""
    feature_index: int
    
    # Sum of squares
    ss_token: float
    ss_behavior: float
    ss_interaction: float
    ss_error: float
    ss_total: float
    
    # Effect sizes (eta-squared)
    eta_sq_token: float
    eta_sq_behavior: float
    eta_sq_interaction: float
    
    # F-statistics
    f_token: float
    f_behavior: float
    f_interaction: float
    
    # P-values
    p_token: float
    p_behavior: float
    p_interaction: float
    
    # Cell means
    mean_A: float  # Reasoning + Has tokens
    mean_B: float  # Reasoning + No tokens
    mean_C: float  # Non-reasoning + Has tokens
    mean_D: float  # Non-reasoning + No tokens
    
    # Sample sizes
    n_per_cell: int
    
    # Decision
    is_token_dominated: bool
    is_behavior_dominated: bool
    dominant_factor: str  # "token", "behavior", "interaction", "none"
    
    def to_dict(self) -> dict:
        return {
            "feature_index": self.feature_index,
            "ss_token": self.ss_token,
            "ss_behavior": self.ss_behavior,
            "ss_interaction": self.ss_interaction,
            "ss_error": self.ss_error,
            "ss_total": self.ss_total,
            "eta_sq_token": self.eta_sq_token,
            "eta_sq_behavior": self.eta_sq_behavior,
            "eta_sq_interaction": self.eta_sq_interaction,
            "f_token": self.f_token,
            "f_behavior": self.f_behavior,
            "f_interaction": self.f_interaction,
            "p_token": self.p_token,
            "p_behavior": self.p_behavior,
            "p_interaction": self.p_interaction,
            "mean_A": self.mean_A,
            "mean_B": self.mean_B,
            "mean_C": self.mean_C,
            "mean_D": self.mean_D,
            "n_per_cell": self.n_per_cell,
            "is_token_dominated": self.is_token_dominated,
            "is_behavior_dominated": self.is_behavior_dominated,
            "dominant_factor": self.dominant_factor,
        }


def compute_anova_for_feature(
    activations_by_condition: dict[str, np.ndarray],
    feature_index: int = 0,
    alpha: float = 0.05,
) -> ANOVAResult:
    """
    Compute two-way ANOVA for a single feature.
    
    Args:
        activations_by_condition: Dictionary mapping condition names to
            activation arrays of shape (n_samples,)
            Keys: "has_tokens_reasoning", "no_tokens_reasoning",
                  "has_tokens_nonreasoning", "no_tokens_nonreasoning"
        feature_index: Index of the feature (for labeling)
        alpha: Significance level
    
    Returns:
        ANOVAResult with full statistics
    """
    from scipy import stats
    
    # Get activations for each cell
    a_A = activations_by_condition.get("has_tokens_reasoning", np.array([]))
    a_B = activations_by_condition.get("no_tokens_reasoning", np.array([]))
    a_C = activations_by_condition.get("has_tokens_nonreasoning", np.array([]))
    a_D = activations_by_condition.get("no_tokens_nonreasoning", np.array([]))
    
    # Balance the cells (use minimum size)
    n = min(len(a_A), len(a_B), len(a_C), len(a_D))
    
    if n == 0:
        # Return empty result
        return ANOVAResult(
            feature_index=feature_index,
            ss_token=0, ss_behavior=0, ss_interaction=0, ss_error=0, ss_total=0,
            eta_sq_token=0, eta_sq_behavior=0, eta_sq_interaction=0,
            f_token=0, f_behavior=0, f_interaction=0,
            p_token=1.0, p_behavior=1.0, p_interaction=1.0,
            mean_A=0, mean_B=0, mean_C=0, mean_D=0,
            n_per_cell=0,
            is_token_dominated=False, is_behavior_dominated=False,
            dominant_factor="none",
        )
    
    # Truncate to balanced size
    a_A, a_B, a_C, a_D = a_A[:n], a_B[:n], a_C[:n], a_D[:n]
    
    # Compute cell means
    mean_A = np.mean(a_A)
    mean_B = np.mean(a_B)
    mean_C = np.mean(a_C)
    mean_D = np.mean(a_D)
    
    # Marginal means
    mean_has_tokens = (mean_A + mean_C) / 2
    mean_no_tokens = (mean_B + mean_D) / 2
    mean_reasoning = (mean_A + mean_B) / 2
    mean_nonreasoning = (mean_C + mean_D) / 2
    grand_mean = (mean_A + mean_B + mean_C + mean_D) / 4
    
    # Sum of squares (balanced design formulas)
    ss_token = 2 * n * ((mean_has_tokens - grand_mean)**2 + (mean_no_tokens - grand_mean)**2)
    ss_behavior = 2 * n * ((mean_reasoning - grand_mean)**2 + (mean_nonreasoning - grand_mean)**2)
    
    ss_interaction = n * sum([
        (mean_A - mean_has_tokens - mean_reasoning + grand_mean)**2,
        (mean_B - mean_no_tokens - mean_reasoning + grand_mean)**2,
        (mean_C - mean_has_tokens - mean_nonreasoning + grand_mean)**2,
        (mean_D - mean_no_tokens - mean_nonreasoning + grand_mean)**2,
    ])
    
    ss_error = (
        np.sum((a_A - mean_A)**2) +
        np.sum((a_B - mean_B)**2) +
        np.sum((a_C - mean_C)**2) +
        np.sum((a_D - mean_D)**2)
    )
    
    ss_total = ss_token + ss_behavior + ss_interaction + ss_error
    
    # Degrees of freedom
    df_token = 1
    df_behavior = 1
    df_interaction = 1
    df_error = 4 * (n - 1)
    
    # Mean squares
    ms_token = ss_token / df_token if df_token > 0 else 0
    ms_behavior = ss_behavior / df_behavior if df_behavior > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 1e-10
    
    # F-statistics
    f_token = ms_token / ms_error
    f_behavior = ms_behavior / ms_error
    f_interaction = ms_interaction / ms_error
    
    # P-values
    p_token = 1 - stats.f.cdf(f_token, df_token, df_error)
    p_behavior = 1 - stats.f.cdf(f_behavior, df_behavior, df_error)
    p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_error)
    
    # Eta-squared (effect sizes)
    eta_sq_token = ss_token / ss_total if ss_total > 0 else 0
    eta_sq_behavior = ss_behavior / ss_total if ss_total > 0 else 0
    eta_sq_interaction = ss_interaction / ss_total if ss_total > 0 else 0
    
    # Decision rules
    # Token-dominated: η²_token > 2 * η²_behavior AND η²_token > 0.06 (medium effect)
    is_token_dominated = (eta_sq_token > 2 * eta_sq_behavior) and (eta_sq_token > 0.06)
    is_behavior_dominated = (eta_sq_behavior > 2 * eta_sq_token) and (eta_sq_behavior > 0.06)
    
    # Determine dominant factor
    if eta_sq_interaction > max(eta_sq_token, eta_sq_behavior) and eta_sq_interaction > 0.06:
        dominant_factor = "interaction"
    elif is_token_dominated:
        dominant_factor = "token"
    elif is_behavior_dominated:
        dominant_factor = "behavior"
    elif max(eta_sq_token, eta_sq_behavior, eta_sq_interaction) < 0.01:
        dominant_factor = "none"
    else:
        dominant_factor = "mixed"
    
    return ANOVAResult(
        feature_index=feature_index,
        ss_token=float(ss_token),
        ss_behavior=float(ss_behavior),
        ss_interaction=float(ss_interaction),
        ss_error=float(ss_error),
        ss_total=float(ss_total),
        eta_sq_token=float(eta_sq_token),
        eta_sq_behavior=float(eta_sq_behavior),
        eta_sq_interaction=float(eta_sq_interaction),
        f_token=float(f_token),
        f_behavior=float(f_behavior),
        f_interaction=float(f_interaction),
        p_token=float(p_token),
        p_behavior=float(p_behavior),
        p_interaction=float(p_interaction),
        mean_A=float(mean_A),
        mean_B=float(mean_B),
        mean_C=float(mean_C),
        mean_D=float(mean_D),
        n_per_cell=n,
        is_token_dominated=is_token_dominated,
        is_behavior_dominated=is_behavior_dominated,
        dominant_factor=dominant_factor,
    )


def compute_anova_summary(results: list[ANOVAResult]) -> dict:
    """Compute summary statistics from multiple ANOVA results."""
    if not results:
        return {}
    
    n_total = len(results)
    n_token_dominated = sum(1 for r in results if r.is_token_dominated)
    n_behavior_dominated = sum(1 for r in results if r.is_behavior_dominated)
    
    dominant_counts = {}
    for r in results:
        dominant_counts[r.dominant_factor] = dominant_counts.get(r.dominant_factor, 0) + 1
    
    return {
        "n_features_analyzed": n_total,
        "n_token_dominated": n_token_dominated,
        "n_behavior_dominated": n_behavior_dominated,
        "pct_token_dominated": 100 * n_token_dominated / n_total,
        "pct_behavior_dominated": 100 * n_behavior_dominated / n_total,
        "dominant_factor_distribution": dominant_counts,
        "mean_eta_sq_token": np.mean([r.eta_sq_token for r in results]),
        "mean_eta_sq_behavior": np.mean([r.eta_sq_behavior for r in results]),
        "mean_eta_sq_interaction": np.mean([r.eta_sq_interaction for r in results]),
        "median_eta_sq_token": np.median([r.eta_sq_token for r in results]),
        "median_eta_sq_behavior": np.median([r.eta_sq_behavior for r in results]),
        "mean_f_token": np.mean([r.f_token for r in results]),
        "mean_f_behavior": np.mean([r.f_behavior for r in results]),
        "pct_significant_token": 100 * sum(1 for r in results if r.p_token < 0.05) / n_total,
        "pct_significant_behavior": 100 * sum(1 for r in results if r.p_behavior < 0.05) / n_total,
    }
