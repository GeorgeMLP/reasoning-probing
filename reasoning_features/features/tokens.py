"""
Top token analysis for SAE features.

This module analyzes which tokens most strongly activate each feature,
helping to understand whether features rely on shallow token cues vs.
deeper reasoning patterns.
"""

from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from transformers import PreTrainedTokenizerBase

from .collector import FeatureActivations


@dataclass
class TokenFeatureAssociation:
    """
    Association between a token and a feature.
    
    Contains multiple metrics to assess how strongly the token
    triggers the feature activation.
    """
    token_id: int
    token_str: str
    feature_index: int
    
    # Basic statistics
    mean_activation: float  # Mean feature activation when token appears
    max_activation: float  # Max feature activation when token appears
    occurrence_count: int  # How many times token appears in dataset
    occurrence_count_reasoning: int  # Occurrences in reasoning samples
    occurrence_count_nonreasoning: int  # Occurrences in non-reasoning samples
    
    # Association metrics
    pmi: float  # Pointwise Mutual Information
    activation_ratio: float  # P(feature fires | token) / P(feature fires)
    
    # Context statistics
    mean_activation_in_reasoning: float  # Mean activation in reasoning context
    mean_activation_in_nonreasoning: float  # Mean activation in non-reasoning context
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "token_str": self.token_str,
            "feature_index": self.feature_index,
            "mean_activation": self.mean_activation,
            "max_activation": self.max_activation,
            "occurrence_count": self.occurrence_count,
            "occurrence_count_reasoning": self.occurrence_count_reasoning,
            "occurrence_count_nonreasoning": self.occurrence_count_nonreasoning,
            "pmi": self.pmi,
            "activation_ratio": self.activation_ratio,
            "mean_activation_in_reasoning": self.mean_activation_in_reasoning,
            "mean_activation_in_nonreasoning": self.mean_activation_in_nonreasoning,
        }


@dataclass
class NgramFeatureAssociation:
    """
    Association between an n-gram (bigram/trigram) and a feature.
    
    Tracks consecutive token sequences that strongly activate features.
    """
    token_ids: tuple  # Tuple of token IDs
    token_strs: tuple  # Tuple of token strings
    ngram_str: str  # Concatenated string representation
    feature_index: int
    n: int  # 2 for bigram, 3 for trigram
    
    # Statistics
    mean_activation: float  # Mean of activations across the n-gram
    max_activation: float
    occurrence_count: int
    occurrence_count_reasoning: int
    occurrence_count_nonreasoning: int
    mean_activation_in_reasoning: float
    mean_activation_in_nonreasoning: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "token_ids": list(self.token_ids),
            "token_strs": list(self.token_strs),
            "ngram_str": self.ngram_str,
            "feature_index": self.feature_index,
            "n": self.n,
            "mean_activation": self.mean_activation,
            "max_activation": self.max_activation,
            "occurrence_count": self.occurrence_count,
            "occurrence_count_reasoning": self.occurrence_count_reasoning,
            "occurrence_count_nonreasoning": self.occurrence_count_nonreasoning,
            "mean_activation_in_reasoning": self.mean_activation_in_reasoning,
            "mean_activation_in_nonreasoning": self.mean_activation_in_nonreasoning,
        }


class TopTokenAnalyzer:
    """
    Analyzes which tokens most strongly activate each SAE feature.
    
    ## Metrics Used
    
    1. **Mean Activation**: Average feature activation when token is present.
    
    2. **Pointwise Mutual Information (PMI)**: Measures co-occurrence strength
       between token and feature activation.
       PMI(token, feature) = log2(P(token, feature_fires) / (P(token) * P(feature_fires)))
    
    3. **Activation Ratio**: How much more likely the feature fires when
       the token is present vs. baseline.
       ratio = P(feature_fires | token) / P(feature_fires)
    
    4. **Context-Aware Activation**: Compares token-feature association in
       reasoning vs. non-reasoning contexts.
    """
    
    def __init__(
        self,
        activations: FeatureActivations,
        tokenizer: PreTrainedTokenizerBase,
        activation_threshold: float = 0.1,
    ):
        """
        Args:
            activations: Collected feature activations
            tokenizer: Tokenizer for decoding token IDs
            activation_threshold: Threshold for considering a feature "active"
                                 (as fraction of max activation)
        """
        self.activations = activations
        self.tokenizer = tokenizer
        self.activation_threshold = activation_threshold
        
        # Precompute useful quantities
        self.reasoning_mask = activations.get_reasoning_mask()
        self._precompute_token_stats()
    
    def _precompute_token_stats(self):
        """Precompute token occurrence statistics."""
        tokens = self.activations.tokens.numpy()
        
        # Token occurrence counts
        self.token_counts = defaultdict(int)
        self.token_positions = defaultdict(list)  # (sample_idx, position)
        
        for sample_idx in range(tokens.shape[0]):
            for pos in range(tokens.shape[1]):
                token_id = tokens[sample_idx, pos]
                self.token_counts[token_id] += 1
                self.token_positions[token_id].append((sample_idx, pos))
        
        self.total_tokens = tokens.size
        self.unique_tokens = len(self.token_counts)
    
    def get_top_tokens_for_feature(
        self,
        feature_index: int,
        top_k: int = 50,
        min_occurrences: int = 5,
    ) -> list[TokenFeatureAssociation]:
        """
        Get the top tokens that most strongly activate a feature.
        
        Args:
            feature_index: Index of the feature to analyze
            top_k: Number of top tokens to return
            min_occurrences: Minimum token occurrences to consider
        
        Returns:
            List of TokenFeatureAssociation sorted by mean activation
        """
        acts = self.activations.activations[:, :, feature_index].numpy()
        tokens = self.activations.tokens.numpy()
        reasoning_mask = self.reasoning_mask.numpy()
        
        # Compute global feature statistics
        feature_max = acts.max()
        threshold = self.activation_threshold * max(feature_max, 1e-10)
        feature_fires_prob = (acts > threshold).mean()
        
        associations = []
        
        for token_id, positions in self.token_positions.items():
            if len(positions) < min_occurrences:
                continue
            
            # Get activations at token positions
            token_acts = []
            token_acts_reasoning = []
            token_acts_nonreasoning = []
            count_reasoning = 0
            count_nonreasoning = 0
            
            for sample_idx, pos in positions:
                act = acts[sample_idx, pos]
                token_acts.append(act)
                
                if reasoning_mask[sample_idx]:
                    token_acts_reasoning.append(act)
                    count_reasoning += 1
                else:
                    token_acts_nonreasoning.append(act)
                    count_nonreasoning += 1
            
            token_acts = np.array(token_acts)
            
            # Basic statistics
            mean_act = float(np.mean(token_acts))
            max_act = float(np.max(token_acts))
            
            # PMI calculation
            p_token = len(positions) / self.total_tokens
            p_feature_fires = feature_fires_prob
            p_joint = (token_acts > threshold).mean()
            
            # Avoid log(0)
            if p_joint > 0 and p_token > 0 and p_feature_fires > 0:
                pmi = np.log2(p_joint / (p_token * p_feature_fires))
            else:
                pmi = -np.inf
            
            # Activation ratio
            p_feature_given_token = (token_acts > threshold).mean()
            if p_feature_fires > 0:
                activation_ratio = p_feature_given_token / p_feature_fires
            else:
                activation_ratio = 0.0
            
            # Context-aware statistics
            mean_reasoning = np.mean(token_acts_reasoning) if token_acts_reasoning else 0.0
            mean_nonreasoning = np.mean(token_acts_nonreasoning) if token_acts_nonreasoning else 0.0
            
            # Decode token
            try:
                token_str = self.tokenizer.decode([token_id])
            except Exception:
                token_str = f"<token_{token_id}>"
            
            associations.append(TokenFeatureAssociation(
                token_id=int(token_id),
                token_str=token_str,
                feature_index=feature_index,
                mean_activation=mean_act,
                max_activation=max_act,
                occurrence_count=len(positions),
                occurrence_count_reasoning=count_reasoning,
                occurrence_count_nonreasoning=count_nonreasoning,
                pmi=float(pmi) if not np.isinf(pmi) else -100.0,
                activation_ratio=float(activation_ratio),
                mean_activation_in_reasoning=float(mean_reasoning),
                mean_activation_in_nonreasoning=float(mean_nonreasoning),
            ))
        
        # Sort by mean activation (descending)
        associations.sort(key=lambda x: x.mean_activation, reverse=True)
        
        return associations[:top_k]
    
    def get_reasoning_specific_tokens(
        self,
        feature_index: int,
        top_k: int = 30,
        min_occurrences: int = 5,
    ) -> list[TokenFeatureAssociation]:
        """
        Get tokens that activate the feature specifically in reasoning contexts.
        
        These are tokens where mean_activation_in_reasoning >> mean_activation_in_nonreasoning.
        """
        all_tokens = self.get_top_tokens_for_feature(
            feature_index, top_k=top_k * 3, min_occurrences=min_occurrences
        )
        
        # Filter and sort by reasoning specificity
        reasoning_tokens = []
        for assoc in all_tokens:
            # Require at least some activation in reasoning
            if assoc.mean_activation_in_reasoning < 0.01:
                continue
            
            # Compute reasoning specificity ratio
            specificity = assoc.mean_activation_in_reasoning / (
                assoc.mean_activation_in_nonreasoning + 1e-10
            )
            
            if specificity > 1.5:  # At least 1.5x more active in reasoning
                reasoning_tokens.append((assoc, specificity))
        
        # Sort by specificity
        reasoning_tokens.sort(key=lambda x: x[1], reverse=True)
        
        return [assoc for assoc, _ in reasoning_tokens[:top_k]]
    
    def analyze_feature_token_dependency(
        self,
        feature_index: int,
        top_k_tokens: int = 20,
    ) -> dict:
        """
        Analyze how much a feature's activation depends on specific tokens.
        
        Returns metrics indicating whether the feature relies on shallow
        token cues vs. deeper patterns.
        """
        acts = self.activations.activations[:, :, feature_index].numpy()
        tokens = self.activations.tokens.numpy()
        
        # Get top tokens
        top_tokens = self.get_top_tokens_for_feature(feature_index, top_k=top_k_tokens)
        top_token_ids = {t.token_id for t in top_tokens}
        
        # Calculate what fraction of high activations come from top tokens
        threshold = self.activation_threshold * max(acts.max(), 1e-10)
        high_act_mask = acts > threshold
        
        # Count high activations from top tokens
        high_acts_from_top_tokens = 0
        total_high_acts = high_act_mask.sum()
        
        for sample_idx in range(acts.shape[0]):
            for pos in range(acts.shape[1]):
                if high_act_mask[sample_idx, pos]:
                    if tokens[sample_idx, pos] in top_token_ids:
                        high_acts_from_top_tokens += 1
        
        # Token concentration metric
        if total_high_acts > 0:
            token_concentration = high_acts_from_top_tokens / total_high_acts
        else:
            token_concentration = 0.0
        
        # Entropy of token distribution for this feature
        token_act_sums = defaultdict(float)
        for sample_idx in range(acts.shape[0]):
            for pos in range(acts.shape[1]):
                token_id = tokens[sample_idx, pos]
                token_act_sums[token_id] += acts[sample_idx, pos]
        
        total_act = sum(token_act_sums.values())
        if total_act > 0:
            probs = np.array([v / total_act for v in token_act_sums.values()])
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(len(token_act_sums))
            normalized_entropy = entropy / max(max_entropy, 1e-10)
        else:
            normalized_entropy = 0.0
        
        return {
            "feature_index": feature_index,
            "top_tokens": [t.to_dict() for t in top_tokens],
            "token_concentration": token_concentration,
            "normalized_entropy": normalized_entropy,
            "is_token_dependent": bool(token_concentration > 0.5),  # >50% from top tokens
            "interpretation": (
                "HIGH token dependency - likely shallow cue" 
                if token_concentration > 0.5 
                else "LOWER token dependency - may capture deeper patterns"
            ),
        }
    
    def get_top_ngrams_for_feature(
        self,
        feature_index: int,
        n: int = 2,
        top_k: int = 30,
        min_occurrences: int = 3,
    ) -> list[NgramFeatureAssociation]:
        """
        Get the top n-grams (consecutive token sequences) that most strongly activate a feature.
        
        Args:
            feature_index: Index of the feature to analyze
            n: N-gram size (2 for bigram, 3 for trigram)
            top_k: Number of top n-grams to return
            min_occurrences: Minimum occurrences to consider
        
        Returns:
            List of NgramFeatureAssociation sorted by mean activation
        """
        acts = self.activations.activations[:, :, feature_index].numpy()
        tokens = self.activations.tokens.numpy()
        reasoning_mask = self.reasoning_mask.numpy()
        
        # Track n-gram statistics: key = tuple of token_ids
        ngram_stats = defaultdict(lambda: {
            'mean_acts': [], 'max_acts': [],
            'reasoning_acts': [], 'nonreasoning_acts': [],
        })
        
        for sample_idx in range(tokens.shape[0]):
            is_reasoning = reasoning_mask[sample_idx]
            seq_len = tokens.shape[1]
            
            for pos in range(seq_len - n + 1):
                ngram_ids = tuple(int(tokens[sample_idx, pos + i]) for i in range(n))
                ngram_acts = acts[sample_idx, pos:pos + n]
                mean_act = float(np.mean(ngram_acts))
                max_act = float(np.max(ngram_acts))
                
                stats = ngram_stats[ngram_ids]
                stats['mean_acts'].append(mean_act)
                stats['max_acts'].append(max_act)
                
                if is_reasoning:
                    stats['reasoning_acts'].append(mean_act)
                else:
                    stats['nonreasoning_acts'].append(mean_act)
        
        # Build associations
        associations = []
        for ngram_ids, stats in ngram_stats.items():
            if len(stats['mean_acts']) < min_occurrences:
                continue
            
            # Decode tokens
            try:
                token_strs = tuple(self.tokenizer.decode([tid]) for tid in ngram_ids)
                ngram_str = ''.join(token_strs)
            except Exception:
                token_strs = tuple(f"<token_{tid}>" for tid in ngram_ids)
                ngram_str = ' '.join(token_strs)
            
            associations.append(NgramFeatureAssociation(
                token_ids=ngram_ids,
                token_strs=token_strs,
                ngram_str=ngram_str,
                feature_index=feature_index,
                n=n,
                mean_activation=float(np.mean(stats['mean_acts'])),
                max_activation=float(np.max(stats['max_acts'])),
                occurrence_count=len(stats['mean_acts']),
                occurrence_count_reasoning=len(stats['reasoning_acts']),
                occurrence_count_nonreasoning=len(stats['nonreasoning_acts']),
                mean_activation_in_reasoning=float(np.mean(stats['reasoning_acts'])) if stats['reasoning_acts'] else 0.0,
                mean_activation_in_nonreasoning=float(np.mean(stats['nonreasoning_acts'])) if stats['nonreasoning_acts'] else 0.0,
            ))
        
        # Sort by mean activation (descending)
        associations.sort(key=lambda x: x.mean_activation, reverse=True)
        return associations[:top_k]
    
    def get_feature_vocabulary(
        self,
        feature_index: int,
        activation_percentile: float = 90,
    ) -> list[str]:
        """
        Get the vocabulary of tokens that trigger high activations for a feature.
        
        Returns unique tokens that appear in positions with activations
        above the given percentile.
        """
        acts = self.activations.activations[:, :, feature_index].numpy()
        tokens = self.activations.tokens.numpy()
        
        threshold = np.percentile(acts[acts > 0], activation_percentile) if (acts > 0).any() else 0
        
        high_act_tokens = set()
        for sample_idx in range(acts.shape[0]):
            for pos in range(acts.shape[1]):
                if acts[sample_idx, pos] > threshold:
                    high_act_tokens.add(tokens[sample_idx, pos])
        
        # Decode tokens
        vocab = []
        for token_id in high_act_tokens:
            try:
                vocab.append(self.tokenizer.decode([token_id]))
            except Exception:
                pass
        
        return vocab
