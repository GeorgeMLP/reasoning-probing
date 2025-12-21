"""
ANOVA dataset construction for disentangling token vs. behavior effects.

This module implements the 2×2 factorial design described in docs/methodology.md.
It creates four conditions by crossing:
- Token Factor: Has reasoning tokens vs. No reasoning tokens
- Behavior Factor: Is reasoning chain vs. Not reasoning chain

The goal is to measure η²_token and η²_behavior for each feature to determine
whether the feature is dominated by token-level patterns or genuine reasoning.
"""

from dataclasses import dataclass
import re
from datasets import load_dataset
import numpy as np


# Reasoning tokens identified from empirical analysis
REASONING_TOKENS = {
    # Mathematical reasoning cues
    "prove", "Prove", "proof", "therefore", "Thus", "hence",
    "Let", "let", "Consider", "consider", "Suppose", "suppose",
    "solve", "Solve", "Problem", "problem",
    
    # Chain-of-thought cues  
    "First", "first", "Second", "second", "Third", "third",
    "Next", "next", "Then", "then", "Finally", "finally",
    "Step", "step",
    
    # Thinking/reasoning verbs
    "think", "Think", "thinking", "reason", "reasoning",
    "analyze", "Analyze", "examining", "considering",
    
    # Hedging/uncertainty (common in reasoning)
    "Maybe", "maybe", "Perhaps", "perhaps", "might", "could",
    "I think", "I believe", "It seems",
    
    # Connectives
    "because", "Because", "since", "Since",
    "therefore", "Therefore", "thus", "Thus",
    "however", "However", "but", "But",
    
    # Need/must (planning cues)
    "need", "Need", "must", "Must", "should", "Should",
    "require", "requires",
}

# Patterns for more complex token detection
REASONING_PATTERNS = [
    r"\blet's\s+think",
    r"\blet\s+me\s+think",
    r"\bstep\s+by\s+step",
    r"\bstep\s+\d+",
    r"\bfirst,?\s+I",
    r"\bI\s+need\s+to",
    r"\bwe\s+need\s+to",
    r"\bto\s+solve\s+this",
    r"\bthe\s+answer\s+is",
    r"<think>",
    r"</think>",
]


@dataclass
class ANOVACondition:
    """Represents one of the four ANOVA conditions."""
    has_reasoning_tokens: bool
    is_reasoning_chain: bool
    texts: list[str]
    
    @property
    def name(self) -> str:
        tokens = "tokens" if self.has_reasoning_tokens else "no_tokens"
        behavior = "reasoning" if self.is_reasoning_chain else "nonreasoning"
        return f"{tokens}_{behavior}"
    
    @property
    def quadrant(self) -> str:
        """A, B, C, or D quadrant identifier."""
        if self.is_reasoning_chain and self.has_reasoning_tokens:
            return "A"
        elif self.is_reasoning_chain and not self.has_reasoning_tokens:
            return "B"
        elif not self.is_reasoning_chain and self.has_reasoning_tokens:
            return "C"
        else:
            return "D"


def count_reasoning_tokens(text: str) -> int:
    """Count occurrences of reasoning tokens and patterns in text."""
    count = 0
    
    # Count token occurrences
    words = set(text.split())
    for token in REASONING_TOKENS:
        if token in words:
            count += 1
    
    # Count pattern matches
    for pattern in REASONING_PATTERNS:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    
    return count


def has_significant_reasoning_tokens(text: str, threshold: float = 0.01) -> bool:
    """
    Check if text has significant reasoning tokens.
    
    Args:
        text: Input text
        threshold: Minimum fraction of words that should be reasoning tokens
    
    Returns:
        True if reasoning token density exceeds threshold
    """
    n_tokens = count_reasoning_tokens(text)
    n_words = len(text.split())
    return n_tokens / max(n_words, 1) > threshold


def remove_reasoning_tokens(text: str) -> str:
    """
    Remove or neutralize reasoning tokens from text.
    
    This transforms Quadrant A (reasoning + tokens) to Quadrant B (reasoning + no tokens).
    """
    result = text
    
    # Replace common patterns with neutral alternatives
    replacements = [
        (r"\bLet's think step by step[.,]?\s*", ""),
        (r"\bLet me think[.,]?\s*", ""),
        (r"\bFirst,?\s*", ""),
        (r"\bSecond,?\s*", ""),
        (r"\bThird,?\s*", ""),
        (r"\bTherefore,?\s*", "So "),
        (r"\bThus,?\s*", "So "),
        (r"\bHence,?\s*", "So "),
        (r"\bI think\s+", ""),
        (r"\bI need to\s+", ""),
        (r"\bWe need to\s+", ""),
        (r"<think>", ""),
        (r"</think>", ""),
        (r"\bStep\s+\d+:?\s*", ""),
        (r"\bProve\b", "Show"),
        (r"\bprove\b", "show"),
        (r"\bConsider\b", "Look at"),
        (r"\bconsider\b", "look at"),
    ]
    
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


def inject_reasoning_tokens(text: str, density: float = 0.02) -> str:
    """
    Inject reasoning tokens into non-reasoning text.
    
    This transforms Quadrant D (no reasoning + no tokens) to 
    Quadrant C (no reasoning + tokens).
    
    Args:
        text: Input non-reasoning text
        density: Target density of reasoning tokens
    """
    sentences = text.split('. ')
    if len(sentences) < 3:
        return f"Let me think about this. {text} Therefore, that's the answer."
    
    # Insert reasoning tokens at intervals
    result_sentences = []
    prefixes = ["First, ", "Next, ", "Then, ", "Also, ", "Therefore, "]
    
    for i, sentence in enumerate(sentences):
        if i == 0:
            result_sentences.append(f"Let me think. {sentence}")
        elif i % 3 == 0 and i < len(sentences) - 1:
            prefix = prefixes[i // 3 % len(prefixes)]
            result_sentences.append(f"{prefix}{sentence}")
        elif i == len(sentences) - 1:
            result_sentences.append(f"Therefore, {sentence}")
        else:
            result_sentences.append(sentence)
    
    return '. '.join(result_sentences)


class ANOVADatasetBuilder:
    """
    Builds the 2×2 ANOVA dataset for disentangling token vs. behavior effects.
    
    ## Usage
    
    ```python
    builder = ANOVADatasetBuilder()
    builder.load_source_data(n_samples=200)
    conditions = builder.build_all_conditions()
    
    for cond in conditions:
        print(f"{cond.name}: {len(cond.texts)} samples")
    ```
    """
    
    def __init__(
        self,
        reasoning_dataset: str = "simplescaling/s1K-1.1",
        nonreasoning_dataset: str = "monology/pile-uncopyrighted",
        reasoning_column: str = "gemini_thinking_trajectory",
    ):
        self.reasoning_dataset = reasoning_dataset
        self.nonreasoning_dataset = nonreasoning_dataset
        self.reasoning_column = reasoning_column
        
        self.reasoning_texts: list[str] = []
        self.nonreasoning_texts: list[str] = []
    
    def load_source_data(
        self,
        n_samples: int = 200,
        min_length: int = 200,
        max_length: int = 2000,
    ):
        """Load source reasoning and non-reasoning texts."""
        # Load reasoning data
        print(f"Loading reasoning data from {self.reasoning_dataset}...")
        reasoning_ds = load_dataset(self.reasoning_dataset, split="train")
        
        for row in reasoning_ds:
            text = row.get(self.reasoning_column, "")
            if min_length <= len(text) <= max_length:
                self.reasoning_texts.append(text)
            if len(self.reasoning_texts) >= n_samples:
                break
        
        # Load non-reasoning data
        print(f"Loading non-reasoning data from {self.nonreasoning_dataset}...")
        nonreasoning_ds = load_dataset(
            self.nonreasoning_dataset, 
            split="train", 
            streaming=True
        )
        
        for row in nonreasoning_ds:
            text = row.get("text", "")
            if min_length <= len(text) <= max_length:
                # Filter out texts that already have reasoning tokens
                if not has_significant_reasoning_tokens(text, threshold=0.005):
                    self.nonreasoning_texts.append(text)
            if len(self.nonreasoning_texts) >= n_samples:
                break
        
        print(f"Loaded {len(self.reasoning_texts)} reasoning texts")
        print(f"Loaded {len(self.nonreasoning_texts)} non-reasoning texts")
    
    def build_quadrant_a(self) -> ANOVACondition:
        """
        Quadrant A: Reasoning + Reasoning Tokens
        
        Natural reasoning text that contains explicit reasoning tokens.
        """
        texts = [
            t for t in self.reasoning_texts
            if has_significant_reasoning_tokens(t)
        ]
        
        return ANOVACondition(
            has_reasoning_tokens=True,
            is_reasoning_chain=True,
            texts=texts,
        )
    
    def build_quadrant_b(self) -> ANOVACondition:
        """
        Quadrant B: Reasoning + No Reasoning Tokens
        
        Reasoning text with explicit reasoning tokens removed/neutralized.
        """
        texts = [
            remove_reasoning_tokens(t)
            for t in self.reasoning_texts
            if has_significant_reasoning_tokens(t)
        ]
        
        # Verify removal was effective
        texts = [t for t in texts if not has_significant_reasoning_tokens(t)]
        
        return ANOVACondition(
            has_reasoning_tokens=False,
            is_reasoning_chain=True,
            texts=texts,
        )
    
    def build_quadrant_c(self) -> ANOVACondition:
        """
        Quadrant C: No Reasoning + Reasoning Tokens
        
        Non-reasoning text injected with reasoning tokens.
        """
        texts = [
            inject_reasoning_tokens(t)
            for t in self.nonreasoning_texts
        ]
        
        return ANOVACondition(
            has_reasoning_tokens=True,
            is_reasoning_chain=False,
            texts=texts,
        )
    
    def build_quadrant_d(self) -> ANOVACondition:
        """
        Quadrant D: No Reasoning + No Reasoning Tokens
        
        Natural non-reasoning text without reasoning tokens.
        """
        return ANOVACondition(
            has_reasoning_tokens=False,
            is_reasoning_chain=False,
            texts=self.nonreasoning_texts,
        )
    
    def build_all_conditions(self) -> list[ANOVACondition]:
        """Build all four ANOVA conditions."""
        return [
            self.build_quadrant_a(),
            self.build_quadrant_b(),
            self.build_quadrant_c(),
            self.build_quadrant_d(),
        ]
    
    def get_balanced_dataset(
        self,
        n_per_condition: int = 100,
    ) -> dict[str, ANOVACondition]:
        """
        Get a balanced dataset with equal samples per condition.
        
        Returns:
            Dictionary mapping condition names to ANOVACondition objects
        """
        conditions = self.build_all_conditions()
        
        balanced = {}
        for cond in conditions:
            if len(cond.texts) >= n_per_condition:
                balanced_cond = ANOVACondition(
                    has_reasoning_tokens=cond.has_reasoning_tokens,
                    is_reasoning_chain=cond.is_reasoning_chain,
                    texts=cond.texts[:n_per_condition],
                )
                balanced[cond.name] = balanced_cond
            else:
                print(f"Warning: {cond.name} only has {len(cond.texts)} samples")
                balanced[cond.name] = cond
        
        return balanced


def compute_anova_for_feature(
    activations_by_condition: dict[str, np.ndarray],
) -> dict:
    """
    Compute two-way ANOVA for a single feature.
    
    Args:
        activations_by_condition: Dictionary mapping condition names to
            activation arrays of shape (n_samples,)
    
    Returns:
        Dictionary with ANOVA results including:
        - ss_token: Sum of squares for token factor
        - ss_behavior: Sum of squares for behavior factor
        - ss_interaction: Sum of squares for interaction
        - ss_error: Sum of squares for error
        - eta_sq_token: η² for token factor
        - eta_sq_behavior: η² for behavior factor
        - is_token_dominated: Whether feature is dominated by token patterns
    """
    # Parse conditions
    conditions = {
        "A": "tokens_reasoning",
        "B": "no_tokens_reasoning", 
        "C": "tokens_nonreasoning",
        "D": "no_tokens_nonreasoning",
    }
    
    # Get activations for each cell
    a_A = activations_by_condition.get("tokens_reasoning", np.array([]))
    a_B = activations_by_condition.get("no_tokens_reasoning", np.array([]))
    a_C = activations_by_condition.get("tokens_nonreasoning", np.array([]))
    a_D = activations_by_condition.get("no_tokens_nonreasoning", np.array([]))
    
    # Compute means
    mean_A = np.mean(a_A) if len(a_A) > 0 else 0
    mean_B = np.mean(a_B) if len(a_B) > 0 else 0
    mean_C = np.mean(a_C) if len(a_C) > 0 else 0
    mean_D = np.mean(a_D) if len(a_D) > 0 else 0
    
    # Marginal means
    mean_tokens = (mean_A + mean_C) / 2  # Has tokens
    mean_no_tokens = (mean_B + mean_D) / 2  # No tokens
    mean_reasoning = (mean_A + mean_B) / 2  # Is reasoning
    mean_nonreasoning = (mean_C + mean_D) / 2  # Not reasoning
    grand_mean = (mean_A + mean_B + mean_C + mean_D) / 4
    
    # Sample sizes (assume balanced for simplicity)
    n = min(len(a_A), len(a_B), len(a_C), len(a_D))
    if n == 0:
        return {
            "ss_token": 0, "ss_behavior": 0, "ss_interaction": 0, "ss_error": 0,
            "eta_sq_token": 0, "eta_sq_behavior": 0, "is_token_dominated": False,
        }
    
    # Sum of squares
    ss_token = 2 * n * ((mean_tokens - grand_mean)**2 + (mean_no_tokens - grand_mean)**2)
    ss_behavior = 2 * n * ((mean_reasoning - grand_mean)**2 + (mean_nonreasoning - grand_mean)**2)
    
    # Interaction
    ss_interaction = n * sum([
        (mean_A - mean_tokens - mean_reasoning + grand_mean)**2,
        (mean_B - mean_no_tokens - mean_reasoning + grand_mean)**2,
        (mean_C - mean_tokens - mean_nonreasoning + grand_mean)**2,
        (mean_D - mean_no_tokens - mean_nonreasoning + grand_mean)**2,
    ])
    
    # Error (within-group variance)
    ss_error = (
        np.sum((a_A[:n] - mean_A)**2) +
        np.sum((a_B[:n] - mean_B)**2) +
        np.sum((a_C[:n] - mean_C)**2) +
        np.sum((a_D[:n] - mean_D)**2)
    )
    
    ss_total = ss_token + ss_behavior + ss_interaction + ss_error
    
    # Eta-squared
    eta_sq_token = ss_token / ss_total if ss_total > 0 else 0
    eta_sq_behavior = ss_behavior / ss_total if ss_total > 0 else 0
    
    # Decision rule: token-dominated if η²_token > 2 * η²_behavior AND η²_token > 0.1
    is_token_dominated = (eta_sq_token > 2 * eta_sq_behavior) and (eta_sq_token > 0.1)
    
    return {
        "ss_token": float(ss_token),
        "ss_behavior": float(ss_behavior),
        "ss_interaction": float(ss_interaction),
        "ss_error": float(ss_error),
        "ss_total": float(ss_total),
        "eta_sq_token": float(eta_sq_token),
        "eta_sq_behavior": float(eta_sq_behavior),
        "is_token_dominated": is_token_dominated,
        "cell_means": {
            "A": float(mean_A),
            "B": float(mean_B),
            "C": float(mean_C),
            "D": float(mean_D),
        },
    }
