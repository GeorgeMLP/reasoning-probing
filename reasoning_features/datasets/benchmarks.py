"""Benchmark loaders for evaluation."""

import re
from typing import Literal
from datasets import load_dataset

from .base import BaseBenchmark, BenchmarkSample


class AIME24Benchmark(BaseBenchmark):
    """
    AIME 2024 benchmark (math-ai/aime24).
    
    Math competition problems with numerical answers in \\boxed{} format.
    """
    
    def __init__(self):
        super().__init__()
    
    def _load_samples(self) -> list[BenchmarkSample]:
        """Load AIME24 samples."""
        dataset = load_dataset(
            path="math-ai/aime24",
            split="test",
        )
        
        samples = []
        for item in dataset:
            # Extract answer from \boxed{} format
            solution = item.get("solution", "")
            answer = self._extract_boxed_answer(solution)
            
            samples.append(BenchmarkSample(
                question=item.get("problem", ""),
                expected_answer=answer,
                metadata={
                    "id": item.get("id"),
                    "url": item.get("url"),
                    "full_solution": solution,
                },
            ))
        
        return samples
    
    def _extract_boxed_answer(self, solution: str) -> str:
        """Extract answer from \\boxed{...} format."""
        # Handle nested braces
        pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        matches = re.findall(pattern, solution)
        if matches:
            return matches[-1].strip()  # Return last boxed answer
        return ""
    
    def check_answer(self, predicted: str, expected: str) -> bool:
        """Check if predicted answer matches expected (numerical comparison)."""
        # Extract number from predicted answer
        pred_answer = self._extract_boxed_answer(predicted)
        if not pred_answer:
            # Try to find a number in the response
            numbers = re.findall(r"(?:^|[^\d])(\d+)(?:[^\d]|$)", predicted)
            if numbers:
                pred_answer = numbers[-1]
        
        # Clean and compare
        pred_clean = pred_answer.strip().replace(",", "")
        exp_clean = expected.strip().replace(",", "")
        
        try:
            # Try numerical comparison
            return float(pred_clean) == float(exp_clean)
        except (ValueError, TypeError):
            # Fall back to string comparison
            return pred_clean == exp_clean
    
    def format_prompt(self, question: str) -> str:
        """Format question into a prompt for the model."""
        return (
            f"Solve the following math problem. "
            f"Provide your final answer in \\boxed{{}} format.\n\n"
            f"Problem: {question}\n\n"
            f"Solution (a number in \\boxed{{}} format only; no other text):"
        )


class GPQADiamondBenchmark(BaseBenchmark):
    """
    GPQA Diamond benchmark (fingertap/GPQA-Diamond).
    
    Graduate-level science questions with multiple choice answers (A/B/C/D).
    """
    
    def __init__(self):
        super().__init__()
    
    def _load_samples(self) -> list[BenchmarkSample]:
        """Load GPQA Diamond samples."""
        dataset = load_dataset(
            path="fingertap/GPQA-Diamond",
            split="test",
        )
        
        samples = []
        for item in dataset:
            samples.append(BenchmarkSample(
                question=item.get("question", ""),
                expected_answer=item.get("answer", "").strip().upper(),
                metadata={},
            ))
        
        return samples
    
    def check_answer(self, predicted: str, expected: str) -> bool:
        """Check if predicted answer matches expected (A/B/C/D)."""
        # Extract letter answer from prediction
        pred_clean = predicted.strip().upper()
        exp_clean = expected.strip().upper()
        
        # Try to find a single letter answer
        if pred_clean in ["A", "B", "C", "D"]:
            return pred_clean == exp_clean
        
        # Look for patterns like "The answer is A" or "(A)"
        patterns = [
            r"(?:answer|choice)[\s:]*(?:is\s+)?([ABCD])\b",
            r"\(([ABCD])\)",
            r"^([ABCD])[\.\)\s]",
            r"([ABCD])$",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, pred_clean, re.IGNORECASE)
            if match:
                return match.group(1).upper() == exp_clean
        
        return False
    
    def format_prompt(self, question: str) -> str:
        """Format question into a prompt for the model."""
        return (
            f"Answer the following multiple choice question. "
            f"Provide only the letter (A, B, C, or D) as your final answer.\n\n"
            f"Question: {question}\n\n"
            f"Answer (a single letter A/B/C/D only; no other text):"
        )


def get_benchmark(
    name: Literal["aime24", "gpqa_diamond"],
) -> BaseBenchmark:
    """
    Factory function to get a benchmark by name.
    
    Args:
        name: Benchmark name - "aime24" or "gpqa_diamond"
    
    Returns:
        A loaded benchmark instance
    """
    benchmarks = {
        "aime24": AIME24Benchmark,
        "gpqa_diamond": GPQADiamondBenchmark,
    }
    
    if name not in benchmarks:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(benchmarks.keys())}")
    
    return benchmarks[name]()
