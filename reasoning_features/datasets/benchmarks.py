"""Benchmark loaders for evaluation."""

import re
from typing import Literal
from datasets import load_dataset

from .base import BaseBenchmark, BenchmarkSample


# One-shot example for math problems (not from any benchmark)
MATH_ONE_SHOT_EXAMPLE = """Example:
Problem: If $x + y = 106$ and $x - y = 488$, what is the value of $x$?

Solution: Let me work through this step by step.

I need to solve this system of equations. Adding the two equations:
$(x + y) + (x - y) = 106 + 488$

Simplifying the left-hand side:
$2x = 594$

Dividing both sides by 2:
$x = 297$

Therefore, the value of $x$ is $\\boxed{297}$.

"""

# One-shot example for multiple choice (not from any benchmark)
MCQ_ONE_SHOT_EXAMPLE = """Example:
Question: Which of the following is the largest planet in our solar system?
A) Earth
B) Mars
C) Jupiter
D) Saturn

Answer: The largest planet in our solar system is Jupiter, which has a mass more than twice that of all other planets combined. Its diameter is about 11 times that of Earth.

\\boxed{C}

"""


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
        # Note: We start the solution with a specific phrase to encourage
        # step-by-step reasoning. Without this, some base models may output
        # placeholder answers like \boxed{?} immediately, especially with
        # low temperature settings.
        return (
            f"Solve the following math problem. "
            f"Provide your final answer in \\boxed{{}} format.\n\n"
            f"{MATH_ONE_SHOT_EXAMPLE}"
            f"Now solve this problem:\n"
            f"Problem: {question}\n\n"
            f"Solution: Let me work through this step by step.\n\n"
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
        # First try to extract from \boxed{}
        boxed_match = re.search(r"\\boxed\{([ABCD])\}", predicted, re.IGNORECASE)
        if boxed_match:
            return boxed_match.group(1).upper() == expected.strip().upper()
        
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
            f"Provide your final answer in \\boxed{{}} format with "
            f"the letter (A, B, C, or D).\n\n"
            f"{MCQ_ONE_SHOT_EXAMPLE}"
            f"Now answer this question:\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )


class MATH500Benchmark(BaseBenchmark):
    """
    MATH-500 benchmark (HuggingFaceH4/MATH-500).
    
    A subset of 500 math problems with diverse answer formats including
    numerical values, algebraic expressions, and text answers.
    
    Uses an LLM judge to evaluate answer equivalence due to the variety
    of answer formats (e.g., "1 \\pm \\sqrt{19}", "\\text{east}", "2k+2").
    """
    
    def __init__(self, use_llm_judge: bool = True):
        """
        Args:
            use_llm_judge: Whether to use LLM judge for answer checking.
                          If False, uses exact string matching.
        """
        super().__init__()
        self.use_llm_judge = use_llm_judge
        self._judge = None
    
    def _get_judge(self):
        """Lazy load the LLM judge."""
        if self._judge is None and self.use_llm_judge:
            from ..utils.llm_judge import LLMJudge
            self._judge = LLMJudge()
        return self._judge
    
    def _load_samples(self) -> list[BenchmarkSample]:
        """Load MATH-500 samples."""
        dataset = load_dataset(
            path="HuggingFaceH4/MATH-500",
            split="test",
        )
        
        samples = []
        for item in dataset:
            samples.append(BenchmarkSample(
                question=item.get("problem", ""),
                expected_answer=item.get("answer", ""),
                metadata={
                    "subject": item.get("subject"),
                    "level": item.get("level"),
                    "unique_id": item.get("unique_id"),
                    "full_solution": item.get("solution"),
                },
            ))
        
        return samples
    
    def _extract_boxed_answer(self, text: str) -> str:
        """Extract answer from \\boxed{...} format, handling nested braces."""
        # Find all \boxed{} occurrences
        pattern = r"\\boxed\{"
        matches = list(re.finditer(pattern, text))
        
        if not matches:
            return ""
        
        # Get the last match (final answer)
        last_match = matches[-1]
        start = last_match.end()
        
        # Find matching closing brace
        brace_count = 1
        pos = start
        while pos < len(text) and brace_count > 0:
            if text[pos] == "{":
                brace_count += 1
            elif text[pos] == "}":
                brace_count -= 1
            pos += 1
        
        if brace_count == 0:
            return text[start:pos-1].strip()
        
        return ""
    
    def check_answer(self, predicted: str, expected: str) -> bool:
        """
        Check if predicted answer matches expected.
        
        Uses LLM judge for mathematical expression equivalence if enabled.
        """
        # Extract answer from \boxed{} if present
        pred_answer = self._extract_boxed_answer(predicted)
        if not pred_answer:
            # Fall back to last line or whole prediction
            lines = predicted.strip().split("\n")
            pred_answer = lines[-1] if lines else predicted
        
        # Clean up the prediction
        pred_answer = pred_answer.strip()
        expected = expected.strip()
        
        # Quick exact match check
        if pred_answer.lower() == expected.lower():
            return True
        
        # Try simple numerical comparison
        try:
            pred_num = float(pred_answer.replace(",", ""))
            exp_num = float(expected.replace(",", ""))
            if pred_num == exp_num:
                return True
        except (ValueError, TypeError):
            pass
        
        # Use LLM judge for complex expressions
        if self.use_llm_judge:
            try:
                judge = self._get_judge()
                return judge.check_equivalence(pred_answer, expected)
            except Exception as e:
                print(f"Warning: LLM judge failed: {e}")
                return False
        
        # Fall back to string comparison
        return pred_answer.lower() == expected.lower()
    
    def format_prompt(self, question: str) -> str:
        """Format question into a prompt for the model."""
        # Note: We start the solution with a specific phrase to encourage
        # step-by-step reasoning. Without this, some base models may output
        # placeholder answers like \boxed{?} immediately, especially with
        # low temperature settings.
        return (
            f"Solve the following math problem. "
            f"Provide your final answer in \\boxed{{}} format.\n\n"
            f"{MATH_ONE_SHOT_EXAMPLE}"
            f"Now solve this problem:\n"
            f"Problem: {question}\n\n"
            f"Solution: Let me work through this step by step.\n\n"
        )


def get_benchmark(
    name: Literal["aime24", "gpqa_diamond", "math500"],
) -> BaseBenchmark:
    """
    Factory function to get a benchmark by name.
    
    Args:
        name: Benchmark name - "aime24", "gpqa_diamond", or "math500"
    
    Returns:
        A loaded benchmark instance
    """
    benchmarks = {
        "aime24": AIME24Benchmark,
        "gpqa_diamond": GPQADiamondBenchmark,
        "math500": MATH500Benchmark,
    }
    
    if name not in benchmarks:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(benchmarks.keys())}")
    
    return benchmarks[name]()
