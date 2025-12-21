"""
LLM-based judge for evaluating mathematical answer equivalence.

Uses OpenRouter API to call LLMs for judging whether two mathematical
expressions are equivalent, handling LaTeX notation and various formats.
"""

import os
import json
import requests
import time
from typing import Optional


class LLMJudge:
    """
    LLM-based judge for checking mathematical expression equivalence.
    
    Uses the OpenRouter API to call models like Gemini 3 Flash for
    determining if two mathematical expressions are equivalent.
    
    ## Usage
    
    ```python
    judge = LLMJudge()
    
    # Check if two expressions are equivalent
    result = judge.check_equivalence("2x + 2", "2(x + 1)")
    print(result)  # True
    
    result = judge.check_equivalence("\\sqrt{4}", "2")
    print(result)  # True
    ```
    """
    
    JUDGE_PROMPT_TEMPLATE = """You are a mathematical equivalence checker. Your task is to determine if two mathematical expressions or answers are equivalent.

Expected answer: {expected}
Model's answer: {predicted}

Consider the following:
1. Mathematical equivalence (e.g., "2x + 2" = "2(x + 1)")
2. Different notations (e.g., "1/2" = "0.5" = "\\frac{{1}}{{2}}")
3. Simplified vs unsimplified forms (e.g., "\\sqrt{{4}}" = "2")
4. Order of terms (e.g., "x + y" = "y + x")
5. Equivalent expressions with different symbols (e.g., "±2" matches if model says "2 or -2")
6. Text answers should match semantically (e.g., "east" = "East" = "EAST")

Respond with ONLY "YES" if the answers are equivalent, or "NO" if they are not.
Do not provide any explanation."""

    def __init__(
        self,
        model: str = "google/gemini-2.0-flash-001",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            model: OpenRouter model name (default: Gemini 2.0 Flash)
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment "
                "variable or pass api_key argument."
            )
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def _call_api(self, prompt: str) -> str:
        """Make API call to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 10,  # We only need YES or NO
            "temperature": 0.0,  # Deterministic
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=30,
                )
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
                
            except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"API call failed after {self.max_retries} attempts: {e}")
        
        return ""
    
    def check_equivalence(self, predicted: str, expected: str) -> bool:
        """
        Check if predicted answer is equivalent to expected answer.
        
        Args:
            predicted: The model's predicted answer
            expected: The expected/ground truth answer
        
        Returns:
            True if answers are equivalent, False otherwise
        """
        # Quick check for exact match (case-insensitive)
        if predicted.strip().lower() == expected.strip().lower():
            return True
        
        # Use LLM judge for more complex comparison
        prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            expected=expected,
            predicted=predicted,
        )
        
        response = self._call_api(prompt)
        return response.upper().startswith("YES")
    
    def batch_check(
        self,
        predictions: list[str],
        expected: list[str],
        verbose: bool = False,
    ) -> list[bool]:
        """
        Check equivalence for multiple prediction-expected pairs.
        
        Args:
            predictions: List of predicted answers
            expected: List of expected answers
            verbose: Print progress
        
        Returns:
            List of boolean results
        """
        results = []
        
        for i, (pred, exp) in enumerate(zip(predictions, expected)):
            if verbose and i % 10 == 0:
                print(f"Checking {i+1}/{len(predictions)}...")
            
            try:
                result = self.check_equivalence(pred, exp)
            except Exception as e:
                print(f"Warning: Judge failed for item {i}: {e}")
                result = False  # Default to incorrect on failure
            
            results.append(result)
        
        return results


# Global judge instance (lazy initialization)
_global_judge: Optional[LLMJudge] = None


def get_judge() -> LLMJudge:
    """Get or create the global LLM judge instance."""
    global _global_judge
    if _global_judge is None:
        _global_judge = LLMJudge()
    return _global_judge


def check_math_equivalence(predicted: str, expected: str) -> bool:
    """
    Convenience function to check mathematical equivalence.
    
    Uses the global LLM judge instance.
    
    Args:
        predicted: The model's predicted answer
        expected: The expected/ground truth answer
    
    Returns:
        True if answers are equivalent, False otherwise
    """
    judge = get_judge()
    return judge.check_equivalence(predicted, expected)
