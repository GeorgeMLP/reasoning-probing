"""Reasoning dataset loaders for reasoning chain text."""

import re
from typing import Optional, Literal
from datasets import load_dataset

from .base import BaseDataset, TextSample


class S1KDataset(BaseDataset):
    """
    Loader for the s1K-1.1 dataset (simplescaling/s1K-1.1).
    
    Contains reasoning traces from Gemini Flash Thinking and DeepSeek R1.
    """
    
    def __init__(
        self,
        max_samples: Optional[int] = None,
        trajectory_source: Literal["gemini", "deepseek", "both"] = "both",
        min_text_length: int = 100,
        max_text_length: int = 4000,
    ):
        """
        Args:
            max_samples: Maximum number of samples to load.
            trajectory_source: Which reasoning trajectories to use:
                - "gemini": Only Gemini Flash Thinking trajectories
                - "deepseek": Only DeepSeek R1 trajectories  
                - "both": Both trajectories (doubles the dataset size)
            min_text_length: Minimum text length to include.
            max_text_length: Maximum text length (longer texts are truncated).
        """
        super().__init__(max_samples)
        self.trajectory_source = trajectory_source
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
    
    def _load_samples(self) -> list[TextSample]:
        """Load samples from the s1K dataset."""
        dataset = load_dataset(
            path="simplescaling/s1K-1.1",
            split="train",
        )
        
        samples = []
        
        for item in dataset:
            cot_type = item.get("cot_type", "unknown")
            
            # Get trajectories based on source preference
            trajectories = []
            if self.trajectory_source in ["gemini", "both"]:
                gemini_traj = item.get("gemini_thinking_trajectory", "")
                if gemini_traj and len(gemini_traj) >= self.min_text_length:
                    trajectories.append(("gemini", gemini_traj))
            
            if self.trajectory_source in ["deepseek", "both"]:
                deepseek_traj = item.get("deepseek_thinking_trajectory", "")
                if deepseek_traj and len(deepseek_traj) >= self.min_text_length:
                    trajectories.append(("deepseek", deepseek_traj))
            
            for source, text in trajectories:
                if len(text) > self.max_text_length:
                    text = text[:self.max_text_length]
                
                samples.append(TextSample(
                    text=text,
                    is_reasoning=True,
                    source=f"s1k_{source}",
                    metadata={
                        "cot_type": cot_type,
                        "trajectory_source": source,
                        "question": item.get("question", ""),
                    },
                ))
        
        return samples


class GeneralInquiryCoTDataset(BaseDataset):
    """
    Loader for the General Inquiry CoT dataset (moremilk/General_Inquiry_Thinking-Chain-Of-Thought).
    
    Contains reasoning chains wrapped in <think>...</think> tags.
    """
    
    def __init__(
        self,
        max_samples: Optional[int] = None,
        extract_thinking: bool = True,
        min_text_length: int = 100,
        max_text_length: int = 4000,
    ):
        """
        Args:
            max_samples: Maximum number of samples to load.
            extract_thinking: If True, extract only the content between <think> tags.
                             If False, include the full reasoning field.
            min_text_length: Minimum text length to include.
            max_text_length: Maximum text length (longer texts are truncated).
        """
        super().__init__(max_samples)
        self.extract_thinking = extract_thinking
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
    
    def _extract_thinking_content(self, text: str) -> str:
        """Extract content between <think> and </think> tags."""
        pattern = r"<think>(.*?)</think>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
    
    def _load_samples(self) -> list[TextSample]:
        """Load samples from the General Inquiry CoT dataset."""
        dataset = load_dataset(
            path="moremilk/General_Inquiry_Thinking-Chain-Of-Thought",
            split="train",
        )
        
        samples = []
        
        for item in dataset:
            metadata = item.get("metadata", {})
            reasoning = metadata.get("reasoning", "")
            
            if not reasoning:
                continue
            
            # Extract thinking content if requested
            if self.extract_thinking:
                text = self._extract_thinking_content(reasoning)
            else:
                text = reasoning
            
            # Filter by length
            if len(text) < self.min_text_length:
                continue
            
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length]
            
            samples.append(TextSample(
                text=text,
                is_reasoning=True,
                source="general_inquiry_cot",
                metadata={
                    "difficulty": metadata.get("difficulty"),
                    "topic": metadata.get("topic"),
                    "question": item.get("question", ""),
                },
            ))
        
        return samples


def get_reasoning_dataset(
    name: Literal["s1k", "general_inquiry_cot", "combined"],
    max_samples: Optional[int] = None,
    **kwargs,
) -> BaseDataset:
    """
    Factory function to get a reasoning dataset by name.
    
    Args:
        name: Dataset name - "s1k", "general_inquiry_cot", or "combined"
        max_samples: Maximum samples to load
        **kwargs: Additional arguments passed to the dataset constructor
    
    Returns:
        A loaded dataset instance
    """
    if name == "s1k":
        return S1KDataset(max_samples=max_samples, **kwargs)
    elif name == "general_inquiry_cot":
        return GeneralInquiryCoTDataset(max_samples=max_samples, **kwargs)
    elif name == "combined":
        # Combine both datasets
        s1k = S1KDataset(max_samples=max_samples // 2 if max_samples else None, **kwargs)
        giq = GeneralInquiryCoTDataset(max_samples=max_samples // 2 if max_samples else None, **kwargs)
        s1k.load()
        giq.load()
        
        # Create a combined dataset
        combined = BaseDataset.__new__(BaseDataset)
        combined._samples = s1k._samples + giq._samples
        combined._loaded = True
        combined.max_samples = max_samples
        if max_samples and len(combined._samples) > max_samples:
            combined._samples = combined._samples[:max_samples]
        return combined
    else:
        raise ValueError(f"Unknown reasoning dataset: {name}")

