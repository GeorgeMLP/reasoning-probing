"""Pile dataset loader for non-reasoning text."""

from typing import Optional
from datasets import load_dataset

from .base import BaseDataset, TextSample


class PileDataset(BaseDataset):
    """
    Loader for the Pile dataset (monology/pile-uncopyrighted).
    
    Used as the non-reasoning baseline dataset.
    """
    
    def __init__(
        self,
        max_samples: Optional[int] = None,
        split: str = "train",
        min_text_length: int = 100,
        max_text_length: int = 2000,
        filter_non_ascii: bool = True,
        streaming: bool = True,
    ):
        super().__init__(max_samples)
        self.split = split
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.filter_non_ascii = filter_non_ascii
        self.streaming = streaming
    
    def _load_samples(self) -> list[TextSample]:
        """Load samples from the Pile dataset."""
        dataset = load_dataset(
            path="monology/pile-uncopyrighted",
            split=self.split,
            streaming=self.streaming,
        )
        
        samples = []
        target_count = self.max_samples * 3 if self.max_samples else 10000
        
        for item in dataset:
            text = item["text"]
            
            # Filter by length
            if len(text) < self.min_text_length:
                continue
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length]
            
            # Filter non-ASCII if requested
            if self.filter_non_ascii and len(text) != len(text.encode()):
                continue
            
            samples.append(TextSample(
                text=text,
                is_reasoning=False,
                source="pile",
                metadata=item.get("meta", {}),
            ))
            
            if len(samples) >= target_count:
                break
        
        return samples
