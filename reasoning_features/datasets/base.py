"""Base dataset interface for reasoning features analysis."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional
from torch import Tensor
from jaxtyping import Int


@dataclass
class TextSample:
    """A single text sample with metadata."""
    text: str
    is_reasoning: bool
    source: str  # Dataset source name
    metadata: Optional[dict] = None


class BaseDataset(ABC):
    """Abstract base class for all datasets."""
    
    def __init__(self, max_samples: Optional[int] = None):
        self.max_samples = max_samples
        self._samples: list[TextSample] = []
        self._loaded = False
    
    @abstractmethod
    def _load_samples(self) -> list[TextSample]:
        """Load samples from the dataset. Override in subclasses."""
        pass
    
    def load(self) -> "BaseDataset":
        """Load the dataset."""
        if not self._loaded:
            self._samples = self._load_samples()
            if self.max_samples is not None:
                self._samples = self._samples[:self.max_samples]
            self._loaded = True
        return self
    
    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._samples)
    
    def __iter__(self) -> Iterator[TextSample]:
        if not self._loaded:
            self.load()
        return iter(self._samples)
    
    def __getitem__(self, idx: int) -> TextSample:
        if not self._loaded:
            self.load()
        return self._samples[idx]
    
    def get_texts(self) -> list[str]:
        """Get all text samples as a list."""
        if not self._loaded:
            self.load()
        return [s.text for s in self._samples]
    
    def tokenize(
        self,
        tokenizer,
        max_length: int = 512,
        truncation: bool = True,
        padding: str = "max_length",
    ) -> Int[Tensor, "batch seq"]:
        """Tokenize all samples."""
        if not self._loaded:
            self.load()
        
        texts = self.get_texts()
        encoded = tokenizer(
            texts,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors="pt",
        )
        return encoded["input_ids"]


@dataclass
class BenchmarkSample:
    """A single benchmark sample with question and expected answer."""
    question: str
    expected_answer: str
    metadata: Optional[dict] = None


class BaseBenchmark(ABC):
    """Abstract base class for evaluation benchmarks."""
    
    def __init__(self):
        self._samples: list[BenchmarkSample] = []
        self._loaded = False
    
    @abstractmethod
    def _load_samples(self) -> list[BenchmarkSample]:
        """Load benchmark samples. Override in subclasses."""
        pass
    
    @abstractmethod
    def check_answer(self, predicted: str, expected: str) -> bool:
        """Check if predicted answer matches expected. Override in subclasses."""
        pass
    
    @abstractmethod
    def format_prompt(self, question: str) -> str:
        """Format question into a prompt. Override in subclasses."""
        pass
    
    def load(self) -> "BaseBenchmark":
        """Load the benchmark."""
        if not self._loaded:
            self._samples = self._load_samples()
            self._loaded = True
        return self
    
    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._samples)
    
    def __iter__(self) -> Iterator[BenchmarkSample]:
        if not self._loaded:
            self.load()
        return iter(self._samples)
    
    def __getitem__(self, idx: int) -> BenchmarkSample:
        if not self._loaded:
            self.load()
        return self._samples[idx]
    
    def evaluate(self, predictions: list[str]) -> dict:
        """Evaluate predictions against expected answers."""
        if not self._loaded:
            self.load()
        
        if len(predictions) != len(self._samples):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) doesn't match "
                f"number of samples ({len(self._samples)})"
            )
        
        correct = 0
        results = []
        for pred, sample in zip(predictions, self._samples):
            is_correct = self.check_answer(pred, sample.expected_answer)
            correct += int(is_correct)
            results.append({
                "question": sample.question,
                "predicted": pred,
                "expected": sample.expected_answer,
                "correct": is_correct,
            })
        
        return {
            "accuracy": correct / len(self._samples),
            "correct": correct,
            "total": len(self._samples),
            "results": results,
        }

