from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from featureinterp.core import ExplanationComponent


@dataclass
class ComplexityExample:
    explanation_component: ExplanationComponent
    """The explanation component that we're trying to predict the complexity of."""
    complexity: float
    """The complexity of the explanation component."""
    complexity_explanation: str
    """A short explanation of why the complexity is what it is."""


class ComplexityExampleSet(Enum):
    """Determines which few-shot examples to use when sampling explanations."""
    ORIGINAL = "original"

    @classmethod
    def from_string(cls, string: str) -> ComplexityExampleSet:
        for example_set in ComplexityExampleSet:
            if example_set.value == string:
                return example_set
        raise ValueError(f"Unrecognized example set: {string}")

    def get_examples(self) -> list[ComplexityExample]:
        """Returns regular examples for use in a few-shot prompt."""
        if self is ComplexityExampleSet.ORIGINAL:
            return ORIGINAL_EXAMPLES
        else:
            raise ValueError(f"Unhandled example set: {self}")


ORIGINAL_EXAMPLES = [
    ComplexityExample(
        explanation_component=ExplanationComponent(
            activates_on="the word 'instruments', specifically in a musical description, catalog, or reference",
            strength=5
        ),
        complexity=1,
        complexity_explanation="Low complexity - only activates on specific words."
    ),
    ComplexityExample(
        explanation_component=ExplanationComponent(
            activates_on="present tense verbs ending in 'ing'",
            strength=5
        ),
        complexity=2,
        complexity_explanation="Moderate complexity - requires understanding verb tenses and specific suffix patterns."
    ),
    ComplexityExample(
        explanation_component=ExplanationComponent(
            activates_on="words related to medical conditions, in the context of movies and filmmaking",
            strength=3
        ),
        complexity=3,
        complexity_explanation="Higher complexity due to the need to recognize medical terminology in metaphorical usage."
    ),
    ComplexityExample(
        explanation_component=ExplanationComponent(
            activates_on="The word 'risk', in the context of medical research/studies",
            strength=2
        ),
        complexity=1,
        complexity_explanation="Moderate complexity - only activates on specific words in medical context."
    ),
    ComplexityExample(
        explanation_component=ExplanationComponent(
            activates_on="expressions of skepticism",
            strength=3
        ),
        complexity=5,
        complexity_explanation="Very high complexity due to the abstract nature of skepticism."
    ),
    ComplexityExample(
        explanation_component=ExplanationComponent(
            activates_on="words that reflect negative judgments",
            strength=4
        ),
        complexity=5,
        complexity_explanation="Very high complexity due to the abstract nature of negative judgments."
    )
]