from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from featureinterp.formatting import FormattedFeatureExpressionRecord
from featureinterp.core import StructuredExplanation


@dataclass
class RecordExample:
    records: list[FormattedFeatureExpressionRecord]
    string_explanation: str
    structured_explanation: StructuredExplanation
    first_revealed_expression_indices: list[int]
    """
    For each expression record, the index of the first token for which the expression value in the
    prompt should be an actual number rather than "unknown".

    Examples all start with the expressions rendered as "unknown", then transition to revealing
    specific normalized expression values. The goal is to lead the model to predict that expression
    sequences will eventually transition to predicting specific expression values instead of just
    "unknown". This lets us cheat and get predictions of expression values for every token in a
    single round of inference by having the expressions in the sequence we're predicting always be
    "unknown" in the prompt: the model will always think that maybe the next token will be a real
    expression.
    
    These are also used as few-shot examples for the explainer.
    """


class RecordExampleSet(Enum):
    """Determines which few-shot examples to use when sampling explanations."""
    ORIGINAL = "original"

    @classmethod
    def from_string(cls, string: str) -> RecordExampleSet:
        for example_set in RecordExampleSet:
            if example_set.value == string:
                return example_set
        raise ValueError(f"Unrecognized example set: {string}")

    def get_examples(self) -> list[RecordExample]:
        """Returns regular examples for use in a few-shot prompt."""
        if self is RecordExampleSet.ORIGINAL:
            return ORIGINAL_EXAMPLES
        else:
            raise ValueError(f"Unhandled example set: {self}")
        

def _parse_record(
    record: list[tuple[str, int, int]]
) -> FormattedFeatureExpressionRecord:

    return FormattedFeatureExpressionRecord(*list(map(list, zip(*record))))


ORIGINAL_EXAMPLES = [
    RecordExample(
        records=[
            _parse_record(
                [
                    (" javascript", 0, 0),
                    (" to", 0, 0),
                    (" provide", 0, 0),
                    ("\n", 0, 0),
                    ("you", 0, 0),
                    (" with", 0, 0),
                    (" a", 0, 0),
                    (" positive", 5, 5),
                    (" online", 0, 0),
                    (" shopping", 0, 0),
                    (" experience", 0, 0),
                    (".", 0, 0),
                    ("\n\n", 0, 0),
                    ("To", 0, 0),
                    (" enable", 0, 0),
                    (" javascript", 0, 0),
                ]
            ),
            _parse_record(
                [
                    (" and", 0, 0),
                    (" ", 0, 0),
                    ("3", 0, 0),
                    (" and", 0, 0),
                    (" Are", 0, 0),
                    (" Negative", 3, 3),
                    (" for", 1, 0),
                    (" Cla", 0, 0),
                    ("udin", 0, 0),
                    (" ", 0, 0),
                    ("4", 0, 0),
                    (".", 0, 0),
                    ("\n", 0, 0),
                    ("In", 0, 0),
                    ("vasive", 0, 0),
                    (" apo", 0, 0),
                ]
            )
        ],
        first_revealed_expression_indices=[10, 3],
        structured_explanation=StructuredExplanation.from_json("""[
    {
        "activates_on": "The words 'positive' and 'negative', in upper and lowercase.",
        "strength": 3
    },
]"""),
        string_explanation="the words 'positive' and 'negative'"
    ),
    RecordExample(
        records=[
            _parse_record(
                [
                    ("\n", 0, 0),
                    ("<", 0, 2),
                    ("title", 0, 4),
                    (">", 0, 0),
                    ("Installation", 0, 1),
                    (" of", 0, 0),
                    (" Less", 0, 0),
                    ("</", 5, 3),
                    ("title", 0, 0),
                    (">", 0, 0),
                    ("\n\n", 0, 0),
                    ("<", 0, 0),
                    ("para", 0, 0),
                    (">", 0, 0),
                    ("Install", 0, 0),
                    (" Less", 0, 0),
                ]
            ),
            _parse_record(
                [
                    ("\n\n", 0, 0),
                    ("<", 0, 1),
                    ("title", 0, 3),
                    (">", 0, 0),
                    ("Code", 0, 0),
                    ("Mirror", 0, 1),
                    (":", 0, 0),
                    (" HTML", 0, 0),
                    (" mixed", 0, 0),
                    (" mode", 0, 0),
                    ("</", 4, 3),
                    ("title", 0, 0),
                    (">", 0, 0),
                    ("\n", 0, 0),
                    ("<", 0, 0),
                    ("meta", 0, 0),
                ]
            ),
        ],
        first_revealed_expression_indices=[5, 14],
        structured_explanation=StructuredExplanation.from_json("""[
    {
        "activates_on": "The HTML tag component '</' when following an opening '<title' tag.",
        "strength": 4
    },
]"""),
        string_explanation="the HTML tag component '</' when following an opening '<title' tag",
    ),
    RecordExample(
        records=[
            _parse_record(
                [
                    (" third", 0, 1),
                    ("-", 0, 0),
                    ("generation", 0, 0),
                    (" cephal", 4, 3),
                    ("ospor", 1, 4),
                    ("in", 2, 1),
                    ("-", 1, 0),
                    ("resistant", 3, 0),
                    (" Escherichia", 0, 0),
                    (" coli", 1, 0),
                    (" from", 0, 0),
                    (" bro", 0, 0),
                    ("ilers", 0, 0),
                    (",", 0, 0),
                    (" swine", 0, 0),
                    (",", 0, 0),
                ]
            ),
            _parse_record(
                [
                    (" NC", 0, 1),
                    ("LLS", 0, 0),
                    (" standard", 0, 0),
                    (" for", 0, 0),
                    (" susceptibility", 2, 4),
                    (" testing", 1, 0),
                    (" of", 0, 0),
                    (" yea", 0, 0),
                    ("sts", 0, 0),
                    ("?", 0, 0),
                    ("].", 0, 0),
                    ("\n", 0, 0),
                    ("The", 0, 0),
                    (" E", 0, 0),
                    ("test", 0, 0),
                    (" and", 0, 0),
                ]
            ),
        ],
        first_revealed_expression_indices=[0, 10],
        structured_explanation=StructuredExplanation.from_json("""[
    {
        "activates_on": "Words relating to susceptibility and resistance in the context of biological testing and scientific research.",
        "strength": 3
    },
    {
        "activates_on": "Antibiotic names.",
        "strength": 4
    }
]"""),
        string_explanation="language related to biological resistance"
    ),
    RecordExample(
        records=[
            _parse_record(
                [
                    ("hage", 0, 0),
                    (" migration", 0, 0),
                    (" inhibitory", 0, 0),
                    (" factor", 0, 0),
                    (" in", 0, 0),
                    (" allergic", 3, 2),
                    (" rhin", 5, 2),
                    ("itis", 2, 0),
                    (":", 0, 0),
                    (" its", 0, 0),
                    (" identification", 0, 0),
                    (" in", 0, 0),
                    (" eosin", 2, 0),
                    ("oph", 0, 0),
                    ("ils", 0, 0),
                    (" at", 0, 0),
                ]
            ),
            _parse_record(
                [
                    (" prevalence", 0, 0),
                    (" of", 0, 0),
                    (" asthma", 2, 0),
                    (" and", 0, 0),
                    (" allergic", 4, 4),
                    (" disorders", 3, 1),
                    (" was", 0, 0),
                    (" assessed", 0, 0),
                    (" in", 0, 0),
                    (" ", 1, 0),
                    ("9", 0, 0),
                    ("-", 0, 0),
                    ("1", 0, 0),
                    ("1", 0, 0),
                    (" year", 0, 0),
                    ("-", 0, 0),
                ]
            ),
        ],
        first_revealed_expression_indices=[3, 5],
        structured_explanation=StructuredExplanation.from_json("""[
    {
        "activates_on": "The word 'allergic'.",
        "strength": 3
    },
    {
        "activates_on": "Diseases and disorders, only if following the word 'allergic'.",
        "strength": 4
    }
]"""),
        string_explanation="language related to diseases, disorders, and allergies"
    ),
    RecordExample(
        records=[
            _parse_record(
                [
                    (" communication", 0, 1),
                    (" system", 0, 1),
                    (" in", 0, 0),
                    (" which", 0, 0),
                    (" Av", 0, 2),
                    (" apparatus", 1, 3),
                    (",", 0, 0),
                    (" such", 0, 2),
                    (" as", 0, 1),
                    (" a", 1, 0),
                    (" video", 5, 3),
                    (" tape", 2, 0),
                    (" recorder", 2, 0),
                    (" (", 0, 0),
                    ("V", 1, 0),
                    ("TR", 2, 0),
                ]
            ),
            _parse_record(
                [
                    (" camera", 0, 4),
                    (" view", 2, 1),
                    (" at", 0, 1),
                    (" the", 0, 0),
                    (" same", 0, 2),
                    (" time", 1, 1),
                    ("?", 1, 0),
                    ("\n\n", 0, 0),
                    ("I", 0, 0),
                    (" have", 0, 0),
                    (" a", 0, 0),
                    (" requirement", 0, 1),
                    (" to", 0, 1),
                    (" show", 1, 1),
                    (" both", 1, 0),
                    (" rear", 2, 2),
                ]
            )
        ],
        first_revealed_expression_indices=[2,12],
        structured_explanation=StructuredExplanation.from_json("""[
    {
        "activates_on": "Words generally related to views, videos, and recording.",
        "strength": 3
    }
]"""),
        string_explanation="language related to views, videos, and recording"
    ),
]
