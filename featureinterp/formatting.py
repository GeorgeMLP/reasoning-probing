from dataclasses import dataclass
import math
from featureinterp.record import FeatureExpressionRecord


UNKNOWN_EXPRESSION_STRING = "unknown"
MAX_FORMATTED_EXPRESSION = 5


@dataclass
class FormattedFeatureExpressionRecord:
    """Collated lists of tokens and their expressions for a single sae index."""

    tokens: list[str]
    """Tokens in the text sequence, represented as strings."""
    expressions: list[int]
    """Formatted expression values on each token in the text sequence."""
    holistic_expressions: list[int]
    """Formatted holistic expression values on each token in the text sequence."""
    
    dataset_index: int | None = None
    """Metadata recording which batch index in the dataset this record came from."""


def calculate_max_expression(records: list[FeatureExpressionRecord]) -> float:
    """Return the maximum expression value of the feature across all the records."""
    return max([max(record.expressions) for record in records])


def _format_expressions(record: list[float], max_feature_expression: float) -> list[int]:
    """Convert raw feature expressions to integers [0, MAX_FORMATTED_EXPRESSION]."""
    if max_feature_expression <= 0:
        return [0 for x in record]
    return [
        max(
            min(
                MAX_FORMATTED_EXPRESSION,
                math.floor(MAX_FORMATTED_EXPRESSION * x / max_feature_expression)
            ),
            0
        )
        for x in record
    ]


def format_record(
    record: FeatureExpressionRecord,
    max_expression: float,
    max_holistic_expression: float,
) -> FormattedFeatureExpressionRecord:

    return FormattedFeatureExpressionRecord(
        tokens=record.tokens,
        expressions=_format_expressions(record.expressions, max_expression),
        holistic_expressions=_format_expressions(
            record.holistic_expressions, max_holistic_expression
        ),
        dataset_index=record.dataset_index,
    )


def format_records(
    records: list[FeatureExpressionRecord],
    max_expression: float,
    max_holistic_expression: float,
) -> list[FormattedFeatureExpressionRecord]:
    
    return [format_record(r, max_expression, max_holistic_expression) for r in records]


def non_zero_expression_proportion(
    records: list[FormattedFeatureExpressionRecord],
) -> float:
    """Return the proportion of expression values that aren't zero."""
    total_expressions_count = sum([len(record.expressions) for record in records])
    formatted_expressions = [record.expressions for record in records]
    non_zero_expressions_count = sum(
        [len([x for x in exprs if x != 0]) for exprs in formatted_expressions]
    )
    return non_zero_expressions_count / total_expressions_count


def stringify_tokens_for_simulation(tokens: list[str]) -> str:
    """Format a list of tokens into a string with each token marked as having
    an "unknown" expression."""
    
    builder = ''
    builder += '\n<start>\n'
    builder += "\n".join([f"{t}\t{UNKNOWN_EXPRESSION_STRING}" for t in tokens]) 
    builder += '\n<end>\n'
    return builder


def stringify_record(
    record: FormattedFeatureExpressionRecord,
    stringify_expression: str,
    start_index: int = 0,
    omit_zeros: bool = False,
) -> str:
    """Format feature expressions into a string, suitable for use in prompts."""
    tokens = record.tokens
    if stringify_expression == 'expression':
        formatted_expressions = record.expressions
    elif stringify_expression == 'holistic':
        formatted_expressions = record.holistic_expressions
    else:
        raise ValueError(f"Invalid stringify_expression: {stringify_expression}")

    if omit_zeros:
        tokens = [t for t, expr in zip(tokens, formatted_expressions) if expr > 0]
        formatted_expressions = [expr for expr in formatted_expressions if expr > 0]

    entries = []
    assert len(tokens) == len(formatted_expressions)
    for index, token, expression in zip(
        range(len(tokens)), tokens, formatted_expressions
    ):
        expression_string = str(int(expression))
        if index < start_index:
            expression_string = UNKNOWN_EXPRESSION_STRING
        entries.append(f"{token}\t{expression_string}")
    return "\n".join(entries)


def stringify_records(
    records: list[FormattedFeatureExpressionRecord],
    include_holistic_expressions: bool,
    omit_zeros: bool = False,
    start_indices: list[int] | None = None,
) -> str:
    """Format a list of expression records into a string."""

    builder = ''
    for i, record in enumerate(records):
        start_index = 0 if start_indices is None else start_indices[i]
        activation_string = stringify_record(record, 'expression', start_index, omit_zeros)
        holistic_string = stringify_record(record, 'holistic', start_index, omit_zeros)
        if (activation_string == '') and (holistic_string == ''):
            continue
        
        if include_holistic_expressions:
            builder += f'\nRECORD START\n'
            builder += '\nActivating tokens'

        builder += '\n<start>\n'
        builder += activation_string 
        builder += '\n<end>\n'

        if include_holistic_expressions:
            builder += '\nActivation-causing tokens'
            builder += '\n<start>\n'
            builder += holistic_string
            builder += '\n<end>\n'
            builder += f'\nRECORD END\n'
    
    return builder


def stringify_contrasting_expressions(
    tokens: list[str],
    expressions_pred_formatted: list[int],
    expressions_true_formatted: list[int],
) -> str:
    """Format feature expressions into a string, suitable for use in prompts."""
    
    assert (
        len(tokens) == len(expressions_pred_formatted) == len(expressions_true_formatted)
    ), "Tokens and expressions must have matching lengths"

    entries = []
    for token, pred_expression, true_expression in zip(
        tokens, expressions_pred_formatted, expressions_true_formatted
    ):
        pred_expression_string = str(int(pred_expression))
        true_expression_string = str(int(true_expression))
        if pred_expression == true_expression:
            error = "none"
        elif pred_expression > true_expression:
            error = f"{pred_expression - true_expression} too high"
        else:
            error = f"{true_expression - pred_expression} too low"
        entries.append(
            f"{token}\t(predicted: {pred_expression_string}) "
            f"(actual: {true_expression_string}) "
            f"(error: {error})"
        )
    return "\n<start>\n" + "\n".join(entries) + "\n<end>\n"
