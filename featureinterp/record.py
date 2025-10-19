from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

from featureinterp import utils


@dataclass
class FeatureExpressionRecord:
    """Collated lists of tokens and their expressions for a single sae index."""

    tokens: List[str]
    """Tokens in the text sequence, represented as strings."""
    expressions: List[float]
    """Raw expression values on each token in the text sequence."""
    holistic_expressions: List[float]
    """Holistic expression values on each token in the text sequence."""
    
    dataset_index: int
    """Metadata recording which batch index in the dataset this record came from."""


@dataclass
class SAEIndexId:
    """Identifier for a SAE feature in an artificial neural network."""

    layer_index: int
    """The index of layer the SAE is in. The first layer has index 0."""
    latent_index: int
    """The latent index of index within the SAE."""


class ComplementaryRecordSource(Enum):
    """Type of complementary records to include in the sample."""
    RANDOM = "random"
    RANDOM_NEGATIVE = "random_negative"
    SIMILAR = "similar"
    SIMILAR_NEGATIVE = "similar_negative"
    SIMILAR_PROJECTED = "similar_projected"
    SIMILAR_NEGATIVE_PROJECTED = "similar_negative_projected"


@dataclass
class RecordSliceParams:
    """How to select splits (train, valid, etc.) of records."""

    positive_examples_per_split: int
    """The number of top-expressing examples to include in each split."""
    
    complementary_examples_per_split: Optional[int] = None
    """The number of complementary examples to include in each split."""
    
    complementary_record_source: Optional[ComplementaryRecordSource] = None
    """The type of complementary records to include."""


@dataclass
class SAEIndexRecord:
    """SAE index activation data, including summary stats and notable records."""

    id: SAEIndexId
    """Identifier for the sae index."""
    
    max_expression: float
    max_holistic_expression: float
    
    most_act_records: list[FeatureExpressionRecord] = field(default_factory=list)
    """Records with the most positive figure of merit value for this feature."""
    
    random_records: list[FeatureExpressionRecord] = field(default_factory=list)
    """Random records from the dataset."""

    random_no_act_records: list[FeatureExpressionRecord] = field(default_factory=list)
    """Random records with negative expression."""

    similar_records: list[FeatureExpressionRecord] = field(default_factory=list)
    """Records that are semantically similar to positive examples."""

    similar_no_act_records: list[FeatureExpressionRecord] = field(default_factory=list)
    """Records that are semantically similar but have negative expression."""
    
    similar_projected_records: list[FeatureExpressionRecord] = field(default_factory=list)
    """Semantically similar records selected using the projection matrix in the similarity retriever."""

    similar_projected_no_act_records: list[FeatureExpressionRecord] = field(default_factory=list)
    """Semantically similar negative records selected using the projection matrix in the similarity retriever."""

    def train_records(
        self, slice_params: RecordSliceParams,
    ) -> list[FeatureExpressionRecord]:
        return self._retrieve_combined_sample(slice_params, "train")

    def valid_records(
        self, slice_params: RecordSliceParams,
    ) -> list[FeatureExpressionRecord]:
        return self._retrieve_combined_sample(slice_params, "valid")

    def test_records(
        self, slice_params: RecordSliceParams,
    ) -> list[FeatureExpressionRecord]:
        return self._retrieve_combined_sample(slice_params, "test")

    def _get_sample(
        self,
        examples_per_split: int | None,
        records: list[FeatureExpressionRecord],
        split: str,
    ) -> list[FeatureExpressionRecord]:

        splits = ["train", "valid", "test"]
        if examples_per_split is None:
            examples_per_split = len(records) // len(splits)
        assert len(records) >= examples_per_split * len(splits)
        return records[self._get_slices_for_splits(splits, examples_per_split)[split]]
    
    def _retrieve_combined_sample(
        self,
        slice_params: RecordSliceParams,
        split: str,
    ) -> list[FeatureExpressionRecord]:
        """A combined sample of the most positive records and complementary records."""

        records = self._get_sample(
            slice_params.positive_examples_per_split,
            self.most_act_records,
            split,
        )
        
        if (
            slice_params.complementary_record_source and
            slice_params.complementary_examples_per_split
        ):
            complementary_records = {
                ComplementaryRecordSource.RANDOM: self.random_records,
                ComplementaryRecordSource.RANDOM_NEGATIVE: self.random_no_act_records,
                ComplementaryRecordSource.SIMILAR: self.similar_records,
                ComplementaryRecordSource.SIMILAR_NEGATIVE: self.similar_no_act_records,
                ComplementaryRecordSource.SIMILAR_PROJECTED: self.similar_projected_records,
                ComplementaryRecordSource.SIMILAR_NEGATIVE_PROJECTED: self.similar_projected_no_act_records,
            }[slice_params.complementary_record_source]
            
            if len(complementary_records) < slice_params.complementary_examples_per_split * 3:
                import warnings
                warnings.warn('Using backup records in _retrieve_combined_sample()', UserWarning)
                backup_records = self.random_no_act_records + self.random_records
                complementary_records = backup_records

            records.extend(self._get_sample(
                slice_params.complementary_examples_per_split,
                complementary_records,
                split,
            ))

        return records

    def _get_slices_for_splits(
        self,
        splits: list[str],
        num_records_per_split: int,
    ) -> dict[str, slice]:
        """
        Get equal-sized interleaved subsets for each of a list of splits, given the
        number of elements to include in each split.
        """
        stride = len(splits)
        num_records_for_even_splits = num_records_per_split * stride
        slices_by_split = {
            split: slice(split_index, num_records_for_even_splits, stride)
            for split_index, split in enumerate(splits)
        }
        self._check_slices(
            slices_by_split=slices_by_split,
            expected_num_values=num_records_for_even_splits,
        )
        return slices_by_split

    def _check_slices(
        self,
        slices_by_split: dict[str, slice],
        expected_num_values: int,
    ) -> None:
        """Assert that the slices are disjoint and fully cover the intended range."""
        indices = set()
        sum_of_slice_lengths = 0
        n_splits = len(slices_by_split.keys())
        for s in slices_by_split.values():
            subrange = range(expected_num_values)[s]
            sum_of_slice_lengths += len(subrange)
            indices |= set(subrange)
        assert (
            sum_of_slice_lengths == expected_num_values
        ), f"{sum_of_slice_lengths=} != {expected_num_values=}"
        stride = n_splits
        expected_indices = set.union(
            *[
                set(range(start_index, expected_num_values, stride))
                for start_index in range(n_splits)
            ]
        )
        assert indices == expected_indices, f"{indices=} != {expected_indices=}"
    
    def __str__(self) -> str:
        groups = {
            "Most positive records": self.most_act_records,
            "Similar records": self.similar_records,
            "Similar records with negative expression": self.similar_no_act_records,
            "Random records": self.random_records,
            "Random records with negative expression": self.random_no_act_records,
            "Similar projected records": self.similar_projected_records,
            "Similar projected records with negative expression": self.similar_projected_no_act_records,
        }
        
        string_builder = ''
        for group_name, records in groups.items():
            string_builder += f'================={group_name}=================\n'
            for record in records[:5]:
                string_builder += f'Record {record.dataset_index}\n'
                
                expressions = np.array(record.expressions)

                string_builder += 'SAE activations\n'
                string_builder += utils.render_expressions(
                    record.tokens, expressions / self.max_expression
                ) + '\n'
                
                # holistic_expressions = np.array(record.holistic_expressions)
                # string_builder += 'Holistic activations\n'
                # string_builder += utils.render_expressions(
                #     record.tokens, holistic_expressions / self.max_holistic_expression
                # ) + '\n'
                
        return string_builder
