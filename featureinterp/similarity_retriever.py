import numpy as np
import torch
from torch import Tensor
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torchvision
from jaxtyping import Float, Int
from einops import pack, reduce, einsum
from einops.layers.torch import Rearrange
from transformers import AutoTokenizer
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from info_nce import InfoNCE
from dataclasses import dataclass
from simple_parsing import Serializable
from functools import partial
from typing import Callable, Literal, TypeAlias
from enum import Enum, auto


Predicate: TypeAlias = Callable[[str], bool]
"""Function that tells whether the input sentence is special."""
SimMeasure: TypeAlias = Callable[
    [Float[Tensor, "b d"], Float[Tensor, "n d"]],
    Float[Tensor, "b"]
]
"""Function that takes the embeddings of some sentences and the special sentences,
and returns the similarities of input sentences to the special sentences."""


class SimilarityMeasure(Enum):
    """Different ways to measure similarity between sentences."""
    EMBEDDING = auto()
    """Use raw sentence embeddings."""
    PROJECTED_EMBEDDING = auto() 
    """Project embeddings before computing similarity."""
    CLASSIFICATION = auto()
    """Use a trained classifier to measure similarity."""


@dataclass
class SimilarityRetrieverConfig(Serializable):
    dataset_path: str | None = None
    """The path to the dataset with pre-computed sentence embeddings."""
    device: str = "cuda"
    """The device to compute similarity and train the projection matrix."""
    batch_size: int = 64
    """The batch size for the dataloader."""
    model_name: str = "all-mpnet-base-v2"
    """Name of the sentence transformer model."""
    tokenizer_name: str = "stanford-crfm/arwen-gpt2-medium-x21"
    """Name of the tokenizer for creating the dataloader."""
    similarity_measure: SimilarityMeasure = SimilarityMeasure.EMBEDDING
    """How to measure similarity between sentences."""
    proj_embedding_length: int = 2048
    """The projected dimension of the sentence embeddings."""
    num_epochs: int = 5
    """Number of epochs to train the projection matrix."""
    learning_rate: float = 100.
    """Learning rate for training the projection matrix."""
    dropout_rate: float = 0.
    """Dropout rate for training the projection matrix."""
    proj_batch_size: int = 64
    """Batch size for training the projection matrix."""
    loss_func_name: Literal["infonce"] = "infonce"
    """Name of the loss function for training the projection matrix."""
    optimizer_name: Literal["sgd", "adam"] = "sgd"
    """Name of the optimizer for training the projection matrix."""


class SimilarityRetriever:
    """
    Split a text dataset into quantiles according to sentence similarity.
    
    Parameters
    ----------
    dataset : Dataset | str
        A huggingface text dataset or a file path where the dataset with
        pre-computed sentence embeddings has been saved. Must have 'input_ids'
        column which works with the tokenizer in the config
    config : SimilarityQuantilesConfig
        Configurations for splitting quantiles.
    save_dataset_to_disk : bool
        If True, pre-compute the sentence embeddings and save the dataset to
        disk.
    """
    def __init__(
        self,
        dataset: Dataset | str,
        config: SimilarityRetrieverConfig,
        save_dataset_to_disk: bool = True,
    ) -> None:

        self.config = config
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        self.device = torch.device(self.config.device)
        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
        )
        if isinstance(dataset, Dataset):
            dataset = dataset.with_format('torch')
            self.dataset = self._compute_sentence_embeddings(dataset)
            if save_dataset_to_disk:
                assert self.config.dataset_path is not None
                self.dataset.save_to_disk(self.config.dataset_path)
        else:  # `dataset` is a file path
            self.dataset = load_from_disk(self.config.dataset_path)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
    
    @torch.no_grad()
    def _compute_sentence_embeddings(
        self, dataset: Dataset, batch_size: int = 128
    ) -> Dataset:
        """Pre-compute the sentence embeddings of the input text dataset."""
        assert 'input_ids' in dataset.column_names, 'Pre-process the dataset first.'

        def _compute_sentence_embeddings(
            batch: dict[str, Float[Tensor, "b ..."]]
        ) -> dict[str, Float[Tensor, "b d"]]:
            input_tokens: Float[Tensor, "b l"] = batch['input_ids'].to(self.device)
            input_strs: list[str] = self._tokenizer.batch_decode(input_tokens)
            batch_embeds: Float[np.ndarray, "b d"] = self.model.encode(
                input_strs, convert_to_tensor=True
            )
            new_batch: dict[str, Float[Tensor, "b d"]] = {'embeddings': batch_embeds}
            return new_batch
        
        return dataset.map(
            _compute_sentence_embeddings,
            batched=True,
            batch_size=batch_size,
        )
    
    @torch.no_grad()
    def get_embeddings(self, sentences: list[str]) -> Float[Tensor, "n d"]:
        """Given a list of sentences, compute their sentence embeddings."""
        return self.model.encode(sentences, convert_to_tensor=True).to(self.device)
    
    def get_similarity_sorted_sentence_indices(
        self,
        special_embeddings: Float[Tensor, "n d"],
        retrieve_n: int,
    ) -> Int[Tensor, "retrieve_n"]:
        """Get a list of indices of retrieve_n sentences in the dataset, sorted by their
        similarity to the special sentences."""
        device = self.device
        
        if self.config.similarity_measure == SimilarityMeasure.EMBEDDING:
            similarity_measure: SimMeasure = self._compute_average_similarity
        elif self.config.similarity_measure == SimilarityMeasure.PROJECTED_EMBEDDING:
            similarity_measure: SimMeasure = partial(
                self._proj_average_similarity,
                proj_matrix=self._train_projection_matrix(special_embeddings)
            )
        else:
            similarity_measure: SimMeasure = partial(
                self._classifier_similarity,
                classifier=self._train_classifier(special_embeddings)
            )
        
        indices, similarities = [], []
        
        for batch in self.dataloader:
            sentence_embeddings: Float[Tensor, "b d"] = batch['embeddings'].to(device)
            input_similarities: Float[Tensor, "b"] = similarity_measure(
                sentence_embeddings, special_embeddings
            )
            
            indices.append(batch['indices'].to(device))
            similarities.append(input_similarities)
            
            if len(indices) * self.config.batch_size >= retrieve_n:
                break
        
        indices = pack(indices, '*')[0]
        similarities = pack(similarities, '*')[0]
        
        assert len(indices) == len(similarities)
        
        _, sort_inds = torch.sort(similarities, dim=0, descending=True)
        return indices[sort_inds[:retrieve_n]]

    def _train_projection_matrix(
        self, special_embeddings: Float[Tensor, "n d"]
    ) -> Float[Tensor, "d p"]:
        """Train the projection matrix of sentence embeddings. The `special_embeddings`
        can be obtained using the `get_special_sentences` function or the
        `get_embeddings` function."""
        if self.config.loss_func_name == 'infonce':
            loss_func = InfoNCE(negative_mode='unpaired')
        else:
            raise NotImplementedError(f'Loss function not implemented.')
        
        def _contrastive_model(
            query: Float[Tensor, "b d"],
            positive_embedding: Float[Tensor, "b d"],
            negative_embeddings: Float[Tensor, "b d"],
            proj_matrix: Float[Tensor, "d p"],
            dropout: float = 0.,
        ) -> Float[Tensor, ""]:
            q: Float[Tensor, "b d"] = F.dropout(query, p=dropout)
            pos: Float[Tensor, "b d"] = F.dropout(positive_embedding, p=dropout)
            neg: Float[Tensor, "b d"] = F.dropout(negative_embeddings, p=dropout)
            proj_q: Float[Tensor, "b p"] = einsum(q, proj_matrix, 'b d, d p -> b p')
            proj_pos: Float[Tensor, "b p"] = einsum(pos, proj_matrix, 'b d, d p -> b p')
            proj_neg: Float[Tensor, "b p"] = einsum(neg, proj_matrix, 'b d, d p -> b p')
            contrastive_loss: Float[Tensor, ""] = loss_func(proj_q, proj_pos, proj_neg)
            return contrastive_loss
        
        positive_dataset = TensorDataset(special_embeddings)
        positive_loader = DataLoader(
            positive_dataset,
            batch_size=self.config.proj_batch_size,
            shuffle=True,
        )
        negative_loader = DataLoader(
            self.dataset,
            batch_size=self.config.proj_batch_size,
            shuffle=True,
        )
        query_loader = DataLoader(
            positive_dataset,
            batch_size=self.config.proj_batch_size,
            shuffle=False,
        )
        embedding_length: int = special_embeddings[0].shape[0]
        torch.manual_seed(0)
        proj_matrix: Float[Tensor, "d p"] = torch.randn(
            embedding_length,
            self.config.proj_embedding_length,
            requires_grad=True,
            device=self.device,
        )
        if self.config.optimizer_name == 'sgd':
            optimizer = SGD([proj_matrix], lr=self.config.learning_rate)
        elif self.config.optimizer_name == 'adam':
            optimizer = Adam([proj_matrix], lr=self.config.learning_rate)
        else:
            raise NotImplementedError('Optimizer not implemented.')

        for _ in range(self.config.num_epochs):
            for query, pos_embed, neg_batch in zip(
                query_loader, positive_loader, negative_loader
            ):
                query: Float[Tensor, "b d"] = query[0].to(self.device)
                pos_embed: Float[Tensor, "b d"] = pos_embed[0].to(self.device)
                neg_embed: Float[Tensor, "n d"] = neg_batch['embeddings'].to(self.device)
                loss: Float[Tensor, ""] = _contrastive_model(
                    query,
                    pos_embed,
                    neg_embed,
                    proj_matrix,
                    self.config.dropout_rate,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return proj_matrix.detach()
    

    def _train_classifier(
        self, special_embeddings: Float[Tensor, "n d"]
    ) -> torch.nn.Module:
        """Train the projection matrix of sentence embeddings. The `special_embeddings`
        can be obtained using the `get_special_sentences` function or the
        `get_embeddings` function."""
        positive_dataset = TensorDataset(special_embeddings)
        positive_loader = DataLoader(
            positive_dataset,
            batch_size=self.config.proj_batch_size,
            shuffle=True,
        )
        negative_loader = DataLoader(
            self.dataset,
            batch_size=self.config.proj_batch_size,
            shuffle=True,
        )
        embedding_length: int = special_embeddings[0].shape[0]
        torch.manual_seed(0)
        
        # MLP using torchvision with initial batchnorm
        classifier = torch.nn.Sequential(
            Rearrange('b d -> b 1 d'),
            torch.nn.BatchNorm1d(embedding_length),
            Rearrange('b 1 d -> b d'),
            torchvision.ops.MLP(
                in_channels=embedding_length,
                hidden_channels=[128, 2],
                activation_layer=torch.nn.ReLU,
            ),
        ).to(self.device)

        if self.config.optimizer_name == 'sgd':
            optimizer = SGD(classifier.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer_name == 'adam':
            optimizer = Adam(classifier.parameters(), lr=self.config.learning_rate)
        else:
            raise NotImplementedError('Optimizer not implemented.')
        
        loss_func = torch.nn.CrossEntropyLoss()
        
        for epoch in range(self.config.num_epochs):
            total_loss = 0.0
            batches = min(len(positive_loader), len(negative_loader))
            for pos_embed, neg_batch in zip(positive_loader, negative_loader):
                pos_embed: Float[Tensor, "b d"] = pos_embed[0].to(self.device)
                neg_embed: Float[Tensor, "b d"] = neg_batch['embeddings'].to(self.device)
                
                whole_batch = torch.cat([pos_embed, neg_embed], dim=0)
                labels = torch.cat([
                    torch.ones(pos_embed.shape[0], dtype=torch.long),
                    torch.zeros(neg_embed.shape[0], dtype=torch.long),
                ], dim=0)
                pred = classifier(whole_batch)

                loss = loss_func(pred, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch + 1} loss: {total_loss / batches}')

        return classifier
    
    @torch.no_grad()
    def _compute_average_similarity(
        self,
        sentence_embeddings: Float[Tensor, "b d"],
        special_embeddings: Float[Tensor, "n d"],
    ) -> Float[Tensor, "b"]:
        """Compute the average similarities of the input sentences to special sentences."""
        similarities: Float[Tensor, "b n"] = self.model.similarity(
            sentence_embeddings,
            special_embeddings,
        )
        avg_sim: Float[Tensor, "b"] = reduce(similarities, 'b n -> b', 'mean')
        return avg_sim
    
    @torch.no_grad()
    def _proj_average_similarity(
        self,
        sentence_embeddings: Float[Tensor, "b d"],
        special_embeddings: Float[Tensor, "n d"],
        proj_matrix: Float[Tensor, "d p"],
    ) -> Float[Tensor, "b"]:
        """Project all the sentence embeddings using `proj_matrix`,
        then compute the average similarities."""
        proj_sentence_embeddings: Float[Tensor, "b p"] = einsum(
            sentence_embeddings,
            proj_matrix,
            'b d, d p -> b p',
        )
        proj_special_embeddings: Float[Tensor, "n p"] = einsum(
            special_embeddings,
            proj_matrix,
            'n d, d p -> n p',
        )
        similarities: Float[Tensor, "b n"] = self.model.similarity(
            proj_sentence_embeddings,
            proj_special_embeddings,
        )
        avg_sim: Float[Tensor, "b"] = reduce(similarities, 'b n -> b', 'mean')
        return avg_sim
    
    @torch.no_grad()
    def _classifier_similarity(
        self,
        sentence_embeddings: Float[Tensor, "b d"],
        special_embeddings: Float[Tensor, "n d"],
        classifier: torch.nn.Module,
    ) -> Float[Tensor, "b"]:
        
        pred: Float[Tensor, "b 2"] = classifier(sentence_embeddings)
        # Class one is for "positive" predictions
        return pred[:, 1]
