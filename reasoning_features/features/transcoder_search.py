"""Streaming feature search utilities for CLT and PLT experiments."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable
import warnings

import numpy as np
import torch
from einops import reduce
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .collector import FeatureActivations
from .detector import FeatureStats
from .runtime import BaseFeatureRuntime
from .tokens import TopTokenAnalyzer


def _pooled_std(
    std_reasoning: float,
    std_nonreasoning: float,
    n_reasoning: int,
    n_nonreasoning: int,
) -> float:
    numerator = ((n_reasoning - 1) * std_reasoning**2) + ((n_nonreasoning - 1) * std_nonreasoning**2)
    denominator = max(n_reasoning + n_nonreasoning - 2, 1)
    return float(np.sqrt(max(numerator / denominator, 0.0)))


def stream_rank_features_by_cohens_d(
    runtime: BaseFeatureRuntime,
    texts: list[str],
    is_reasoning: list[bool],
    layer_index: int,
    max_length: int,
    batch_size: int,
    max_features: int | None = None,
) -> dict[str, np.ndarray | int]:
    """Stream over texts once and rank all features by Cohen's d."""

    n_total_features = runtime.get_num_features(layer_index)
    n_features = min(n_total_features, max_features) if max_features is not None else n_total_features

    sum_reasoning = np.zeros(n_features, dtype=np.float64)
    sumsq_reasoning = np.zeros(n_features, dtype=np.float64)
    sum_nonreasoning = np.zeros(n_features, dtype=np.float64)
    sumsq_nonreasoning = np.zeros(n_features, dtype=np.float64)
    n_reasoning = 0
    n_nonreasoning = 0

    effective_batch_size = 1 if n_features > 100_000 else max(1, batch_size)

    for start in tqdm(range(0, len(texts), effective_batch_size), desc="Ranking features by Cohen's d"):
        batch_texts = texts[start : start + effective_batch_size]
        batch_labels = is_reasoning[start : start + effective_batch_size]
        batch = runtime.get_layer_activations(
            texts=batch_texts,
            layer_index=layer_index,
            max_length=max_length,
            feature_indices=list(range(n_features)) if max_features is not None else None,
            apply_activation_function=True,
        )
        batch_max = batch.activations.amax(dim=1).numpy()

        for sample_max, label in zip(batch_max, batch_labels, strict=True):
            if label:
                sum_reasoning += sample_max
                sumsq_reasoning += np.square(sample_max)
                n_reasoning += 1
            else:
                sum_nonreasoning += sample_max
                sumsq_nonreasoning += np.square(sample_max)
                n_nonreasoning += 1

    mean_reasoning = sum_reasoning / max(n_reasoning, 1)
    mean_nonreasoning = sum_nonreasoning / max(n_nonreasoning, 1)

    var_reasoning = np.clip(
        (sumsq_reasoning - (np.square(sum_reasoning) / max(n_reasoning, 1))) / max(n_reasoning - 1, 1),
        a_min=0.0,
        a_max=None,
    )
    var_nonreasoning = np.clip(
        (sumsq_nonreasoning - (np.square(sum_nonreasoning) / max(n_nonreasoning, 1)))
        / max(n_nonreasoning - 1, 1),
        a_min=0.0,
        a_max=None,
    )
    std_reasoning = np.sqrt(var_reasoning)
    std_nonreasoning = np.sqrt(var_nonreasoning)

    pooled_var = (
        ((n_reasoning - 1) * var_reasoning) + ((n_nonreasoning - 1) * var_nonreasoning)
    ) / max(n_reasoning + n_nonreasoning - 2, 1)
    cohens_d = (mean_reasoning - mean_nonreasoning) / np.sqrt(np.clip(pooled_var, a_min=1e-10, a_max=None))

    return {
        "n_total_features": n_total_features,
        "n_features": n_features,
        "n_reasoning": n_reasoning,
        "n_nonreasoning": n_nonreasoning,
        "mean_reasoning": mean_reasoning,
        "mean_nonreasoning": mean_nonreasoning,
        "std_reasoning": std_reasoning,
        "std_nonreasoning": std_nonreasoning,
        "cohens_d": cohens_d,
    }


def collect_selected_feature_data(
    runtime: BaseFeatureRuntime,
    texts: list[str],
    is_reasoning: list[bool],
    sources: list[str],
    layer_index: int,
    max_length: int,
    batch_size: int,
    selected_feature_indices: list[int],
) -> tuple[np.ndarray, FeatureActivations]:
    """Collect exact sample maxima and token activations for a small selected subset."""

    n_samples = len(texts)
    n_selected = len(selected_feature_indices)
    effective_batch_size = 1 if runtime.get_num_features(layer_index) > 100_000 else max(1, batch_size)

    sample_maxes = np.zeros((n_samples, n_selected), dtype=np.float32)
    activations = torch.zeros((n_samples, max_length, n_selected), dtype=torch.float32)
    tokens = torch.zeros((n_samples, max_length), dtype=torch.long)

    cursor = 0
    for start in tqdm(
        range(0, len(texts), effective_batch_size),
        desc="Collecting selected feature activations",
    ):
        batch_texts = texts[start : start + effective_batch_size]
        batch = runtime.get_layer_activations(
            texts=batch_texts,
            layer_index=layer_index,
            max_length=max_length,
            feature_indices=selected_feature_indices,
            apply_activation_function=True,
        )
        batch_size_local = batch.tokens.shape[0]
        batch_max = batch.activations.amax(dim=1).numpy()

        sample_maxes[cursor : cursor + batch_size_local] = batch_max
        activations[cursor : cursor + batch_size_local] = batch.activations
        tokens[cursor : cursor + batch_size_local] = batch.tokens
        cursor += batch_size_local

    feature_activations = FeatureActivations(
        activations=activations,
        tokens=tokens,
        is_reasoning=is_reasoning,
        sources=sources,
        layer_index=layer_index,
        model_name=runtime.model_name,
        sae_name=getattr(runtime, "transcoder_set", getattr(runtime, "sae_name", "unknown")),
    )
    return sample_maxes, feature_activations


def compute_exact_feature_stats(
    sample_maxes: np.ndarray,
    is_reasoning: list[bool],
    selected_feature_indices: list[int],
    score_weights: dict[str, float],
) -> list[FeatureStats]:
    """Compute the exact feature statistics for a selected subset."""

    labels = np.array(is_reasoning, dtype=bool)
    reasoning_acts = sample_maxes[labels]
    nonreasoning_acts = sample_maxes[~labels]
    n_reasoning = int(labels.sum())
    n_nonreasoning = int((~labels).sum())

    stats: list[FeatureStats] = []
    for local_idx, feature_index in enumerate(selected_feature_indices):
        r = reasoning_acts[:, local_idx]
        nr = nonreasoning_acts[:, local_idx]

        mean_r = float(np.mean(r))
        mean_nr = float(np.mean(nr))
        std_r = float(np.std(r))
        std_nr = float(np.std(nr))
        pooled_std = _pooled_std(std_r, std_nr, n_reasoning, n_nonreasoning)
        cohens_d = (mean_r - mean_nr) / (pooled_std + 1e-10)
        log_fc = float(np.log2((mean_r + 1e-10) / (mean_nr + 1e-10)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                u_stat, u_pval = mannwhitneyu(r, nr, alternative="two-sided")
            except ValueError:
                u_stat, u_pval = 0.0, 1.0

            try:
                t_stat, t_pval = ttest_ind(r, nr, equal_var=False)
            except ValueError:
                t_stat, t_pval = 0.0, 1.0

        try:
            roc_auc = float(roc_auc_score(labels.astype(int), sample_maxes[:, local_idx]))
        except ValueError:
            roc_auc = 0.5

        threshold = 0.01 * max(float(np.max(sample_maxes[:, local_idx])), 1e-10)
        freq_r = float(np.mean((r > threshold).astype(float)))
        freq_nr = float(np.mean((nr > threshold).astype(float)))

        stats.append(
            FeatureStats(
                feature_index=feature_index,
                mean_reasoning=mean_r,
                mean_nonreasoning=mean_nr,
                std_reasoning=std_r,
                std_nonreasoning=std_nr,
                cohens_d=float(cohens_d),
                log_fold_change=log_fc,
                mannwhitney_u=float(u_stat),
                mannwhitney_p=float(u_pval),
                ttest_t=float(t_stat) if not np.isnan(t_stat) else 0.0,
                ttest_p=float(t_pval) if not np.isnan(t_pval) else 1.0,
                roc_auc=roc_auc,
                freq_active_reasoning=freq_r,
                freq_active_nonreasoning=freq_nr,
                _score_weights=score_weights,
            )
        )

    n_tests = max(len(stats), 1)
    for stat in stats:
        stat.mannwhitney_p = min(stat.mannwhitney_p * n_tests, 1.0)
        stat.ttest_p = min(stat.ttest_p * n_tests, 1.0)

    stats.sort(key=lambda item: item.reasoning_score, reverse=True)
    return stats


def select_reasoning_features(
    stats: list[FeatureStats],
    *,
    no_filter: bool,
    top_k_features: int,
    min_auc: float,
    max_pvalue: float,
    min_effect_size: float,
) -> list[FeatureStats]:
    """Apply the same thresholding logic used by the SAE pipeline."""

    if no_filter:
        return stats[:top_k_features]

    filtered = [
        stat
        for stat in stats
        if stat.roc_auc >= min_auc
        and stat.mannwhitney_p <= max_pvalue
        and stat.cohens_d >= min_effect_size
        and stat.mean_reasoning > stat.mean_nonreasoning
    ]
    if filtered:
        return filtered[:top_k_features]
    return stats[:top_k_features]


def analyze_selected_tokens(
    feature_activations: FeatureActivations,
    tokenizer,
    stats: Iterable[FeatureStats],
    *,
    top_k_tokens: int,
    min_token_occurrences: int,
    top_k_bigrams: int,
    min_bigram_occurrences: int,
    top_k_trigrams: int,
    min_trigram_occurrences: int,
) -> list[dict]:
    """Run the existing token analyzer on a local feature subset and remap indices."""

    analyzer = TopTokenAnalyzer(feature_activations, tokenizer)
    selected_stats = list(stats)
    feature_analyses: list[dict] = []

    for local_idx, stat in enumerate(selected_stats):
        analysis = analyzer.analyze_feature_token_dependency(local_idx, top_k_tokens=top_k_tokens)
        analysis["feature_index"] = stat.feature_index
        analysis["feature_stats"] = stat.to_dict()

        for token in analysis["top_tokens"]:
            token["feature_index"] = stat.feature_index

        top_bigrams = analyzer.get_top_ngrams_for_feature(
            local_idx,
            n=2,
            top_k=top_k_bigrams,
            min_occurrences=min_bigram_occurrences,
        )
        analysis["top_bigrams"] = []
        for bigram in top_bigrams:
            entry = bigram.to_dict()
            entry["feature_index"] = stat.feature_index
            analysis["top_bigrams"].append(entry)

        top_trigrams = analyzer.get_top_ngrams_for_feature(
            local_idx,
            n=3,
            top_k=top_k_trigrams,
            min_occurrences=min_trigram_occurrences,
        )
        analysis["top_trigrams"] = []
        for trigram in top_trigrams:
            entry = trigram.to_dict()
            entry["feature_index"] = stat.feature_index
            analysis["top_trigrams"].append(entry)

        feature_analyses.append(analysis)

    return feature_analyses


def build_token_analysis_summary(feature_token_analyses: list[dict]) -> dict:
    """Mirror the summary structure used by the SAE token analysis output."""

    if not feature_token_analyses:
        return {
            "total_features_analyzed": 0,
            "high_token_dependency_count": 0,
            "high_token_dependency_percentage": 0.0,
            "mean_token_concentration": 0.0,
            "mean_normalized_entropy": 0.0,
        }

    high_token_dependency_count = sum(
        1 for analysis in feature_token_analyses if analysis["token_concentration"] > 0.5
    )
    return {
        "total_features_analyzed": len(feature_token_analyses),
        "high_token_dependency_count": high_token_dependency_count,
        "high_token_dependency_percentage": high_token_dependency_count / len(feature_token_analyses) * 100,
        "mean_token_concentration": float(
            reduce(np.array([analysis["token_concentration"] for analysis in feature_token_analyses]), "f ->", "mean")
        ),
        "mean_normalized_entropy": float(
            reduce(np.array([analysis["normalized_entropy"] for analysis in feature_token_analyses]), "f ->", "mean")
        ),
    }


def feature_stats_to_dicts(stats: Iterable[FeatureStats]) -> list[dict]:
    """Serialize feature stats for JSON output."""

    return [asdict(stat) | {"reasoning_score": stat.reasoning_score} for stat in stats]
