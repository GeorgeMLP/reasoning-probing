from colorama import Fore, Style
import dacite
import numpy as np
import orjson
import torch
import gc
from dataclasses import dataclass
from typing import Any, Callable
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

import warnings
from transformers import logging

from featureinterp.complexity import ComplexityAnalyzer
from featureinterp.local_inference import LocalInferenceManager
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from featureinterp import formatting, utils
from featureinterp.record import RecordSliceParams, SAEIndexRecord
from featureinterp.explainer import FeatureExplainer
from featureinterp.core import StructuredExplanation
from featureinterp.core import ScoredSimulation
from featureinterp.scoring import simulate_and_score
from featureinterp.simulator import CacheState, ExplanationFeatureSimulator, make_simulation_header


@dataclass
class ExperimentResult:
    layer_index: int
    latent_index: int
    
    max_expression: float
    max_holistic_expression: float

    score: float

    explanation: StructuredExplanation
    explanation_extra_data: dict[str, Any]
    explanation_complexities: list[float] | None
    """Complexity of each component in the explanation."""
    
    scored_simulation: ScoredSimulation | None


def get_dataset_path(model_arg: str) -> str:
    DATASET_PATHS = {
        'gemma': 'data/pile-uncopyrighted_gemma-2-9b/records',
        'gpt2': 'data/pile-uncopyrighted_gpt2/records',
        'llama': 'data/pile-uncopyrighted_Llama-3.1-8B/records'
    }
    return DATASET_PATHS[model_arg]


def get_layer_indices(
    model_arg: str,
    subpoints: int | None = 8,
) -> list[int]:

    LAYERS = {
        'gemma': 42,
        'gpt2': 12,
        'llama': 32,
    }
    
    if subpoints is None:
        return list(range(LAYERS[model_arg]))
    else:
        return np.round(np.linspace(0, LAYERS[model_arg] - 1, subpoints)).astype(int)


def load_sae_index_record(
    layer_index: int, latent_index: int, records_path: str
) -> SAEIndexRecord:
    
    # Find folder containing the layer index
    layer_folder_pattern = f".{layer_index}."
    matching = [f for f in os.listdir(records_path) if layer_folder_pattern in f]
    if not matching:
        raise ValueError(f"No folder found for layer index '{layer_folder_pattern}'")
    assert len(matching) == 1
    layer_folder_name = matching[0]
    
    file_path = "/".join([records_path, layer_folder_name, f"{latent_index}.json"])
    with open(file_path, 'r') as f:
        sae_index_record = orjson.loads(f.read())
        return dacite.from_dict(data_class=SAEIndexRecord, data=sae_index_record)


# Global cache for tokenizers and models
_MODEL_CACHE: dict[str, tuple[AutoTokenizer, AutoModelForCausalLM]] = {}


def load_tokenizer_and_model(
    model_name: str,
    model_cache_dir: str | None = None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    
    # Return cached model if available
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    
    bnb_4bit_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    bnb_8bit_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=False,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.bfloat16
    )
    
    quantization_configs = {
        "google/gemma-2-27b-it": bnb_4bit_config,
        "google/gemma-2-9b-it": bnb_8bit_config,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_configs[model_name],
        attn_implementation="sdpa",
        cache_dir=model_cache_dir,
    )
 
    # Cache the model and tokenizer
    _MODEL_CACHE[model_name] = (tokenizer, model)
    
    return tokenizer, model


def load_simulator_factory(
    model_name: str,
    batch_size: int,
    model_cache_dir: str | None = None,
) -> Callable[[StructuredExplanation | str], ExplanationFeatureSimulator]:
    
    probability_temperatures = {
        "google/gemma-2-27b-it": 1.0,
        "google/gemma-2-9b-it": 1.0,
    }
    
    emphasize_numerical_activations = {
        "google/gemma-2-27b-it": False,
        "google/gemma-2-9b-it": False,
    }
    
    global_cache_states = {
        "google/gemma-2-27b-it": CacheState.HEADER_CACHE,
        "google/gemma-2-9b-it": CacheState.HEADER_CACHE,
    }
    
    simulator_cache_states = {
        "google/gemma-2-27b-it": CacheState.HEADER_EXPLANATION_CACHE,
        "google/gemma-2-9b-it": CacheState.HEADER_EXPLANATION_CACHE,
    }
    
    tokenizer, model = load_tokenizer_and_model(model_name, model_cache_dir)
    
    global_cache_state = global_cache_states[model_name]
    simulator_cache_state = simulator_cache_states[model_name]
    
    inference_manager = LocalInferenceManager(
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
    )
    if global_cache_state == CacheState.NO_CACHE:
        pass
    elif global_cache_state == CacheState.HEADER_CACHE:
        inference_manager.append_to_cache(make_simulation_header())
    else:
        raise ValueError(f"Invalid global cache state: {global_cache_state}")
    
    def factory(explanation: StructuredExplanation | str) -> ExplanationFeatureSimulator:
        return ExplanationFeatureSimulator(
            inference_manager=LocalInferenceManager.clone(inference_manager),
            cache_state=global_cache_state,
            desired_cache_state=simulator_cache_state,
            explanation=explanation,
            probability_temperature=probability_temperatures[model_name],
            emphasize_numerical_activations=emphasize_numerical_activations[model_name],
        )

    return factory


def load_complexity_analyzer(
    model_name: str,
    batch_size: int,
    model_cache_dir: str | None = None,
) -> ComplexityAnalyzer:

    use_kv_caches = {
        "google/gemma-2-27b-it": True,
        "google/gemma-2-9b-it": True,
    }

    tokenizer, model = load_tokenizer_and_model(model_name, model_cache_dir)
    return ComplexityAnalyzer(
        inference_manager=LocalInferenceManager(
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
        ),
        use_kv_cache=use_kv_caches[model_name],
    )


async def run_experiment(
    sae_index_record: SAEIndexRecord,
    train_record_params: RecordSliceParams,
    valid_record_params: RecordSliceParams,
    test_record_params: RecordSliceParams,
    explainer: FeatureExplainer,

    simulator_factory: Callable[
        [StructuredExplanation], ExplanationFeatureSimulator
    ] | None = None,
    complexity_analyzer: ComplexityAnalyzer | None = None,

    log=False,
) -> ExperimentResult | None:

    train_records = formatting.format_records(
        sae_index_record.train_records(train_record_params),
        max_expression=sae_index_record.max_expression,
        max_holistic_expression=sae_index_record.max_holistic_expression,
    )
    valid_records = formatting.format_records(
        sae_index_record.train_records(valid_record_params),
        max_expression=sae_index_record.max_expression,
        max_holistic_expression=sae_index_record.max_holistic_expression,
    )
    test_records = sae_index_record.test_records(test_record_params)
    
    if (
        sae_index_record.max_expression <= 0
        or len(train_records) == 0
        or len(test_records) == 0
    ):
        return None
    
    explanations, explanation_extra_data = await explainer.generate_explanations(
        train_records=train_records,
        valid_records=valid_records,
    )
    assert len(explanations) == 1
    explanation = explanations[0]
    
    score = None
    if simulator_factory is not None:
        simulator = simulator_factory(explanation)
        scored_simulation = await simulate_and_score(simulator, test_records)
        score = scored_simulation.get_preferred_score()
    
    complexities = None
    if complexity_analyzer is not None and isinstance(explanation, StructuredExplanation):
        complexities = complexity_analyzer.analyze_complexity(explanation)

    result = ExperimentResult(
        layer_index=sae_index_record.id.layer_index,
        latent_index=sae_index_record.id.latent_index,
        max_expression=sae_index_record.max_expression,
        max_holistic_expression=sae_index_record.max_holistic_expression,
        score=score,
        explanation=explanation,
        explanation_extra_data=explanation_extra_data,
        explanation_complexities=complexities,
        scored_simulation=scored_simulation,
    )

    if log:
        print(f'\n\n{Fore.BLUE}===== Train expression records ====={Style.RESET_ALL}')
        for record in train_records[:5]:
            norm_exprs = np.array(record.expressions) / 5
            print(f'\n{utils.render_expressions(record.tokens, norm_exprs)}')
        
        print(f'\n\n{Fore.BLUE}===== Explanation generation ====={Style.RESET_ALL}')
        
        if isinstance(explanation, StructuredExplanation):
            print(f"\nExplanation:\n{explanation.to_json()}")
        else:
            print(f"\nExplanation:\n{explanation}")
        print(f"\nComponent complexities:\n{complexities}")
        
        print(f'\n\n{Fore.BLUE}===== Simulation ====={Style.RESET_ALL}')

        if simulator_factory is not None:
            for seqsim in scored_simulation.scored_sequence_simulations[:5]:
                print(f'\n\n{Fore.GREEN}True expressions{Style.RESET_ALL}')
                print(utils.render_expressions(
                    seqsim.simulation.tokens,
                    np.array(seqsim.true_expressions) / sae_index_record.max_expression,
                ))
                print(f'{Fore.RED}Simulated expressions{Style.RESET_ALL}')
                print(utils.render_expressions(
                    seqsim.simulation.tokens,
                    np.array(seqsim.simulation.expected_expressions) / 5,
                ))

            score = scored_simulation.get_preferred_score()
            print(f"\n\n{Fore.BLUE}Final score: {score:.2f}{Style.RESET_ALL}")

    del explainer
    if simulator_factory is not None:
        del simulator, scored_simulation
    gc.collect()
    torch.cuda.empty_cache()

    return result
