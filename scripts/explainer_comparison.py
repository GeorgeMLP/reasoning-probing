import asyncio
import os
from pathlib import Path
from typing import Callable
import click
import pickle
import orjson
import tqdm
import warnings
from transformers import logging

from featureinterp.core import StructuredExplanation
from featureinterp.explainer import OneShotExplainer, OneShotExplainerParams, TreeExplainer, TreeExplainerParams
from featureinterp.record import ComplementaryRecordSource, RecordSliceParams
from featureinterp.simulator import ExplanationFeatureSimulator
from scripts import experiment


logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


async def write_results(
    explainer: OneShotExplainer | TreeExplainer,
    simulator_factory: Callable[[StructuredExplanation], ExplanationFeatureSimulator],
    include_train_negatives: bool,
    save_dir: Path,
    save_file_stem: str,
    model_arg: str,
    use_cache: bool,
    cache_dir: Path,
) -> list[list[experiment.ExperimentResult]]:

    train_record_params = RecordSliceParams(
        positive_examples_per_split=10,
        complementary_examples_per_split=10,
        complementary_record_source=(
            ComplementaryRecordSource.SIMILAR_NEGATIVE
            if include_train_negatives
            else None
        ),
    )
    valid_record_params = RecordSliceParams(
        positive_examples_per_split=10,
        complementary_examples_per_split=10,
        complementary_record_source=ComplementaryRecordSource.SIMILAR_NEGATIVE,
    )
    test_record_params = RecordSliceParams(
        positive_examples_per_split=10,
        complementary_examples_per_split=10,
        complementary_record_source=ComplementaryRecordSource.SIMILAR_NEGATIVE,
    )

    dataset_path = experiment.get_dataset_path(model_arg)
    layer_indices = experiment.get_layer_indices(model_arg)
    feature_indices = range(30)

    layer_results: list[list[experiment.ExperimentResult]] = []
    for layer_index in tqdm.tqdm(layer_indices, desc='Layer'):
        layer_experiments = []
        for feature_index in tqdm.tqdm(feature_indices, desc='Feature', leave=False):
            cache_path = cache_dir / Path('_'.join((
                save_file_stem,
                str(layer_index),
                str(feature_index),
            )) + '.pkl')

            if use_cache and os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    result: experiment.ExperimentResult = pickle.load(f)
            else:
                sae_index_record = experiment.load_sae_index_record(
                    layer_index=layer_index,
                    latent_index=feature_index,
                    records_path=dataset_path,
                )
                try:
                    result = await asyncio.wait_for(
                        experiment.run_experiment(
                            sae_index_record=sae_index_record,
                            train_record_params=train_record_params,
                            valid_record_params=valid_record_params,
                            test_record_params=test_record_params,
                            explainer=explainer,
                            simulator_factory=simulator_factory,
                            complexity_analyzer=None,
                        ),
                        timeout=1500,
                    )
                except asyncio.exceptions.TimeoutError:
                    print(f"Timeout on layer {layer_index}, feature {feature_index}")
                    continue
                if use_cache and result is not None:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(result, f)
            
            if result is not None:
                layer_experiments.append(result)
            
        layer_results.append(layer_experiments)

            
    with open(save_dir / f'{save_file_stem}.json', 'wb') as f:
        f.write(orjson.dumps(layer_results))

    with open(save_dir / f'{save_file_stem}.pkl', 'wb') as f:
        pickle.dump(layer_results, f)
        

async def async_main(
    model_arg: str,
    explainer_type: str,
    structured_explanations: bool,
    include_train_negatives: bool,
    include_holistic_expressions: bool,
    use_cache: bool,
):
    results_path = Path('results', f'explainer_comparison_{model_arg}')
    results_path.mkdir(exist_ok=True, parents=True)
    version = 1
    cache_dir = Path(f'cache', f'explainer_comparison_{model_arg}_v{version}')
    cache_dir.mkdir(exist_ok=True, parents=True)

    SIMULATOR_MODEL_NAME = "google/gemma-2-27b-it"
    
    simulator_factory = experiment.load_simulator_factory(
        SIMULATOR_MODEL_NAME,
        batch_size=2,
    )
    
    explainer_model_name = "meta-llama/llama-4-scout"
    
    print(
        f"Running experiments for {explainer_type}, "
        f"model: {explainer_model_name}, "
        f"structured explanations: {structured_explanations}, "
        f"include train negatives: {include_train_negatives}, "
        f"include holistic expressions: {include_holistic_expressions}, "
        f"dataset: {model_arg}"
    )
    
    if explainer_type == "one-shot":
        explainer = OneShotExplainer(
            model_name=explainer_model_name,
            params=OneShotExplainerParams(
                rule_cap=5,
                include_holistic_expressions=include_holistic_expressions,
                structured_explanations=structured_explanations,
            ),
        )
    else:
        explainer = TreeExplainer(
            model_name=explainer_model_name,
            simulator_factory=simulator_factory,
            params=TreeExplainerParams(
                print_explanations=False,
                rule_cap=5,
                depth=2,
                width=2,
                include_holistic_expressions=include_holistic_expressions,
                structured_explanations=structured_explanations,
            ),
        )
    
    model_name = explainer_model_name.split('/')[-1]
    await write_results(
        explainer=explainer,
        simulator_factory=simulator_factory,
        include_train_negatives=include_train_negatives,
        save_dir=results_path,
        save_file_stem=f"data{model_arg}_explainer{model_name}_method{explainer_type}_structured{structured_explanations}_trainnegatives{include_train_negatives}_holisticexpressions{include_holistic_expressions}",
        model_arg=model_arg,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )


@click.command()
@click.option('--model', type=click.Choice(['gemma', 'gpt2', 'llama']), default='gemma')
@click.option('--explainer_type', type=click.Choice(['one-shot', 'tree']), default='one-shot')
@click.option('--structured_explanations', is_flag=True)
@click.option('--include_train_negatives', is_flag=True)
@click.option('--include_holistic_expressions', is_flag=True)
@click.option('--use-cache/--no-use-cache', default=True)
def main(
    model: str,
    explainer_type: str,
    structured_explanations: bool,
    include_train_negatives: bool,
    include_holistic_expressions: bool,
    use_cache: bool,
):
    asyncio.run(async_main(
        model,
        explainer_type,
        structured_explanations,
        include_train_negatives,
        include_holistic_expressions,
        use_cache,
    ))


if __name__ == "__main__":
    main()
