import asyncio
import time
import warnings
from transformers import logging

from featureinterp.explainer import OneShotExplainer, OneShotExplainerParams, TreeExplainerParams, TreeExplainer
from featureinterp.record import ComplementaryRecordSource, RecordSliceParams
from scripts import experiment


logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


async def main():
    INFERENCE_BATCH_SIZE = 2
    STRUCTURED_EXPLANATIONS = True
    
    EXPLAINER_MODEL_NAME = "meta-llama/llama-4-scout"
    COMPLEXITY_MODEL_NAME = "google/gemma-2-27b-it"
    SIMULATOR_MODEL_NAME = "google/gemma-2-27b-it"

    dataset_path = 'data/pile-uncopyrighted_gemma-2-9b/records'
    
    train_record_params = RecordSliceParams(
        positive_examples_per_split=10,
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

    sae_index_record = experiment.load_sae_index_record(
        layer_index=10,
        latent_index=10,
        records_path=dataset_path,
    )
    
    simulator_factory = experiment.load_simulator_factory(
        SIMULATOR_MODEL_NAME,
        batch_size=INFERENCE_BATCH_SIZE,
    )
    complexity_analyzer = experiment.load_complexity_analyzer(
        COMPLEXITY_MODEL_NAME,
        batch_size=1,
    )

    explainer = OneShotExplainer(
        model_name=EXPLAINER_MODEL_NAME,
        params=OneShotExplainerParams(
            include_holistic_expressions=True,
            structured_explanations=STRUCTURED_EXPLANATIONS,
            rule_cap=1,
        )
    )
    # explainer = TreeExplainer(
    #     model_name=EXPLAINER_MODEL_NAME,
    #     simulator_factory=simulator_factory,
    #     params=TreeExplainerParams(
    #         rule_cap=5,
    #         width=2,
    #         depth=2,
    #         branching_factor=2,
    #         include_holistic_expressions=False,
    #         structured_explanations=STRUCTURED_EXPLANATIONS,
    #         print_explanations=False,
    #     ),
    # )
    
    t0 = time.time()
    result = await experiment.run_experiment(
        sae_index_record=sae_index_record,
        train_record_params=train_record_params,
        valid_record_params=valid_record_params,
        test_record_params=test_record_params,
        explainer=explainer,
        simulator_factory=simulator_factory,
        complexity_analyzer=complexity_analyzer,
        log=True,
    )

    print(f"Time: {time.time() - t0:.3f}s")


if __name__ == "__main__":  
    asyncio.run(main())
