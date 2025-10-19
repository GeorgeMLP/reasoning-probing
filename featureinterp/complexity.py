from __future__ import annotations

from featureinterp.core import (
    ExplanationComponent,
    StructuredExplanation,
)
from featureinterp.complexity_examples import ComplexityExampleSet
from featureinterp.local_inference import LocalInferenceManager
from featureinterp.prompt_builder import (
    Message,
    PromptBuilder,
    Role,
)
from featureinterp.simulator import (
    compute_expected_value,
)


class ComplexityAnalyzer:
    def __init__(
        self,
        inference_manager: LocalInferenceManager,
        few_shot_example_set: ComplexityExampleSet = ComplexityExampleSet.ORIGINAL,
        use_kv_cache: bool = True,
    ):
        assert inference_manager.batch_size == 1
        self.inference_manager = inference_manager
        self.few_shot_example_set = few_shot_example_set

        if use_kv_cache:
            self.inference_manager.append_to_cache(self._prompt_header())

    def __del__(self):
        # ComplexityAnalyzer is responsible for cleaning up the inference
        # manager upon destruction. Not quite sure why this memory leak is happening.
        del self.inference_manager.cached_inputs
        del self.inference_manager.prompt_cache

    def analyze_complexity(
        self,
        explanation: StructuredExplanation,
    ) -> list[float]:

        complexities = []
        for component in explanation.components:
            complexity = self._analyze_component(component)
            complexities.append(complexity)
        return complexities
        
    def _analyze_component(
        self,
        explanation_component: ExplanationComponent,
    ) -> float:

        if self.inference_manager.using_kv_cache:
            prompt = self._prompt_suffix(explanation_component)
        else:
            prompt = self._prompt_header() + self._prompt_suffix(explanation_component)
        
        token_ids, logprobs = self.inference_manager.run_batched_inference([prompt])[0]

        response_tokens = self.inference_manager.tokenizer.batch_decode(
            token_ids, clean_up_tokenization_spaces=False
        )
        
        # Find the index of the last "unknown" token
        unknown_idx = None
        for i in range(len(response_tokens) - 1, -1, -1):
            if "unknown" == response_tokens[i]:
                unknown_idx = i
                break
        if unknown_idx is None:
            raise ValueError("Could not find 'unknown' token in response")
        
        analysis_logprobs = logprobs[unknown_idx:unknown_idx + 1]

        complexity = compute_expected_value(
            logprobs=analysis_logprobs,
            temperature=1.0,
            tokenizer=self.inference_manager.tokenizer,
        )
        
        return complexity.item()

    def _prompt_header(self) -> list[Message]:
        prompt_builder = PromptBuilder()
        
        # We aren't using "strength" here because we're not predicting activations.
        prompt_builder.add_message(
            Role.SYSTEM,
            f"""We're studying features in a neural network. Each feature has an activation rule describing how it activates:

Given an explanation of a feature's activation rule, predict how complex it is.

The complexity is a number between 0 and 5, where 0 is the simplest and 5 is the most complex.

The complexity is determined by the level of abstraction required to understand the activation rule.

Having a specific activation context is more complex than not having an "Any" context.

Activation rules which activate only on a list of specific tokens are simpler than rules which activate on tokens capturing abstract concepts.
""",
        )

        few_shot_examples = self.few_shot_example_set.get_examples()
        for example in few_shot_examples:
            prompt_builder.add_message(
                Role.USER,
                f"Explanation:\n"
                f"{self._format_explanation_component(example.explanation_component)}",
            )
            prompt_builder.add_message(
                Role.ASSISTANT,
                f"Complexity:\t{example.complexity}\n"
                f"Complexity explanation:\t{example.complexity_explanation}"
            )
        
        return prompt_builder.build()
    
    def _prompt_suffix(
        self,
        explanation_component: ExplanationComponent,
    ) -> list[Message]:
        """A simulation prompt suffix which includes the tokens we want to simulate."""

        prompt_builder = PromptBuilder()
        prompt_builder.add_message(
            Role.USER,
            f"Explanation:\n"
            f"{self._format_explanation_component(explanation_component)}"
        )
        prompt_builder.add_message(
            Role.ASSISTANT,
            f"Complexity:\tunknown"
        )
        return prompt_builder.build(check_expected_role=False)
    
    def _format_explanation_component(
        self, explanation_component: ExplanationComponent
    ) -> str:
        
        return explanation_component.activates_on

        