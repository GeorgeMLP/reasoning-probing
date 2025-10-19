from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
import numpy as np
import copy
import asyncio

from featureinterp.formatting import (
    MAX_FORMATTED_EXPRESSION,
    FormattedFeatureExpressionRecord,
    stringify_contrasting_expressions,
    stringify_record,
    stringify_records,
    non_zero_expression_proportion,
)
from featureinterp.scoring import (
    aggregate_scored_sequence_simulations,
    simulate_and_score,
)
from featureinterp.simulator import ExplanationFeatureSimulator
from featureinterp.api_client import ApiClient
from featureinterp.core import (
    STRING_EXPLANATION_PREFIX,
    STRUCTURED_EXPLANATION_PREFIX,
    ExplanationComponent,
    ScoredSequenceSimulation,
    StructuredExplanation,
)
from featureinterp.record_examples import RecordExampleSet
from featureinterp.prompt_builder import (
    Message,
    PromptBuilder,
    Role,
)

logger = logging.getLogger(__name__)


def rule_cap_str(rule_cap: int | None) -> str:
    if rule_cap is None:
        return ""
    return (
        f"The strict maximum number of rules is {rule_cap}. "
        "Do not generate more than this number of rules. "
        "You should not try to fill up the rule cap, only add rules if they are "
        "actually necessary and try to keep the list of rules as short as possible."
    )


def add_system_prompt(
    prompt_builder: PromptBuilder,
    include_holistic_expressions: bool,
    structured_explanations: bool,
    rule_cap: int | None = None,
) -> None:
    
    holistic_explanation_str = ""
    if include_holistic_expressions:
        if structured_explanations:
            holistic_explanation_str = (
                "Activation records consist of two parts: activating tokens and activation-causing tokens. "
                "Activating tokens are the tokens that the feature activates on. "
                "Activation-causing tokens are the tokens that cause the feature to activate on later activating tokens."
            )
        else:
            holistic_explanation_str = (
                "Activation records consist of two parts: activating tokens and activation-causing tokens. "
                "Activating tokens are the tokens that the neuron activates on. "
                "Activation-causing tokens are the tokens that cause the neuron to activate on later activating tokens."
            )

    if structured_explanations:
        prompt_builder.add_message(
            Role.SYSTEM,
            "We're studying features in a neural network. Each feature looks for specific patterns "
            "in text. Analyze the parts of the text where the feature activates and explain its "
            "behavior in a structured format.\n\n"
            "For each feature, provide a list of rules, where each rule "
            "consists of two fields:\n"
            "1. 'activates_on' (string): The specific tokens on which the activation occurs. This must be a string -- NOT a list of strings.\n"
            "2. 'strength' (int): The strength of the activation, from 0 to 5. Only put a single integer here, no additional text.\n\n"
            "Each rule should consist of a single human-interpretable concept. Do not try to pack completely unrelated activating tokens into the same rule.\n"
            "For example, if the feature activates on the word 'stop' and also on the word 'cookie', you should put them in different rules.\n"
            "But sufficiently similar or conceptually related activating tokens can be grouped together in the same rule.\n"
            "For example, if the feature activates on the word 'car' and also on the word 'truck', you should put them in the same rule.\n\n"
            f"The activation format is token<tab>activation. Values range from 0 to {MAX_FORMATTED_EXPRESSION}. "
            "Non-zero activations indicate the feature found what it's looking for. Higher values "
            "indicate stronger matches.\n"
            f"{holistic_explanation_str}\n"
            "Try to keep the 'activates_on' field short.\n"
            "Also keep the list of rules as short as possible. "
            "Only add rules to the list if they are really necessary; "
            "i.e., only add a rule if the feature activates on it and it's not already in the list.\n"
            f"{rule_cap_str(rule_cap)}\n\n"
            "Format your response as a JSON list of dictionaries with 'activates_on' and 'strength' fields.",
        )
    else:
        prompt_builder.add_message(
            Role.SYSTEM,
            "We're studying neurons in a neural network. Each neuron looks for some particular "
            "thing in a short document. Look at the parts of the document the neuron activates for "
            "and summarize in a single sentence what the neuron is looking for. Don't list "
            "examples of words.\n\nThe activation format is token<tab>activation. Activation "
            f"values range from 0 to {MAX_FORMATTED_EXPRESSION}. A neuron finding what it's looking for is represented by a "
            "non-zero activation value. The higher the activation value, the stronger the match.\n\n"
            f"{holistic_explanation_str}",
        )


def add_per_feature_explanation_prompt(
    prompt_builder: PromptBuilder,
    records: list[FormattedFeatureExpressionRecord],
    feature_index: int | None,
    explanation: Optional[StructuredExplanation | str],
    include_holistic_expressions: bool,
    structured_explanations: bool,
    repeat_non_zero_expressions: bool = True,
) -> None:
    
    feature_string = f"Feature {feature_index + 1}" if feature_index is not None else ""
    
    stringified_records = stringify_records(
        records,
        include_holistic_expressions=include_holistic_expressions,
        omit_zeros=False,
    )

    user_message = (
        f"{feature_string}\n\n"
        f"Activation records:\n{stringified_records}"
    )
    # We repeat the non-zero expressions only if it was requested and if the proportion of
    # non-zero expressions isn't too high.
    if (
        repeat_non_zero_expressions
        and non_zero_expression_proportion(records) < 0.2
    ):
        no_zero_stringified_records = stringify_records(
            records,
            include_holistic_expressions=include_holistic_expressions,
            omit_zeros=True,
        )
        user_message += (
            f"\nSame records, but with all zeros filtered out:\n"
            f"{no_zero_stringified_records}"
        )
    
    if structured_explanations:
        prefix = STRUCTURED_EXPLANATION_PREFIX
    else:
        prefix = STRING_EXPLANATION_PREFIX
    user_message += f"\n\n{feature_string}\n{prefix}"
    prompt_builder.add_message(Role.USER, user_message)

    assistant_message = ""
    if explanation is not None:
        if structured_explanations:
            assistant_message += f"{explanation.to_json()}"
        else:
            assistant_message += explanation
    if assistant_message:
        prompt_builder.add_message(Role.ASSISTANT, assistant_message)


def parse_structured_explanation(completion: str) -> StructuredExplanation | None:
    match = re.search(r'\[(.*)\]', completion, re.DOTALL)
    if match is None:
        return None

    json_str = match.group(0)
    return StructuredExplanation.from_json(json_str)


def rule_truncate(
    explanation: StructuredExplanation,
    rule_cap: int,
) -> StructuredExplanation:

    explanation = copy.deepcopy(explanation)
    explanation.components = explanation.components[:rule_cap]
    return explanation


class FeatureExplainer(ABC):
    def __init__(
        self,
        model_name: str,
        max_concurrent: Optional[int] = 30,
    ):
        self.model_name = model_name
        self.client = ApiClient(model_name=model_name, max_concurrent=max_concurrent)

    @abstractmethod
    async def generate_explanations(
        self,
        train_records: list[FormattedFeatureExpressionRecord],
        valid_records: list[FormattedFeatureExpressionRecord],
        num_samples: int,
    ) -> tuple[list[StructuredExplanation] | list[str], dict[str, Any]]:
        ...


@dataclass
class OneShotExplainerParams:
    max_tokens: int = 1500
    temperature: float = 1.0
    top_p: float = 1.0
    rule_cap: int = 10
    include_holistic_expressions: bool = False
    structured_explanations: bool = True


class OneShotExplainer(FeatureExplainer):
    def __init__(
        self,
        model_name: str,
        few_shot_example_set: RecordExampleSet = RecordExampleSet.ORIGINAL,
        params: OneShotExplainerParams = OneShotExplainerParams(),
    ):
        super().__init__(model_name=model_name)
        self.few_shot_example_set = few_shot_example_set
        self.params = params

    async def generate_explanations(
        self,
        train_records: list[FormattedFeatureExpressionRecord],
        valid_records: list[FormattedFeatureExpressionRecord],
        num_samples: int = 1,
        recursion_depth: int = 0,
    ) -> tuple[list[StructuredExplanation] | list[str], dict[str, Any]]:

        builder = self._explanation_prompt(records=train_records)
        prompt = builder.build()

        assert isinstance(prompt, list)

        response = await self.client.make_request(
            n=num_samples,
            max_tokens=self.params.max_tokens,
            temperature=self.params.temperature,
            top_p=self.params.top_p,
            messages=prompt,
        )
        
        if 'choices' not in response:
            raise ValueError("No choices in response")
        
        explanations = [x["message"]["content"] for x in response["choices"]]
        if self.params.structured_explanations:
            explanations = [parse_structured_explanation(x) for x in explanations]
            # Could get a None if explanation is misformed
            explanations = [e for e in explanations if e is not None]
            explanations = [rule_truncate(e, self.params.rule_cap) for e in explanations]

        if len(explanations) < num_samples:
            if recursion_depth > 5:
                print('Recursion depth exceeded')
                if self.params.structured_explanations:
                    dummy = StructuredExplanation(components=[
                        ExplanationComponent(activates_on="", strength=0)
                    ])
                else:
                    dummy = "No explanation found"

                explanations.extend([dummy] * (num_samples - len(explanations)))
                return explanations, {}
            
            extra_explanations, _ = await self.generate_explanations(
                train_records=train_records,
                valid_records=valid_records,
                num_samples=num_samples - len(explanations),
                recursion_depth=recursion_depth + 1,
            )
            explanations.extend(extra_explanations)
        
        return explanations, {}

    def _explanation_prompt(
        self,
        records: list[FormattedFeatureExpressionRecord],
    ) -> PromptBuilder:

        prompt_builder = PromptBuilder()
        add_system_prompt(
            prompt_builder,
            rule_cap=self.params.rule_cap,
            include_holistic_expressions=self.params.include_holistic_expressions,
            structured_explanations=self.params.structured_explanations,
        )
        few_shot_examples = self.few_shot_example_set.get_examples()
        for i, few_shot_example in enumerate(few_shot_examples):
            few_shot_expression_records = few_shot_example.records
            if self.params.structured_explanations:
                explanation = few_shot_example.structured_explanation
            else:
                explanation = few_shot_example.string_explanation

            add_per_feature_explanation_prompt(
                prompt_builder,
                few_shot_expression_records,
                i,
                explanation=explanation,
                include_holistic_expressions=self.params.include_holistic_expressions,
                structured_explanations=self.params.structured_explanations,
            )
        add_per_feature_explanation_prompt(
            prompt_builder,
            records,
            len(few_shot_examples),
            explanation=None,
            include_holistic_expressions=self.params.include_holistic_expressions,
            structured_explanations=self.params.structured_explanations,
        )

        return prompt_builder


# Adapted from https://github.com/dreadnode/parley

@dataclass
class ExplanationAttempt:
    improvement: Optional[str]
    explanation: StructuredExplanation | str


@dataclass
class TreeNode:
    children: list["TreeNode"]
    """The children of this node."""
    conversation: list[Message]
    """The conversation history of this node."""
    attempt: Optional[ExplanationAttempt]
    """An attempt at explaining the feature."""
    feedback: Optional[str]
    """Contains the validation score and most incorrect validation examples
    for the next iteration of the explanation generation."""
    score_train: Optional[float]
    """The score of the explanation."""
    score_valid: Optional[float]
    """The validation score of the explanation."""
    iteration: int


@dataclass
class TreeExplainerParams:
    root_nodes: int = 3
    branching_factor: int = 2
    explainer_temperature: float = 1.2
    explainer_top_p: float = 1.0
    width: int = 3
    depth: int = 3
    stop_score: float = 0.9
    rule_cap: int = 10
    k_worst_feedback_records: int = 1
    include_holistic_expressions: bool = False
    structured_explanations: bool = True

    print_explanations: bool = False
    print_truncation: int = 100


class TreeExplainer(FeatureExplainer):
    def __init__(
        self,
        model_name: str,
        simulator_factory: Callable[
            [StructuredExplanation | str], ExplanationFeatureSimulator
        ],
        few_shot_example_set: RecordExampleSet = RecordExampleSet.ORIGINAL,
        params: TreeExplainerParams = TreeExplainerParams(),
    ):
        super().__init__(model_name=model_name)
        
        self.simulator_factory = simulator_factory
        self.few_shot_example_set = few_shot_example_set
        self.params = params

    async def generate_explanations(
        self,
        train_records: list[FormattedFeatureExpressionRecord],
        valid_records: list[FormattedFeatureExpressionRecord],
        num_samples: int = 1,
    ) -> tuple[list[StructuredExplanation] | list[str], dict[str, Any]]:

        async def score_and_feedback_fn(
            explanation: StructuredExplanation
        ) -> tuple[float, float, str]:
            return await self._score_and_feedback(
                explanation=explanation,
                records=train_records,
                valid_records=valid_records,
            )

        root_nodes = await self._one_shot_root_nodes(train_records=train_records)
        
        await self._add_score_and_feedback(
            root_nodes,
            score_and_feedback_fn=score_and_feedback_fn,
            log_prefix="Root",
        )

        # All nodes which have ever been generated
        all_nodes: list[TreeNode] = copy.deepcopy(root_nodes)
        # The current nodes being processed at this level of the tree
        current_nodes: list[TreeNode] = copy.deepcopy(root_nodes)
        
        self._log("Starting best score: " + str(max(node.score_train for node in root_nodes)))
        
        self._log("[+] Beginning TAP ...")
        for iteration in range(self.params.depth):
            self._log(f" |- Iteration {iteration + 1} with {len(current_nodes)} nodes")
            
            for i, node in enumerate(current_nodes):
                self._append_feedback_to_conversation(node)
                
            await asyncio.gather(*[
                self._create_child_explanation_attempts(node)
                for node in current_nodes
            ])

            for i, node in enumerate(current_nodes):
                log_messages = await self._add_score_and_feedback(
                    node.children,
                    score_and_feedback_fn=score_and_feedback_fn,
                    log_prefix=f"{iteration + 1}->{i + 1}",
                    log=False,
                )
                for message in log_messages:
                    self._log(message)

            children = [child for node in current_nodes for child in node.children]
            children = self._sort_nodes_by_score(children, train_or_valid='train')

            if len(children) == 0:
                self._log("\n[!] No more nodes to explore\n")
                raise RuntimeError("No more nodes to explore")

            all_nodes.extend(copy.deepcopy(children))
            current_nodes = children[:self.params.width]
            
            best_score = current_nodes[0].score_train
            # assert best_score == max(n.score_train for n in current_nodes)
            
            self._log(f"=== New best score: {best_score}")
            
            if best_score >= self.params.stop_score:
                self._log("\n[!] Found a good explanation!\n")
                break
        
        all_nodes = self._sort_nodes_by_score(all_nodes, train_or_valid='valid')
        explanations = [node.attempt.explanation for node in all_nodes[:num_samples]]

        return explanations, {
            'all_explanations': [node.attempt.explanation for node in all_nodes],
            'all_train_scores': [float(node.score_train) for node in all_nodes],
            'all_valid_scores': [float(node.score_valid) for node in all_nodes],
            'all_iterations': [node.iteration for node in all_nodes],
        }
        
    def _print_format(self, text: str) -> str:
        return ' '.join(text[:self.params.print_truncation].split())
    
    def _sort_nodes_by_score(
        self,
        nodes: list[TreeNode],
        train_or_valid: str = 'train',
        filter_nan: bool = True,
    ) -> list[TreeNode]:
        """Sort nodes by score, with None scores at the end."""
        
        def get_score(node: TreeNode) -> float:
            assert train_or_valid in ['train', 'valid']
            score = node.score_train if train_or_valid == 'train' else node.score_valid
            if score is None or (filter_nan and np.isnan(score)):
                return float("-inf")
            return score
        
        return sorted(nodes, key=get_score, reverse=True)
    
    async def _one_shot_root_nodes(
        self,
        train_records: list[FormattedFeatureExpressionRecord],
    ) -> list[TreeNode]:

        header_builder = PromptBuilder()
        add_system_prompt(
            header_builder,
            rule_cap=self.params.rule_cap,
            include_holistic_expressions=self.params.include_holistic_expressions,
            structured_explanations=self.params.structured_explanations,
        )
        add_per_feature_explanation_prompt(
            header_builder,
            train_records,
            feature_index=None,
            explanation=None,
            include_holistic_expressions=self.params.include_holistic_expressions,
            structured_explanations=self.params.structured_explanations,
        )

        one_shot_explainer = OneShotExplainer(
            model_name=self.model_name,
            few_shot_example_set=self.few_shot_example_set,
            params=OneShotExplainerParams(
                rule_cap=self.params.rule_cap,
                include_holistic_expressions=self.params.include_holistic_expressions,
                structured_explanations=self.params.structured_explanations,
            ),
        )
        explanations, _ = await one_shot_explainer.generate_explanations(
            train_records=train_records,
            valid_records=None, # Not used for one-shot
            num_samples=self.params.root_nodes,
        )
            
        def stringify(explanation: str | StructuredExplanation) -> str:
            if isinstance(explanation, str):
                return explanation
            return explanation.to_json()
        
        root_nodes: list[TreeNode] = [
            TreeNode(
                children=[],
                conversation=header_builder.build() + [
                    Message(role=Role.ASSISTANT, content=stringify(explanation))
                ],
                attempt=ExplanationAttempt(improvement=None, explanation=explanation),
                feedback=None,
                score_train=None,
                score_valid=None,
                iteration=0,
            )
            for explanation in explanations
        ]
        return root_nodes
    
    def _append_feedback_to_conversation(self, node: TreeNode) -> None:
        next_message = Message(
            role=Role.USER,
            content=(
                f"Feedback on this explanation attempt:\n"
                f"{node.feedback}\n{self._feedback_suffix()}"
            )
        )
        node.conversation.append(next_message)
    
    async def _create_child_explanation_attempts(self, node: TreeNode) -> None:
        """Adds branching_factor child conversation completions to the node."""

        completions = await asyncio.gather(*[
            self._get_completion(node.conversation)
            for _ in range(self.params.branching_factor)
        ])
        
        for completion in completions:
            try:
                improvement, explanation = self._parse_completion(completion)
            except ValueError as e:
                self._log(f"  |> Attack generation failed: {e}")
                self._log(completion)
                continue

            conversation = copy.deepcopy(node.conversation)
            conversation.append(
                Message(role=Role.ASSISTANT, content=completion)
            )

            node.children.append(
                TreeNode(
                    children=[],
                    conversation=conversation,
                    attempt=ExplanationAttempt(
                        improvement=improvement,
                        explanation=explanation,
                    ),
                    feedback=None,
                    score_train=None,
                    score_valid=None,
                    iteration=node.iteration + 1,
                )
            )
            
    async def _add_score_and_feedback(
        self,
        nodes: list[TreeNode],
        score_and_feedback_fn: Callable[[StructuredExplanation], tuple[float, str]],
        log_prefix: str,
        log: bool = True,
    ) -> list[str]:

        # 3 - Perform the inference + evaluations
        log_messages_lst: list[list[str]] = []
        for k, node in enumerate(nodes):
            messages = await self._evaluate_single_node(
                node, score_and_feedback_fn, f'{log_prefix}->{k + 1}'
            )
            log_messages_lst.append(messages)

        log_messages: list[str] = [
            message for messages in log_messages_lst for message in messages
        ]
        if log:
            for message in log_messages:
                self._log(message)
        return log_messages

    async def _evaluate_single_node(
        self,
        node: TreeNode,
        score_and_feedback_fn: Callable[[StructuredExplanation], tuple[float, str]],
        log_prefix: str,
    ) -> list[str]:

        log_messages: list[str] = []
        log_messages.append(f"  |= {log_prefix}")
        if node.attempt.improvement is not None:
            # This is the case for root nodes, there's only a one-shot explanation
            log_messages.append(f'   |- Improvement: "{self._print_format(node.attempt.improvement)}"')

        explanation = node.attempt.explanation
        if self.params.structured_explanations:
            explanation = explanation.to_json()
        log_messages.append(f'   |- Explanation: "{self._print_format(explanation)}"')
        
        score, score_valid, feedback = await score_and_feedback_fn(node.attempt.explanation)

        node.feedback = feedback
        node.score_train = score
        node.score_valid = score_valid

        log_messages.append(f'   |- Feedback:    "{self._print_format(node.feedback)}"')
        log_messages.append(f"   |- Score:       {node.score_train}")
        log_messages.append(f"   |- Validation score: {node.score_valid}")
        return log_messages

    def _feedback_suffix(self) -> str:
        """Text appended to the feedback to guide the explanation generation."""
        
        if self.params.structured_explanations:
            return (
                "Try generating a better explanation, taking into account this feedback. "
                "Be creative; if you have been trying something and it isn't working, try something else.\n"
                "Format your response as including first an improvement (in natural language), then the explanation.\n"
                "You must make your improvement precise and specific; here are a few examples:\n"
                "- \"The feature is activating on the word 'cat', but my explanation doesn't capture this. I should add another rule to my explanation list.\"\n"
                "- \"My score is much lower than previous attempts. I should remove the rules I added recently, perhaps they are too long and confusing.\"\n"
                "- \"My last rule to says that 'cat' activates in any context, but this examples show that it doesn't activate when 'dog' appears previously in the text. I should add this required context to the rule.\"\n"
                "- \"My rule says to activate on 'cat' with a strength of 2, but the examples have higher activations. I don't need to add any new rules, but I should increase the strength of my rule.\"\n"
                f"{rule_cap_str(self.params.rule_cap)}\n"
                "Structure your response as follows:\n\n"
                "IMPROVEMENT: ... \n\n"
                "NEW EXPLANATION: [ { \"activates_on\": ..., \"strength\": ... }, ... ]\n\n"
                "Note that the \"activates_on\" field is a single string, not a list of strings.\n"
                "Do not include any comment in the JSON list."
            )
        else:
            return (
                "Try generating a better explanation, taking into account this feedback. "
                "Be creative; if you have been trying something and it isn't working, try something else.\n"
                "Format your response as including first an improvement (in natural language), then the explanation.\n"
                "You must make your improvement precise and specific; here are a few examples:\n"
                "- \"The neuron is activating on the word 'cat', but my explanation doesn't capture this. I should amend my explanation.\"\n"
                "- \"My score is much lower than previous attempts. I should undo my recent changes.\"\n"
                "- \"My last explanation says that 'cat' activates in any context, but this examples show that it doesn't activate when 'dog' appears previously in the text. I should narrow the context of my explanation.\"\n"
                "Structure your response as follows:\n\n"
                "IMPROVEMENT: ... \n\n"
                "NEW EXPLANATION: ...\n\n"
            )
    
    def _parse_completion(
        self,
        completion: str,
    ) -> tuple[str, StructuredExplanation | str]:
        """Parse the completion into an improvement and explanation."""

        try:
            completion = completion.split("IMPROVEMENT:")[1] 
            improvement = completion.split("NEW EXPLANATION:")[0].strip()
            explanation = completion.split("NEW EXPLANATION:")[1].strip()
            if self.params.structured_explanations:
                explanation = parse_structured_explanation(explanation)
                explanation = rule_truncate(explanation, self.params.rule_cap)
                if explanation is None:
                    raise ValueError("Failed to parse explanation")
            return improvement, explanation
        except (IndexError, AttributeError):
            raise ValueError("Failed to parse completion")

    async def _score_and_feedback(
        self,
        explanation: StructuredExplanation | str,
        records: list[FormattedFeatureExpressionRecord],
        valid_records: list[FormattedFeatureExpressionRecord],
    ) -> tuple[float, float, str]:
        """Feedback consists of the original score and the worst-simulated records."""
        
        k_worst_records = self.params.k_worst_feedback_records
        
        simulator = self.simulator_factory(explanation)
        
        scored_simulation = await simulate_and_score(simulator, records)
        scored_simulation_valid = await simulate_and_score(simulator, valid_records)

        original_score = scored_simulation.get_preferred_score()
        original_score_valid = scored_simulation_valid.get_preferred_score()

        improvements: list[tuple[float, ScoredSequenceSimulation]] = []
        for i, sim in enumerate(scored_simulation.scored_sequence_simulations):
            remaining_sims = scored_simulation.scored_sequence_simulations[:i] + \
                           scored_simulation.scored_sequence_simulations[i+1:]
            
            new_scored_sim = aggregate_scored_sequence_simulations(remaining_sims)
            new_score = new_scored_sim.get_preferred_score()
            improvements.append((new_score - original_score, sim))
        
        # Sort such that the first element is the largest score improvement
        # A large score improvement means the explanation did poorly on this sequence
        improvements.sort(reverse=True, key=lambda x: x[0])
        
        low_score_sims = improvements[:k_worst_records]
        
        feedback_str = f"Overall score (out of 1.0 max): {original_score:.2f}\n"
        if k_worst_records == 1:
            feedback_str += f"Most incorrect record:\n"
        else:
            feedback_str += f"Top {k_worst_records} most incorrect records:\n"

        for _, sim in low_score_sims:
            if k_worst_records > 1:
                feedback_str += f"Record {i + 1}\n"

            tokens = sim.simulation.tokens
            pred_expressions = sim.simulation.expected_expressions
            true_expressions = sim.true_expressions
            
            pred_expressions = np.array(pred_expressions).round().astype(int).tolist()
            
            if self.params.include_holistic_expressions:
                feedback_str += 'Activation-causing tokens (review)\n'
                feedback_str += '<start>\n'
                feedback_str += stringify_record(sim.record, 'holistic')
                feedback_str += '\n<end>\n\n'
            
            feedback_str += 'Activating token errors'
            feedback_str += stringify_contrasting_expressions(
                tokens, pred_expressions, true_expressions
            )
        
        return original_score, original_score_valid, feedback_str

    async def _get_completion(self, prompt: list[Message]) -> str:
        response = await self.client.make_request(
            n=1,
            max_tokens=500 + self.params.rule_cap * 200,
            temperature=self.params.explainer_temperature,
            top_p=self.params.explainer_top_p,
            messages=prompt
        )
        if 'choices' not in response:
            print(response)
            raise ValueError("No choices in response")
        # Don't want it to have terminated because of length
        if response["choices"][0]["finish_reason"] not in ["stop", "eos"]:
            print('Response terminated because of length')
        return [x["message"]["content"] for x in response["choices"]][0]

    def _log(self, message: str) -> None:
        """Print message only if print_explanations is enabled in tap_args."""
        if self.params.print_explanations:
            print(message)
