import os
import re
import copy
import json
import backoff
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, Dataset
from openai import OpenAI
from typing import Any


prompt = (
    "Please split the following reasoning chain of an LLM into annotated parts "
    "using labels and the following format [\"label\"]...[\"end-section\"]. "
    "A sentence should be split into multiple parts if it incorporates "
    "multiple behaviours indicated by the labels.\n\n"
    "Available labels:\n"
    "0. initializing -> The model is rephrasing the given task and states "
    "initial thoughts.\n"
    "1. deduction -> The model is performing a deduction step based on its "
    "current approach and assumptions.\n"
    "2. adding-knowledge -> The model is enriching the current approach with "
    "recalled facts.\n"
    "3. example-testing -> The model generates examples to test its current "
    "approach.\n"
    "4. uncertainty-estimation -> The model is stating its own uncertainty.\n"
    "5. backtracking -> The model decides to change its approach.\n\n"
    "The reasoning chain to analyze:\n"
    "{thinking_process}\n\n"
    "Answer only with the annotated text. Only use the labels outlined above. "
    "Remember to add [\"end-section\"] at the end of each part."
)
valid_labels = [
    'initializing',
    'deduction',
    'adding-knowledge',
    'example-testing',
    'uncertainty-estimation',
    'backtracking',
]


def backoff_hdlr(details: dict[str, Any]) -> None:
    print("Backing off {wait:0.1f} seconds after {tries} tries".format(**details))


@backoff.on_exception(backoff.expo, Exception, max_tries=5, on_backoff=backoff_hdlr)
def annotate_reasoning_chain(reasoning_chain: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt.format(thinking_process=reasoning_chain),
                },
            ],
        },
    ]
    completion = client.chat.completions.create(
        model='openai/gpt-4o',
        messages=messages,
    )
    return completion.choices[0].message.content


def split_annotated_reasoning_chain(
    chain: str,
    valid_labels: list[str] = valid_labels,
) -> list[tuple[str, str]]:
    chain.replace('[end-section]', '["end-section"]')
    for label in valid_labels:  # fix missing quotes
        chain.replace(f'[{label}]', f'["{label}"]')
    pattern = r'\["(.*?)"\](.*?)\["end-section"\]'
    matches: list[tuple[str, str]] = re.findall(pattern, chain, re.DOTALL)
    matches = list(filter(lambda t: t[0] in valid_labels, matches))
    matches = [(label, content.lstrip()) for label, content in matches]
    return matches


if __name__ == '__main__':
    save_path = Path('data', 'annotated_dataset.json')
    dataset: Dataset = load_dataset("simplescaling/s1K-1.1")["train"]
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv('OPENROUTER_API_KEY'),
    )

    annotated_dataset: dict[str, dict[str, Any]] = {}
    for i, data_instance in tqdm(enumerate(dataset), total=len(dataset)):
        data: dict[str, Any] = copy.deepcopy(data_instance)
        annotated_gemini_reasoning_chain = annotate_reasoning_chain(
            data['gemini_thinking_trajectory']
        )
        annotated_deepseek_reasoning_chain = annotate_reasoning_chain(
            data['deepseek_thinking_trajectory']
        )
        data['annotated_gemini_thinking_trajectory'] = \
            split_annotated_reasoning_chain(
                annotated_gemini_reasoning_chain,
                valid_labels=valid_labels,
            )
        data['annotated_deepseek_thinking_trajectory'] = \
            split_annotated_reasoning_chain(
                annotated_deepseek_reasoning_chain,
                valid_labels=valid_labels,
            )
        annotated_dataset[str(i)] = data
    
        with open(save_path, 'w') as f:
            json.dump(annotated_dataset, f, indent=4)
