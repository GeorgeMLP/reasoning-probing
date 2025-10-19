from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum

import tiktoken


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role | str
    content: str


class PromptBuilder:
    """Class for accumulating components of a prompt and then formatting them into an output."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add_message(self, role: Role, message: str) -> None:
        self._messages.append(Message(role=role, content=message))

    def prompt_length_in_tokens(self) -> int:
        # TODO(sbills): Make the model/encoding configurable. This implementation assumes GPT-4.
        encoding = tiktoken.get_encoding("cl100k_base")
        # Approximately-correct implementation adapted from this documentation:
        # https://platform.openai.com/docs/guides/chat/introduction
        num_tokens = 0
        for message in self._messages:
            num_tokens += (
                4  # every message follows <|im_start|>{role/name}\n{content}<|im_end|>\n
            )
            num_tokens += len(encoding.encode(message.content, allowed_special="all"))
        num_tokens += 2  # every reply is primed with <|im_start|>assistant
        return num_tokens

    def build(
        self,
        *,
        allow_extra_system_messages: bool = False,
        check_expected_role: bool = True,
    ) -> list[Message]:
        """
        Validates the messages added so far (reasonable alternation of assistant vs. user, etc.)
        and returns a list of dictionaries suitable for use with the /chat/completions endpoint.

        The `allow_extra_system_messages` parameter allows the caller to specify that the prompt
        should be allowed to contain system messages after the very first one.
        """
        messages = [copy.deepcopy(message) for message in self._messages]

        expected_next_role = Role.SYSTEM
        for message in messages:
            role = message.role
            if check_expected_role:
                assert role == expected_next_role or (
                    allow_extra_system_messages and role == Role.SYSTEM
                ), f"Expected message from {expected_next_role} but got message from {role}"

            if role == Role.SYSTEM:
                expected_next_role = Role.USER
            elif role == Role.USER:
                expected_next_role = Role.ASSISTANT
            elif role == Role.ASSISTANT:
                expected_next_role = Role.USER
                
        return messages
