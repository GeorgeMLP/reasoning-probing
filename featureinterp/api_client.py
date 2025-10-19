import asyncio
import contextlib
import os
import random
import traceback
from asyncio import Semaphore
from functools import wraps
from typing import Any, Callable, Optional

import httpx

from featureinterp.prompt_builder import Message, Role


def is_api_error(err: Exception) -> bool:
    if isinstance(err, httpx.HTTPStatusError):
        response = err.response
        if response.status_code in [400, 404, 415]:
            error_data = response.json().get("error", {})
            error_message = error_data.get("message")
            if error_data.get("type") == "idempotency_error":
                print(f"Retrying after idempotency error: {error_message} ({response.url})")
                return True
            else:
                # Invalid request
                return False
                # return True # Sometimes, the API returns a 404 for no reason.
        else:
            # print(f"Retrying after API error: {error_message} ({response.url})")
            return True

    elif isinstance(err, httpx.ConnectError):
        print(f"Retrying after connection error... ({err.request.url})")
        return True

    elif isinstance(err, httpx.TimeoutException):
        print(f"Retrying after a timeout error... ({err.request.url})")
        return True

    elif isinstance(err, httpx.ReadError):
        print(f"Retrying after a read error... ({err.request.url})")
        return True

    print(f"Retrying after an unexpected error: {repr(err)}")
    traceback.print_tb(err.__traceback__)
    return True


def exponential_backoff(
    retry_on: Callable[[Exception], bool] = lambda err: True
) -> Callable[[Callable], Callable]:
    """
    Returns a decorator which retries the wrapped function as long as the specified retry_on
    function returns True for the exception, applying exponential backoff with jitter after
    failures, up to a retry limit.
    """
    init_delay_s = 1.0
    max_delay_s = 10.0
    # Roughly 30 minutes before we give up.
    max_tries = 200
    backoff_multiplier = 2.0
    jitter = 0.2

    def decorate(f: Callable) -> Callable:
        assert asyncio.iscoroutinefunction(f)

        @wraps(f)
        async def f_retry(*args: Any, **kwargs: Any) -> None:
            delay_s = init_delay_s
            for i in range(max_tries):
                try:
                    return await f(*args, **kwargs)
                except Exception as err:
                    if not retry_on(err) or i == max_tries - 1:
                        raise
                    jittered_delay = random.uniform(delay_s * (1 - jitter), delay_s * (1 + jitter))
                    await asyncio.sleep(jittered_delay)
                    delay_s = min(delay_s * backoff_multiplier, max_delay_s)

        return f_retry

    return decorate


class ApiClient:
    def __init__(
        self,
        model_name: str,
        # If set, no more than this number of HTTP requests will be made concurrently.
        max_concurrent: Optional[int] = None,
    ):

        self.model_name = model_name

        if 'gemini' in model_name:
            if self.model_name == "google/gemini-flash-1.5-8b":
                api_model_name = "gemini-1.5-flash-8b"
            elif self.model_name == "google/gemini-2.0-flash":
                api_model_name = "gemini-2.0-flash"
            else:
                raise ValueError(f"Unsupported Gemini model: {model_name}")

            self.api_key = os.getenv("GOOGLE_API_KEY")
            self.base_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{api_model_name}:generateContent?key={self.api_key}"
            
            self.api_http_headers = {
                "Content-Type": "application/json",
            }
        elif 'openai' in model_name:
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_api_url = "https://api.openai.com/v1"
            self.model_name = model_name.split('/')[-1]

            self.api_http_headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            }
        else:
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.base_api_url = "https://openrouter.ai/api/v1"

            self.api_http_headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            }

        if max_concurrent is not None:
            self._concurrency_check: Optional[Semaphore] = Semaphore(max_concurrent)
        else:
            self._concurrency_check = None

        assert self.api_key, "Please set the api key environment variable"
    
    async def make_request(self, **kwargs: Any) -> dict[str, Any]:
        if 'gemini' in self.model_name:
            return await self.make_request_gemini(**kwargs)
        else:
            return await self.make_request_openrouter(**kwargs)
    
    @exponential_backoff(retry_on=is_api_error)
    async def make_request_gemini(
        self,
        timeout_seconds: Optional[int] = None,
        **kwargs: Any
    ) -> dict[str, Any]:

        messages = kwargs['messages']
        
        def process_message(message: Message) -> dict[str, Any]:
            if message.role == Role.SYSTEM:
                return { 'parts': [{ "text": message.content }] }
            elif message.role == Role.USER:
                return { 'role': 'user', 'parts': [{ "text": message.content }] }
            elif message.role == Role.ASSISTANT:
                return { 'role': 'model', 'parts': [{ "text": message.content }] }
            else:
                raise ValueError(f"Invalid message role: {message.role}")
        
        request = {
            "systemInstruction": process_message(messages[0]),
            "contents": [process_message(message) for message in messages[1:]],
            "generationConfig": {
                "maxOutputTokens": kwargs['max_tokens'],
                "temperature": kwargs['temperature'],
                "topP": kwargs['top_p'],
                "candidateCount": kwargs['n'],
            },
        }
        
        async with contextlib.AsyncExitStack() as stack:
            if self._concurrency_check is not None:
                await stack.enter_async_context(self._concurrency_check)

            http_client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=timeout_seconds)
            )
            response = await http_client.post(
                self.base_api_url,
                headers=self.api_http_headers,
                json=request,
            )

        try:
            response.raise_for_status()
        except Exception as e:
            # print(response.json())
            raise e
        
        return {
            'choices': [
                {
                    'finish_reason': candidate['finishReason'].lower(),
                    'message': { 'content': candidate['content']['parts'][0]['text'] },
                }
                for candidate in response.json()['candidates']
            ]
        }
    
    @exponential_backoff(retry_on=is_api_error)
    async def make_request_openrouter(
        self,
        timeout_seconds: Optional[int] = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        
        if 'messages' in kwargs:
            def convert_message(message: Message) -> dict[str, Any]:
                if isinstance(message.role, Role):
                    role = message.role.value
                elif isinstance(message.role, str):
                    role = message.role
                else:
                    raise ValueError(f"Invalid message role: {message.role}")
                return { 'role': role, 'content': message.content }

            kwargs['messages'] = [
                convert_message(message) for message in kwargs['messages']
            ]

        async with contextlib.AsyncExitStack() as stack:
            if self._concurrency_check is not None:
                await stack.enter_async_context(self._concurrency_check)

            http_client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=timeout_seconds)
            )
            url = self.base_api_url + "/chat/completions"
            kwargs["model"] = self.model_name
            
            # This should force the cheapest https://openrouter.ai/docs/provider-routing
            # kwargs["provider"] = { "allow_fallbacks": False }
            response = await http_client.post(url, headers=self.api_http_headers, json=kwargs)
        
        # The response json has useful information but the exception doesn't include it, so print it
        # out then reraise.
        try:
            response.raise_for_status()
        except Exception as e:
            print(response.json())
            raise e
        return response.json()


if __name__ == "__main__":
    async def main() -> None:
        client = ApiClient(model_name="deepseek/deepseek-chat", max_concurrent=1)
        messages = [
            Message(Role.SYSTEM, "You are a helpful assistant."),
            Message(Role.USER, "Why did the chicken cross the road?"),
        ]
        print(await client.make_request(messages=messages))

    asyncio.run(main())
