import logging
from typing import Any, Optional, Union
import requests
import json
import copy
import asyncio
import openai
import aiohttp
from openai import OpenAI

from evals.api import CompletionFn, CompletionResult
from evals.base import CompletionFnSpec
from evals.prompt.base import (
    ChatCompletionPrompt,
    CompletionPrompt,
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
)
from evals.record import record_sampling
from evals.utils.api_utils import create_retrying

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

logger = logging.getLogger(__name__)

# 通过ip:port发请求时设置的超时时间
TIMEOUT = 3 * 3600
# 发送请求的HEADERS
HEADERS = {"User-Agent": "Benchmark Client", "Content-Type": "application/json"}

def openai_completion_create_retrying(client: OpenAI, *args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    result = create_retrying(
        client.completions.create, retry_exceptions=OPENAI_TIMEOUT_EXCEPTIONS, *args, **kwargs
    )
    if "error" in result:
        logger.warning(result)
        raise openai.APIError(result["error"])
    return result


def openai_chat_completion_create_retrying(client: OpenAI, *args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    result = create_retrying(
        client.chat.completions.create, retry_exceptions=OPENAI_TIMEOUT_EXCEPTIONS, *args, **kwargs
    )
    if "error" in result:
        logger.warning(result)
        raise openai.APIError(result["error"])
    return result


class OpenAIBaseCompletionResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any):
        self.raw_data = raw_data
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        raise NotImplementedError


class OpenAIChatCompletionResult(OpenAIBaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data:
            for choice in self.raw_data.choices:
                if choice.message.content is not None:
                    completions.append(choice.message.content)
        return completions


class OpenAICompletionResult(OpenAIBaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data:
            for choice in self.raw_data.choices:
                completions.append(choice.text)
        return completions


class OpenAICompletionFn(CompletionFn):
    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        n_ctx: Optional[int] = None,
        extra_options: Optional[dict] = {},
        **kwargs,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> OpenAICompletionResult:
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"

            prompt = CompletionPrompt(
                raw_prompt=prompt,
            )

        openai_create_prompt: OpenAICreatePrompt = prompt.to_formatted_prompt()

        result = openai_completion_create_retrying(
            OpenAI(api_key=self.api_key, base_url=self.api_base),
            model=self.model,
            prompt=openai_create_prompt,
            **{**kwargs, **self.extra_options},
        )
        result = OpenAICompletionResult(raw_data=result, prompt=openai_create_prompt)
        record_sampling(
            prompt=result.prompt,
            sampled=result.get_completions(),
            model=result.raw_data.model,
            usage=result.raw_data.usage,
        )
        return result


class OpenAIChatCompletionFn(CompletionFnSpec):
    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        n_ctx: Optional[int] = None,
        extra_options: Optional[dict] = {},
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> OpenAIChatCompletionResult:
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"

            prompt = ChatCompletionPrompt(
                raw_prompt=prompt,
            )

        openai_create_prompt: OpenAICreateChatPrompt = prompt.to_formatted_prompt()

        result = openai_chat_completion_create_retrying(
            OpenAI(api_key=self.api_key, base_url=self.api_base),
            model=self.model,
            messages=openai_create_prompt,
            **{**kwargs, **self.extra_options},
        )
        result = OpenAIChatCompletionResult(raw_data=result, prompt=openai_create_prompt)
        record_sampling(
            prompt=result.prompt,
            sampled=result.get_completions(),
            model=result.raw_data.model,
            usage=result.raw_data.usage,
        )
        return result

class UserChatCompletionResult(OpenAIBaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data:
            for choice in self.raw_data.get("choices", []):
                if choice.get("message", {}).get("content", '') is not None:
                    content = choice.get("message", {}).get("content", '')
                elif choice.get("message", {}).get("reasoning_content", '') is not None:
                    reasoning_content = choice.get("message", {}).get("reasoning_content", '')
                output = content if content != "" else reasoning_content
                logger.info(f"output: {output}")
                request_id = choice.get("message", {}).get("id", "request_id not found")
                if output == "":
                    logger.error(f"Request_id为{request_id}的请求返回信息为空，请检查具体原因!!!")
                    raise Exception(f"Request_id为{request_id}的请求返回信息为空，请检查具体原因!!!")
                completions.append(output)
        return completions

class UserChatCompletionFn(CompletionFnSpec):
    def __init__(
        self,
        model: Optional[str] = None,
        api_base: str = "",
        payload: dict = {},
        enable_pc_offload: bool = False,
        extra_options: Optional[dict] = {},
    ):
        self.model = model
        self.api_base = api_base
        assert self.api_base, "api_base is required"
        self.url = f"http://{self.api_base}/v1/chat/completions"
        self.payload = payload
        # 如果payload为str类型，将其转为dict
        if isinstance(self.payload, str):
            self.payload = json.loads(self.payload)
        assert self.payload, "payload is required"
        self.enable_pc_offload = enable_pc_offload
        self.extra_options = extra_options
    
    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> UserChatCompletionResult:
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"

            prompt = ChatCompletionPrompt(
                raw_prompt=prompt,
            )

        # 这里多并发情况下每个每个请求需要有自己独立的payload
        per_request_payload = copy.deepcopy(self.payload)
        user_create_prompt: OpenAICreateChatPrompt = prompt.to_formatted_prompt()
        per_request_payload = self._update_payload(per_request_payload, user_create_prompt, self.model, self.enable_pc_offload)
        # result = asyncio.run(self._async_do_request(per_request_payload))
        # 这里由于eval.py中的eval_all_samples通过线性池进行调用，这里采用同步发送请求的方式
        result = self._do_request(per_request_payload)

        result = UserChatCompletionResult(raw_data=result, prompt=user_create_prompt)
        record_sampling(
            prompt=result.prompt,
            sampled=result.get_completions(),
            model=result.raw_data.get("model"),
            usage=result.raw_data.get("usage"),
        )
        return result

    def _update_payload(self, payload: dict, prompt: str, model: str, enable_pc_offload: bool) -> None:
        """
        对payload进行更新, 主要需要对prompt、model名, 以及enable_pc_ooffload进行更新
        """
        payload.update({"messages": prompt})
        payload.update({"model": model})
        payload.update({"enable_pc_offload": enable_pc_offload})
        return payload

    def _do_request(self, payload: dict):
        """
        同步发请求, eval.py中的eval_all_samples通过线性池调用时使用该函数
        """
        try:
            response = requests.post(self.url, headers=HEADERS, json=payload)
            response.raise_for_status()
            return json.loads(response.text)
        except Exception as err:
            self._handle_request_error(err)
    
    async def _async_do_request(self, payload: dict):
        """
        异步发请求, 当采用eval中的async_eval_all_samples运行实际请求时使用该函数
        """
        try:
            timeout = aiohttp.ClientTimeout(total=TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(ssl=False)) as session:
                async with session.post(self.url, headers=HEADERS, json=payload) as response:
                    response.raise_for_status()
                    return await response.json
        except Exception as err:
            self._handle_request_error(err)
    
    def _handle_request_error(self, err: Exception) -> None:
        """
        用于对_do_request及_async_do_request的异常进行处理, 使用户能够更清楚出现异常的原因
        """
        if isinstance(err, (requests.exceptions.ConnectionError, aiohttp.ClientConnectionError)):
            logger.error(f"无法连接到服务器{self.api_base}, 检查网络是否可达")
            raise ConnectionError(f"无法连接到服务器{self.api_base}, 检查网络是否可达")
        elif isinstance(err, (requests.exceptions.Timeout, aiohttp.ServerTimeoutError)):
            logger.error("请求超时，检查服务端状态")
            raise TimeoutError("请求超时，检查服务端状态")
        elif isinstance(err, (requests.exceptions.HTTPError, aiohttp.ClientResponseError)):
            status_code = err.response.status_code if hasattr(err, "response") else err.status
            if status_code == 404:
                logger.error(f"请求资源不存在或是model名称错误")
            else:
                logger.error(f"HTTP错误, 状态码: {status_code}")
            raise Exception(f"HTTP错误, 状态码: {status_code}")
        else:
            logger.error(f"其他未知错误: {err}")
            raise Exception(f"其他未知错误: {err}")