import inspect
import json
from typing import Any, cast

from anthropic import AsyncAnthropic

from .base_provider import BaseModel
from .toolcall_utility import TOOL_REGISTRY


class AnthropicProvider(BaseModel):
    def __init__(
        self,
        api_key: str,
        model_id: str,
        max_token_count: int,
        system_prompt: str | None,
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tool_rounds: int = 10,
    ):
        self.llm = AsyncAnthropic(api_key=api_key)
        self.model_id = model_id
        self.max_token_count = max_token_count
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_tool_rounds = max_tool_rounds
        self.messages: list[dict[str, Any]] = []
        self.input_token_count = 0
        self.output_token_count = 0

    def reset(self) -> None:
        self.messages = []
        self.input_token_count = 0
        self.output_token_count = 0

    async def invoke(self, message: str, *, reset: bool = False) -> str:
        if reset:
            self.reset()

        self.messages.append({"role": "user", "content": message})
        response_text: list[str] = []
        tool_rounds = 0

        while True:
            if tool_rounds >= self.max_tool_rounds:
                raise RuntimeError(
                    f"Exceeded max tool-call rounds ({self.max_tool_rounds}). "
                    "Consider reducing tool chaining or increasing max_tool_rounds."
                )
            tool_rounds += 1

            response = await self.llm.messages.create(
                max_tokens=self.max_token_count,
                messages=self.messages,
                model=self.model_id,
                **({"system": self.system_prompt} if self.system_prompt else {}),
                **({"tools": self.tools} if self.tools else {}),
            )

            self.input_token_count += response.usage.input_tokens
            self.output_token_count += response.usage.output_tokens

            self.messages.append(
                {
                    "role": "assistant",
                    "content": [
                        block.model_dump(exclude_none=True)
                        for block in response.content
                    ],
                }
            )

            tool_results = []
            for block in response.content:
                if block.type == "text":
                    response_text.append(block.text)
                elif block.type == "tool_use":
                    result_content, is_error = await self._invoke_tool(
                        block.name,
                        cast(dict[str, Any], block.input),
                    )
                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_content,
                    }
                    if is_error:
                        tool_result["is_error"] = True
                    tool_results.append(tool_result)

            if not tool_results:
                return "".join(response_text)

            self.messages.append({"role": "user", "content": tool_results})

    def token_count(self) -> tuple[int, int]:
        return (self.input_token_count, self.output_token_count)

    async def _invoke_tool(self, tool_name: str, tool_args: dict[str, Any]) -> tuple[str, bool]:
        try:
            tool = TOOL_REGISTRY.get(tool_name)
            if tool is None:
                raise RuntimeError(f"No implementation is registered for tool {tool_name!r}")

            result = tool(**tool_args)
            if inspect.isawaitable(result):
                result = await result

            if hasattr(result, "model_dump"):
                payload = result.model_dump(mode="json")
            elif hasattr(result, "dict"):
                payload = result.dict()
            elif isinstance(result, (dict, list, tuple, set)):
                payload = result
            else:
                payload = result

            result_content = (
                result if isinstance(result, str)
                else json.dumps(payload, default=str, ensure_ascii=False)
            )
            is_error = False
        except Exception as error:  # noqa: BLE001
            result_content = f"{type(error).__name__}: {error}"
            is_error = True

        return (result_content, is_error)