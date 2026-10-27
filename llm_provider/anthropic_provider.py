import inspect
import json
from typing import Any

from .base_provider import BaseModel
from .toolcall_utility import TOOL_REGISTRY
from anthropic import AsyncAnthropic

class AnthropicProvider(BaseModel):
    def __init__(
        self,
        api_key: str,
        model_id: str,
        max_token_count: int,
        system_prompt: str | None,
        tools: list[dict[str, Any]] | None = None,
    ):
        self.llm = AsyncAnthropic(api_key=api_key)
        self.model_id = model_id
        self.max_token_count = max_token_count
        self.system_prompt = system_prompt
        self.tools = tools
        self.messages: list[dict[str, Any]] = []
        self.input_token_count = 0
        self.output_token_count = 0

    async def invoke(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        response_text = []

        ## start the loop to interpret tool calls
        while True:
            response = await self.llm.messages.create(
                max_tokens=self.max_token_count,
                messages=self.messages,
                model=self.model_id,
                **({"system": self.system_prompt} if self.system_prompt else {}),
                **({"tools": self.tools} if self.tools is not None else {}),
            )

            self.input_token_count += response.usage.input_tokens
            self.output_token_count += response.usage.output_tokens

            self.messages.append({
                "role": "assistant",
                "content": [
                    block.model_dump(exclude_none=True)
                    for block in response.content
                ],
            })

            tool_results = []
            for block in response.content:
                if block.type == "text":
                    response_text.append(block.text)
                elif block.type == "tool_use":
                    try:
                        tool = TOOL_REGISTRY.get(block.name)
                        if tool is None:
                            raise RuntimeError(
                                f"No implementation is registered for tool {block.name!r}"
                            )

                        result = tool(**block.input)
                        if inspect.isawaitable(result):
                            result = await result

                        result_content = (
                            result
                            if isinstance(result, str)
                            else json.dumps(result, default=str)
                        )
                        is_error = False
                    except Exception as error:
                        result_content = f"{type(error).__name__}: {error}"
                        is_error = True

                    ## append the tool result. It will be sent to the llm in the next call.
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