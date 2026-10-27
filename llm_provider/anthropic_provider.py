from .base_provider import BaseModel, Message, Role
from anthropic import AsyncAnthropic

class AnthropicProvider(BaseModel):
    def __init__(self, api_key: str, model_id: str, max_token_count: int, system_prompt: str | None):
        self.llm = AsyncAnthropic(api_key=api_key)
        self.model_id = model_id
        self.max_token_count = max_token_count
        self.system_prompt = system_prompt
        self.messages: list[Message] = []
        self.input_token_count = 0
        self.output_token_count = 0

    async def invoke(self, message: str) -> str:
        self.messages.append(Message(role=Role.USER, message=message))
        request_messages = [
            {"role": item.role, "content": item.message}
            for item in self.messages
        ]

        response = await self.llm.messages.create(
            max_tokens=self.max_token_count,
            messages=request_messages,
            model=self.model_id,
            **({"system": self.system_prompt} if self.system_prompt else {}),
        )

        self.input_token_count += response.usage.input_tokens
        self.output_token_count += response.usage.output_tokens

        ret_message = []
        for block in response.content:
            if block.type == "text":
                ret_message.append(block.text)

        response_message = "".join(ret_message)
        self.messages.append(Message(role=Role.ASSISTANT, message=response_message))

        return response_message

    def token_count(self) -> tuple[int, int]:
        return (self.input_token_count, self.output_token_count)