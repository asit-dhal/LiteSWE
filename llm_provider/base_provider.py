from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, ConfigDict


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    role: Role
    message: str


class BaseModel(ABC):
    @abstractmethod
    async def invoke(self, message: str, *, reset: bool = False) -> str:
        pass

    @abstractmethod
    def token_count(self) -> tuple[int, int]:
        pass

    def reset(self) -> None:
        """Clear conversation state and counters."""
        self.messages = []
        self.input_token_count = 0
        self.output_token_count = 0
