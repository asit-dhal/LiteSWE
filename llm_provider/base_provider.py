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
    async def invoke(self, message: str) -> str:
        pass

    @abstractmethod
    def token_count(self) -> tuple[int, int]:
        pass
