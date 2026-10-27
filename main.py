import asyncio
import datetime
import os
import time

from dotenv import load_dotenv

from llm_provider import AnthropicProvider
from llm_provider.toolcall_utility import tool

load_dotenv()


@tool
async def get_local_timezone() -> str:
    """Get the current local timezone name."""
    tz_name = time.tzname
    if isinstance(tz_name, (list, tuple)):
        return tz_name[0] if tz_name else "UTC"
    return tz_name or "UTC"


@tool
async def get_current_datetime(format: str = "%d.%m.%y %H:%M:%S") -> str:
    """Return the current local datetime formatted using a Python strftime pattern."""
    return datetime.datetime.now().strftime(format)


async def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    provider = AnthropicProvider(
        api_key=api_key,
        model_id="claude-opus-5-5",
        max_token_count=1024,
        system_prompt="You are a helpful assistant. You always give straight forward answer and no additional text.",
        tools=[get_current_datetime.schema, get_local_timezone.schema],
    )
    print(await provider.invoke("What is the time and date now?", reset=True))
    print(f"Total token spent: {provider.token_count()[0] + provider.token_count()[1]}")


if __name__ == "__main__":
    asyncio.run(main())
