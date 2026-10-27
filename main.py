from dotenv import load_dotenv
import os
import asyncio
from llm_provider import AnthropicProvider

load_dotenv()

async def main():
    provider = AnthropicProvider(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        model_id="claude-opus-5-5",
        max_token_count=1024,
        system_prompt="You are a helpful assistant.")
    print(await provider.invoke("What is LLM?"))
    print(await provider.invoke("Explain reference engineering"))
    print(f"Total token spent: { provider.token_count()[0] + provider.token_count()[1]}")

if __name__ == "__main__":
    asyncio.run(main())
