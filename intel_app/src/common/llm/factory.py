from src.config.base_config import config
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI

def get_llm_client(async_mode: bool = False):
    if config.LLM_PROVIDER.lower() == "openai":
        if async_mode:
            return AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        else:
            return OpenAI(api_key=config.OPENAI_API_KEY)

    elif config.LLM_PROVIDER.lower() == "azure":
        if async_mode:
            return AsyncAzureOpenAI(
                    api_version="2024-12-01-preview",
                    azure_endpoint=config.AZURE_OPENAI_BASE_URL,
                    api_key=config.AZURE_OPENAI_API_KEY,
                )

        else:
            return AzureOpenAI(
                    api_version="2024-12-01-preview",
                    azure_endpoint=config.AZURE_OPENAI_BASE_URL,
                    api_key=config.AZURE_OPENAI_API_KEY,
                )


    else:
        raise ValueError(f"Unsupported LLM Provider: {config.LLM_PROVIDER}")