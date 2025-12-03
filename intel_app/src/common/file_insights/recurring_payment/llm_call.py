import json
import time
from typing import List
from openai import OpenAI
from src.common.file_insights.metadata_extractor.json_parser import LLMOutputParser
from src.config.base_config import config
from src.common.logger import _log_message
import mlflow
# mlflow.config.enable_async_logging()
# mlflow.langchain.autolog()
mlflow.openai.autolog()
from opentelemetry import context as ot_context
from src.common.llm.factory import get_llm_client

# Load OpenAI API credentials
OPENAI_API_KEY = config.OPENAI_API_KEY
OPENAI_MODEL_NAME = config.OPENAI_MODEL_NAME


client = get_llm_client(async_mode=False)

MODULE_NAME = "llm_call.py"

# OpenAI model pricing details (in USD per million tokens)
PRICING = {
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.150, "cached_input": 0.075, "output": 0.600}
}

@mlflow.trace(name="Recurring Payment - LLM Call")
def open_ai_llm_call(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    temperature: float,
    function_name: str,
    logger
) -> str:
    """
    Calls OpenAI's LLM API with the provided prompts and logs relevant details.
    """
    try:
        start_time = time.perf_counter()
        logger.info(_log_message(f"Invoking OpenAI LLM API: {function_name}", function_name, MODULE_NAME))

        # Log API configuration details
        logger.debug(_log_message(f"Model: {model_name}, Temperature: {temperature}", function_name, MODULE_NAME))
        logger.debug(_log_message(f"System Prompt: {system_prompt}", function_name, MODULE_NAME))
        logger.debug(_log_message(f"User Prompt: {user_prompt}", function_name, MODULE_NAME))

        # Make the API call
        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        llm_answer = response.choices[0].message.content

        # Log and validate response
        logger.info(_log_message(f"LLM Response: {llm_answer}", function_name, MODULE_NAME))

        # Token cost computation
        if hasattr(response, "usage"):
            compute_costs(response, model_name, function_name, logger)

        duration = time.perf_counter() - start_time
        logger.info(_log_message(f"LLM call completed in {duration:.2f} seconds", function_name, MODULE_NAME))

        return llm_answer

    except Exception as e:
        logger.error(_log_message(f"Error during OpenAI API call: {e}", function_name, MODULE_NAME))
        return None

@mlflow.trace(name="Recurring Payment - Compute Costs")
def compute_costs(response, model_name, function_name, logger):
    """
    Computes and logs the cost breakdown of the API call.
    """
    try:
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        cached_tokens = response.usage.prompt_tokens_details.cached_tokens
        uncached_tokens = prompt_tokens - cached_tokens

        model_pricing = PRICING.get(model_name)
        if not model_pricing:
            raise ValueError(f"Pricing not defined for model: {model_name}")

        cost_cached_input = (cached_tokens / 1_000_000) * model_pricing["cached_input"]
        cost_uncached_input = (uncached_tokens / 1_000_000) * model_pricing["input"]
        cost_output = (completion_tokens / 1_000_000) * model_pricing["output"]
        total_cost = cost_cached_input + cost_uncached_input + cost_output

        logger.debug(_log_message("########### Token Usage Details ###########", function_name, MODULE_NAME))
        logger.debug(_log_message(f"Cached Tokens: {cached_tokens}, Cost: ${cost_cached_input:.6f}", function_name, MODULE_NAME))
        logger.debug(_log_message(f"Uncached Tokens: {uncached_tokens}, Cost: ${cost_uncached_input:.6f}", function_name, MODULE_NAME))
        logger.debug(_log_message(f"Output Tokens: {completion_tokens}, Cost: ${cost_output:.6f}", function_name, MODULE_NAME))
        logger.debug(_log_message(f"Total Cost: ${total_cost:.6f}", function_name, MODULE_NAME))
    except Exception as e:
        logger.error(_log_message(f"Error computing API costs: {e}", function_name, MODULE_NAME))



@mlflow.trace(name="Recurring Payment - Prepare Prompts")
def llm_call(query: str, retrieved_chunks: List[str], file_id, user_id, org_id, logger, feedback: str = "") -> str:
    """
    Prepares the prompts and invokes the OpenAI LLM API.
    """
    retrieved_chunks_text = "\n\n".join(retrieved_chunks)
    question = list(query.keys())[0]
    instructions = list(query.values())[0]

    user_prompt = f"""Answer the query based on the instructions.
        Question: {question} 
        {instructions}
        Here is relevant information:
        {retrieved_chunks_text}. 
        Output should be a minimal one line json, DO NOT provide any extra words. {feedback}.
        ###Outpput Format:
        Strictly do not include "```" or "```json" or any markers in your response."""

    system_prompt = """You are an assistant that extracts precise and relevant information from contracts and agreements, providing minimal and accurate answers in a clear format."""

    logger.debug(_log_message(f"Calling llm_call with Query: {query}", "llm_call", MODULE_NAME))

    return open_ai_llm_call(system_prompt, user_prompt, "gpt-4o", 0, "llm_call", logger)

@mlflow.trace(name="Recurring Payment - LLM Wrapper Function")
def call_llm(query: str, retrieved_chunks: List[str], file_id, user_id, org_id, logger) -> str:
    """
    Wrapper function to handle retries and error scenarios for the LLM API call.
    """
    logger.info(_log_message("Starting call_llm...", "call_llm", MODULE_NAME))
    retries = 2
    backoff_factor = 2

    for attempt in range(retries):
        try:
            llm_response = llm_call(query, retrieved_chunks, file_id, user_id, org_id, logger)
            logger.debug(_log_message(f"Received LLM Response (Attempt {attempt+1}): {llm_response}", "call_llm", MODULE_NAME))
            parser = LLMOutputParser(logger)
            json_response = parser.parse(llm_response)
            logger.info(_log_message(f"Parsed JSON Response: {json_response}", "call_llm", MODULE_NAME))
            return json_response

        except Exception as e:
            logger.warning(
                _log_message(f"Attempt {attempt+1} failed with error: {e}. Retrying...", "call_llm", MODULE_NAME)
            )
            time.sleep(backoff_factor ** attempt)

    logger.error(_log_message("All retries failed. Exiting call_llm.", "call_llm", MODULE_NAME))
    return None
