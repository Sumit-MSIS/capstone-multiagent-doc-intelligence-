from src.config.base_config import config
from openai import OpenAI
import time
from src.common.logger import _log_message
import mlflow
# mlflow.config.enable_async_logging()
from src.common.llm.factory import get_llm_client
# mlflow.langchain.autolog()
mlflow.openai.autolog()
from opentelemetry import context as ot_context


#  Load OpenAI API credentials
OPENAI_API_KEY = config.OPENAI_API_KEY
OPENAI_MODEL_NAME = config.OPENAI_MODEL_NAME

client = get_llm_client(async_mode=False)

MODULE_NAME = "llm_call.py"

# OpenAI model pricing details (in USD per million tokens)
PRICING = {
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.150, "cached_input": 0.075, "output": 0.600}
}

@mlflow.trace(name="LLM Call")
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

@mlflow.trace(name="Compute Costs")
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