from openai import OpenAI
from src.config.base_config import config
from src.common.logger import request_logger, _log_message, flush_all_logs
import mlflow
# mlflow.config.enable_async_logging()
# mlflow.langchain.autolog()
mlflow.openai.autolog()
from opentelemetry import context as ot_context
from src.common.llm.factory import get_llm_client

client = get_llm_client(async_mode=False)

OPENAI_API_KEY = config.OPENAI_API_KEY
OPENAI_MODEL_NAME = config.OPENAI_MODEL_NAME
module_name = "llm_call.py"

@mlflow.trace(name="Generic LLM Call")
def open_ai_llm_call(
    file_id, 
    user_id: str, 
    org_id: str, 
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    temperature: float,
    function_name: str,
    logger
) -> str:
    
    try:
        logger.info(_log_message(f"Invoking open_ai_llm_call from function: {function_name}", 'open_ai_llm_call', module_name))
        logger.info(_log_message(f"Calling OpenAI LLM with system prompt: {system_prompt} and user prompt: {user_prompt}", 'open_ai_llm_call', module_name))        
        logger.info(_log_message(f"OpenAI Model Name: {model_name} | Temperature: {temperature}", 'open_ai_llm_call', module_name))
        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        llm_answer = response.choices[0].message.content

        logger.info(_log_message(f"OpenAI LLM response: {llm_answer}", 'open_ai_llm_call', module_name))

        #usage


        """Calculate the cost of an OpenAI API call based on token usage and model pricing."""
        pricing = {
            "gpt-4o": {
                "input": 2.50,
                "cached_input": 1.25,
                "output": 10.00
            },
            "gpt-4o-mini": {
                "input": 0.150,
                "cached_input": 0.075,
                "output": 0.600
            }
        }

        # Extract token usage
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        cached_tokens = response.usage.prompt_tokens_details.cached_tokens

        # Calculate uncached tokens
        uncached_tokens = prompt_tokens - cached_tokens

        # Get pricing for the model
        model_pricing = pricing.get(model_name)
        if not model_pricing:
            raise ValueError(f"Pricing not available for model: {model_name}")

        # Calculate costs
        cost_cached_input = (cached_tokens / 1_000_000) * model_pricing['cached_input']
        cost_uncached_input = (uncached_tokens / 1_000_000) * model_pricing['input']
        cost_output = (completion_tokens / 1_000_000) * model_pricing['output']

        total_cost = cost_cached_input + cost_uncached_input + cost_output

        logger.info(_log_message(f"################################### Token Usage start  ###################################", 'open_ai_llm_call', module_name))
        logger.info(_log_message(f"OpenAI API Cost Breakdown:", 'open_ai_llm_call', module_name))
        logger.info(_log_message(f"Token Usage: {response.usage}", 'open_ai_llm_call', module_name))
        logger.info(_log_message(f"Cached Input Tokens: {cached_tokens} | Cost: ${cost_cached_input:.6f}", 'open_ai_llm_call', module_name))
        logger.info(_log_message(f"Uncached Input Tokens: {uncached_tokens} | Cost: ${cost_uncached_input:.6f}", 'open_ai_llm_call', module_name))
        logger.info(_log_message(f"Output Tokens: {completion_tokens} | Cost: ${cost_output:.6f}", 'open_ai_llm_call', module_name))
        logger.info(_log_message(f"OpenAI API Cost: ${total_cost:.6f}", 'open_ai_llm_call', module_name))
        logger.info(_log_message(f"################################### Token Usage End  ###################################", 'open_ai_llm_call', module_name))

        return llm_answer
    
    except Exception as e:
        error_message = f"Error restructuring answer for normal Q&A: {e}"
        logger.error(_log_message(error_message, 'open_ai_llm_call', module_name))
        return None
    
# def log_token_usgae(
#     file_id,
#     user_id: str, 
#     org_id: str, 
#     token_usage: dict, 
# ):
#     logger = request_logger(f"{file_id}-{org_id}-{user_id}", str(config.CONCEPTUAL_SEARCH_LOG_DIR_NAME), "CONCEPTUAL_SEARCH")
#     logger.info(_log_message(f"Token Usage: {token_usage}", "log_token_usage", module_name))
#     return



# {
#     "id": "chatcmpl-abc123",
#     "object": "chat.completion",
#     "created": 1677858242,
#     "model": "gpt-4o-mini",
#     "usage": {
#         "prompt_tokens": 13,
#         "completion_tokens": 7,
#         "total_tokens": 20,
#         "completion_tokens_details": {
#             "reasoning_tokens": 0,
#             "accepted_prediction_tokens": 0,
#             "rejected_prediction_tokens": 0
#         }
#     },
#     "choices": [
#         {
#             "message": {
#                 "role": "assistant",
#                 "content": "\n\nThis is a test!"
#             },
#             "logprobs": null,
#             "finish_reason": "stop",
#             "index": 0
#         }
#     ]
# }