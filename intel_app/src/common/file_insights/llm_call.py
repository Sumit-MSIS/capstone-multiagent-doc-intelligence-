import json
import time
from typing import List
from openai import OpenAI
from src.common.file_insights.metadata_extractor.json_parser import LLMOutputParser
from src.config.base_config import config
from src.common.logger import _log_message
import mlflow
from opentelemetry import context as ot_context
from src.common.file_insights.metadata_extractor.others.tcv_module.all_prompts import all_function_map, add, subtract, multiply, divide
from src.common.file_insights.metadata_extractor.others.tcv_module.all_prompts import convert_currency, calculate_amortization_monthly, calculate_area_based_payment, calculate_escalation, convert_quarterly_to_monthly, convert_yearly_to_monthly 
from src.common.llm.factory import get_llm_client

# mlflow.config.enable_async_logging()

from datetime import datetime

# mlflow.langchain.autolog()
mlflow.openai.autolog()

# Load OpenAI API credentials
OPENAI_API_KEY = config.OPENAI_API_KEY

# Initialize OpenAI client
client = get_llm_client(async_mode=False)

MODULE_NAME = "llm_call.py"

# OpenAI model pricing details (in USD per million tokens)
PRICING = {
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.150, "cached_input": 0.075, "output": 0.600},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00}
}

@mlflow.trace(name="LLM Call - Generate MetaData")
def open_ai_llm_call(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    temperature: float,
    json_schema = {},
    function_name = "open_ai_llm_call",
    logger=None
) -> str:
    """
    Calls OpenAI's LLM API with the provided prompts and logs relevant details.
    """
    try:
        model_name = "gpt-4o"
        start_time = time.perf_counter()
        logger.info(_log_message(f"Invoking OpenAI LLM API: {function_name}", function_name, MODULE_NAME))

        # Log API configuration details
        logger.debug(_log_message(f"Model: {model_name}, Temperature: {temperature}", function_name, MODULE_NAME))
        # logger.debug(_log_message(f"System Prompt: {system_prompt}", function_name, MODULE_NAME))
        # logger.debug(_log_message(f"User Prompt: {user_prompt}", function_name, MODULE_NAME))


        if json_schema:
            # Ensure the JSON schema is properly formatted
            # if not isinstance(json_schema, dict):
            #     raise ValueError("json_schema must be a dictionary.")
            # if "type" not in json_schema or json_schema["type"] != "json_schema":
            #     raise ValueError("json_schema must have 'type' set to 'json_schema'.")
            # if "json_schema" not in json_schema:
            #     raise ValueError("json_schema must contain 'json_schema' key with schema details.")

            response = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                top_p=1,
                response_format=json_schema
            )
        else:
            # Make the API call
            response = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                top_p=1
                # response_format # update it to return json object only
            )

        llm_answer = response.choices[0].message.content.strip()

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



@mlflow.trace(name="LLM Call with Tools")
def openai_llm_call_with_tools(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    temperature: float,
    tools: list = None,
    max_iterations: int = 20,
    json_schema = {},
    function_name = "open_ai_llm_call_with_tools",
    logger=None
):
    """
    Calls the OpenAI API with function calling tools and returns the final result.
    Handles multiple tool calls and iterations.
    """
    model_name = "gpt-4o"
    start_time = time.perf_counter()
    logger.info(_log_message(f"Invoking OpenAI LLM API: {function_name}", function_name, MODULE_NAME))

    # Log API configuration details
    logger.debug(_log_message(f"Model: {model_name}, Temperature: {temperature}", function_name, MODULE_NAME))
    # logger.debug(_log_message(f"System Prompt: {system_prompt}", function_name, MODULE_NAME))
    # logger.debug(_log_message(f"User Prompt: {user_prompt}", function_name, MODULE_NAME))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    iteration = 0
    calculations = []
    while iteration < max_iterations:
        try:
            # Make the API call
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=temperature,
                top_p=1,
                messages=messages,
                tools=tools if tools else [],
                response_format=json_schema if json_schema else None,
                tool_choice="auto" if tools else None
            )
            
            message = response.choices[0].message
            # return message
            
            # Check if there are tool calls
            if message.tool_calls:
                # Add assistant message to conversation
                messages.append(message)
                
                # Process each tool call
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    # Execute the function
                    """Execute a function call with given arguments"""
                    if function_name in all_function_map:
                        if function_name in ["add", "subtract", "multiply", "divide"]:
                            result =all_function_map[function_name](arguments["numbers"])
                        else:
                            result = all_function_map[function_name](arguments)
                    else:
                        raise ValueError(f"Unknown function: {function_name}")
                    
                    # result = execute_function_call(function_name, arguments)
                    calculations.append({
                        "function": function_name,
                        "arguments": arguments,
                        "result": result
                    })
                    
                    # Add function result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
            else:
                # No more tool calls, return the final response
                llm_response = message.content
                logger.info(_log_message(f"LLM Response: {llm_response}", function_name, MODULE_NAME))
                logger.info(_log_message(f"Calculations: {calculations}", function_name, MODULE_NAME))
                if hasattr(response, "usage"):
                    compute_costs(response, model_name, function_name, logger)
                
                duration = time.perf_counter() - start_time
                logger.info(_log_message(f"LLM call completed in {duration:.2f} seconds", function_name, MODULE_NAME))
                answer = {
                    "final_response": llm_response,
                    "calculations": calculations,
                    "total_iterations": iteration + 1
                }
                return answer["final_response"]

            iteration += 1
            
        except Exception as e:
            print(f"Error in function calling iteration {iteration}: {e}")
            return {
                "error": str(e),
                "calculations": calculations,
                "total_iterations": iteration
            }
    
    return {
        "error": "Maximum iterations reached",
        "calculations": calculations,
        "total_iterations": iteration
    }


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


@mlflow.trace(name="LLM Call - Prepare Prompts")
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


    logger.debug(_log_message(f"User Prompt: {user_prompt}", "llm_call", MODULE_NAME))
    # logger.debug(_log_message(f"Retrieved Chunks: {retrieved_chunks_text}", "llm_call", MODULE_NAME))
    system_prompt = """You are an assistant that extracts precise and relevant information from contracts and agreements, providing minimal and accurate answers in a clear format."""

    logger.debug(_log_message(f"Calling llm_call with Query: {query}", "llm_call", MODULE_NAME))

    return open_ai_llm_call(system_prompt, user_prompt, "gpt-4o", 0, "llm_call", logger)

@mlflow.trace(name="LLM Call - Dates")
def llm_call_for_dates(retrieved_chunks: str, date_extraction_prompt, file_id, user_id, org_id, logger, feedback: str = "") -> str:
    """
    Calls the LLM API specifically for date-related queries.
    """
    system_prompt = """You are an assistant that understands and extracts relevant dates from the legal agreement's and contract's context provided to you.
        You also provide the dates extractes in a specific format as per the instructions provided to you."""

    user_prompt = f"""
    {date_extraction_prompt}
    Below is the text paragraph consisting of all the dates filtered from the legal agreement's and contract's context provided to you.
    {retrieved_chunks}. 
    Output should be a minimal json, DO NOT provide any extra words. {feedback}"""

    retries = 2
    backoff_factor = 2
    logger.debug(_log_message(f"Calling llm_call_for_dates", "llm_call_for_dates", MODULE_NAME))
    llm_response = open_ai_llm_call(system_prompt, user_prompt, "gpt-4o", 0, "llm_call_for_dates", logger)
    for attempt in range(retries):
        try:
            logger.debug(_log_message(f"Received LLM Response (Attempt {attempt+1}): {llm_response}", "llm_call_for_dates", MODULE_NAME))
            parser = LLMOutputParser(logger)
            json_response = parser.parse(llm_response)
            logger.info(_log_message(f"Parsed JSON Response: {json_response}", "llm_call_for_dates", MODULE_NAME))
            return json_response

        except Exception as e:
            feedback = f"Previous error encountered: {e}."
            logger.warning(
                _log_message(f"Attempt {attempt+1} failed with error: {e}. Retrying...", "llm_call_for_dates", MODULE_NAME)
            )
            time.sleep(backoff_factor ** attempt)

    logger.error(_log_message("All retries failed. Exiting call_llm.", "call_llm", MODULE_NAME))
    return None


@mlflow.trace(name="LLM Call - Dynamic")
def dynamic_llm_call(user_prompt, system_prompt, json_schema, logger, feedback: str = "") -> str:
    """
    Calls the LLM API specifically for date-related queries.
    """
    retries = 2
    backoff_factor = 2
    logger.debug(_log_message(f"Calling generatal_llm_call", "generatal_llm_call", MODULE_NAME))

    current_date = time.strftime("%Y-%m-%d %H:%M:%S (%A)", time.localtime())
    system_prompt = f"{system_prompt}. Today's date is: {current_date} for your reference."
    llm_response = open_ai_llm_call(system_prompt, user_prompt + feedback, "gpt-4o", 0, json_schema, "generatal_llm_call", logger)
    for attempt in range(retries):
        try:
            logger.debug(_log_message(f"Received LLM Response (Attempt {attempt+1}): {llm_response}", "generatal_llm_call", MODULE_NAME))
            parser = LLMOutputParser(logger)
            json_response = parser.parse(llm_response)
            logger.info(_log_message(f"Parsed JSON Response: {json_response}", "generatal_llm_call", MODULE_NAME))
            return json_response

        except Exception as e:
            feedback = f"Previous error encountered: {e}."
            logger.warning(
                _log_message(f"Attempt {attempt+1} failed with error: {e}. Retrying...", "generatal_llm_call", MODULE_NAME)
            )
            time.sleep(backoff_factor ** attempt)

    logger.error(_log_message("All retries failed. Exiting generatal_llm_call.", "generatal_llm_call", MODULE_NAME))
    return None

@mlflow.trace(name="LLM Call - Jurisdiction")
def llm_call_for_jurisdiction(retrieved_chunks: str, jurisdiction_extraction_prompt, file_id, user_id, org_id, logger, feedback: str = "") -> str:
    """
    Calls the LLM API specifically for date-related queries.
    """
    system_prompt = """You are an assistant that understands and extracts jurisdiction from the legal contract context"""

    user_prompt = f"""
    {jurisdiction_extraction_prompt}
    Below is the text paragraph consisting of all the required context filtered from the legal agreement's, contract's context and is provided to you.
    {retrieved_chunks}. 
    Output should be a minimal json, DO NOT provide any extra words. {feedback}"""

    retries = 2
    backoff_factor = 2
    logger.debug(_log_message(f"Calling llm_call_for_dates", "llm_call_for_dates", MODULE_NAME))
    llm_response = open_ai_llm_call(system_prompt, user_prompt, "gpt-4o", 0, "llm_call_for_dates", logger)
    for attempt in range(retries):
        try:
            logger.debug(_log_message(f"Received LLM Response (Attempt {attempt+1}): {llm_response}", "llm_call_for_dates", MODULE_NAME))
            parser = LLMOutputParser(logger)
            json_response = parser.parse(llm_response)
            logger.info(_log_message(f"Parsed JSON Response: {json_response}", "llm_call_for_dates", MODULE_NAME))
            return json_response

        except Exception as e:
            feedback = f"Previous error encountered: {e}."
            logger.warning(
                _log_message(f"Attempt {attempt+1} failed with error: {e}. Retrying...", "llm_call_for_dates", MODULE_NAME)
            )
            time.sleep(backoff_factor ** attempt)

    logger.error(_log_message("All retries failed. Exiting call_llm.", "call_llm", MODULE_NAME))
    return None

@mlflow.trace(name="LLM Call - Contract Value")
def llm_call_for_cv(system_prompt: str , user_prompt: str, tools, json_schema, file_id, user_id, org_id, logger, feedback: str = "") -> str:
    """
    Calls the LLM API specifically for contract value extraction.
    """

    retries = 2
    backoff_factor = 2
    logger.debug(_log_message(f"Calling llm_call_for_cv", "llm_call_for_cv", MODULE_NAME))
    llm_response = openai_llm_call_with_tools(system_prompt, user_prompt, "gpt-4o", 0, tools, max_iterations=20, json_schema=json_schema, function_name= "llm_call_for_cv", logger=logger)
    for attempt in range(retries):
        try:
            logger.debug(_log_message(f"Received LLM Response (Attempt {attempt+1}): {llm_response}", "llm_call_for_dates", MODULE_NAME))
            parser = LLMOutputParser(logger)
            json_response = parser.parse(llm_response)
            logger.info(_log_message(f"Parsed JSON Response: {json_response}", "llm_call_for_dates", MODULE_NAME))
            return json_response

        except Exception as e:
            feedback = f"Previous error encountered: {e}."
            logger.warning(
                _log_message(f"Attempt {attempt+1} failed with error: {e}. Retrying...", "llm_call_for_dates", MODULE_NAME)
            )
            time.sleep(backoff_factor ** attempt)

    logger.error(_log_message("All retries failed. Exiting call_llm.", "call_llm", MODULE_NAME))
    return None


# @mlflow.trace(name="LLM Call - Contract Value")
# def llm_call_for_cv(retrieved_chunks: str, cv_extraction_prompt, file_id, user_id, org_id, logger, feedback: str = "") -> str:
#     """
#     Calls the LLM API specifically for date-related queries.
#     """
#     system_prompt = """You are an assistant that understands, calculates and extracts the contract value from the legal agreement's, contract's context provided to you."""

#     user_prompt = f"""
#     {cv_extraction_prompt}
#     Below is the text paragraph consisting of all the dates filtered from the legal agreement's and contract's context provided to you.
#     {retrieved_chunks}. 
#     Output should be a minimal json, DO NOT provide any extra words. {feedback}"""

#     retries = 2
#     backoff_factor = 2
#     logger.debug(_log_message(f"Calling llm_call_for_dates", "llm_call_for_dates", MODULE_NAME))
#     llm_response = open_ai_llm_call(system_prompt, user_prompt, "gpt-4o", 0, "llm_call_for_dates", logger)
#     for attempt in range(retries):
#         try:
#             logger.debug(_log_message(f"Received LLM Response (Attempt {attempt+1}): {llm_response}", "llm_call_for_dates", MODULE_NAME))
#             parser = LLMOutputParser(logger)
#             json_response = parser.parse(llm_response)
#             logger.info(_log_message(f"Parsed JSON Response: {json_response}", "llm_call_for_dates", MODULE_NAME))
#             return json_response

#         except Exception as e:
#             feedback = f"Previous error encountered: {e}."
#             logger.warning(
#                 _log_message(f"Attempt {attempt+1} failed with error: {e}. Retrying...", "llm_call_for_dates", MODULE_NAME)
#             )
#             time.sleep(backoff_factor ** attempt)

#     logger.error(_log_message("All retries failed. Exiting call_llm.", "call_llm", MODULE_NAME))
#     return None



@mlflow.trace(name="LLM Call - Wrapper Function")
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

@mlflow.trace(name="Validate Payment Due Date")
def payment_due_date_validatior(is_recursive, payment_due_date, expiry_date, current_date, file_id, user_id, org_id, logger):
    if expiry_date == "null":
        expiry_date = None
    if payment_due_date == "null":
        payment_due_date = None
    # if expiry_date:
    #     expiry_date = datetime.strftime(expiry_date, "%Y-%m-%d")
    # if payment_due_date:
    #     payment_due_date = datetime.strftime(payment_due_date, "%Y-%m-%d")
    if expiry_date is None or payment_due_date is None:
        logger.info(_log_message(f"Expiry date or payment due date is None for file: {file_id}. Cannot proceed.", "payment_due_date_validator", MODULE_NAME))
        is_recursive = "No"
        payment_due_date = "null"
        return is_recursive, payment_due_date
    
    if current_date > expiry_date:
        logger.info(_log_message(f"Contract has expired for file: {file_id}. No need to update.", "payment_due_date_validator", MODULE_NAME))
        is_recursive = "No"
        payment_due_date = "null"
        return is_recursive, payment_due_date

    elif current_date < payment_due_date:
        logger.info(_log_message(f"Payment Due Date is in the future for file: {file_id}. No need to update.", "payment_due_date_validator", MODULE_NAME))
        is_recursive = "Yes"
        return is_recursive, payment_due_date
    elif current_date < expiry_date and current_date > payment_due_date:
        is_recursive = "Yes"
        return is_recursive, payment_due_date
