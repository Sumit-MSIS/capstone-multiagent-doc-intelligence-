import json
import time
import asyncio
import mlflow
from opentelemetry import context as ot_context
from typing import List, Dict, Any
from src.config.base_config import config
from src.intel.services.intel_chat.utils import get_chat_session, set_chat_session
from src.common.logger import request_logger, _log_message, flush_all_logs
from src.common.llm_status_handler.status_handler import set_ai_answer, set_websocket_stream_db
from src.intel.services.intel_chat.agent.agent_runner import get_answer_from_agent
from src.common.token_cost_tracker import (
    record_llm_usage, get_total_cost, calculate_model_cost
)
# === Configuration ===
TRACKING_URI = config.MLFLOW_TRACKING_URI
ENV = config.MLFLOW_ENV
MODULE_NAME = "lambda_function.py"

# === MLFlow Setup ===
mlflow.set_tracking_uri(TRACKING_URI)
# mlflow.config.enable_async_logging()
mlflow.openai.autolog()


# === OTEL Helper for Executors ===
def map_with_context(executor, fn, iterable):
    """
    Preserves OpenTelemetry context for parallel execution.
    """
    parent_ctx = ot_context.get_current()

    def wrapped(item):
        token = ot_context.attach(parent_ctx)
        try:
            return fn(item)
        finally:
            ot_context.detach(token)

    return list(executor.map(wrapped, iterable))


def submit_with_context(executor, fn, *args, **kwargs):
    """
    Submits a function with OpenTelemetry context in concurrent execution.
    """
    parent_ctx = ot_context.get_current()

    def wrapped():
        token = ot_context.attach(parent_ctx)
        try:
            return fn(*args, **kwargs)
        finally:
            ot_context.detach(token)

    return executor.submit(wrapped)


# === Lambda Handler ===
async def generate_answer(event, logger) -> Dict[str, Any]:
    """
    Lambda handler for Intel Chat requests. Supports action: GetAnswer.
    Logs and tracks MLFlow metrics and handles routing and session updates.
    """
    mlflow.set_experiment(f"Intel Chat - {ENV}")
    with mlflow.start_span(name="Intel Chat") as span:
        # === Span Inputs ===
        span.set_inputs({
            "file_id": event.get("file_ids"),
            "question": event.get("question"),
        })

        # === Span Tags ===
        span.set_attributes({
            "org_id": event.get("org_id"),
            "user_id": event.get("user_id"),
            "chat_id": event.get("chat_id"),
            "session_id": event.get("session_id"),
            "parent_session_id": event.get("parent_session_id"),
            "enable_agent": event.get('enable_agent', False),
            "request_id": event.get('request_id'),
            "connection_id": event.get("connection_id"),
            "client_id": event.get("client_id"),
            "tag_ids": event.get("tag_ids", [])
        })

        tags = {
            "org_id": event.get("org_id"),
            "user_id": event.get("user_id"),
            "machine": ENV,
            "cost": False,
            "chat_id": event.get("chat_id"),
            "agent": event.get('enable_agent', False)
        }

        if event.get("org_id") == 802 and event.get("user_id") == 671:
            tags["test_set"] = 10

        mlflow.update_current_trace(tags=tags)

        # === Extract Inputs ===
        session_id = event.get('session_id')
        parent_session_id = event.get('parent_session_id')
        chat_id = event.get('chat_id')
        user_id = event.get('user_id')
        org_id = event.get('org_id')
        file_ids = event.get('file_ids')
        question = event.get('question')
        action = event.get('action')
        connection_id = event.get('connection_id')
        request_id = event.get('request_id')
        enable_agent = event.get('enable_agent', False)
        client_id = event.get('client_id')
        tag_ids = event.get('tag_ids', [])

        if not all([user_id, org_id, question, chat_id, session_id, client_id, tag_ids]):
            error_msg = "Missing required parameters!"
            span.set_outputs({"success": False, "error": error_msg})
            return error_response(error_msg, file_ids, user_id, org_id, session_id, client_id, parent_session_id, chat_id, question, enable_agent, connection_id, request_id,logger)

        if action not in ['GetAnswer', 'test-request']:
            error_msg = "Invalid action specified! Supported action: GetAnswer"
            span.set_outputs({"success": False, "error": error_msg})
            return error_response(error_msg, file_ids, user_id, org_id, session_id, client_id, parent_session_id, chat_id, question, enable_agent, connection_id, request_id,logger)

        start_time = time.time()
        logger.info(_log_message("Processing GetAnswer", "lambda_handler", MODULE_NAME))

        try:
            user_id = int(user_id)
            org_id = int(org_id)
            chat_id = int(chat_id)
            session_id = str(session_id)
            question = str(question).strip()


            result = await process_files(user_id, org_id, question, file_ids, session_id, client_id, parent_session_id, chat_id, connection_id, request_id, enable_agent, tag_ids, logger)

            if isinstance(result, str):
                result = json.loads(result)

            if not result.get("success"):
                span.set_outputs(result)
                return error_response(result.get("error"), file_ids, user_id, org_id, session_id, client_id, parent_session_id, chat_id, question, enable_agent, connection_id, request_id,logger)

            # set_ai_answer(user_id, org_id, session_id, parent_session_id, chat_id, result.get("data", {}), "", enable_agent, client_id, logger)

            set_websocket_stream_db(connection_id, request_id, result.get("data", {}), chat_id, client_id, enable_agent)

            elapsed_time = round(time.time() - start_time, 2)
            logger.info(_log_message(f"Total time taken: {elapsed_time} seconds", "lambda_handler", MODULE_NAME))
            logger.info(_log_message(f"Successfully generated response: {result}", "lambda_handler", MODULE_NAME))

            span.set_outputs(result)

            cost_summary = get_total_cost(chat_id, clear_after=True)

            span.set_attributes({
                'session_id': session_id,
                'parent_session_id': parent_session_id,
                "file_ids": file_ids,
                "chat_id": chat_id,
                "user_query": question,  # Assuming client_id is not used here
                "user_id": user_id,
                "org_id": org_id,
                "enable_agent": enable_agent,
                "cost_summary": cost_summary
            })
            
            return result

        except (ValueError, json.JSONDecodeError) as ve:
            return error_response(f"Validation Error: {ve}", file_ids, user_id, org_id, session_id, client_id, parent_session_id, chat_id, question, enable_agent, connection_id, request_id,logger)
        except Exception as e:
            return error_response(f"Processing Error: {e}", file_ids, user_id, org_id, session_id, client_id, parent_session_id, chat_id, question, enable_agent, connection_id, request_id,logger)


# === Error Response Builder ===
@mlflow.trace(name="Error Response")
def error_response(
    error_message: str,
    file_ids: List[str],
    user_id: int,
    org_id: int,
    session_id: str,
    client_id: str, 
    parent_session_id: str,
    chat_id: int,
    question: str,
    enable_agent: bool,
    connection_id: str,
    request_id: str,
    logger
) -> Dict[str, Any]:
    """
    Returns a standard error response with logging and fallback answer.
    """
    logger.error(_log_message(error_message, "lambda_handler", MODULE_NAME))
    response = {
        "success": False,
        "error": error_message,
        "data": {
            "answers": [{
                "answer": "Unable to process your request. Please try again later.",
                "page_no": 1,
                "paragraph_no": 1,
                "score": 1
            }],
            "related_questions": []
        }
    }
    set_ai_answer(user_id, org_id, session_id, parent_session_id, chat_id, response["data"], error_message, enable_agent, client_id, logger)
    set_websocket_stream_db(connection_id, request_id, response["data"], chat_id, client_id, enable_agent)
    
    logger.info(_log_message("########################### INTEL CHAT PROCESS END ###########################", "generate_insights", MODULE_NAME))
    return response


# === File Processor ===
@mlflow.trace(name="Process Files")
async def process_files(
    user_id: int,
    org_id: int,
    question: str,
    file_ids: List[str],
    session_id: str,
    client_id: str, 
    parent_session_id: str,
    chat_id: int,
    connection_id: str,
    request_id: str,
    enable_agent: bool,
    tag_ids: List[str],
    logger
) -> Dict[str, Any]:
    """
    Main file-based routing processor with session context and history update.
    """
    try:

        if enable_agent:
            try:
                logger.info(_log_message("Starting agent response generation", "process_files", MODULE_NAME))
                session_data = await get_chat_session(session_id, logger)
                logger.info(_log_message(f"Session data retrieved: {session_data}", "process_files", MODULE_NAME))

                agent_current_history = list(session_data.get("llm_history", []))  # create a copy

                result = await get_answer_from_agent(question, org_id, user_id, file_ids, session_id, client_id, chat_id, True, connection_id, request_id, agent_current_history.copy(), tag_ids, logger)
                logger.info(_log_message(f"Agent response: {result}", "process_files", MODULE_NAME))
                if isinstance(result, str):
                    result = json.loads(result)

                if not result.get("success"):
                    return error_response(result.get("error"), file_ids, user_id, org_id, session_id, client_id, parent_session_id, chat_id, question, enable_agent, connection_id, request_id, logger)
                logger.info(_log_message(f"Successfully generated response - {result}", "lambda_handler", MODULE_NAME))
                
                # current_history.append({"role": "user", "content": question})
                agent_current_history.append({"role": "user", "content": question})
                agent_response = result["data"]["answers"][0]["answer"]
                agent_current_history.append({"role": "assistant", "content": agent_response})

                await set_chat_session(session_id, user_id, org_id, {"llm_history": agent_current_history}, logger)

                logger.info(_log_message("##################################################### INTEL CHAT PROCESS END #####################################################", "generate_insights", MODULE_NAME))
                return result
            
            except Exception as e:
                logger.error(_log_message(f"Error occurred: {e}", "lambda_handler", MODULE_NAME))
                return error_response(f"Processing Error: {e}", file_ids, user_id, org_id, session_id, client_id,  parent_session_id, chat_id, question, enable_agent, connection_id, request_id,logger)

    except Exception as e:
        logger.error(_log_message(f"Error in process_files: {e}", "process_files", MODULE_NAME))
        return {
            "success": False,
            "error": f"Error in process_files: {e}",
            "data": {
                "answers": [{
                    "answer": "Unable to process your request. Please try again later.",
                    "page_no": 1,
                    "paragraph_no": 1,
                    "score": 1
                }],
                "related_questions": []
            }
        }
