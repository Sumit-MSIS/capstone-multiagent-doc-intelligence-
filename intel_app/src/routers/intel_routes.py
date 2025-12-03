import re
import mlflow
import uuid
import asyncio
import contextvars
from datetime import datetime
from mlflow import MlflowClient
from src.config.base_config import config
from src.common.logger import request_logger, _log_message, flush_all_logs
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from src.intel.models.intel_chat_schemas import IntelChatRequestStream, IntelChatResponseStream
from src.intel.services.intel_chat.get_answer import generate_answer

intel_router = APIRouter()
MODULE_NAME = "intel_routes.py"


def sanitize_for_log_stream(text: str, max_length: int = 80) -> str:
    """
    Sanitize any input for safe CloudWatch log stream names.
    - Keeps only allowed chars: A-Z a-z 0-9 . _ / # + -
    - Replaces everything else with '_'
    - Truncates to max_length to avoid errors
    """
    sanitized = re.sub(r"[^A-Za-z0-9\._/#\+\-]", "_", text)
    return sanitized[:max_length]

@intel_router.post("/get-answer-stream", response_model=IntelChatResponseStream, summary="Stream Intel Chat Response")
async def intel_chat_lambda_handler_stream(intel_chat_request: IntelChatRequestStream):
    """
    Initiates a streaming task for Intel Chat and responds immediately.
    
    Args:
        intel_chat_request (IntelChatRequestStream): Request containing chat details.
    
    Returns:
        IntelChatResponseStream: Immediate response indicating streaming task initiation.
    
    Raises:
        HTTPException: If the operation fails.
    """
    mlflow.set_experiment(f"Intel Chat - {config.MLFLOW_ENV}")
    question_snippet = sanitize_for_log_stream(intel_chat_request.question[:30])
    logger = request_logger(
        f"session_id-{intel_chat_request.session_id} | chat_id-{intel_chat_request.chat_id} | question-{question_snippet}",
        str(config.INTELCHAT_LOG_DIR_NAME),
        "INTEL_CHAT"
    )
    try:
        logger.start(_log_message(
            "##################################################### INTEL CHAT PROCESS START #####################################################",
            "intel_chat_lambda_handler_stream",
            MODULE_NAME
        ))
        event = {
            "action": intel_chat_request.action,
            "client_id": intel_chat_request.client_id,
            "session_id": intel_chat_request.session_id,
            "parent_session_id": intel_chat_request.parent_session_id,
            "chat_id": intel_chat_request.chat_id,
            "user_id": intel_chat_request.user_id,
            "org_id": intel_chat_request.org_id,
            "question": intel_chat_request.question,
            "file_ids": intel_chat_request.file_ids,
            "connection_id": intel_chat_request.connection_id,
            "request_id": intel_chat_request.request_id,
            "enable_agent": intel_chat_request.enable_agent,
            "tag_ids": [str(t) for t in intel_chat_request.tag_ids] if intel_chat_request.tag_ids else []
        }
        logger.info(_log_message(f"Event received - Stream Mock: {event}", "intel_chat_lambda_handler_stream", MODULE_NAME))
        is_test_required = intel_chat_request.action
        if is_test_required == "test-request":
            response = await run_streaming_task(event, logger)
            # return response
            return IntelChatResponseStream(
                    success=True,
                    error="",
                    data=str(response)
                )
        else:
            asyncio.create_task(run_streaming_task(event, logger))
            return IntelChatResponseStream(
                success=True,
                error="",
                data="Request received and started processing streaming..."
            )
    except Exception as e:
        logger.error(_log_message(f"An error occurred: {str(e)}", "intel_chat_lambda_handler_stream", MODULE_NAME))
        flush_all_logs(
            f"session_id-{intel_chat_request.session_id} | chat_id-{intel_chat_request.chat_id} | question-{question_snippet}",
            str(config.INTELCHAT_LOG_DIR_NAME),
            "INTEL_CHAT"
        )
        return IntelChatResponseStream(
            success=False,
            error=f"An error occurred: {str(e)}",
            data=""
        )

async def run_streaming_task(event: dict, logger):
    """
    Runs the streaming task for Intel Chat in the background.
    
    Args:
        event (dict): Event data for the streaming task.
        logger (logging.Logger): Logger instance for tracking the process.
    """
    try:
        response = await generate_answer(event, logger)
        return response
    except Exception as e:
        logger.error(_log_message(f"An error occurred: {str(e)}", "run_streaming_task", MODULE_NAME))