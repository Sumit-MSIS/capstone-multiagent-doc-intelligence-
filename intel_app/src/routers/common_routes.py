import mlflow
import uuid
import asyncio
import contextvars
from datetime import datetime
from mlflow import MlflowClient
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from src.config.base_config import config
from src.common.logger import request_logger, _log_message, flush_all_logs
from src.common.models.insights_schemas import InsightRequest, InsightResponse
from src.common.file_insights.file_handler.file_downloader import fetch_file_async, _cleanup_temp_file_async
from src.common.file_insights.insights_generator import generate_insights
from src.common.llm_status_handler.status_handler import set_llm_file_status
from src.common.models.page_counter_schemas import GetPageCountRequest
from src.common.page_counter.page_counter import get_page_count
from src.common.models.conceptual_search_schemas import ConceptualSearchFileRequest, ConceptualSearchFileResponse
from src.common.conceptual_search.filter_contracts import filter_relevant_file_ids
from src.common.models.delete_file_schemas import DeleteFileRequest, DeleteFileResponse
from src.common.delete_files.data_remover import file_remover
from src.common.models.toc_schemas import GetTableOfContentsRequest, GetTableOfContentsResponse
from src.common.file_insights.toc_extractor.get_table_of_contents import extract_toc
from src.common.models.recurring_payment_schemas import RecurringPaymentDueRequest, RecurringPaymentDueResponse
from src.common.file_insights.recurring_payment.is_recurrsive import due_date_updater
from src.common.models.access_tag_schemas import UpdateTagsRequest, UpdateTagsResponse, FileTagUpdateResult
from src.common.access_control.file_tag_manager import update_file_tags

# Initialize FastAPI router
common_router = APIRouter()

# MLflow configuration
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
cost_tags = {"cost": "False"} # Cost tracking tags for MLflow
client = MlflowClient(config.MLFLOW_TRACKING_URI)

MODULE_NAME = "intel_routes.py"

async def process_insights_task(insight_request: InsightRequest, temp_file_path: str, logger):
    """
    Processes insights generation in the background, handling medical and non-medical organizations.
    
    Args:
        insight_request (InsightRequest): Request containing document details.
        temp_file_path (str): Path to the temporary file.
        logger (logging.Logger): Logger instance for tracking the process.
    
    Raises:
        Exception: If insight generation fails.
    """
    file_id = insight_request.file_id
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        request_data = insight_request.dict()
        request_data["url"] = temp_file_path
        request_data["file_type"] = insight_request.file_type.value
        request_data["tag_ids"] = [str(t) for t in insight_request.tag_ids] if insight_request.tag_ids else []

        mlflow.set_experiment(f"Generate Insights - {config.MLFLOW_ENV}")
        with mlflow.start_span(name="Generate Insights") as span:
            span.set_inputs({
                "file_id": request_data["file_id"],
                "file_name": request_data["file_name"],
                "file_type": request_data["file_type"],
                "tag_ids": request_data["tag_ids"]
            })
            span.set_attributes({
                "org_id": request_data["org_id"],
                "user_id": request_data["user_id"],
                "start_time": start_datetime
            })
            mlflow.update_current_trace({
                "org_id": request_data["org_id"],
                "user_id": request_data["user_id"],
                "tag_ids": request_data["tag_ids"],
                "machine": config.MLFLOW_ENV,
                "cost": "False"
            })

            set_llm_file_status(
                ci_file_guid=file_id,
                user_id=insight_request.user_id,
                org_id=insight_request.org_id,
                process_type=1,
                retryable_file=True,
                retry_no=insight_request.retry_no,
                completed_steps=1,
                error_msg="",
                process_start_time=start_datetime,
                process_end_time=None,
                immediate_retry_required=True,
                isMock=False,
                in_queue=True,
                execution_progress_percent=0,
                logger=logger
            )

            ctx = contextvars.copy_context()
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                ctx.run,
                generate_insights,
                request_data["file_id"],
                request_data["file_name"],
                request_data["file_type"],
                request_data["user_id"],
                request_data["org_id"],
                request_data["url"],
                request_data.get("retry_no", 0),
                request_data.get("retry_process_id", []),
                request_data["target_metadata_fields"],
                request_data["tag_ids"],
                logger
            )

            span.set_outputs(response)
            logger.info(_log_message(
                f"Insight generation completed for file_id: {file_id}",
                "process_insights_task",
                MODULE_NAME
            ))

            return response

    except Exception as e:
        set_llm_file_status(
            ci_file_guid=file_id,
            user_id=insight_request.user_id,
            org_id=insight_request.org_id,
            process_type=1,
            retryable_file=True,
            retry_no=insight_request.retry_no,
            completed_steps=2,
            error_msg=f"Unexpected error for file_id: {file_id}: {str(e)}",
            process_start_time=start_datetime,
            process_end_time=None,
            immediate_retry_required=True,
            isMock=False,
            in_queue=False,
            execution_progress_percent=0,
            logger=logger
        )
        logger.error(_log_message(
            f"Unexpected error for file_id: {file_id}: {str(e)}",
            "process_insights_task",
            MODULE_NAME
        ))
        raise

    finally:
        if temp_file_path:
            try:
                logger.info(_log_message(
                    f"Cleaning up temporary file: {temp_file_path} for file_id: {file_id}",
                    "process_insights_task",
                    MODULE_NAME
                ))
                await _cleanup_temp_file_async(temp_file_path)
                flush_all_logs(file_id, config.INSIGHTS_LOG_DIR_NAME, "CONTRACT")
            except Exception as e:
                logger.error(_log_message(
                    f"Failed to clean up temporary file for file_id: {file_id}: {str(e)}",
                    "process_insights_task",
                    MODULE_NAME
                ))


@common_router.post(
    "/get-insights",
    response_model=InsightResponse,
    summary="Generate Insights from Document",
    description="Queues an async task to process a document through file parsing, embedding generation, metadata extraction, and summarization."
)
async def get_insights(insight_request: InsightRequest):
    """
    Queues an asynchronous task to generate insights from a document.
    
    Args:
        insight_request (InsightRequest): Request containing document details.
        background_tasks (BackgroundTasks): FastAPI background tasks for async processing.
    
    Returns:
        InsightResponse: Response indicating whether the request was queued successfully.
    
    Raises:
        HTTPException: If required fields are missing or processing fails.
    """
    file_id = insight_request.file_id
    logger = request_logger(file_id, config.INSIGHTS_LOG_DIR_NAME, "CONTRACT")
    temp_file_path = None

    try:
        logger.info(_log_message(
            f"Received request for file_id: {insight_request.dict()}",
            "get_insights",
            MODULE_NAME
        ))

        if not all([insight_request.file_id, insight_request.file_name, insight_request.file_type, insight_request.url, insight_request.tag_ids]):
            logger.error(_log_message(
                f"Validation failed: Missing required fields for file_id: {file_id}",
                "get_insights",
                MODULE_NAME
            ))
            raise HTTPException(status_code=400, detail="Missing required fields: file_id, file_name, file_type, or url")

        if not insight_request.target_metadata_fields:
            insight_request.target_metadata_fields = ["*"]

        logger.info(_log_message(
            f"Downloading file from URL: {insight_request.url} for file_id: {file_id}",
            "get_insights",
            MODULE_NAME
        ))
        temp_file_path = await fetch_file_async(insight_request.url, insight_request.file_type.value, logger=logger)
        logger.info(_log_message(
            f"File downloaded to: {temp_file_path} for file_id: {file_id}",
            "get_insights",
            MODULE_NAME
        ))

        response = await process_insights_task(insight_request, temp_file_path, logger)

        return response

    except asyncio.CancelledError as e:
        logger.error(_log_message(
            f"Client disconnected! Task cancelled for file_id: {file_id}: {str(e)}",
            "get_insights",
            MODULE_NAME
        ))
        flush_all_logs(file_id, config.INSIGHTS_LOG_DIR_NAME, "CONTRACT")
        return InsightResponse(
            success=False,
            error=f"Client disconnected! Task was cancelled - {str(e)}",
            data=""
        )

    except Exception as e:
        logger.error(_log_message(
            f"Unexpected error for file_id: {file_id}: {str(e)}",
            "get_insights",
            MODULE_NAME
        ))
        flush_all_logs(file_id, config.INSIGHTS_LOG_DIR_NAME, "CONTRACT")
        return InsightResponse(
            success=False,
            error=f"Failed to queue request: {str(e)}",
            data=""
        )
    

@common_router.post("/get-page-count", summary="Get Page Count of Document")
async def get_pages_count(page_count_request: GetPageCountRequest):
    """
    Retrieves the page count of a document from the specified URL.
    
    Args:
        page_count_request (GetPageCountRequest): Request containing URL and file type.
    
    Returns:
        dict: Page count response.
    
    Raises:
        HTTPException: If the page count retrieval fails.
    """
    try:
        response = await get_page_count(page_count_request.url, page_count_request.file_type)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@common_router.post("/search-contracts", response_model=ConceptualSearchFileResponse, summary="Conceptual File Search")
async def search_contracts(request: ConceptualSearchFileRequest):
    """
    Filters and searches contracts based on provided search text.
    
    Args:
        request (ConceptualSearchFileRequest): Request containing search parameters.
    
    Returns:
        ConceptualSearchFileResponse: Search results.
    
    Raises:
        HTTPException: If the operation fails.
    """
    mlflow.set_experiment(f"Search Contracts - {config.MLFLOW_ENV}")
    try:
        with mlflow.start_span("Search Contracts") as span:
            event = {
                "user_id": request.user_id,
                "org_id": request.org_id,
                "search_text": request.search_text,
                "search_id": request.search_id,
                "tag_ids": [str(t) for t in request.tag_ids] if request.tag_ids else []
            }
            span.set_inputs({"search_text": event['search_text']})
            span.set_attributes({"org_id": event['org_id'], "user_id": event['user_id']})
            client.set_trace_tag(span.request_id, "org_id", event['org_id'])
            client.set_trace_tag(span.request_id, "user_id", event['user_id'])
            client.set_trace_tag(span.request_id, "machine", config.MLFLOW_ENV)
            mlflow.update_current_trace(tags=cost_tags)
            response = filter_relevant_file_ids(event, only_contract=True)
            span.set_outputs(response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@common_router.post("/delete-files", response_model=DeleteFileResponse, summary="Delete Files")
async def delete_files(request: DeleteFileRequest):
    """
    Deletes specified files from the database and Pinecone.
    
    Args:
        request (DeleteFileRequest): Request containing file deletion details.
    
    Returns:
        DeleteFileResponse: Deletion result.
    
    Raises:
        HTTPException: If the operation fails.
    """
    try:
        event = {
            "user_id": request.user_id,
            "org_id": request.org_id,
            "file_ids": request.file_ids
        }
        return file_remover(event)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@common_router.post("/get-table-of-contents", response_model=GetTableOfContentsResponse, summary="Fetch Table of Contents")
async def get_toc(request: GetTableOfContentsRequest):
    """
    Fetches the table of contents for specified files from the database.
    
    Args:
        request (GetTableOfContentsRequest): Request containing user and file details.
    
    Returns:
        GetTableOfContentsResponse: Table of contents data.
    
    Raises:
        HTTPException: If the operation fails.
    """
    try:
        event = {
            "user_id": request.user_id,
            "org_id": request.org_id,
            "file_ids": request.file_ids
        }
        return extract_toc(event)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@common_router.post("/recurring_payment", response_model=RecurringPaymentDueResponse, summary="Classify and Update Recurring Payment")
async def recurring_payment(recurring_payment_request: RecurringPaymentDueRequest):
    """
    Classifies and updates recurring payment details for a contract.
    
    Args:
        recurring_payment_request (RecurringPaymentDueRequest): Request containing payment details.
    
    Returns:
        RecurringPaymentDueResponse: Payment classification result.
    
    Raises:
        HTTPException: If the operation fails.
    """
    mlflow.set_experiment(f"Recurring Payment - {config.MLFLOW_ENV}")
    try:
        with mlflow.start_span("Recurring Payment") as span:
            span.set_inputs({"file_id": recurring_payment_request.file_id})
            span.set_attributes({
                "org_id": recurring_payment_request.org_id,
                "user_id": recurring_payment_request.user_id
            })
            client.set_trace_tag(span.request_id, "org_id", recurring_payment_request.org_id)
            client.set_trace_tag(span.request_id, "user_id", recurring_payment_request.user_id)
            client.set_trace_tag(span.request_id, "machine", config.MLFLOW_ENV)
            mlflow.update_current_trace(tags=cost_tags)
            response = due_date_updater(
                recurring_payment_request.file_id,
                recurring_payment_request.org_id,
                recurring_payment_request.user_id
            )
            span.set_outputs(response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@common_router.post("/update-access-tags", response_model=UpdateTagsResponse)
async def update_tags_route(request: UpdateTagsRequest):
    try:
        unique_id = str(uuid.uuid4())
        logger = request_logger(f"{unique_id}", config.ACCESS_TAG_LOG_DIR_NAME, "UPDATE_ACCESS_TAG")
        logger.info(_log_message("################################################# UPDATE TAGS START ############################################################", "update_tags_route", MODULE_NAME))
        logger.debug(_log_message(f"Request Paylload: {request.dict}", "update_tags_route", MODULE_NAME))
        results = await update_file_tags(request.org_id, [f.dict() for f in request.file_tag_list], logger)
        logger.info(_log_message(f"Tags Updated Successfully! - {results}", "update_tags_route", MODULE_NAME))
        return UpdateTagsResponse(success=True, data=[FileTagUpdateResult(**r) for r in results])
    except Exception as e:
        logger.error(_log_message(f"Error in update_tags_route: {str(e)}", "update_tags_route", MODULE_NAME))
        return UpdateTagsResponse(success=False, error=f"Error in update_tags_route: {str(e)}", data=None)
    finally:
        flush_all_logs(f"{unique_id}", config.ACCESS_TAG_LOG_DIR_NAME, "UPDATE_ACCESS_TAG")