import time
import orjson
import gc
import psutil
import mlflow
from typing import List
from src.config.base_config import config
from src.common.logger import request_logger, _log_message, flush_all_logs
from src.common.file_insights.file_parser import FileParser
from src.common.file_insights.doc_summariser.doc_summary_and_template import process_document
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from src.common.file_insights.metadata_extractor.extract_metadata_parallely import extract_metadata as get_metadata_parallely
from threading import Thread
from src.common.llm_status_handler.status_handler import set_llm_file_status
from datetime import datetime
import orjson
import mlflow
import json
from opentelemetry import context as ot_context
from src.common.file_insights.utils import store_extracted_text, retrieve_extracted_text
from src.common.file_insights.hybrid_vector_handler.vector_handler import process_and_upsert
from src.common.file_insights.hybrid_vector_handler.bm25_handler import calculate_tf_update_db
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.common.template_updator.update_avivo_templates import update_avivo_template

# mlflow.config.enable_async_logging()

TRACKING_URI = config.MLFLOW_TRACKING_URI
ENV = config.MLFLOW_ENV
MODULE_NAME = "insights_generator.py"
COST_TAGS = {"cost": "False"}
DOCUMENT_PARSING_ENVIRONMENT = config.DOCUMENT_PARSING_ENVIRONMENT
mlflow.set_tracking_uri(TRACKING_URI)

def log_time(stage_name, start_time, logger):
    duration = time.perf_counter() - start_time
    logger.info(_log_message(f"{stage_name} completed in {duration:.2f} seconds.", "generate_insights", MODULE_NAME))
    return duration

def submit_with_context(executor, fn, *args, **kwargs):
    """Helper method to submit functions with OpenTelemetry context preservation"""
    parent_ctx = ot_context.get_current()
    
    def wrapped():
        token = ot_context.attach(parent_ctx)
        try:
            return fn(*args, **kwargs)
        finally:
            ot_context.detach(token)
    
    return executor.submit(wrapped)

def process_1(file_id: str, file_name: str, file_type: str, user_id: int, org_id: int,
              url: str, retry_count: int, start_datetime: str, logger) -> dict:
    """Process 1: Extract text and return chunks"""
    try:
        logger.info(_log_message("Starting Process 1: Text Extraction", "process_1", MODULE_NAME))
        file_parser = FileParser(logger)
        
        def update_status(step_code):
            set_llm_file_status(file_id, user_id, org_id, 1, True, retry_count, 1, "", start_datetime, "", True, False, False, step_code, logger)

        update_status(20)

        is_paid = DOCUMENT_PARSING_ENVIRONMENT in ["QA", "UAT"]

        # Try retrieving from DB
        stage_start = time.perf_counter()
        retrieved_file_text, unstructured_elements = retrieve_extracted_text(file_id, logger)
        if retrieved_file_text and unstructured_elements:
            logger.info(_log_message(f"Retrieved existing extracted text for file_id: {file_id}", "process_1", MODULE_NAME))
            complete_file_text = retrieved_file_text
            complete_file_markdown_text = file_parser._convert_paid_unstructured_elements_to_markdown(unstructured_elements) if is_paid else file_parser._convert_elements_to_markdown(unstructured_elements)
            complete_file_toc = file_parser._extract_toc_from_paid_unstructured_elements(unstructured_elements) if is_paid else file_parser.extract_toc_from_elements(unstructured_elements)
            logger.info(_log_message(
                f"Using existing data: {len(complete_file_text)} chars | "
                f"{len(complete_file_markdown_text)} markdown chars | "
                f"{len(complete_file_toc)} TOC sections",
                "process_1", MODULE_NAME
            ))
        else:
            update_status(30)

            
            parse_method = (
                file_parser._unstructured_file_parser_paid if is_paid else file_parser.process_file
            )
            toc_method = (
                file_parser._extract_toc_from_paid_unstructured_elements if is_paid else file_parser.extract_toc_from_elements
            )
            text_method = (
                file_parser._extract_text_from_paid_unstructured_elements if is_paid else file_parser._extract_text_from_elements
            )
            markdown_method = (
                file_parser._convert_paid_unstructured_elements_to_markdown if is_paid else file_parser._convert_elements_to_markdown
            )

            # Extract unstructured elements
            try:
                unstructured_elements = parse_method(url, file_type) if is_paid else parse_method(url)
                logger.info(_log_message(
                    f"Type of elements extracted: {type(unstructured_elements)} elements from {file_name}",
                    "process_1", MODULE_NAME
                ))
                # unstructured_elements = json.dumps(element_dicts)
            except Exception as e:
                error_message = f"Error extracting unstructured elements: {e}"
                logger.error(_log_message(error_message, "process_1", MODULE_NAME))
                return {
                    "success": False, "error": error_message,
                    "chunks": [], "complete_file_markdown_text": "", "complete_file_toc": ""
                }
            update_status(40)
            logger.info(_log_message(
                f"Unstructured elements extracted: {file_name} | Type: {type(unstructured_elements)}",
                "process_1", MODULE_NAME
            ))

            # Concurrently extract text, markdown, TOC
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_text = submit_with_context(executor, text_method, unstructured_elements)
                future_markdown = submit_with_context(executor, markdown_method, unstructured_elements)
                future_toc = submit_with_context(executor, toc_method, unstructured_elements)

                complete_file_text = future_text.result()
                complete_file_markdown_text = future_markdown.result()
                complete_file_toc = future_toc.result()

            logger.info(_log_message(
                f"Text extracted: {len(complete_file_text)} chars | Markdown extracted: {len(complete_file_markdown_text)} chars | TOC extracted: {len(complete_file_toc)} sections",
                "process_1", MODULE_NAME
            ))

            if complete_file_text:
                store_extracted_text(file_id, file_name, file_type, complete_file_text, unstructured_elements, logger)

        update_status(60)
        log_time("Text and Markdown extraction", stage_start, logger)

        # Chunk the text
        chunks_start = time.perf_counter()
        chunks = file_parser.extract_and_chunk_text(
            complete_file_text, file_name, file_type, file_id, user_id, org_id, retry_count, start_datetime
        )

        if not chunks:
            msg = "No text content extracted; unable to proceed."
            logger.error(_log_message(msg, "process_1", MODULE_NAME))
            return {
                "success": False, "error": msg,
                "chunks": [], "complete_file_markdown_text": "", "complete_file_toc": ""
            }

        logger.info(_log_message(
            f"Text chunking completed in {time.perf_counter() - chunks_start:.2f} seconds.",
            "process_1", MODULE_NAME
        ))

        logger.info(_log_message("Process 1: Text Extraction completed successfully", "process_1", MODULE_NAME))
        return {
            "success": True,
            "error": "",
            "chunks": chunks,
            "complete_file_markdown_text": complete_file_markdown_text,
            "complete_file_toc": complete_file_toc
        }

    except Exception as e:
        error_message = f"Error in Process 1 - Text Extraction: {e}"
        logger.error(_log_message(error_message, "process_1", MODULE_NAME))
        return {
            "success": False, "error": error_message,
            "chunks": [], "complete_file_markdown_text": "", "complete_file_toc": ""
        }



@mlflow.trace(name="File Parser - Update Encoder and Upsert Vectors")
def update_encoder_and_upsert_vectors(chunks, file_name, file_type, file_id, user_id, org_id, retry_count, tag_ids, logger):
    function_name = "update_encoder_and_upsert_vectors"
    overall_start_time = time.perf_counter()
    process = psutil.Process()
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        status_start_time = time.perf_counter()
        set_llm_file_status(file_id, user_id, org_id, 2, True, retry_count, 1, "", start_datetime, "", True, False, False, 50, logger)
        llm_status_time = round(time.perf_counter() - status_start_time, 2)

        vector_start_time = time.perf_counter()
        logger.info(_log_message("Retrieving master BM25 encoder...", function_name, MODULE_NAME))

        cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        all_chunks_tf_sum = calculate_tf_update_db(org_id, user_id, file_id, cleaned_chunks, logger)

        process_and_upsert(org_id, user_id, file_id, file_name, cleaned_chunks, all_chunks_tf_sum, tag_ids, logger)

        set_llm_file_status(file_id, user_id, org_id, 2, True, retry_count, 1, "", start_datetime, "", True, False, False, 75, logger)

        vector_processing_time = round(time.perf_counter() - vector_start_time, 2)
        logger.info(_log_message(f"BM25 and vector processing completed in {vector_processing_time:.2f} seconds.", function_name, MODULE_NAME))

        status_end_time = time.perf_counter()
        set_llm_file_status(file_id, user_id, org_id, 2, True, retry_count, 3, "", start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), True, False, False, 100, logger)
        status_completion_time = round(time.perf_counter() - status_end_time, 2)

        total_time = round(time.perf_counter() - overall_start_time, 2)
        memory_usage = process.memory_info().rss / (1024 * 1024)
        cpu_usage = process.cpu_percent(interval=0.1)

        log_data = {
            "file_id": file_id,
            "file_name": file_name,
            "user_id": user_id,
            "org_id": org_id,
            "cpu_usage": round(cpu_usage, 2),
            "memory_usage": round(memory_usage, 2),
            "chunks_count": len(chunks),
            "time_taken": {
                "llm_status_initiate_2_time": llm_status_time,
                "vector_upsertion_time": vector_processing_time,
                "llm_status_completion_2_time": status_completion_time,
                "total_vector_upsertion_execution_time": total_time
            }
        }
        logger.info(_log_message(f"VECTOR UPSERTION SUMMARY: {orjson.dumps(log_data).decode()}", function_name, MODULE_NAME))
        return {"success": True, "error": ""}

    except Exception as e:
        error_msg = f"Exception occurred: {e}"
        logger.error(_log_message(error_msg, function_name, MODULE_NAME))
        set_llm_file_status(file_id, user_id, org_id, 2, True, retry_count, 2, error_msg, start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), True, False, False, 100, logger)
        return {"success": False, "error": error_msg}
        
def process_2(file_id: str, file_name: str, file_type: str, user_id: int, org_id: int, 
                           chunks: list, retry_count: int, tag_ids: List[int], logger) -> dict:
    """Process 2: Vector upsert function to upsert the vectors"""
    try:
        logger.info(_log_message("Starting Process 2: Vector Upsert", "process_2", MODULE_NAME))
        
        if not chunks:
            msg = "No chunks provided for vector upsert"
            logger.error(_log_message(msg, "process_2", MODULE_NAME))
            return {"success": False, "error": msg}

        stage_start = time.perf_counter()
        
        # Call vector upsert function
        upsert_result = update_encoder_and_upsert_vectors(
            chunks, file_name, file_type, file_id, user_id, org_id, retry_count, tag_ids, logger
        )
        
        log_time("Vector Upsert", stage_start, logger)
        
        if not upsert_result.get("success", False):
            error_message = f"Vector upsert failed: {upsert_result.get('error', 'Unknown error')}"
            logger.error(_log_message(error_message, "process_2", MODULE_NAME))
            return {"success": False, "error": error_message}

        logger.info(_log_message("Process 2: Vector Upsert completed successfully", "process_2", MODULE_NAME))
        return {"success": True, "error": ""}

    except Exception as e:
        error_message = f"Error in Process 2 - Vector Upsert: {e}"
        logger.error(_log_message(error_message, "process_2", MODULE_NAME))
        return {"success": False, "error": error_message}

def process_3(file_id: str, file_name: str, file_type: str, user_id: int, org_id: int, 
                                 chunks: list, complete_file_text: str, retry_count: int, target_metadata_fields, tag_ids, logger) -> dict:
    """Process 3: Metadata extraction"""
    try:
        logger.info(_log_message("Starting Process 3: Metadata Extraction", "process_3", MODULE_NAME))
        
        if not chunks:
            msg = "No chunks provided for metadata extraction"
            logger.error(_log_message(msg, "process_3", MODULE_NAME))
            return {"success": False, "error": msg, "metadata": {}}

        stage_start = time.perf_counter()
        metadata = get_metadata_parallely(file_id, file_name, file_type, user_id, org_id, retry_count, chunks, complete_file_text, target_metadata_fields, tag_ids, logger)
  
        metadata_duration = log_time("Metadata Extraction", stage_start, logger)

        logger.info(_log_message("Process 3: Metadata Extraction completed successfully", "process_3", MODULE_NAME))
        return {"success": True, "error": "", "metadata": metadata}

    except Exception as e:
        error_message = f"Error in Process 3 - Metadata Extraction: {e}"
        logger.error(_log_message(error_message, "process_3", MODULE_NAME))
        return {"success": False, "error": error_message, "metadata": {}}

def process_4(file_id: str, file_name: str, file_type: str, user_id: int, org_id: int, 
                                   complete_file_markdown_text: str, complete_file_toc: str, retry_count: int, logger) -> dict:
    """Process 4: Document summarization"""
    try:
        logger.info(_log_message("Starting Process 4: Document Summarization", "process_4", MODULE_NAME))
        
        if not complete_file_markdown_text:
            msg = "No markdown text provided for document summarization"
            logger.error(_log_message(msg, "process_4", MODULE_NAME))
            return {"success": False, "error": msg}

        stage_start = time.perf_counter()
        
        # Run process_document in a separate thread (fire-and-forget)
        Thread(
            target=process_document,
            args=(file_id, file_name, file_type, user_id, org_id, complete_file_markdown_text, complete_file_toc, retry_count),
            daemon=True
        ).start()
        
        log_time("Document Summarization (Thread Started)", stage_start, logger)
        
        logger.info(_log_message("Process 4: Document Summarization thread started successfully", "process_4", MODULE_NAME))
        return {"success": True, "error": ""}

    except Exception as e:
        error_message = f"Error in Process 4 - Document Summarization: {e}"
        logger.error(_log_message(error_message, "process_4", MODULE_NAME))
        return {"success": False, "error": error_message}

@mlflow.trace(name="Generate Insights")
def generate_insights(file_id: str, file_name: str, file_type: str, user_id: int, org_id: int, url: str, retry_count: int, retry_processes, target_metadata_fields, tag_ids, logger) -> dict:
    global_start_time = time.perf_counter()
    process = psutil.Process()
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        set_llm_file_status(file_id, user_id, org_id, 1, True, retry_count, 1, "", start_datetime, "", True, False, False, 10, logger)
        logger.info(_log_message("Initiating insights generation.", "generate_insights", MODULE_NAME))

        logger.info(_log_message(
            f"Request Details -> URL: {url}, FileID: {file_id}, FileName: {file_name}, FileType: {file_type}, "
            f"UserID: {user_id}, OrgID: {org_id}, RetryCount: {retry_count}",
            "generate_insights", MODULE_NAME
        ))

        # Initialize variables
        chunks = []
        complete_file_markdown_text = ""
        complete_file_toc = ""
        metadata = {}

        # Determine which processes to run based on retry_count
        if retry_count == 0:
            # First time: run processes 1, 2, 4, 3
            processes_to_run = [1, 2, 4, 3]
            logger.info(_log_message("First-time processing: Running processes 1, 2, 4, 3", "generate_insights", MODULE_NAME))
        else:
            # Retry: run specific processes from retry_processes configuration
            processes_to_run = retry_processes
            if not processes_to_run:
                logger.info(_log_message("No retry processes specified, defaulting to all processes", "generate_insights", MODULE_NAME))
                processes_to_run = [1, 2, 3, 4]
            logger.info(_log_message(f"Retry processing: Running processes {processes_to_run}", "generate_insights", MODULE_NAME))

        # Check if we need to retrieve existing data for retry processes
        need_chunks = any(proc in processes_to_run for proc in [2, 3])
        need_markdown = any(proc in processes_to_run for proc in [4])
        run_process_1 = 1 in processes_to_run

        # If we need data but process 1 is not running, retrieve existing data
        if (need_chunks or need_markdown) and not run_process_1:
            logger.info(_log_message("Retrieving existing chunks and markdown data for retry processing", "generate_insights", MODULE_NAME))
            
            # Retrieve existing chunks from database/storage
            # Process 1: Text Extraction
            existing_data = process_1(
                file_id, file_name, file_type, user_id, org_id, url, retry_count, start_datetime, logger
            )

            if existing_data["success"]:
                chunks = existing_data.get("chunks", [])
                complete_file_markdown_text = existing_data.get("complete_file_markdown_text", "")
                complete_file_toc = existing_data.get("complete_file_toc", "")
                
                logger.info(_log_message(
                    f"Retrieved existing data: {len(chunks)} chunks, "
                    f"markdown length: {len(complete_file_markdown_text)}, "
                    f"TOC length: {len(complete_file_toc)}", 
                    "generate_insights", MODULE_NAME
                ))
            else:
                logger.warning(_log_message(
                    f"Failed to retrieve existing data: {existing_data.get('error', 'Unknown error')}", 
                    "generate_insights", MODULE_NAME
                ))
                # If we can't retrieve existing data, we might need to run process 1
                if not run_process_1:
                    logger.info(_log_message("Adding process 1 to run due to missing existing data", "generate_insights", MODULE_NAME))
                    processes_to_run = [1] + processes_to_run
                    run_process_1 = True

        # Execute processes in order
        for process_num in processes_to_run:
            if process_num == 1:
                logger.info(_log_message(f"Starting Process {process_num}: Text Extraction", "generate_insights", MODULE_NAME))
                # Process 1: Text Extraction
                result = process_1(
                    file_id, file_name, file_type, user_id, org_id, url, retry_count, start_datetime, logger
                )
                if not result["success"]:
                    set_llm_file_status(file_id, user_id, org_id, 1, True, retry_count, 2, result["error"], start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), True, False, False, 100, logger)
                    return {"success": False, "error": f"Process 1 failed: {result['error']}", "data": ""}
                
                chunks = result["chunks"]
                complete_file_markdown_text = result["complete_file_markdown_text"]
                complete_file_toc = result["complete_file_toc"]
                logger.info(_log_message(
                    f"Process {process_num} completed successfully: {len(chunks)} chunks, "
                    f"markdown length: {len(complete_file_markdown_text)}, "
                    f"TOC length: {len(complete_file_toc)}", 
                    "generate_insights", MODULE_NAME
                ))

            elif process_num == 2:
                # Process 2: Vector Upsert
                logger.info(_log_message(f"Starting Process {process_num}: Vector Upsert", "generate_insights", MODULE_NAME))
                if not chunks:
                    logger.warning(_log_message("No chunks available for Process 2, skipping vector upsert", "generate_insights", MODULE_NAME))
                    continue
                
                result = process_2(
                    file_id, file_name, file_type, user_id, org_id, chunks, retry_count, tag_ids, logger
                )
                logger.info(_log_message(
                    f"Process {process_num} completed successfully: {result.get('success', False)}", 
                    "generate_insights", MODULE_NAME
                ))
                if not result["success"]:
                    set_llm_file_status(file_id, user_id, org_id, 2, True, retry_count, 2, result["error"], start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), True, False, False, 100, logger)
                    return {"success": False, "error": f"Process 2 failed: {result['error']}", "data": ""}

            elif process_num == 3:
                # Process 3: Metadata Extraction
                logger.info(_log_message(f"Starting Process {process_num}: Metadata Extraction", "generate_insights", MODULE_NAME))
                if not chunks:
                    logger.warning(_log_message("No chunks available for Process 3, skipping metadata extraction", "generate_insights", MODULE_NAME))
                    continue
                
                result = process_3(
                    file_id, file_name, file_type, user_id, org_id, chunks, complete_file_markdown_text, retry_count, target_metadata_fields, tag_ids, logger
                )

                if not result["success"]:
                    set_llm_file_status(file_id, user_id, org_id, 3, True, retry_count, 2, result["error"], start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), True, False, False, 100, logger)
                    return {"success": False, "error": f"Process 3 failed: {result['error']}", "data": ""}
                
                metadata = result["metadata"]
                logger.info(_log_message(
                    f"Process {process_num} completed successfully: {len(metadata)} metadata items extracted", 
                    "generate_insights", MODULE_NAME
                ))

            elif process_num == 4:
                # Process 4: Document Summarization
                logger.info(_log_message(f"Starting Process {process_num}: Document Summarization", "generate_insights", MODULE_NAME))
                if not complete_file_markdown_text:
                    logger.warning(_log_message("No markdown text available for Process 4, skipping document summarization", "generate_insights", MODULE_NAME))
                    continue
                
                result = process_4(
                    file_id, file_name, file_type, user_id, org_id, complete_file_markdown_text, complete_file_toc, retry_count, logger
                )
                logger.info(_log_message(
                    f"Process {process_num} completed successfully: {result.get('success', False)}", 
                    "generate_insights", MODULE_NAME
                ))
                if not result["success"]:
                    set_llm_file_status(file_id, user_id, org_id, 4, True, retry_count, 2, result["error"], start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), True, False, False, 100, logger)
                    return {"success": False, "error": f"Process 4 failed: {result['error']}", "data": ""}

        try:
            if org_id == config.AVIVO_TEMPLATE_ORG_ID:
                update_avivo_template(org_id, user_id, url, file_id, file_name, file_type)
                logger.info(_log_message("Avivo templates updated successfully.", "generate_insights", MODULE_NAME))
        except Exception as e:
            logger.error(_log_message(f"Error updating Avivo templates: {e}", "generate_insights", MODULE_NAME))

        # System stats and logging
        memory_usage = round(process.memory_info().rss / (1024 * 1024), 2)
        cpu_usage = round(process.cpu_percent(interval=0.1), 2)
        total_duration = round(time.perf_counter() - global_start_time, 2)

        log_summary = {
            "file_id": file_id,
            "file_name": file_name,
            "file_type": file_type,
            "user_id": user_id,
            "org_id": org_id,
            "retry_count": retry_count,
            "processes_executed": processes_to_run,
            "chunks_count": len(chunks),
            "metadata_summary": metadata,
            "memory_usage_MB": memory_usage,
            "cpu_usage_percent": cpu_usage,
            "total_time_taken": total_duration
        }

        logger.debug(_log_message(f"INSIGHTS PROCESS SUMMARY: {orjson.dumps(log_summary).decode()}", "generate_insights", MODULE_NAME))
        logger.info(_log_message("Insights generation completed successfully.", "generate_insights", MODULE_NAME))
        return {"success": True, "error": "", "data": ""}

    except Exception as e:
        error_duration = round(time.perf_counter() - global_start_time, 2)
        memory_usage = round(process.memory_info().rss / (1024 * 1024), 2)
        cpu_usage = round(process.cpu_percent(interval=0.1), 2)
        error_message = f"Error during processing: {e}"

        logger.error(_log_message(error_message, "generate_insights", MODULE_NAME))
        logger.info(_log_message("Insights generation failed.", "generate_insights", MODULE_NAME))
        return {"success": False, "error": error_message, "data": ""}

    finally:
        gc.collect()