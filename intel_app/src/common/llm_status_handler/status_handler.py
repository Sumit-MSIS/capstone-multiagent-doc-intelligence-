from typing import Optional, Dict, Any, List
import requests
from src.config.base_config import config
import mlflow
import threading
import logging
from src.common.db_connection_manager import DBConnectionManager

# mlflow.config.enable_async_logging()

MODULE_NAME = "llm_status_handler.py"

def _log_message(message: str, function_name: str) -> str:
    return f"[function={function_name} | module={MODULE_NAME}] - {message}"

@mlflow.trace(name="Status Handler - Send Post Request")
def send_post_request(url: str, payload: dict, headers: dict, logger) -> None:
    def _post():
        try:
            logger.debug(_log_message(f"Sending POST request to {url} with payload: {payload}", "send_post_request"))
            # requests.post(url, json=payload, headers=headers)
        except Exception as e:
            logger.error(_log_message(f"Failed to send POST request to {url}: {str(e)}", "send_post_request"))

    thread = threading.Thread(target=_post)
    thread.daemon = True  # Doesn't block program exit
    thread.start()


@mlflow.trace(name="Status Handler - Set LLM File Status")
def set_llm_file_status(ci_file_guid: str, user_id: int, org_id: int, process_type: int, retryable_file: bool, retry_no: int, completed_steps: int, error_msg: str = "", process_start_time=None, process_end_time=None, immediate_retry_required: bool = False, isMock: bool = False, in_queue = False, execution_progress_percent = 0, logger=None):
    payload = {
        "info": {
            "ci_file_guid": ci_file_guid,
            "process_type": process_type,
            "retryable_file": retryable_file,
            "retry_no": retry_no,
            "completed_steps": completed_steps,
            "error_msg": error_msg,
            "process_start_time": process_start_time,
            "process_end_time": process_end_time if process_end_time and process_end_time != "" else None,
            "immediate_retry_required": immediate_retry_required,
            "isMock": isMock,
            "in_queue": in_queue,
            "execution_progress_percent": execution_progress_percent
        }
    }

    logger.info(_log_message(f"Initialising LLMFileStatus update to Process Type: {process_type}, Completed Steps: {completed_steps}, Retryable: {retryable_file}, Error Message: {error_msg}, Request: {payload}", "set_llm_file_status"))

    url = config.SET_LLM_STATUS_URL
    if not url:
        logger.error(_log_message("SET_LLM_STATUS_URL environment variable is not set.", "set_llm_file_status"))
        return

    headers = {'Content-Type': 'application/json'}

    send_post_request(url, payload, headers, logger)

    return



@mlflow.trace(name="Status Handler - Set AIAnswer")
def set_ai_answer(user_id: int, org_id: int, session_id: int, parent_session_id: int, chat_id: int, answer_json: dict, error_msg: str = "", enable_agent=False, client_id=None, logger=None):
    payload = {
        "info": {
            "client_id": client_id, 
            "user_id": user_id,
            "org_id": org_id,
            "session_id": session_id,
            "parent_session_id": parent_session_id,
            "chat_id": chat_id,
            "answer": answer_json,
            "error": error_msg,
            "enable_agent": enable_agent  # Store whether this is an agent response
        }
    }

    logger.info(_log_message(f"Initializing AI answer update: {payload}", "set_ai_answer"))

    url = config.SET_AI_ANSWER_URL
    if not url:
        logger.error(_log_message("SET_AI_ANSWER_URL environment variable is not set.", "set_ai_answer"))
        return

    headers = {'Content-Type': 'application/json'}

    send_post_request(url, payload, headers, logger)
    return


@mlflow.trace(name="Status Handler - Set Contract")
def set_contract(document_id: str, document_content: str, logger=None):
    payload = {
        "info": {
            "document_id": document_id,
            "document_content": document_content
        }
    }

    logger.info(_log_message(f"Initializing contract update: {payload}", "set_contract"))

    url = config.SET_TRADITIONAL_CONTRACT_URL
    if not url:
        logger.error(_log_message("SET_TRADITIONAL_CONTRACT_URL environment variable is not set.", "set_contract"))
        return

    headers = {'Content-Type': 'application/json'}

    send_post_request(url, payload, headers, logger)
    return


@mlflow.trace(name="Store Extracted Metadata")
def store_metadata(ci_file_guid: str, metadata, tag_ids, logger):
    function_name = "store_metadata"
    try:
        with DBConnectionManager(config.CONTRACT_INTEL_DB, logger) as conn:
            if conn is None:
                logger.error(_log_message("Failed to establish database connection.", function_name))
                raise ConnectionError("Failed DB Connection")

            logger.info(_log_message("DB connection established.", function_name))

            # Metadata lists
            dates_metadata = metadata.get("metadata", {}).get("dates", []) or []
            others_metadata = metadata.get("metadata", {}).get("others", []) or []

            with conn.cursor() as cursor:

                # -------------------------------------------------
                # 1️⃣ PROCESS OTHERS METADATA (file_metadata table)
                # -------------------------------------------------
                for item in others_metadata:
                    title = item.get("title")
                    value = item.get("value")

                    if not title:
                        continue  # skip invalid

                    cursor.execute("""
                        SELECT COUNT(*) AS count
                        FROM file_metadata
                        WHERE ci_file_guid = %s AND metadata_title = %s
                    """, (ci_file_guid, title))

                    exists_row = cursor.fetchone()
                    exists = exists_row.get("count") if isinstance(exists_row, dict) else exists_row[0]

                    if exists > 0:
                        cursor.execute("""
                            UPDATE file_metadata
                            SET metadata_text_value = %s,
                                tag_id = %s,
                                updated_on = NOW()
                            WHERE ci_file_guid = %s AND metadata_title = %s
                        """, (value, int(tag_ids[0]), ci_file_guid, title))
                    else:
                        cursor.execute("""
                            INSERT INTO file_metadata (ci_file_guid, metadata_title, metadata_text_value, tag_id, updated_on)
                            VALUES (%s, %s, %s, %s, NOW())
                        """, (ci_file_guid, title, value, int(tag_ids[0])))


                # -------------------------------------------------
                # 2️⃣ PROCESS DATE METADATA (file_metadata_dates table)
                # -------------------------------------------------
                for item in dates_metadata:
                    date_title = item.get("title")
                    date_value = item.get("value")

                    if not date_title:
                        continue

                    cursor.execute("""
                        SELECT COUNT(*) AS count
                        FROM file_metadata_dates
                        WHERE ci_file_guid = %s AND date_title = %s
                    """, (ci_file_guid, date_title))

                    exists_row = cursor.fetchone()
                    exists = exists_row.get("count") if isinstance(exists_row, dict) else exists_row[0]

                    if exists > 0:
                        cursor.execute("""
                            UPDATE file_metadata_dates
                            SET date_value = %s,
                                updated_on = NOW()
                            WHERE ci_file_guid = %s AND date_title = %s
                        """, (date_value, ci_file_guid, date_title))
                    else:
                        cursor.execute("""
                            INSERT INTO file_metadata_dates (ci_file_guid, date_title, date_value, updated_on)
                            VALUES (%s, %s, %s, NOW())
                        """, (ci_file_guid, date_title, date_value))

                conn.commit()
                logger.info(_log_message(f"Metadata stored successfully for ci_file_guid: {ci_file_guid}", function_name))

    except Exception as e:
        logger.error(_log_message(f"Error storing metadata: {str(e)}", function_name))
        raise



    

@mlflow.trace(name="Status Handler - Set Metadata")
def set_meta_data(ci_file_guid: str, user_id: int, org_id: int, process_type: int, process_start_time: str, process_end_time: str, completed_steps: int, immediate_retry_required: bool, error_msg: str, retry_no: int, meta_data, in_queue, execution_progress_percent, tag_ids, logger):
    """
    Updates the LLM file processing status by sending data to a specified API endpoint.

    Args:
        ci_file_guid (str): The unique GUID of the CI file.
        metadata (dict): The metadata details to be sent to the API.
    """

    logger.info(_log_message(f"Initiating metadata update for ci_file_guid: {ci_file_guid}, user_id: {user_id}, org_id: {org_id}", "set_meta_data"))

    meta_data_list = [meta_data["metadata"]]

    payload = {
        "info": {
            "ci_file_guid": ci_file_guid,
            "user_id": user_id,
            "process_type": process_type,
            "process_start_time": process_start_time,
            "process_end_time": process_end_time,
            "completed_steps": completed_steps,
            "immediate_retry_required": immediate_retry_required,
            "error_msg": error_msg,
            "retry_no": retry_no,
            "in_queue": in_queue,
            "execution_progress_percent": execution_progress_percent,
            "metadata": meta_data_list
        }
    }

    store_metadata(ci_file_guid, meta_data, tag_ids, logger)

    # Retrieve the API endpoint from environment variables
    url = config.SET_META_DATA_URL
    if not url:
        logger.error(_log_message("SET_META_DATA_URL environment variable is not set.", "set_meta_data"))
        return

    headers = {'Content-Type': 'application/json'}

    send_post_request(url, payload, headers, logger)

    return

    # response_json = send_post_request(url, payload, headers, logger)

    # if response_json and response_json.get("success"):
    #     logger.info(_log_message(f"Successfully updated Metadata for ci_file_guid: {ci_file_guid}", "set_meta_data"))
    # else:
    #     logger.error(_log_message(f"API response indicates failure for ci_file_guid: {ci_file_guid}. Response: {response_json}", "set_meta_data"))




@mlflow.trace(name="Status Handler - Set Comparison Result")
def set_comparison_result(chat_id: int, comparison_markdown: str, error_msg="", logger=None):
    payload = {
            "info": {
                "chat_id": chat_id,
                "answers": {
                    "comparison": comparison_markdown
                }
                
            },
            "error": error_msg
        }

    logger.info(_log_message(f"Initializing comparison markdown update: {payload}", "set_comparison_result"))

    url = config.SET_COMPARISION_RESULT_URL
    if not url:
        logger.error(_log_message("SET_COMPARISION_RESULT_URL environment variable is not set.", "set_comparison_result"))
        return

    headers = {'Content-Type': 'application/json'}

    send_post_request(url, payload, headers, logger)
    return


# @mlflow.trace(name="Status Handler - Send Post Request MySQL")
def send_post_request_mysql(url: str, payload: dict, headers) -> None:
    # @mlflow.trace(name="Status Handler - Send Post Request MySQL - Thread")
    def _post():
        requests.post(url, json=payload, headers=headers)

    thread = threading.Thread(target=_post)
    thread.daemon = True  # Doesn't block program exit
    thread.start()

@mlflow.trace(name="Status Handler - Set WebSocket Stream DB")
def set_websocket_stream_db(connection_id, request_id, chunk_answer, chat_id, client_id, enable_agent):
    # logger.info(_log_message(f"Initializing WebSocket stream update: connection_id: {connection_id}, request_id: {request_id}, session_id: {session_id}, chunk_order: {chunk_order}, chunk_answer: {chunk_answer}, stream_completed_status: {stream_completed_status}, chat_id: {chat_id}, related_questions: {related_questions}", "set_websocket_stream_db"))
    payload = {
            "info": {
                "connection_id": connection_id,
                "request_id": request_id,
                "client_id": client_id,
                "chat_id": chat_id,
                "enable_agent": enable_agent,
                "groupByChunkAnswer": chunk_answer
            }
        }
   

    # Retrieve the API endpoint from environment variables
    url = config.SET_WEBSOCKET_STREAM_URL
    if not url:
        # logger.error(_log_message("SET_WEBSOCKET_STREAM_URL environment variable is not set.", "set_meta_data"))
        return

    headers = {'Content-Type': 'application/json'}

    send_post_request_mysql(url, payload, headers)
    return



@mlflow.trace(name="Status Handler - Set Conceptual Search Result")
def set_conceptual_search_results(search_id: int, relevant_ids: List[str], logger=None):
    payload = {
        "info": {
            "search_id": search_id,
            "relevant_ids": relevant_ids
        }
    }

   

    url = config.SET_CONCEPTUAL_SEARCH_RESULT_URL
    if not url:
        logger.error(_log_message("SET_TRADITIONAL_CONTRACT_URL environment variable is not set.", "set_contract"))
        return

    headers = {'Content-Type': 'application/json'}

    send_post_request(url, payload, headers, logger)
    return