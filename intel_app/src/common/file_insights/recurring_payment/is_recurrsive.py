import pymysql
import logging
from openai import OpenAI
from datetime import datetime
from src.config.base_config import config
from pymysql.cursors import DictCursor
from src.common.file_insights.hybrid_vector_handler.master_bm25_retrieval import retrieve_master_bm25
from src.common.hybrid_retriever.hybrid_search_retrieval import get_context_from_pinecone
from src.common.file_insights.recurring_payment.llm_call import call_llm
from src.common.logger import request_logger, _log_message, flush_all_logs
from src.common.db_connection_manager import DBConnectionManager
import mlflow
from src.common.llm.factory import get_llm_client

mlflow.openai.autolog()
from opentelemetry import context as ot_context


MODULE_NAME = "is_recurrsive.py"
CONTRACT_INTEL_DB = config.CONTRACT_INTEL_DB
pinecone_index_name = config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX
pinecone_api_key = config.PINECONE_API_KEY


client = get_llm_client(async_mode=False)

@mlflow.trace(name="Recurring Payment - Update Due Date")
def due_date_updater(file_ids, org_id, user_id):
    results = []
    for file_id in file_ids:
        logger = request_logger(str(file_id), config.RECURRING_PAYMENT_LOG_DIR_NAME, "RECURRING_PAYMENT")
        logger.start(_log_message("RECURRING PAYMENT REQUEST STARTED", "due_date_updater", MODULE_NAME))
        logger.info(_log_message(f"Processing question for file_id: {file_id}, org_id: {org_id}, user_id: {user_id}", "due_date_updater", MODULE_NAME))
        
        try:
            logger.debug(_log_message("BM25 encoder retrieved successfully", "due_date_updater", MODULE_NAME))
            
            payment_due_date, expiry_date = access_metadata_dates(file_id, logger)

            current_date = datetime.now().strftime("%Y-%m-%d")

            question = {"What is the payment due date?":f"""Instructions:
Two parts, part A and part B:
Part A:
    Determine if the contract supports recurring payments based on the given text.

    Criteria:
    1. Identify if the contract mentions payments on a recurring basis using keywords such as "monthly," "quarterly," "annually," or phrases like:
    - "Payments due on the [specific date] of each month."
    - "Quarterly payments due on [specific months]."
    - "Annual payment due on [specific date] every year."
    
    2. Extract the contract expiry date.
    3. Extract the payment due dates.
    4. If the payment due dates extend beyond the contract expiry date, return "false".
    5. If the contract explicitly states recurring payments and they fall within the contract period, return "true".
    6. If no recurring payment terms are found, return "false".

    Return only a single word: "true" or "false".

Part B:    
    Given the following relevant contract information, determine the next immediate payment due date based on the present date that is: {current_date}.

    Consider the following scenarios:
    1. **Fixed Payment Due Date:** If the contract specifies a one-time or fixed payment due date, return that date if it is in the future or have passed.
    2. **Recurring Payment Due Dates:** If the contract specifies recurring payments (e.g., monthly, quarterly, annually), identify the next immediate due date considering the present date. The recurrence pattern can be in formats such as:
    - "Payments due on the 15th of each month"
    - "Quarterly payments due on the first of January, April, July, and October"
    - "Annual payment due on June 30 every year"

    **Instructions:**  
    - If no payment due date is found or it's unclear, return `"null"` as a string.
    - The output should be a JSON-style object with the following structure:
    {{"Payment Due Date": "YYYY-MM-DD"}}

OUTPUT:
{{
  "flag": true/false,
  "Payment Due Date": "YYYY-MM-DD"
}}
###If Expiry date({expiry_date}) < "Current Date ({current_date})", then return "flag" = false and "Payment Due Date" = "null" 
###If Expiry date({expiry_date}) < "Payment Due Date", then return "flag" = false and "Payment Due Date" = "null" 
Now, analyze the following contract text and return the next immediate payment due date in the required format.

""" }
            if expiry_date:
                expiry_date = datetime.strftime(expiry_date, "%Y-%m-%d")
            if payment_due_date:
                payment_due_date = datetime.strftime(payment_due_date, "%Y-%m-%d")
            print(f"Current date: {current_date}")
            print(f"expiry_date: {expiry_date}")
            print(f"previous_due_date: {payment_due_date}")
            if expiry_date is None or payment_due_date is None:
                logger.info(_log_message(f"Expiry date or payment due date is None for file: {file_id}. Cannot proceed.", "due_date_updater", MODULE_NAME))
                # flush_all_logs(str(file_id), config.RECURRING_PAYMENT_LOG_DIR_NAME, "RECURRING_PAYMENT")
                results.append({
                    "file_id": file_id,
                    "success": False,
                    "error": "Expiry date or payment due date is None.",
                    "data": {
                        "is_recursive": False,
                        "updated_payment_due_date": "null"
                    }
                })
                continue
            
            if current_date > expiry_date:
                logger.info(_log_message(f"Contract has expired for file: {file_id}. No need to update.", "due_date_updater", MODULE_NAME))
                # flush_all_logs(str(file_id), config.RECURRING_PAYMENT_LOG_DIR_NAME, "RECURRING_PAYMENT")
                results.append({
                    "file_id": file_id,
                    "success": True,
                    "error": "",
                    "data": {
                        "is_recursive": False,
                        "updated_payment_due_date": payment_due_date
                    }
                })
                continue

            elif current_date < payment_due_date:
                logger.info(_log_message(f"Payment Due Date is in the future for file: {file_id}. No need to update.", "due_date_updater", MODULE_NAME))
                # flush_all_logs(str(file_id), config.RECURRING_PAYMENT_LOG_DIR_NAME, "RECURRING_PAYMENT")
                results.append({
                    "file_id": file_id,
                    "success": True,
                    "error": "",
                    "data": {
                        "is_recursive": True,
                        "updated_payment_due_date": payment_due_date
                    }
                })
                continue

            elif current_date < expiry_date and current_date > payment_due_date:
                logger.info(_log_message("Payment Due Date is in the past. Updating payment field...", "due_date_updater", MODULE_NAME))
                top_k = 3
                custom_filter = {"file_id": {"$eq": file_id}}
                query = list(question.keys())[0]
                
                logger.info(_log_message("Retrieving context from Pinecone", "due_date_updater", MODULE_NAME))
                context_chunks = get_context_from_pinecone(pinecone_index_name, pinecone_api_key, custom_filter, top_k, query, file_id, user_id, org_id, logger)
                
                matches = context_chunks.get('matches', [])
                if not matches:
                    logger.warning(_log_message("No matching context found in Pinecone", "due_date_updater", MODULE_NAME))
                    results.append({
                        "file_id": file_id,
                        "success": False,
                        "error": "No matching context found in Pinecone",
                        "data": {
                            "is_recursive": False,
                            "updated_payment_due_date": "null"
                        }
                    })
                    continue

                context = [match['metadata']['text'] for match in matches]
                logger.debug(_log_message(f"Retrieved {len(context)} context chunks", "due_date_updater", MODULE_NAME))
                
                logger.info(_log_message("Calling LLM for response", "due_date_updater", MODULE_NAME))
                response = call_llm(question, context, file_id, user_id, org_id, logger)
                
                flag_value = response.get("flag")
                payment_due_date = response.get("Payment Due Date")
                
                update_payment_field(file_id, payment_due_date, logger)
                logger.info(_log_message(f"Updated payment field with Payment Due Date: {payment_due_date}", "due_date_updater", MODULE_NAME))
                # flush_all_logs(str(file_ids), config.RECURRING_PAYMENT_LOG_DIR_NAME, "RECURRING_PAYMENT")
                results.append({
                    "file_id": file_id,
                    "success": True,
                    "error": "",
                    "data": {
                        "is_recursive": flag_value,
                        "updated_payment_due_date": payment_due_date
                    }
                })
        except ValueError as ve:
            error_msg = f"ValueError in process_question: {str(ve)}"
            logger.error(_log_message(f"{error_msg}", "due_date_updater", MODULE_NAME))
            results.append({
                "file_id": file_id,
                "success": False,
                "error": error_msg,
                "data": {
                    "is_recursive": False,
                    "updated_payment_due_date": "null"
                }
            })
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(_log_message(f"{error_msg}", "due_date_updater", MODULE_NAME))
            results.append({
                "file_id": file_id,
                "success": False,
                "error": error_msg,
                "data": {
                    "is_recursive": False,
                    "updated_payment_due_date": "null"
                }
            })
    logger.info(_log_message(f"Response generated Successfully: {results}", "due_date_updater", MODULE_NAME))
    logger.info(_log_message(f"RECURRING PAYMENT REQUEST COMPLETED", "due_date_updater", MODULE_NAME))
    flush_all_logs(str(file_id), config.RECURRING_PAYMENT_LOG_DIR_NAME, "RECURRING_PAYMENT")
    return results

@mlflow.trace(name="Recurring Payment - Access Metadata Dates")
def access_metadata_dates(file_id: str, logger):
    try:
        
        logger.info(_log_message(f"Accessing metadata dates for file_id: {file_id}", "access_metadata_dates", MODULE_NAME))
        with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            with conn.cursor() as cursor:
                query = "SELECT date_title, date_value FROM file_metadata_dates WHERE ci_file_guid = %s"
                cursor.execute(query, (file_id,))
                rows = cursor.fetchall()
        
        previous_due_date = next((item['date_value'] for item in rows if item['date_title'] == 'Payment Due Date'), None)
        expiration_date = next((item['date_value'] for item in rows if item['date_title'] == 'Expiration Date'), None)
        
        logger.debug(_log_message(f"Retrieved Previous Payment Due Date: {previous_due_date}, Expiration Date: {expiration_date}", "access_metadata_dates", MODULE_NAME))
        return previous_due_date, expiration_date
    except Exception as e:
        logger.error(_log_message(f"Error accessing metadata dates: {str(e)}", "access_metadata_dates", MODULE_NAME))
        raise e
    
@mlflow.trace(name="Recurring Payment - Update Payment Field")
def update_payment_field(file_id: str, new_date: str, logger) -> bool:
    if new_date == "null":
        logger.warning(_log_message(f"Skipping update: Payment Due Date is null for file_id {file_id}", "update_payment_field", MODULE_NAME))
        return False
    
    logger.info(_log_message(f"Updating payment field for file_id: {file_id} with date: {new_date}", "update_payment_field", MODULE_NAME))
    query = """
        UPDATE file_metadata_dates
        SET date_value = %s
        WHERE date_title = 'Payment Due Date' AND ci_file_guid = %s;
    """
    try:
        with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (new_date, file_id))
                logger.info(_log_message(f"Successfully updated Payment Due Date for file_id: {file_id}", "update_payment_field", MODULE_NAME))
                return True
    except Exception as e:
        logger.error(_log_message(f"Error updating Payment Due Date: {str(e)}", "update_payment_field", MODULE_NAME))
        raise e



# file_id = ["5818afce-162b-4133-ad32-1e38a9eefa56"]
# org_id = 463
# user_id = 599
# print(due_date_updater(file_id, org_id, user_id))
