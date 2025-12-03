import json
import pinecone
import logging
from src.config.base_config import config
import pymysql
from src.common.logger import request_logger, _log_message, flush_all_logs
from src.common.db_connection_manager import DBConnectionManager

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)


CONTRACT_GENERATION_DB = config.CONTRACT_GENERATION_DB
CONTRACT_INTEL_DB = config.CONTRACT_INTEL_DB
MODULE_NAME = "get_table_of_contents.py"


def fetch_file_toc_from_mysql(file_ids, logger):
    """Fetches file TOC data from the specified database and table."""
    try:
        with DBConnectionManager(CONTRACT_GENERATION_DB, logger) as conn:
            if conn is None:
                raise ConnectionError("Failed to establish database connection.")
            logger.info(_log_message("Database connection successful.", "fetch_file_toc", MODULE_NAME))

            with conn.cursor() as cursor:
                column_name = "file_toc"
                table_name = "generate_contract_markdown"
                query = f"""
                    SELECT distinct file_id, {column_name} 
                    FROM {table_name} 
                    WHERE file_id IN ({','.join(['%s'] * len(file_ids))})
                """
                logger.info(_log_message(f"Executing query - {query}", "fetch_file_toc", MODULE_NAME))
                cursor.execute(query, file_ids)
                result = cursor.fetchall()
                logger.info(_log_message(f"Results for TOC on line 43: {result}", "fetch_file_toc", MODULE_NAME))
                logger.info(_log_message(f"Type of Results for TOC on line 44: {type(result)} and length {len(result)}", "fetch_file_toc", MODULE_NAME))
                # Convert results into a dictionary
                # toc_data = {row[0]: row[1] for row in result}
                toc_data = [json.loads(item['file_toc']) for item in result]
                return toc_data
    except Exception as e:
        logger.error(_log_message(f"Error fetching TOC from {CONTRACT_GENERATION_DB}: {e}", "fetch_file_toc", MODULE_NAME))
        return []




def extract_toc(event):
    file_ids = list(set(event.get('file_ids', [])))  
    user_id = event.get('user_id')  
    org_id = event.get('org_id')

    # Initialize logger
    log_target = file_ids[0] if file_ids else f"org_id-{org_id} | user_id-{user_id}"
    logger = request_logger(log_target, str(config.DOC_SUMMARY_TEMPLATE_LOG_DIR_NAME), "FETCH_FILES_TOC")

    try:
        logger.info(_log_message("FETCHING TOC OPERATION STARTED", "extract_toc", MODULE_NAME))
        logger.info(_log_message(f"Received Event: {event}", "extract_toc", MODULE_NAME))

        # Fetching Table of Contents from MySQL database
        db_results = fetch_file_toc_from_mysql(file_ids, logger) 
        
        logger.info(_log_message("Fetching Table Of Contents Ended", "extract_toc", MODULE_NAME))
        result = {
            "success": True,
            "error": None,
            "data": db_results
        }
        logger.info(_log_message(f"Result on line 74: {result}", "extract_toc", MODULE_NAME))
        return result
        
    except Exception as e:
        error_message = f"An error occurred while fetching file TOC data: {str(e)}"
        logger.error(_log_message(error_message, "extract_toc", MODULE_NAME))
        return {
            "success": False,
            "error": error_message
        }
