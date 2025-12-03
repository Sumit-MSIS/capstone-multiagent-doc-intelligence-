import pymysql
import pickle
import os
import logging
from src.config.base_config import config
from src.common.logger import _log_message
import time
from src.common.db_connection_manager import DBConnectionManager
import mlflow
# mlflow.config.enable_async_logging()
# mlflow.langchain.autolog()
mlflow.openai.autolog()
from opentelemetry import context as ot_context

module_name ="master_bm25_retrieval.py"

DB_HOST = config.DB_HOST
DB_USER = config.DB_USER
DB_PASSWORD = config.DB_PASSWORD
DB_PORT = int(config.DB_PORT)
CONTEXTUAL_RAG_DB = config.CONTEXTUAL_RAG_DB

@mlflow.trace(name="Retrieve Master BM25")
def retrieve_master_bm25(org_id: int, logger) -> bytes:
    """
    Fetch and deserialize BM25 index from MySQL using PyMySQL.

    Args:
        org_id (int): Organization ID for identification.
        logger: Logger instance for logging.

    Returns:
        bytes: Deserialized BM25 index if found, otherwise None.
    """
    function_name = "retrieve_master_bm25"
    module_name = __name__
    
    logger.info(_log_message(f"Fetching BM25 index for org_id={org_id}...", function_name, module_name))
    
    start_time = time.perf_counter()  # More accurate timing

    try:
        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            if conn is None:
                raise ConnectionError("Failed to establish database connection.")
            with conn.cursor() as cursor:
                query = """SELECT corpus_data 
                           FROM bm25_corpus 
                           WHERE org_id = %s 
                           ORDER BY id DESC 
                           LIMIT 1;"""
                logger.debug(_log_message(f"Executing query: {query} with org_id={org_id}", function_name, module_name))
                
                cursor.execute(query, (org_id,))
                result = cursor.fetchone()

                

                if not result:
                    logger.warning(_log_message(f"No BM25 index found for org_id={org_id}.", function_name, module_name))
                    # raise ValueError(f"No BM25 index found for org_id={org_id}.")
                    return None
                # Handle DictCursor case
                if isinstance(result, dict):
                    serialized_index = result.get("corpus_data")  # Use .get() to avoid KeyError
                else:
                    serialized_index = result[0]  # Tuple case

                if serialized_index is None:
                    logger.warning(_log_message(f"BM25 index not found for org_id={org_id}.", function_name, module_name))
                    # raise ValueError(f"BM25 index not found for org_id={org_id}.")
                    return None


                try:
                    bm25_index = pickle.loads(serialized_index)
                except pickle.UnpicklingError as e:
                    logger.error(_log_message(f"Pickle UnpicklingError: {e}", function_name, module_name))
                    raise
                except Exception as e:
                    logger.error(_log_message(f"Unknown deserialization error: {repr(e)}", function_name, module_name))
                    raise
                execution_time = time.perf_counter() - start_time  # More precise execution time
                logger.info(_log_message(f"BM25 index successfully retrieved and deserialized in {execution_time:.6f} seconds.", function_name, module_name))
                return bm25_index
            
    
    except Exception as e:
        execution_time = time.perf_counter() - start_time
        logger.error(_log_message(f"Unexpected Error: {repr(e)} (Execution time: {execution_time:.6f} seconds)", function_name, module_name))
        raise


@mlflow.trace(name="Retrieve Master Metadata BM25")
def retrieve_master_metadata_bm25(org_id: int, logger) -> bytes:
    """
    Fetch and deserialize BM25 metadata index from MySQL.

    Args:
        file_id (str): File ID for identification.
        user_id (int): User ID for identification.
        org_id (int): Organization ID for identification.
        logger: Logger instance.

    Returns:
        bytes: Deserialized BM25 metadata index if found, otherwise None.
    """
    function_name = "retrieve_master_metadata_bm25"

    logger.info(_log_message("Fetching Master BM25 Metadata index from MySQL...", function_name, module_name))
    start_time = time.perf_counter()

    try:
        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            if conn is None:
                raise ConnectionError("Failed to establish database connection.")
            logger.info(_log_message("Database connection successful.", function_name, module_name))

            with conn.cursor() as cursor:
                query = """SELECT metadata_corpus_data 
                           FROM bm25_corpus 
                           WHERE org_id = %s 
                           ORDER BY id DESC 
                           LIMIT 1;"""

                logger.debug(_log_message(f"Executing query: {query} with org_id={org_id}", function_name, module_name))
                
                cursor.execute(query, (org_id,))
                result = cursor.fetchone()

                if not result:
                    logger.warning(_log_message(f"No Meta Data BM25 index found for org_id={org_id}.", function_name, module_name))
                    # raise ValueError(f"No Meta Data BM25 index found for org_id={org_id}.")
                    return None

                # Handle DictCursor case
                if isinstance(result, dict):
                    serialized_index = result.get("metadata_corpus_data")  # Use .get() to avoid KeyError
                else:
                    serialized_index = result[0]  # Tuple case

                if serialized_index is None:
                    logger.warning(_log_message(f"Meta Data BM25 index not found for org_id={org_id}.", function_name, module_name))
                    # raise ValueError(f"Meta Data BM25 index not found for org_id={org_id}.")
                    return None


                try:
                    bm25_index = pickle.loads(serialized_index)
                except pickle.UnpicklingError as e:
                    logger.error(_log_message(f"Pickle UnpicklingError: {e}", function_name, module_name))
                    raise
                except Exception as e:
                    logger.error(_log_message(f"Unknown deserialization error: {repr(e)}", function_name, module_name))
                    raise
                execution_time = time.perf_counter() - start_time  # More precise execution time
                logger.info(_log_message(f"Meta Data BM25 index successfully retrieved and deserialized in {execution_time:.6f} seconds.", function_name, module_name))
                return bm25_index
                
    except Exception as e:
        execution_time = time.perf_counter() - start_time
        logger.error(_log_message(f"Unexpected Error: {e} (Execution time: {execution_time:.6f} seconds)", function_name, module_name))
        raise


# data = retrieve_master_metadata_bm25(6)
# print(type(data))
