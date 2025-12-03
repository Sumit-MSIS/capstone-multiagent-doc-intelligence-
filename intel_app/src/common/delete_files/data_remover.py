import json
import pinecone
import logging
from src.config.base_config import config
from src.common.delete_files.update_bm25_corpus import update_corpus
import pymysql
from src.common.logger import request_logger, _log_message, flush_all_logs
from src.common.db_connection_manager import DBConnectionManager
from requests.exceptions import RequestException
import requests
import threading
import requests
from requests.exceptions import RequestException

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)




# Configuration
PINECONE_INDICES = {
    "DOCUMENT_SUMMARY_AND_CHUNK_INDEX": config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX,
    "DOCUMENT_METADATA_INDEX": config.DOCUMENT_METADATA_INDEX,
    "DOCUMENT_SUMMARY_INDEX": config.DOCUMENT_SUMMARY_INDEX,
    "DOCUMENT_SUMMARY_AND_CHUNK_INDEX_SPARSE": config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX_SPARSE
}

CONTRACT_GENERATION_DB = config.CONTRACT_GENERATION_DB
CONTRACT_INTEL_DB = config.CONTRACT_INTEL_DB

MODULE_NAME = "data_remover.py"
DB_CONFIGS = {
    "CONTRACT_GENERATION_DB": {"db": config.CONTRACT_GENERATION_DB, "column": "file_id", "table": config.CONTRACT_MARKDOWN_TABLE},
    # "contextual_rag": {"db": config.CONTEXTUAL_RAG_DB, "column": "file_id", "table": "contextual_rag_table"},
    "CONTRACT_INTEL_DB": {"db": config.CONTRACT_INTEL_DB, "column": "file_id", "table": config.MEDICAL_TABLE_NAME},
    "BM25_INDEX": {"db": config.CONTEXTUAL_RAG_DB, "column": "file_id", "table": "bm25_index"}
}



def bm25_deletion(org_id, file_ids, current_chunk_count, all_chunks_tf_sum, logger):
    function_name = "bm25_deletion"
    url = config.GET_BM25_ORG_AVGDL_URL

    payload = {
        "org_id": str(org_id),
        "file_id": str(file_ids),  # used just for debugging in logs, no use in api yet
        "total_chunks": current_chunk_count,
        "total_document_length": all_chunks_tf_sum,
        "action": "DELETE"
    }

    def fire_and_forget():
        for attempt in range(3):
            try:
                response = requests.post(url, json=payload)
                logger.debug(f"post request sent to /bm25/get-org-avgdl for {org_id} | response {response}")
                if response.status_code == 200:
                    data = response.json()
                    avgdl = data.get("avgdl")
                    logger.info(_log_message(
                        f"[API] Got avgdl={avgdl} from external API",
                        function_name,
                        MODULE_NAME
                    ))
                    return
                else:
                    logger.warning(_log_message(
                        f"[API] Failed attempt {attempt+1}: {response.status_code} {response.text}",
                        function_name,
                        MODULE_NAME
                    ))
            except RequestException as e:
                logger.warning(_log_message(
                    f"[API] Exception on attempt {attempt+1}: {e}",
                    function_name,
                    MODULE_NAME
                ))

    # Run in background thread (fire and forget)
    threading.Thread(target=fire_and_forget, daemon=True).start()


def get_deletion_stats(file_ids, logger):
    """Fetches current chunk count and total term frequency sum for BM25 recalculation."""
    try:
        db_info = DB_CONFIGS["BM25_INDEX"]
        with DBConnectionManager(db_info["db"], logger) as conn:
            if conn is None:
                raise ConnectionError("Failed to establish database connection.")
            logger.info(_log_message("Database connection successful.", "get_deletion_stats", MODULE_NAME))

            with conn.cursor() as cursor:
                column_name = db_info["column"]
                table_name = db_info["table"]
                query = f"""
                    SELECT COUNT(*) AS chunk_count, SUM(tf) AS tf_sum
                    FROM {table_name}
                    WHERE {column_name} IN ({','.join(['%s'] * len(file_ids))})
                """ 
                logger.info(_log_message(f"Executing query - {query}", "get_deletion_stats", MODULE_NAME))
                cursor.execute(query, file_ids)
                result = cursor.fetchone()

                chunk_count = int(result['chunk_count']) if result['chunk_count'] else 0
                tf_sum = int(result['tf_sum']) if result['tf_sum'] else 0
                logger.info(_log_message(f"Fetched deletion stats - chunk_count: {chunk_count}, tf_sum: {tf_sum}", "get_deletion_stats", MODULE_NAME))
                return chunk_count, tf_sum

    except Exception as e:
        logger.error(_log_message(f"Error fetching deletion stats: {e}", "get_deletion_stats", MODULE_NAME))
        return 0, 0

def remove_avivo_default_templates(file_ids, logger):
    """Removes Avivo default templates from the database."""
    try:
        with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            if conn is None:
                raise ConnectionError("Failed to establish database connection.")
            logger.info(_log_message("Database connection successful.", "remove_avivo_default_templates", MODULE_NAME))

            with conn.cursor() as cursor:
                query = f"DELETE FROM templates WHERE fileId IN ({','.join(['%s'] * len(file_ids))})"
                logger.info(_log_message(f"Executing query - {query}", "remove_avivo_default_templates", MODULE_NAME))
                cursor.execute(query, file_ids)
                deleted_rows = cursor.rowcount
                logger.info(_log_message(f"Deleted {deleted_rows} Avivo default templates.", "remove_avivo_default_templates", MODULE_NAME))
                return {"deleted_rows": deleted_rows}
    except Exception as e:
        logger.error(_log_message(f"Error deleting Avivo templates: {e}", "remove_avivo_default_templates", MODULE_NAME))
        return {"deleted_rows": 0, "error": str(e)}
        



def execute_db_deletion(db_key, file_ids, logger):
    """Deletes data from the specified database and table."""
    try:
        if db_key == "BM25_INDEX":
            db_info = DB_CONFIGS[db_key]
            with DBConnectionManager(db_info["db"], logger) as conn:
                if conn is None:
                    raise ConnectionError("Failed to establish database connection.")
                logger.info(_log_message("Database connection successful.", "execute_db_deletion", MODULE_NAME))

                with conn.cursor() as cursor:
                    column_name = db_info["column"]
                    table_name = db_info["table"]
                    query = f"UPDATE {table_name} SET is_archived = 1 WHERE {column_name} IN ({','.join(['%s'] * len(file_ids))})"
                    logger.info(_log_message(f"Executing query - {query}", "execute_db_deletion", MODULE_NAME))
                    cursor.execute(query, file_ids)
                    deleted_rows = cursor.rowcount


                    
                    return {"deleted_rows": deleted_rows}
           
        db_info = DB_CONFIGS[db_key]
        with DBConnectionManager(db_info["db"], logger) as conn:
            if conn is None:
                raise ConnectionError("Failed to establish database connection.")
            logger.info(_log_message("Database connection successful.", "execute_db_deletion", MODULE_NAME))

            with conn.cursor() as cursor:
                column_name = db_info["column"]
                table_name = db_info["table"]
                query = f"DELETE FROM {table_name} WHERE {column_name} IN ({','.join(['%s'] * len(file_ids))})"
                logger.info(_log_message(f"Executing query - {query}", "execute_db_deletion", MODULE_NAME))
                cursor.execute(query, file_ids)
                deleted_rows = cursor.rowcount
                return {"deleted_rows": deleted_rows}

        
    except Exception as e:
        logger.error(_log_message(f"Error deleting from {db_key}: {e}", "execute_db_deletion", MODULE_NAME))
        return {"deleted_rows": 0, "error": str(e)}


def get_pinecone_index(index_name):
    """Initialize and return a Pinecone index."""
    return pinecone.Pinecone(api_key=config.PINECONE_API_KEY).Index(index_name)

def delete_vectors(index_type: str, user_id: int, org_id: int, file_ids, logger):
    """Deletes vectors from Pinecone based on provided parameters."""
    try:
        logger.info(_log_message(f"Deleting vectors from Pinecone: {index_type}", "delete_vectors", MODULE_NAME))
        index = get_pinecone_index(PINECONE_INDICES[index_type])

        logger.info(_log_message(f"LLM Debug: User ID: {user_id}, Org ID: {org_id}, File IDs: {file_ids}", "delete_vectors", MODULE_NAME))
        logical_partition = f"org_id_{org_id}#"
        deleted_ids = []
        deleted_file_ids = []
        total_chunks_deleted = 0

        if file_ids:
            for file_id in file_ids:
                prefix = f"{org_id}#{file_id}#"

                logger.info(_log_message(f"Prefix: {prefix}", "delete_vectors", MODULE_NAME))
                
                # Check if the prefix exists in the namespace
                existing_ids = [ids for ids in index.list(prefix=prefix, namespace=logical_partition)]
                logger.info(_log_message(f"[{index_type}] Existing IDs for prefix {prefix}: {existing_ids}", "delete_vectors", MODULE_NAME))
                if not existing_ids:
                    logger.warning(_log_message(f"[{index_type}] No data found for File ID: {file_id}", "delete_vectors", MODULE_NAME))
                    continue  # Skip if no data found

                for ids in existing_ids:
                    logger.info(_log_message(f"[{index_type}] LLM Debug: Deleting IDs: {ids}", "delete_vectors", MODULE_NAME))
                    index.delete(ids=ids, namespace=logical_partition)
                    deleted_ids.extend(ids)
                    total_chunks_deleted += len(ids)
                deleted_file_ids.append(file_id)  # Ensure this is within the loop

            logger.info(_log_message(f"[{index_type}] Deleted IDs for File ID {file_id}: {deleted_ids}", "delete_vectors", MODULE_NAME))
        else:
            logger.warning(_log_message("No file IDs provided", "delete_vectors", MODULE_NAME))

        logger.info(_log_message(f"Total chunks deleted: {total_chunks_deleted}", "delete_vectors", MODULE_NAME))
        return {"deleted_ids": deleted_file_ids, "total_chunks_deleted": total_chunks_deleted}
    except Exception as e:
        logger.error(_log_message(f"Error deleting vectors: {e}", "delete_vectors", MODULE_NAME))
        return {"deleted_ids": [], "total_chunks_deleted": 0, "error": str(e)}


def file_remover(event):
    file_ids = list(set(event.get('file_ids', [])))  # Ensure uniqueness
    user_id = event.get('user_id')  
    org_id = event.get('org_id')

    # Initialize logger
    log_target = file_ids[0] if file_ids else f"org_id-{org_id} | user_id-{user_id}"
    logger = request_logger(log_target, str(config.DELETE_FILES_LOG_DIR_NAME), "DELETE_FILES")

    try:
        logger.info(_log_message("DELETION OPERATION STARTED", "file_remover", MODULE_NAME))
        logger.info(_log_message(f"Received Event: {event}", "file_remover", MODULE_NAME))

        # Deleting vectors from Pinecone
        delete_results = {
            "chunk_summary": delete_vectors("DOCUMENT_SUMMARY_AND_CHUNK_INDEX", user_id, org_id, file_ids, logger),
            "contract_metadata": delete_vectors("DOCUMENT_METADATA_INDEX", user_id, org_id, file_ids, logger),
            "summary": delete_vectors("DOCUMENT_SUMMARY_INDEX", user_id, org_id, file_ids, logger),
            "chunk_summary_sparse": delete_vectors("DOCUMENT_SUMMARY_AND_CHUNK_INDEX_SPARSE", user_id, org_id, file_ids, logger),

        }

        # Deleting data from databases
        db_results = {
            db_key: execute_db_deletion(db_key, file_ids, logger) for db_key in DB_CONFIGS
        }

        # Log details of each deletion result
        for key, result in delete_results.items():
            logger.info(f"{key.capitalize()} Result: {result}")

        # Consolidate deletion results
        deleted_ids = list(set(sum([result["deleted_ids"] for result in delete_results.values()], [])))
        total_chunks_deleted = sum([result["total_chunks_deleted"] for result in delete_results.values()])

        # update_corpus(file_ids, user_id, org_id, logger)

        deleted_chunk_count, deleted_chunks_tf_sum = get_deletion_stats(file_ids, logger)
        logger.info(_log_message(f"Chunks to be deleted from BM25 corpus: {deleted_chunk_count}, TF sum: {deleted_chunks_tf_sum}", "file_remover", MODULE_NAME))
        
        if org_id == config.AVIVO_TEMPLATE_ORG_ID:
            removed_templates_count = remove_avivo_default_templates(file_ids, logger)
            logger.info(_log_message(f"Removed Avivo default templates count: {removed_templates_count}", "file_remover", MODULE_NAME))
        if deleted_chunk_count > 0 and deleted_chunks_tf_sum > 0:
            bm25_deletion(org_id,file_ids, deleted_chunk_count, deleted_chunks_tf_sum, logger)

        logger.info(_log_message("DELETION OPERATION ENDED", "file_remover", MODULE_NAME))

        return {
            "success": True,
            "error": "Success",
            "data": {
                "deleted_ids": deleted_ids,
                "total_chunks_deleted": total_chunks_deleted,
                "db_results": db_results
            }
        }
    except Exception as e:
        error_message = f"An error occurred while deleting file data: {str(e)}"
        logger.error(_log_message(error_message, "file_remover", MODULE_NAME))
        return {
            "success": False,
            "error": error_message,
            "data": {
                "deleted_ids": [],
                "total_chunks_deleted": 0,
                "db_results": {}
            }
        }
    finally:
        flush_all_logs(log_target, str(config.DELETE_FILES_LOG_DIR_NAME), "DELETE_FILES")
