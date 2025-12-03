import pymysql
# from src.utils.settings import config
from src.config.base_config import config
from src.common.db_connection_manager import DBConnectionManager
from src.common.logger import _log_message
import mlflow
# mlflow.config.enable_async_logging()
# mlflow.langchain.autolog()
mlflow.openai.autolog()
from opentelemetry import context as ot_context

CONTEXTUAL_RAG_DB = config.CONTEXTUAL_RAG_DB 
module_name = "upload_master_bm25.py"


def upload_hybrid_bm25(serialized_index: bytes, user_id: int, org_id: int, file_id: str, file_name: str):
    DB_HOST = config.DB_HOST
    DB_USER = config.DB_USER
    DB_PASSWORD = config.DB_PASSWORD
    DB_NAME = config.CONTEXTUAL_RAG_DB
    DB_PORT = int(config.DB_PORT)
    
    try:
        # Use a context manager for the connection
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        print("Database connection successful!")

        with connection.cursor() as cursor:
            query = """
            INSERT INTO hybrid_bm25 (file_name, ci_file_id, user_id, org_id, serialized_index)
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (file_name, file_id, user_id, org_id, serialized_index))
            connection.commit()

    except pymysql.MySQLError as err:
        print(f"MySQL Error: {err}")
    finally:
        if connection:
            connection.close()



@mlflow.trace(name="Upload BM25 Master")
def upload_bm25_master(serialized_index: bytes, org_id: int, logger):
    
    try:
        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            if conn is None:
                raise ConnectionError("Failed to establish database connection.")
            logger.info(_log_message("Database connection successful.", "upload_bm25_master", module_name))

            with conn.cursor() as cursor:
                # Ensure table exists with LONGTEXT for corpus_data and an auto-incrementing primary key id
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS bm25_corpus (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    org_id INT NOT NULL,
                    corpus_data LONGBLOB,
                    metadata_corpus_data LONGBLOB,
                    summary_corpus_data LONGBLOB)
                """)

                # Check if org_id exists in the table
                cursor.execute("SELECT COUNT(*) FROM bm25_corpus WHERE org_id = %s", (org_id,))
                row_count = cursor.fetchone()
                logger.info(_log_message(f"Row count for org_id {org_id}: {row_count}", "upload_bm25_master", module_name))

                if isinstance(row_count, dict):
                    row_count = row_count.get("COUNT(*)")
                else:
                    row_count = row_count[0]

                if row_count > 0:
                    logger.info(_log_message(f"Existing row found for org_id {org_id}. Proceeding to update.", "upload_bm25_master", module_name))
                    # Update existing row
                    update_query = """
                    UPDATE bm25_corpus 
                    SET corpus_data = %s 
                    WHERE org_id = %s
                    """
                    cursor.execute(update_query, (serialized_index, org_id))  # Decode bytes to string
                    conn.commit()
                    logger.info(_log_message(f"Updated corpus_data for org_id {org_id}.", "upload_bm25_master", module_name))
                else:
                    logger.info(_log_message(f"No existing row for org_id {org_id}, inserting new row.", "upload_bm25_master", module_name))
                    # Insert new row
                    insert_query = """
                    INSERT INTO bm25_corpus (org_id, corpus_data)
                    VALUES (%s, %s)
                    """
                    cursor.execute(insert_query, (org_id, serialized_index))  # Decode bytes to string
                    conn.commit()
                    logger.info(_log_message(f"Inserted new row for org_id {org_id}.", "upload_bm25_master", module_name))


    except Exception as e:
        logger.error(_log_message(f"Error in upload_bm25_master: {e}", "upload_bm25_master", module_name))
        raise





@mlflow.trace(name="Upload BM25 Metadata Master")
def upload_bm25_metadata_master(serialized_index: bytes, org_id: int, logger):

    try:
        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            if conn is None:
                raise ConnectionError("Failed to establish database connection.")
            logger.info(_log_message("Database connection successful.", "upload_bm25_metadata_master", module_name))

            with conn.cursor() as cursor:
                # Ensure table exists with LONGTEXT for corpus_data and an auto-incrementing primary key id
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS bm25_corpus (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    org_id INT NOT NULL,
                    corpus_data LONGBLOB,
                    metadata_corpus_data LONGBLOB,
                    summary_corpus_data LONGBLOB)
                """)

                # Check if org_id exists in the table
                cursor.execute("SELECT COUNT(*) FROM bm25_corpus WHERE org_id = %s", (org_id,))
                row_count = cursor.fetchone()
                logger.info(_log_message(f"Row count for org_id {org_id}: {row_count}", "upload_bm25_metadata_master", module_name))

                if isinstance(row_count, dict):
                    row_count = row_count.get("COUNT(*)")
                else:
                    row_count = row_count[0]

                if row_count > 0:
                    logger.info(_log_message(f"Existing row found for org_id {org_id}. Proceeding to update.", "upload_bm25_metadata_master", module_name))
                    # Update existing row
                    update_query = """
                    UPDATE bm25_corpus 
                    SET metadata_corpus_data = %s 
                    WHERE org_id = %s
                    """
                    cursor.execute(update_query, (serialized_index, org_id))  # Decode bytes to string
                    conn.commit()
                    logger.info(_log_message(f"Updated corpus_data for org_id {org_id}.", "upload_bm25_metadata_master", module_name))
                else:
                    # logger.info(_log_message(f"No existing row for org_id {org_id}, inserting new row.", "upload_bm25_metadata_master", module_name))
                    # # Insert new row
                    # insert_query = """
                    # INSERT INTO bm25_corpus (org_id, metadata_corpus_data)
                    # VALUES (%s, %s)
                    # """
                    # cursor.execute(insert_query, (org_id, serialized_index))  # Decode bytes to string
                    # conn.commit()
                    # logger.info(_log_message(f"Inserted new row for org_id {org_id}.", "upload_bm25_metadata_master", module_name))
                    raise ValueError(f"No existing row for org_id {org_id}. Cannot update metadata_corpus_data without an existing corpus_data row.")


    except Exception as e:
        logger.error(_log_message(f"Error in upload_bm25_metadata_master: {e}", "upload_bm25_metadata_master", module_name))
        raise
