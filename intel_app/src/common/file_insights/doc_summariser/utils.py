from src.common.db_connection_manager import DBConnectionManager
from typing import List
import time
from pinecone import Pinecone
from src.common.logger import _log_message
from src.config.base_config import config
import mlflow
# mlflow.config.enable_async_logging()
# mlflow.langchain.autolog()
import json
import re
mlflow.openai.autolog()
from opentelemetry import context as ot_context



CONTRACT_GENERATION_DB_NAME = config.CONTRACT_GENERATION_DB
CONTRACT_INTEL_DB = config.CONTRACT_INTEL_DB

MODULE_NAME = "utils.py"

@mlflow.trace(name="Save Markdown to DB")
def save_markdown_to_db(data: dict, markdown_text: str, tags: str = "", section_name: str = "", chunk_count: int = 1, extracted_clause = "", logger=None):
    start_time = time.perf_counter()

    try:
        with DBConnectionManager(CONTRACT_GENERATION_DB_NAME, logger) as conn:
            if conn is None:
                raise ConnectionError("Failed to establish database connection.")
            logger.info(_log_message("Database connection successful.", "save_markdown_to_db", MODULE_NAME))

            with conn.cursor() as cursor:
                query = """
                INSERT INTO generate_contract_markdown (id, user_id, org_id, file_id, file_name, file_type, file_markdown, tags, section_name, chunk_count, extracted_clause)
                VALUES (NULL, %(user_id)s, %(org_id)s, %(file_id)s, %(file_name)s, %(file_type)s, %(file_markdown)s, %(tags)s, %(section_name)s, %(chunk_count)s, %(extracted_clause)s)
                """
                cursor.execute(query, {
                    'user_id': data['user_id'],
                    'org_id': data['org_id'],
                    'file_id': data['file_id'],
                    'file_name': data['file_name'],
                    'file_type': data['file_type'],
                    'file_markdown': markdown_text,
                    'tags': tags,
                    'section_name': section_name,
                    'chunk_count': chunk_count,
                    'extracted_clause': extracted_clause
                })

                logger.info(_log_message(f"Data saved to the database for chunk {chunk_count}", "save_markdown_to_db", MODULE_NAME))
                end_time = time.perf_counter()
                logger.info(_log_message(f"Time taken to save markdown to DB: {end_time - start_time:.4f} seconds", "save_markdown_to_db", MODULE_NAME))

    except Exception as e:
        logger.error(_log_message(f"Database error: {e}", "save_markdown_to_db", MODULE_NAME))
        raise f"Database error while inserting markdown: {e}"
    
@mlflow.trace(name="Save Table of contents to DB")
def update_toc_to_db(file_id="", extracted_toc="", logger=None):
    start_time = time.perf_counter()

    try:
        with DBConnectionManager(CONTRACT_GENERATION_DB_NAME, logger) as conn:
            if conn is None:
                raise ConnectionError("Failed to establish database connection.")
            logger.info(_log_message("Database connection successful.", "update_toc_to_db", MODULE_NAME))
            # logger.info(_log_message(f"Before Extracted TOC: {extracted_toc} and type {type(extracted_toc)}", "update_toc_to_db", MODULE_NAME))
            extracted_toc = json.dumps(extracted_toc)
            # logger.info(_log_message(f"After Dump Extracted TOC: {extracted_toc} and type {type(extracted_toc)}", "saveupdate_toc_to_db_markdown_to_db", MODULE_NAME))
            with conn.cursor() as cursor:
                query = """
                update generate_contract_markdown set file_toc = %(file_toc)s where file_id = %(file_id)s
                """
                cursor.execute(query, {
                    'file_toc': extracted_toc,
                    'file_id': file_id
                })

                logger.info(_log_message(f"File TOC is stored in the MYSQL DB", "update_toc_to_db", MODULE_NAME))
                end_time = time.perf_counter()
                logger.info(_log_message(f"Time taken to save markdown to DB: {end_time - start_time:.4f} seconds", "update_toc_to_db", MODULE_NAME))

    except Exception as e:
        logger.error(_log_message(f"Database error: {e}", "update_toc_to_db", MODULE_NAME))
        raise f"Database error while inserting markdown: {e}"


@mlflow.trace(name="Highlight Placeholders")
def highlight_placeholders(html_content, logger=None):
    try:
        logger.info(_log_message("Highlighting placeholders in HTML content.", "highlight_placeholders", MODULE_NAME))
        style = 'color: #025dfa; font-weight: bold;'  # Customize style as needed
        
        # Highlight [[placeholders]]
        html_content = re.sub(r'(\[\[.*?\]\])', rf'<span style="{style}">\1</span>', html_content)
        
        # Highlight {{placeholders}}
        html_content = re.sub(r'(\{\{.*?\}\})', rf'<span style="{style}">\1</span>', html_content)
        
        # Highlight long underscores (blanks)
        html_content = re.sub(r'(_{5,})', rf'<span style="{style}">\1</span>', html_content)
        
        # Highlight [placeholders] (excluding markdown links)
        html_content = re.sub(r'(?<!\()(\[[^\[\]]+?\])(?!\))', rf'<span style="{style}">\1</span>', html_content)

        return html_content
    except Exception as e:
        logger.error(_log_message(f"Error highlighting placeholders: {e}", "highlight_placeholders", MODULE_NAME))
        return html_content  # Return original content if error occurs


@mlflow.trace(name="Update HTML to DB")
def update_html_to_db(file_id, html_content, logger=None):
    start_time = time.perf_counter()

    try:
        html_content = highlight_placeholders(html_content, logger)
        with DBConnectionManager(CONTRACT_GENERATION_DB_NAME, logger) as conn:
            if conn is None:
                raise ConnectionError("Failed to establish database connection.")

            logger.info(_log_message("Database connection successful.", "save_html_to_db", MODULE_NAME))

            with conn.cursor() as cursor:
                # Step 1: Create the table if it doesn't exist
                create_table_query = """
                CREATE TABLE IF NOT EXISTS contract_html (
                    file_id VARCHAR(255) PRIMARY KEY,
                    html_content LONGTEXT
                )
                """
                cursor.execute(create_table_query)
                logger.info(_log_message("Ensured contract_html table exists.", "save_html_to_db", MODULE_NAME))

                # Step 2: Check if row already exists
                check_query = "SELECT COUNT(*) FROM contract_html WHERE file_id = %(file_id)s"
                cursor.execute(check_query, {'file_id': file_id})
                row = cursor.fetchone()

                count = int(row['COUNT(*)']) if row else 0
                # count = row.get("count") if isinstance(row, dict) else row[0]

                # Step 3: Update or Insert accordingly
                if count > 0:
                    update_query = """
                    UPDATE contract_html SET html_content = %(html_content)s WHERE file_id = %(file_id)s
                    """
                    cursor.execute(update_query, {
                        'html_content': html_content,
                        'file_id': file_id
                    })
                    logger.info(_log_message(f"Updated HTML content for file_id: {file_id}", "save_html_to_db", MODULE_NAME))
                else:
                    insert_query = """
                    INSERT INTO contract_html (file_id, html_content) VALUES (%(file_id)s, %(html_content)s)
                    """
                    cursor.execute(insert_query, {
                        'file_id': file_id,
                        'html_content': html_content
                    })
                    logger.info(_log_message(f"Inserted new HTML content for file_id: {file_id}", "save_html_to_db", MODULE_NAME))

                end_time = time.perf_counter()
                logger.info(_log_message(f"Time taken to save HTML to DB: {end_time - start_time:.4f} seconds", "save_html_to_db", MODULE_NAME))

    except Exception as e:
        logger.error(_log_message(f"Database error: {e}", "save_html_to_db", MODULE_NAME))
        raise Exception(f"Database error while saving HTML content: {e}")


@mlflow.trace(name="Save Summary to DB")
def save_summary_to_db(file_name: str, file_type: str, ci_file_guid: str, user_id: int, org_id: int, summary_plain_text: str, summary_html_text: str, logger=None):
    import time
    start_time = time.perf_counter()

    try:
        with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            if conn is None:
                raise ConnectionError("Failed to establish database connection.")
            logger.info(_log_message("Database connection successful.", "save_summary_to_db", MODULE_NAME))

            with conn.cursor() as cursor:
                # Step 1: Check if data already exists
                check_query = "SELECT COUNT(*) FROM file_summary WHERE ci_file_guid = %s"
                cursor.execute(check_query, (ci_file_guid,))
                result = cursor.fetchone()
                
                count = result.get("count") if isinstance(result, dict) else result[0]
                

                if count == 0:
                    # Insert new record
                    insert_query = """
                        INSERT INTO file_summary 
                        (file_name, file_type, ci_file_guid, user_id, org_id, summary_plain_text, summary_html_text) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (file_name, file_type, ci_file_guid, user_id, org_id, summary_plain_text, summary_html_text))
                    logger.info(_log_message("Inserted new summary into file_summary.", "save_summary_to_db", MODULE_NAME))
                else:
                    # Update existing record
                    update_query = """
                        UPDATE file_summary 
                        SET file_name = %s, file_type = %s, user_id = %s, org_id = %s, 
                            summary_plain_text = %s, summary_html_text = %s
                        WHERE ci_file_guid = %s
                    """
                    cursor.execute(update_query, (file_name, file_type, user_id, org_id, summary_plain_text, summary_html_text, ci_file_guid))
                    logger.info(_log_message("Updated existing summary in file_summary.", "save_summary_to_db", MODULE_NAME))

                end_time = time.perf_counter()
                logger.info(_log_message(f"Time taken to save summary to DB: {end_time - start_time:.4f} seconds", "save_summary_to_db", MODULE_NAME))

    except Exception as e:
        logger.error(_log_message(f"Database error: {e}", "save_summary_to_db", MODULE_NAME))
        raise Exception(f"Database error while inserting/updating markdown: {e}")

@mlflow.trace(name="Get Pineocne Client")
def get_pinecone_client(pinecone_api_key):
    return Pinecone(api_key=pinecone_api_key)

@mlflow.trace(name="Get Pineocone Index")
def get_pinecone_index(pinecone_client, index_name):
    try:
        return pinecone_client.Index(index_name)
    except Exception as e:
        raise f"Error in getting Pinecone index: {str(e)}"