from src.config.base_config import config
from src.common.db_connection_manager import DBConnectionManager
from src.common.logger import _log_message
import json
import mlflow
mlflow.openai.autolog()

CONTRACT_GENERATION_DB = config.CONTRACT_GENERATION_DB 
module_name = "utils.py"

import pickle

@mlflow.trace(name="Store Extracted Text")
def store_extracted_text(ci_file_guid: str, file_name: str, file_type: str, extracted_text: str, unstructured_elements, logger):
    function_name = "store_unstructured_elements"
    try:
        if unstructured_elements:
            logger.info(_log_message(f"Storing unstructured elements for ci_file_guid: {ci_file_guid}", function_name, module_name))
            dict_elements = [el for el in unstructured_elements]
            unstructured_elements_blob = pickle.dumps(dict_elements)
        else:
            logger.warning(_log_message(f"No unstructured elements extracted for ci_file_guid: {ci_file_guid}", function_name, module_name))
            unstructured_elements_blob = pickle.dumps([])

        with DBConnectionManager(CONTRACT_GENERATION_DB, logger) as conn:
            if conn is None:
                logger.error(_log_message("Failed to establish database connection.", function_name, module_name))
                raise ConnectionError("Failed to establish database connection.")
            logger.info(_log_message("Database connection established.", function_name, module_name))

            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM file_extractions WHERE ci_file_guid = %s", (ci_file_guid,))
                result = cursor.fetchone()
                count = result.get("count") if isinstance(result, dict) else result[0]

                if count > 0:
                    cursor.execute("""
                        UPDATE file_extractions
                        SET unstructured_elements = %s,
                            extracted_text = %s,
                            file_name = %s,
                            file_type = %s
                        WHERE ci_file_guid = %s
                    """, (unstructured_elements_blob, extracted_text, file_name, file_type, ci_file_guid))
                else:
                    cursor.execute("""
                        INSERT INTO file_extractions (ci_file_guid, file_name, file_type, extracted_text, unstructured_elements)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (ci_file_guid, file_name, file_type, extracted_text, unstructured_elements_blob))

                conn.commit()
                logger.info(_log_message(f"Transaction committed for ci_file_guid: {ci_file_guid}", function_name, module_name))

    except Exception as e:
        logger.error(_log_message(f"Error occurred during storing unstructured elements: {str(e)}", function_name, module_name))
        raise e

@mlflow.trace(name="Retrieve Extracted Text")
def retrieve_extracted_text(ci_file_guid: str, logger) -> tuple[str, list | None]:
    function_name = "retrieve_extracted_text"

    try:
        with DBConnectionManager(CONTRACT_GENERATION_DB, logger) as conn:
            if conn is None:
                msg = "Failed to establish database connection."
                logger.error(_log_message(msg, function_name, module_name))
                raise ConnectionError(msg)
            logger.info(_log_message("Database connection established.", function_name, module_name))

            with conn.cursor() as cursor:
                query = """
                    SELECT extracted_text, unstructured_elements
                    FROM file_extractions
                    WHERE ci_file_guid = %s
                """
                logger.info(_log_message(f"Executing query for ci_file_guid: {ci_file_guid}", function_name, module_name))
                cursor.execute(query, (ci_file_guid,))
                result = cursor.fetchone()

                if not result:
                    logger.warning(_log_message(f"No record found for ci_file_guid: {ci_file_guid}", function_name, module_name))
                    return "", None

                extracted_text = result.get("extracted_text", "")
                unstructured_elements_raw = result.get("unstructured_elements")
                logger.info(_log_message(f"Retrieved extracted_text length - {len(extracted_text)}", function_name, module_name))

                elements = None
                if unstructured_elements_raw:
                    try:
                        elements = pickle.loads(unstructured_elements_raw)
                        logger.info(_log_message(f"Pickle loaded unstructured_elements successfully for ci_file_guid: {ci_file_guid}", function_name, module_name))
                    except pickle.UnpicklingError as pickle_err:
                        logger.error(_log_message(f"Pickle unpickling error: {pickle_err}", function_name, module_name))

                logger.info(_log_message(f"Retrieved data for ci_file_guid: {ci_file_guid}", function_name, module_name))
                return extracted_text, elements

    except Exception as e:
        logger.error(_log_message(f"Unhandled exception: {str(e)}", function_name, module_name))
        raise e
