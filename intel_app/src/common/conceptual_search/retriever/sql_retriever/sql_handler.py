import mysql.connector
import json
from src.config.base_config import config
from typing import List, Dict, Any, Optional
from src.common.logger import _log_message
from src.common.db_connection_manager import DBConnectionManager
import mlflow
# mlflow.config.enable_async_logging()
# mlflow.langchain.autolog()
mlflow.openai.autolog()
from opentelemetry import context as ot_context

module_name = "sql_handler.py"
DB_HOST = config.DB_HOST
DB_PORT = config.DB_PORT
DB_USER = config.DB_USER
DB_PASSWORD = config.DB_PASSWORD
DB_NAME = config.CONTRACT_INTEL_DB
CONTRACT_INTEL_DB = config.CONTRACT_INTEL_DB


# def _log_message(message: str, function_name: str, module_name: str) -> str:
#     return f"[function={function_name} | module={module_name}] - {message}"

@mlflow.trace(name="Create View for SQL Agent")
def create_view_for_agent(org_id: str, logger) -> bool:
    """
    Creates a view in the MySQL database for the SQL agent.

    Args:
        user_id (str): User ID for logging.
        org_id (str): Organization ID.
        logger (logging.Logger): Logger instance.

    Returns:
        bool: True if the view was created successfully, False otherwise.
    """
    try:
        with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            with conn.cursor() as cursor:

                # Query to create the view
                create_view_query = f"""
                    CREATE VIEW file_metadata_view_{org_id} AS
                        WITH metadata_pivot AS (
                            SELECT 
                                fm.ci_file_guid,
                                MAX(CASE WHEN fm.metadata_title = 'Title of the Contract' THEN CAST(fm.metadata_text_value AS CHAR CHARACTER SET utf8mb4) END) AS title_of_contract,
                                MAX(CASE WHEN fm.metadata_title = 'Scope of Work' THEN CAST(fm.metadata_text_value AS CHAR CHARACTER SET utf8mb4) END) AS scope_of_Work,
                                MAX(CASE WHEN fm.metadata_title = 'Parties Involved' THEN CAST(fm.metadata_text_value AS CHAR CHARACTER SET utf8mb4) END) AS parties_involved,
                                MAX(CASE WHEN fm.metadata_title = 'Contract Type' THEN CAST(fm.metadata_text_value AS CHAR CHARACTER SET utf8mb4) END) AS contract_type,
                                MAX(CASE WHEN fm.metadata_title = 'File Type' THEN CAST(fm.metadata_text_value AS CHAR CHARACTER SET utf8mb4) END) AS file_type,
                                MAX(CASE WHEN fm.metadata_title = 'Jurisdiction' THEN CAST(fm.metadata_text_value AS CHAR CHARACTER SET utf8mb4) END) AS jurisdiction,
                                MAX(CASE WHEN fm.metadata_title = 'Version Control' THEN CAST(fm.metadata_text_value AS CHAR CHARACTER SET utf8mb4) END) AS version_control,
                                MAX(CASE WHEN fm.metadata_title = 'Contract Duration' THEN CAST(fm.metadata_text_value AS CHAR CHARACTER SET utf8mb4) END) AS contract_duration,
                                MAX(CASE 
                                    WHEN fm.metadata_title = 'Contract Value' 
                                    THEN CAST(COALESCE(CONVERT(fm.metadata_int_value, CHAR), fm.metadata_text_value) AS CHAR CHARACTER SET utf8mb4) 
                                END) AS contract_value,
                                MAX(CASE WHEN fm.metadata_title = 'Risk Mitigation Score' THEN fm.metadata_int_value END) AS risk_mitigation_score
                            FROM file_metadata fm
                            GROUP BY fm.ci_file_guid
                        ),
                        date_pivot AS (
                            SELECT 
                                fmd.ci_file_guid,
                                MAX(CASE WHEN fmd.date_title = 'Effective Date' THEN fmd.date_value END) AS effective_date,
                                MAX(CASE WHEN fmd.date_title = 'Term Date' THEN fmd.date_value END) AS term_date,
                                MAX(CASE WHEN fmd.date_title = 'Payment Due Date' THEN fmd.date_value END) AS payment_due_date,
                                MAX(CASE WHEN fmd.date_title = 'Delivery Date' THEN fmd.date_value END) AS delivery_date,
                                MAX(CASE WHEN fmd.date_title = 'Termination Date' THEN fmd.date_value END) AS termination_date,
                                MAX(CASE WHEN fmd.date_title = 'Renewal Date' THEN fmd.date_value END) AS renewal_date,
                                MAX(CASE WHEN fmd.date_title = 'Expiration Date' THEN fmd.date_value END) AS expiration_date
                            FROM file_metadata_dates fmd
                            GROUP BY fmd.ci_file_guid
                        )
                        SELECT 
                            f.ci_file_guid,
                            f.name AS file_name,
                            f.user_id AS created_by,
                            f.created_date AS uploaded_date,
                            mp.title_of_contract,
                            mp.scope_of_Work,
                            mp.parties_involved,
                            mp.contract_type,
                            mp.file_type,
                            mp.jurisdiction,
                            mp.version_control,
                            mp.contract_duration,
                            mp.contract_value,
                            mp.risk_mitigation_score,
                            dp.effective_date,
                            dp.term_date,
                            dp.payment_due_date,
                            dp.delivery_date,
                            dp.termination_date,
                            dp.renewal_date,
                            dp.expiration_date
                        FROM (
                            SELECT DISTINCT ci_file_guid, name, user_id, created_date
                            FROM files 
                            WHERE ci_org_guid IN (SELECT ci_org_guid FROM organizations WHERE org_id = {org_id}) 
                            AND is_archived IS NULL AND type = 1
                        ) f
                        LEFT JOIN metadata_pivot mp ON f.ci_file_guid = mp.ci_file_guid
                        LEFT JOIN date_pivot dp ON f.ci_file_guid = dp.ci_file_guid;"""
                
                logger.info(_log_message("Executing query to create view for SQL agent.", "create_view_for_agent", module_name))


                # Execute the query
                cursor.execute(create_view_query)
                conn.commit()
                logger.info(_log_message("View created successfully for SQL agent.", "create_view_for_agent", module_name))
                return True
    except Exception as e:
        logger.error(_log_message(f"Error creating view for SQL agent: {str(e)}", "create_view_for_agent", module_name))
        return False


@mlflow.trace(name="Get Distinct Contract Types")
def get_distinct_contract_types(org_id: int, logger) -> List[str]:
    """
    Fetches distinct contract types from the file_metadata_view_{org_id} view.

    Args:
        org_id (int): Organization ID.
        logger (logging.Logger): Logger instance.

    Returns:
        List[str]: List of distinct contract types.
    """
    try:
        with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            with conn.cursor() as cursor:
                query = f"""
                    SELECT DISTINCT LOWER(contract_type) AS contract_type
                    FROM file_metadata_view_{org_id}
                    WHERE contract_type IS NOT NULL AND contract_type != '';
                """
                cursor.execute(query)
                result = cursor.fetchall()
                distinct_contract_types = [row['contract_type'] for row in result]
                return distinct_contract_types
    except Exception as e:
        logger.error(_log_message(f"Error fetching distinct contract types: {str(e)}", "get_distinct_contract_types", module_name))
        return []

@mlflow.trace(name="Fetch metadata for SQL Agent")
def execute_sql_query(query:str, org_id: str, logger) -> bool:
    """
    Drops the view created for the SQL agent in the MySQL database.

    Args:
        user_id (str): User ID for logging.
        org_id (str): Organization ID.
        logger (logging.Logger): Logger instance.

    Returns:
        bool: True if the view was dropped successfully, False otherwise.
    """
    try:
        # Connect to the MySQL database
        with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            with conn.cursor() as cursor:

                # Query to execute
                logger.info(_log_message(f"Executing query: {query}", "execute_sql_query", module_name))
                cursor.execute(query)

                result = cursor.fetchall()


            
                # # Query to drop the view
                # drop_view_query = f"DROP VIEW IF EXISTS file_metadata_view_{org_id};"
                # logger.info(_log_message("Executing query to drop view for SQL agent.", "drop_view_for_agent", module_name))

                # # Execute the query
                # cursor.execute(drop_view_query)
                # logger.info(_log_message("View dropped successfully for SQL agent.", "drop_view_for_agent", module_name))

                if result:  # Ensuring result is not None or empty
                    logger.info(_log_message(f"Query executed successfully. Result: {result}", "execute_sql_query", module_name))
                    if isinstance(result, list) and all(isinstance(row, dict) for row in result):
                        file_ids = [row.get("ci_file_guid") for row in result if "ci_file_guid" in row]  # Extract values safely

                    else:
                        logger.warning(_log_message(f"Unexpected result type: {type(result)} - {result}", "execute_sql_query", module_name))
                        return []

                    logger.info(_log_message(f"Query executed successfully. Result: {file_ids}", "execute_sql_query", module_name))
                    return file_ids
                else:
                    logger.warning(_log_message("Query executed but returned no results.", "execute_sql_query", module_name))
                    return []
    except Exception as e:
        logger.error(_log_message(f"Error executing query for SQL agent: {str(e)}", "execute_sql_query", module_name))
        return []
    # finally:
    #     execute_drop_view(org_id, logger)  # Ensure the view is dropped after execution


@mlflow.trace(name="Drop View for SQL Agent")
def execute_drop_view(org_id: str, logger) -> bool:
    """
    Drops the view created for the SQL agent in the MySQL database.

    Args:
        user_id (str): User ID for logging.
        org_id (str): Organization ID.
        logger (logging.Logger): Logger instance.

    Returns:
        bool: True if the view was dropped successfully, False otherwise.
    """
    try:
        # Connect to the MySQL database
        with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            with conn.cursor() as cursor:            
                # Query to drop the view
                drop_view_query = f"DROP VIEW IF EXISTS file_metadata_view_{org_id};"
                logger.info(_log_message("Executing query to drop view for SQL agent.", "execute_drop_view", module_name))

                # Execute the query
                cursor.execute(drop_view_query)
                logger.info(_log_message("View dropped successfully for SQL agent.", "execute_drop_view", module_name))
    except Exception as e:
        logger.error(_log_message(f"Error executing query for SQL agent: {str(e)}", "execute_drop_view", module_name))


@mlflow.trace(name="Fetch MetaData from FileIDs")
def get_metadata_from_sql(
    user_file_ids: List[str], user_id: str, org_id: str, logger
) -> Dict[str, Any]:
    """
    Fetches metadata for the given file IDs from the MySQL database.

    Args:
        user_file_ids (List[str]): List of file GUIDs for which metadata is required.
        logger: Logger instance for logging operations.

    Returns:
        Dict[str, Any]: A dictionary containing metadata for each file, along with total file count.
    """
    try:
        logger.info(_log_message(f"User {user_id} requested metadata retrieval for file IDs: {user_file_ids}", 'get_metadata_from_sql', module_name))
        
        with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            with conn.cursor() as cursor:

                file_ids_str = ', '.join(f"'{file_id}'" for file_id in user_file_ids)
                
                # Query to fetch metadata for the given file IDs
                present_file_ids_query = f"""
                    SELECT 
                        f.ci_file_guid AS file_id,
                        f.name AS file_name,
                        JSON_ARRAYAGG(
                            JSON_OBJECT(
                                'title', title,
                                'value', value
                            )
                        ) AS metadata
                    FROM (
                        SELECT 
                            ci_file_guid, 
                            metadata_title AS title, 
                            CASE 
                                WHEN metadata_title = 'Contract Value' THEN CAST(metadata_int_value AS CHAR)
                                ELSE metadata_text_value 
                            END AS value
                        FROM 
                            file_metadata 
                        UNION ALL 
                        SELECT 
                            ci_file_guid, 
                            date_title AS title, 
                            date_value AS value 
                        FROM 
                            file_metadata_dates
                    ) combined
                    JOIN files f 
                        ON combined.ci_file_guid = f.ci_file_guid
                    WHERE 
                        f.ci_file_guid IN ({file_ids_str})
                    GROUP BY 
                        f.ci_file_guid, f.name;
                """
                logger.info(_log_message(f"Executing metadata query for file ID {user_file_ids}: {present_file_ids_query}", 'get_metadata_from_sql', module_name))
                
                # Execute the query
                cursor.execute(present_file_ids_query)
                result = cursor.fetchall()

                # Process results into a structured dictionary
                structured_data: Dict[str, Any] = {}
                total_files = 0
                
                for row in result:
                    file_name = row['file_name']
                    metadata_json = row['metadata']

                    if metadata_json:  # Check if metadata_json is not empty
                        logger.info(_log_message(f"Metadata for file {file_name}: {metadata_json}", 'get_metadata_from_sql', module_name))  # Log metadata content
                        try:
                            metadata_list = json.loads(metadata_json)
                            # Filter out null values
                            filtered_metadata = {item["title"]: item["value"] for item in metadata_list if item["value"] is not None}
                            structured_data[file_name] = filtered_metadata
                            total_files += 1
                        except json.JSONDecodeError as jde:
                            logger.error(_log_message(f"JSON decoding error for file {file_name}: {str(jde)}", 'get_metadata_from_sql', module_name))
                            logger.error(_log_message(f"Invalid metadata JSON for file {file_name}: {metadata_json}", 'get_metadata_from_sql', module_name))  # Log the invalid JSON
                    else:
                        logger.warning(_log_message(f"No metadata found for file {file_name}", 'get_metadata_from_sql', module_name))
                
                structured_data["total_files"] = total_files
                logger.info(_log_message(f"Successfully retrieved metadata for {total_files} files for user {user_id}", 'get_metadata_from_sql', module_name))
                return structured_data
    
    except Exception as e:
        logger.error(_log_message(f"Error retrieving metadata for file IDs {user_file_ids} requested by user {user_id}: {str(e)}", 'get_metadata_from_sql', module_name))
        return None



@mlflow.trace(name="Get File Names from FileIds")
def get_file_names_from_mysql(
    file_ids: Optional[List[str]], org_id: str, user_id: str, logger
) -> Dict[str, str]:
    """
    Retrieve file names from the MySQL database for the given file IDs.

    Args:
        file_ids (Optional[List[str]]): List of file GUIDs. If empty, fetch from the database.
        org_id (str): Organization ID.
        user_id (str): User ID for logging.
        logger (logging.Logger): Logger instance.

    Returns:
        Dict[str, str]: Dictionary mapping file GUIDs to file names.
    """
    if file_ids is None:
        file_ids = []

    conn = None
    file_dict = {}

    try:
        with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            with conn.cursor() as cursor:
                logger.info(_log_message("Connected to MySQL.", "get_file_names_from_mysql", module_name))

                # If file_ids list is empty, fetch valid file IDs for the organization
                if not file_ids:
                    cursor.execute("""
                        SELECT f.ci_file_guid 
                        FROM files AS f
                        LEFT JOIN llm_process_status AS lm 
                            ON f.ci_file_guid = lm.ci_file_guid 
                            AND lm.process_type = (
                                SELECT MAX(process_type) 
                                FROM llm_process_status AS sub_lm 
                                WHERE sub_lm.ci_file_guid = f.ci_file_guid
                            )
                        WHERE f.ci_org_guid IN (
                                SELECT ci_org_guid FROM organizations WHERE org_id = %s
                            ) 
                            AND f.type != 2 
                            AND f.upload_state = 3 
                            AND (f.status = 1 OR f.status IS NULL) 
                            AND f.is_archived IS NULL 
                        ORDER BY f.file_id DESC;
                    """, (org_id,))
                    file_ids = [row['ci_file_guid'] for row in cursor.fetchall()]

                    if not file_ids:
                        logger.info(_log_message("No valid file IDs found.", "get_file_names_from_mysql", module_name))
                        return {}

                    logger.info(_log_message(f"Fetched {len(file_ids)} file IDs.", "get_file_names_from_mysql", module_name))

                # Fetch file names for the retrieved file IDs
                cursor.execute(f"""
                    SELECT ci_file_guid, name FROM files WHERE ci_file_guid IN ({', '.join(['%s'] * len(file_ids))});
                """, tuple(file_ids))
                file_dict = {row['ci_file_guid']: row['name'] for row in cursor.fetchall()}

            logger.info(_log_message(f"Retrieved {len(file_dict)} file names.", "get_file_names_from_mysql", module_name))

    except mysql.connector.Error as err:
        logger.error(_log_message(f"MySQL error: {err}", "get_file_names_from_mysql", module_name))
    finally:
        if conn:
            conn.close()
            logger.info(_log_message("MySQL connection closed.", "get_file_names_from_mysql", module_name))

    return file_dict


@mlflow.trace(name="Fetch Meta Data and Date for FileID")
def fetch_meta_data(file_id: str, user_id: str, org_id: str, logger) -> Optional[str]:
    """Fetches metadata and date information for a given file from MySQL."""

    query = """
        SELECT metadata_title, metadata_text_value FROM file_metadata WHERE ci_file_guid = %s
        UNION ALL
        SELECT date_title, date_value FROM file_metadata_dates WHERE ci_file_guid = %s
    """

    try:
        # Use a single connection and cursor to optimize performance
        with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            with conn.cursor() as cursor:
            
                # Execute the combined query with parameterized input
                cursor.execute(query, (file_id, file_id))  
                results = cursor.fetchall()

                # Filter out null or empty values, format the output
                meta_data_list = [f"{row['metadata_title']} - '{row['metadata_text_value']}'" 
                                  for row in results 
                                  if row['metadata_text_value'] and row['metadata_text_value'].lower() != "null"]

                # Log the number of retrieved metadata entries
                logger.info(_log_message(f"Retrieved {len(meta_data_list)} metadata entries.", 'fetch_meta_data', module_name))

                return '\n'.join(meta_data_list) if meta_data_list else None

    except Exception as err:
        logger.error(_log_message(f"Error fetching metadata: {err}", 'fetch_meta_data', module_name))
        return None