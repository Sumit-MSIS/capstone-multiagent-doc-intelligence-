import httpx
import re
from src.config.base_config import config
import requests
from src.common.logger import _log_message
import json
import ast
from typing import List, Dict, Optional, Union
import mlflow
from opentelemetry import context as ot_context
from src.common.async_db_connection_manager import DBConnectionManager
from src.common.db_connection_manager import DBConnectionManager as SyncDBConnectionManager
from typing import Any, Dict, List
from pinecone import Pinecone
import asyncio
# mlflow.config.enable_async_logging()
# mlflow.langchain.autolog()
mlflow.openai.autolog()

PINECONE_API_KEY = config.PINECONE_API_KEY

module_name = "utils.py"
CONTRACT_INTEL_DB = config.CONTRACT_INTEL_DB


@mlflow.trace(name="Get Pinecone Index")
def get_pinecone_index(pinecone_index_name):
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(pinecone_index_name)
    return index


@mlflow.trace(name="Validate Summarization Keywords")
def has_summarization_keywords(question):
    summarization_keywords = [
        "summarize", "summary", "condense", "abbreviate",
        "summarise", "gist", "synopsis", "compress", "sumarise", "sumarize"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in summarization_keywords)


@mlflow.trace(name="Clean Generated Questions")
def clean_generated_questions(response: str) -> Union[dict, list, str]:
    """
    Cleans unwanted formatting from a response and converts it into a JSON object if possible.

    Args:
        response (str): The raw response string.

    Returns:
        dict | list | str: Parsed JSON object if valid, otherwise cleaned string.
    """
    # Define unwanted patterns
    unwanted_patterns = [r'```json', r'```']
    
    # Remove unwanted patterns
    for pattern in unwanted_patterns:
        response = re.sub(pattern, '', response)

    # Trim whitespace and remove empty lines
    cleaned_response = '\n'.join(line.strip() for line in response.split('\n') if line.strip())
    cleaned_response = re.sub(r'[ \t]+\n', '\n', cleaned_response)  # Remove spaces before \n
    cleaned_response = re.sub(r'\n[ \t]+', '\n', cleaned_response)  # Remove spaces after \n
    try:
        # Attempt to parse JSON
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        return cleaned_response



@mlflow.trace(name="Clean SQL Agent Response")
def clean_sql_agent_response(response: str) -> List[str]:
    """
    Cleans the response string from a SQL agent to extract a flat list of file IDs.

    Args:
        response (str): The response string from the SQL agent.

    Returns:
        List[str]: A flat list of file IDs.
    """
    try:
        # Remove unwanted patterns
        response = re.sub(r'```(?:json)?', '', response).strip()
        
        # Extract the part that matches a list structure
        match = re.search(r'\[.*\]', response)
        
        if not match:
            return []  # No valid list found
        
        raw_data = match.group()
        
        # Replace tuples with lists for safe parsing
        raw_data = raw_data.replace('(', '[').replace(')', ']')

        # Safely convert string to Python list
        parsed_data = ast.literal_eval(raw_data)

        # Ensure parsed_data is a list
        if not isinstance(parsed_data, list):
            return []
        
        # Handle empty list cases
        if parsed_data == ['']:
            return []

        # Flatten nested lists
        flat_list = []
        for item in parsed_data:
            if isinstance(item, (list, tuple)):
                flat_list.extend(item)
            else:
                flat_list.append(item)

        # Convert everything to strings (ensuring uniformity)
        return [str(file_id) for file_id in flat_list]
    
    except (SyntaxError, ValueError, TypeError, IndexError, Exception) as e:
        print(f"Error parsing the response: {e}")
        return []


@mlflow.trace(name="Validate Web Search Keywords")
def has_call_web_words(answer):
    # Check if all the words "call", "web", and "search" are in the answer
    return all(word in answer for word in ["call", "web", "search"])





# Load the JSON structure from the prompts.json file
@mlflow.trace(name="Load Prompts")
def load_prompts(filename=r"src/intel/services/intel_chat/prompts.json"):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError as e:
        print(f"prompt library file is not found.")
        raise e
    
@mlflow.trace(name="Get Prompt")
async def get_prompt(service_name, category_name, prompt_identifier):
    try:
        # Load the prompt library
        prompt_library = load_prompts()

        # Check if the service and category exist in the prompt library
        if service_name not in prompt_library or category_name not in prompt_library[service_name]:
            return None  

        # Search for the prompt
        for prompt_data in prompt_library[service_name][category_name]:
            if prompt_data["prompt_identifier"] == prompt_identifier:
                return prompt_data["prompt"]
        
        # If no prompt is found, print the error message once
        print(f"No prompt found in prompt library for identifier: {prompt_identifier}")
        return None

    except Exception as e:
        print(f"Error in get_prompt: {e}")
        raise e



@mlflow.trace(name="Get Chat Session")
async def get_chat_session(session_id, logger):

    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    default_history = {
        "llm_history": [
            {
                "role": "system",
                "content": "You are an intelligent assistant specializing in legal contracts and agreements. Provide detailed, accurate responses to user questions regarding legal documents, using a conversational and user-friendly tone. Ensure your answers draw from a comprehensive understanding of all relevant information available to you. \n\n###Steps:\n**1) Understand the Question**: Carefully read and analyze the user's question and details provided from the user selected legal contracts or agreements.\n\n**2) Research**: Utilize your full knowledge of legal terms, contract structures, and common legal practices to inform your response, considering all available information. \n\n**3) Draft a Response**: Create a detailed answer that addresses the user’s question, simplifying legal terminology for easy understanding. \n\n**4) Clarify if Necessary**: If the user's question is unclear, politely suggest ways they could provide more details or ask follow-up questions to ensure an accurate response. \n\n###Output Format:\nProvide a detailed, accurate responses explaining legal terms when needed. Include a brief explanation of the answer if it aids understanding. Provide a clear and user-friendly response and enhance readability by including new line characters (\n) where needed. Focus on user-friendly explanations while maintaining accuracy. Strictly do not disclose the information that has provided as context or metadata or mention internal processes, data sources, or how you generate answers or terms like the retrieved context,etc. when no answer found or in answer. Strictly do not use '###' for formatting markdown text. Ensure the answer is well-structured and easy for the user to understand."
            }
        ]
    }

    try:        
        async with DBConnectionManager(config.CONTRACT_GENERATION_DB, logger) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT session_history FROM chat_sessions WHERE session_id = %s", 
                    (session_id,)
                )
                session_data = await cursor.fetchone()
                
                if session_data is None:
                    logger.info(f"No session data found for session_id: {session_id}, returning default history")
                    return default_history

                history_json = session_data.get("session_history", "{}")
                if history_json:
                    history_dict = json.loads(history_json)
                else:
                    history_dict = {}

                logger.info(f"Session history successfully retrieved for session_id: {session_id}")
                return history_dict.get(session_id, {"llm_history": []})

    except Exception as e:
        logger.error(f"Error retrieving chat session: {str(e)}")
        return default_history
            

@mlflow.trace(name="Set Chat Session")
async def set_chat_session(session_id, user_id, org_id, history, logger):
    try:
        session_history_json = json.dumps({session_id: history})

        async with DBConnectionManager(config.CONTRACT_GENERATION_DB, logger) as conn:
            async with conn.cursor() as cursor:
                # Check if session exists
                await cursor.execute("SELECT COUNT(*) AS count FROM chat_sessions WHERE session_id = %s", (session_id,))
                result = await cursor.fetchone()

                if result and result.get("count", 0) > 0:
                    # Update
                    await cursor.execute(
                        "UPDATE chat_sessions SET session_history = %s WHERE session_id = %s",
                        (session_history_json, session_id)
                    )
                    logger.info("Session history updated in DB.")
                else:
                    # Insert
                    await cursor.execute(
                        """
                        INSERT INTO chat_sessions (user_id, org_id, session_id, session_history) 
                        VALUES (%s, %s, %s, %s)
                        """,
                        (user_id, org_id, session_id, session_history_json)
                    )
                    logger.info("New session history inserted into DB.")
                # await conn.commit()
    except Exception as e:
        logger.error(f"Error storing chat session to DB: {e}")


@mlflow.trace(name="Get Metadata")
async def get_metadata(
    user_file_ids: List[str], user_id: str, org_id: str, logger
) -> Dict[str, Any]:
    """
    Fetches metadata for the given file IDs from the MySQL database.

    Args:
        user_file_ids (List[str]): List of file GUIDs for which metadata is required.
        logger: Logger instance for logging operations.

    Returns:
        Dict[str, Any]: A dictionary with file_id as key and metadata details including file_name.
    """
    try:
        logger.info(_log_message(f"User {user_id} requested metadata retrieval for file IDs: {user_file_ids}", 'get_metadata', module_name))
        
        async with DBConnectionManager(config.CONTRACT_INTEL_DB, logger) as conn:
            async with conn.cursor() as cursor:

                file_ids_str = ', '.join(f"'{file_id}'" for file_id in user_file_ids)
                
                # Updated query to include file_id (ci_file_guid)
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
                            metadata_title COLLATE utf8mb4_unicode_ci AS title, 
                            CASE 
                                WHEN metadata_title = 'Contract Value' THEN CAST(metadata_int_value AS CHAR)
                                ELSE metadata_text_value 
                            END COLLATE utf8mb4_unicode_ci AS value
                        FROM 
                            file_metadata 
                        UNION ALL 
                        SELECT 
                            ci_file_guid, 
                            date_title COLLATE utf8mb4_unicode_ci AS title, 
                            date_value COLLATE utf8mb4_unicode_ci AS value 
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
                logger.info(_log_message(f"Executing metadata query for file IDs {user_file_ids}: {present_file_ids_query}", 'get_metadata', module_name))
                
                await cursor.execute(present_file_ids_query)
                result = await cursor.fetchall()

                structured_data: Dict[str, Any] = {}
                total_files = 0
                
                for row in result:
                    file_id = row['file_id']
                    file_name = row['file_name']
                    metadata_json = row['metadata']

                    if metadata_json:
                        try:
                            metadata_list = json.loads(metadata_json)
                            filtered_metadata = {item["title"]: item["value"] for item in metadata_list if item["value"] is not None}
                            structured_data[file_id] = {
                                "file_name": file_name,
                                "metadata": filtered_metadata
                            }
                            total_files += 1
                        except json.JSONDecodeError as jde:
                            logger.error(_log_message(f"JSON decoding error for file {file_name} (ID: {file_id}): {str(jde)}", 'get_metadata', module_name))
                            logger.error(_log_message(f"Invalid metadata JSON: {metadata_json}", 'get_metadata', module_name))
                    else:
                        logger.warning(_log_message(f"No metadata found for file {file_name} (ID: {file_id})", 'get_metadata', module_name))
                
                structured_data["total_files"] = total_files
                logger.info(_log_message(f"Successfully retrieved metadata for {total_files} files for user {user_id}", 'get_metadata', module_name))
                return structured_data

    except Exception as e:
        logger.error(_log_message(f"Error retrieving metadata for file IDs {user_file_ids} requested by user {user_id}: {str(e)}", 'get_metadata', module_name))
        return None



@mlflow.trace(name="Get File Names from FileIds")
async def get_file_names_from_mysql(
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
    file_dict = {}

    try:
        async with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            async with conn.cursor() as cursor:
                logger.info(_log_message("Connected to MySQL.", "get_file_names_from_mysql", module_name))

                # Fetch file names for the retrieved file IDs
                await cursor.execute(f"""
                    SELECT ci_file_guid, name FROM files WHERE ci_file_guid IN ({', '.join(['%s'] * len(file_ids))}) AND is_archived IS NULL;
                """, tuple(file_ids))
                file_dict = {row['ci_file_guid']: row['name'] for row in await cursor.fetchall()}

            logger.info(_log_message(f"Retrieved {len(file_dict)} file names.", "get_file_names_from_mysql", module_name))

    except Exception as err:
        logger.error(_log_message(f"MySQL error: {err}", "get_file_names_from_mysql", module_name))
        
    return file_dict





@mlflow.trace(name="Fetch MetaData from FileIDs")
async def get_metadata_from_sql(
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
        
        async with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            async with conn.cursor() as cursor:

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
                            metadata_title COLLATE utf8mb4_unicode_ci AS title, 
                            CASE 
                                WHEN metadata_title = 'Contract Value' THEN CAST(metadata_int_value AS CHAR)
                                ELSE metadata_text_value 
                            END COLLATE utf8mb4_unicode_ci AS value
                        FROM 
                            file_metadata 
                        UNION ALL 
                        SELECT 
                            ci_file_guid, 
                            date_title COLLATE utf8mb4_unicode_ci AS title, 
                            date_value COLLATE utf8mb4_unicode_ci AS value 
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
                await cursor.execute(present_file_ids_query)
                result = await cursor.fetchall()

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
                logger.info(_log_message(f"Successfully retrieved metadata for {total_files} files for user {user_id} - {structured_data}", 'get_metadata_from_sql', module_name))
                return structured_data
    
    except Exception as e:
        logger.error(_log_message(f"Error retrieving metadata for file IDs {user_file_ids} requested by user {user_id}: {str(e)}", 'get_metadata_from_sql', module_name))
        return None



@mlflow.trace(name="Fetch Meta Data and Date for FileID")
async def fetch_meta_data(file_id: str, user_id: str, org_id: str, logger) -> Optional[str]:
    """Fetches metadata and date information for a given file from MySQL."""

    query = """
        SELECT metadata_title, metadata_text_value FROM file_metadata WHERE ci_file_guid = %s
        UNION ALL
        SELECT date_title, date_value FROM file_metadata_dates WHERE ci_file_guid = %s
    """

    try:
        # Use a single connection and cursor to optimize performance
        async with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            async with conn.cursor() as cursor:
            
                # Execute the combined query with parameterized input
                await cursor.execute(query, (file_id, file_id))  
                results = await cursor.fetchall()

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


@mlflow.trace(name="Get File Names")
async def get_file_names(file_ids, logger):
    try:
        async with DBConnectionManager(config.CONTRACT_INTEL_DB, logger) as connection:
            # Use a dictionary cursor
            async with connection.cursor() as cursor:
                file_ids_str = ', '.join(f"'{file_id}'" for file_id in file_ids)
                query = f"""
                SELECT ci_file_guid, name
                FROM files
                WHERE ci_file_guid IN ({file_ids_str});
                """
                await cursor.execute(query)
                results = await cursor.fetchall()

                # Convert list of dicts to {file_id: name}
                file_dict = {row["ci_file_guid"]: row["name"] for row in results}
                logger.info(_log_message(f"Retrieved {file_dict} file names from the database.", "get_file_names", module_name))
                return file_dict

    except Exception as e:
        logger.error(_log_message(f"Error in get_file_names: {str(e)}", "get_file_names", module_name))
        return {}

@mlflow.trace(name="Get User Details")
async def get_user_details(user_id, logger):
    """
    Fetches user details from the database.

    Args:
        user_id (str): The ID of the user.
        logger: Logger instance for logging operations.

    Returns:
        Dict[str, Any]: A dictionary containing user details.
    """
    try:
        async with DBConnectionManager(config.CONTRACT_INTEL_DB, logger) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT firstname, lastname, created_date, last_login, email, phone_no FROM users WHERE user_id = %s", (user_id,))
                result = await cursor.fetchone()
                if result:
                    return dict(result)
                else:
                    logger.warning(_log_message(f"No user found with ID {user_id}", 'get_user_details', module_name))
                    return {}
    except Exception as e:
        logger.error(_log_message(f"Error fetching user details for ID {user_id}: {str(e)}", 'get_user_details', module_name))
        return {}

@mlflow.trace(name="Filter Contracts")
def filter_contracts(file_ids, logger):
    try:
        if not file_ids:
            logger.info(_log_message("No file IDs provided for filtering templates.", "filter_templates", module_name))
            return []
        with SyncDBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            with conn.cursor() as cursor:
                placeholders = ', '.join(['%s'] * len(file_ids))
                query = f"""
                SELECT ci_file_guid
                FROM files
                WHERE ci_file_guid IN ({placeholders}) AND is_template IN (0, 2) AND type = 1 AND is_archived IS NULL;
                """
                cursor.execute(query, file_ids)
                result = cursor.fetchall()
                matching_file_ids = [record["ci_file_guid"] for record in result]
                return matching_file_ids

    except Exception as e:
        logger.error(_log_message(f"Error fetching contract HTML: {e}", "filter_templates", module_name))
        return []


