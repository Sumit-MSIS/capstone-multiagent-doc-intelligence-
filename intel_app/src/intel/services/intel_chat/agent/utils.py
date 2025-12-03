from typing import List, Dict, Any
from src.intel.services.intel_chat.agent.async_db_connection_manager import DBConnectionManager
import json
import os
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()
import mlflow
CONTRACT_INTEL_DB = os.getenv("CONTRACT_INTEL_DB", "contract_intel_db")
mlflow.openai.autolog()

module_name = "utils.py"

@mlflow.trace(name="Retrieve All File IDs")
async def retrieve_all_file_ids(org_id: int, tag_ids: List[int]) -> List[str]:
    """
    Retrieve file IDs for a given organization that meet specific conditions
    and are linked to any of the provided tag_ids.

    Steps:
    1. Query eligible file_ids from `files` (organization, type, template, etc.).
    2. Query file_temp_ids from `file_tags` for provided tag_ids.
    3. Return intersection of both sets.

    :param org_id: Organization ID
    :param user_id: User ID
    :param tag_ids: List of tag IDs to filter files by
    :param logger: Logger instance
    :return: List of matching file GUIDs
    """
    try:
        async with DBConnectionManager(CONTRACT_INTEL_DB) as connection:
            async with connection.cursor() as cursor:

                # --- Step 1: Get all valid file IDs ---
                file_query = """
                    SELECT DISTINCT f.ci_file_guid
                    FROM files f
                    JOIN organizations o ON f.ci_org_guid = o.ci_org_guid
                    WHERE o.org_id = %s
                      AND f.is_archived IS NULL
                      AND f.type = 1
                      AND f.is_template IN (0, 2)
                      AND f.name NOT LIKE 'temp/%%'
                      AND o.is_archived IS NULL
                      AND EXISTS (
                          SELECT 1
                          FROM llm_process_status lps
                          WHERE lps.ci_file_guid = f.ci_file_guid
                            AND lps.process_type = 3
                            AND lps.completed_steps = 3
                      );
                """
                await cursor.execute(file_query, (org_id,))
                results = await cursor.fetchall()
                file_ids = [row["ci_file_guid"] for row in results]

                if not file_ids:
                    return []

                # --- Step 2: Get file_temp_ids for given tag_ids ---
                if tag_ids:
                    tag_query = """
                        SELECT DISTINCT file_temp_id
                        FROM file_tags
                        WHERE tag_id IN %s;
                    """
                    await cursor.execute(tag_query, (tuple(tag_ids),))
                    tag_results = await cursor.fetchall()
                    file_temp_ids = [row["file_temp_id"] for row in tag_results]


                    if not file_temp_ids:
                        return []

                    # --- Step 3: Filter by tag-linked file IDs ---
                    filtered_ids = [fid for fid in file_ids if fid in file_temp_ids]
                    return filtered_ids
                else:
                    return file_ids

    except Exception as e:
        raise RuntimeError(f"Failed to retrieve file IDs: {str(e)}")

@mlflow.trace(name="Filter Contracts")
async def filter_contracts(file_ids: List[str], tag_ids: List[int]) -> List[str]:
    """
    Filter contracts based on provided file_ids and tag_ids.

    Steps:
    1. Validate provided file_ids (must exist, be active, and not archived).
    2. If tag_ids are given, further filter by file_temp_ids linked to those tags.

    :param file_ids: List of candidate file GUIDs
    :param tag_ids: List of tag IDs to filter by
    :return: Filtered list of ci_file_guid values
    """

    try:
        if not file_ids:
            return []

        async with DBConnectionManager(CONTRACT_INTEL_DB) as connection:
            async with connection.cursor() as cursor:

                # --- Step 1: Filter valid contracts ---
                placeholders = ', '.join(['%s'] * len(file_ids))
                query = f"""
                    SELECT ci_file_guid
                    FROM files
                    WHERE ci_file_guid IN ({placeholders})
                      AND is_template IN (0, 2)
                      AND type = 1
                      AND is_archived IS NULL;
                """
                await cursor.execute(query, file_ids)
                results = await cursor.fetchall()
                valid_file_ids = [record["ci_file_guid"] for record in results]

                if not valid_file_ids:
                    return []

                # --- Step 2: If tag filtering is required ---
                if tag_ids:
                    placeholders_tag = ', '.join(['%s'] * len(tag_ids))
                    tag_query = f"""
                        SELECT DISTINCT file_temp_id
                        FROM file_tags
                        WHERE tag_id IN ({placeholders_tag});
                    """
                    await cursor.execute(tag_query, tag_ids)
                    tag_results = await cursor.fetchall()
                    tag_file_ids = [row["file_temp_id"] for row in tag_results]

                    if not tag_file_ids:
                        return []

                    # --- Step 3: Intersection ---
                    filtered_file_ids = [fid for fid in valid_file_ids if fid in tag_file_ids]

                    return filtered_file_ids
                
                # If no tag filtering needed
                return valid_file_ids
    except Exception as e:
        return []

@mlflow.trace(name="Get File Metadata")
async def get_metadata(
    user_file_ids: List[str]) -> Dict[str, Any]:
    """
    Fetches metadata for the given file IDs from the MySQL database.

    Args:
        user_file_ids (List[str]): List of file GUIDs for which metadata is required.

    Returns:
        Dict[str, Any]: A dictionary with file_id as key and metadata details including file_name.
    """
    try:
        
        async with DBConnectionManager(CONTRACT_INTEL_DB) as conn:
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
                            continue
                structured_data["total_files"] = total_files
                return structured_data

    except Exception as e:
        raise RuntimeError(f"Failed to fetch metadata: {str(e)}")


@mlflow.trace(name="Get File Names")
async def get_all_file_names(file_ids: List[str]) -> Dict[str, str]:
    """Helper function to get file names for given file IDs."""
    try:
        async with DBConnectionManager(CONTRACT_INTEL_DB) as connection:
            async with connection.cursor() as cursor:
                if not file_ids:
                    return {}
                
                placeholders = ', '.join(['%s'] * len(file_ids))
                query = f"""
                    SELECT ci_file_guid, name
                    FROM files
                    WHERE ci_file_guid IN ({placeholders});
                """
                await cursor.execute(query, file_ids)
                results = await cursor.fetchall()
                
                return {row["ci_file_guid"]: row["name"] for row in results}
    except Exception as e:
        raise RuntimeError(f"Failed to fetch file names: {str(e)}")
            



@mlflow.trace(name="Get Page Count")
async def get_page_count(file_id:str):
    try:
        async with DBConnectionManager(CONTRACT_INTEL_DB) as connection:
            async with connection.cursor() as cursor:

                # --- Step 1: Get all valid file IDs ---
                file_query = """
                    SELECT page_number AS total_pages
                    FROM files 
                    WHERE ci_file_guid = %s
                """
                await cursor.execute(file_query, (file_id,))
                results = await cursor.fetchall()
                total_pages = results[0]['total_pages'] if results else 0
                return total_pages
    except Exception as e:
        print(f"Error in get_page_count: {e}")
        return 0


@mlflow.trace(name="Read Summary Key-Points JSON")
def read_summary_json() -> dict:
    try:
        file_path = r"src/intel/services/intel_chat/summariser/contract_type_points.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return {}

@mlflow.trace(name="Get Pinecone Index")
def get_pinecone_index(pinecone_index_name):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(pinecone_index_name)
    return index