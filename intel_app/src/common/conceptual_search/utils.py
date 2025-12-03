from src.common.db_connection_manager import DBConnectionManager
from typing import List
import time
import re
import logging
from src.config.base_config import config
from src.common.logger import _log_message
from src.common.conceptual_search.llm_call import open_ai_llm_call
import mlflow
from typing import List, Dict, Optional, Union, Any
from openai import OpenAI
import json
# mlflow.config.enable_async_logging()
# mlflow.langchain.autolog()
mlflow.openai.autolog()
from opentelemetry import context as ot_context
from src.common.llm.factory import get_llm_client

client = get_llm_client(async_mode=False)

CONTRACT_GENERATION_DB_NAME= config.CONTRACT_GENERATION_DB
CONTRACT_INTEL_DB = config.CONTRACT_INTEL_DB
module_name = "utils.py"

@mlflow.trace(name="Check File ID Present")
def is_file_id_present(file_ids: List[str], only_contract, tag_ids, logger: logging.Logger) -> List[str]:
    """
    Checks whether given file IDs exist in the database and are not archived.

    :param file_ids: List of file IDs to check.
    :param user_id: User ID making the request.
    :param org_id: Organization ID.
    :param logger: Logger instance.
    :return: List of matching file IDs found in the database.
    """
    start_time = time.perf_counter()

    if not isinstance(file_ids, list) or not all(isinstance(file_id, str) for file_id in file_ids):
        logger.error(_log_message(f"Input Error: 'file_ids' must be a list of strings.", "is_file_id_present", module_name))
        return []

    try:

        # Step 1: Get file_temp_ids values from file_tags
        with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT file_temp_id 
                    FROM file_tags 
                    WHERE tag_id IN %s
                    """,
                    (tuple(tag_ids),)
                )
                file_temp_ids = [row["file_temp_id"] for row in cursor.fetchall()]  # collect as list

        if not file_temp_ids:
            logger.info(_log_message("No temp file IDs found for given tags.", "is_file_id_present", module_name))
            return []

        logger.info(_log_message(f"Temp file IDs from tags: {file_temp_ids}", "is_file_id_present", module_name))

        file_ids = [fid for fid in file_ids if fid in file_temp_ids] if file_ids else []

        if not file_ids:
            logger.info(_log_message("No matching file IDs found after filtering with tags.", "is_file_id_present", module_name))
            return []


        with DBConnectionManager(CONTRACT_INTEL_DB, logger) as connection:
            logger.info(_log_message(f"Checking file IDs in repository | File IDs: {file_ids}", "is_file_id_present", module_name))

            with connection.cursor() as cursor:
                query = """
                SELECT ci_file_guid 
                FROM files 
                WHERE ci_file_guid IN %s 
                AND is_archived IS NULL;
                """ if not only_contract else """
                SELECT ci_file_guid 
                FROM files 
                WHERE ci_file_guid IN %s 
                AND is_archived IS NULL AND is_template IN (0, 2);
                """
                cursor.execute(query, (tuple(file_ids),))
                records = cursor.fetchall()
                matching_file_ids = [record["ci_file_guid"] for record in records]

                execution_time = time.perf_counter() - start_time
                logger.info(_log_message(f"Found {len(matching_file_ids)} matching file IDs. Execution time: {execution_time:.4f} seconds.", "is_file_id_present", module_name))
                return matching_file_ids

    except Exception as e:
        logger.error(_log_message(f"Error in is_file_id_present: {e}", "is_file_id_present", module_name))
        return []

@mlflow.trace(name="Get Similar Files")
def get_similar_files(user_query: str, org_id: int, logger: logging.Logger) -> List[str]:
    """
    Check whether similar text is present in file_markdown using SQL full-text search.
    If any similar text is found, return the corresponding file_ids.

    :param user_query: User input string.
    :param org_id: Organization ID.
    :param logger: Logger instance.
    :return: List of matching file IDs.
    """
    start_time = time.perf_counter()

    try:
        with DBConnectionManager(CONTRACT_GENERATION_DB_NAME, logger) as connection:
            logger.info(_log_message("Checking for similar files in the database.", "get_similar_files", module_name))

            with connection.cursor() as cursor:
                query = """
                SELECT DISTINCT file_id FROM generate_contract_markdown 
                WHERE org_id = %s AND file_markdown LIKE %s;
                """
                cursor.execute(query, (org_id, f"%{user_query}%"))
                matching_files = [row["file_id"] for row in cursor.fetchall()]

                execution_time = time.perf_counter() - start_time
                logger.info(_log_message(f"Found {len(matching_files)} similar files. Execution time: {execution_time:.4f} seconds.", "get_similar_files", module_name))
                return matching_files

    except Exception as e:
        logger.error(_log_message(f"Error in get_similar_files: {str(e)}", "get_similar_files", module_name))
        return 

@mlflow.trace(name="Generate Clause Regex")
def generate_clause_regex(extracted_clause: str, logger: logging.Logger) -> str:
    """
    Generates a regex pattern for exact legal clause matching in SQL queries.
    Handles edge cases: 
    - Prevents subword matches (e.g., "lease" won't match "release")
    - Accommodates hyphenated terms (e.g., "force-majeure" matches "force majeure")
    - Escapes special characters
    
    :param extracted_clause: Single clause word extracted by LLM
    :return: Safe regex pattern string
    """
    try:
        # Escape special regex characters
        clause = re.escape(extracted_clause.lower())
        
        # Handle hyphenated terms (allow space/hyphen variations)
        # if '-' in clause:
        #     # Replace escaped hyphens with flexible separator pattern
        #     clause = clause.replace(r"\-", r"(?:[-\s]|\\b)")
        if '-' in clause:
            clause = clause.replace(r"\-", r"[-\s]")
            
        # Create word boundary pattern
        # Uses negative lookarounds to prevent subword matches
        return rf"[^a-zA-Z0-9]{clause}[^a-zA-Z0-9]|^{clause}[^a-zA-Z0-9]|[^a-zA-Z0-9]{clause}$|^{clause}$"
    
    except Exception as e:
        logger.error(_log_message(f"Regex generation error: {str(e)}", "generate_clause_regex", module_name))
        return r"(?!x)x"  # Fallback to never-match pattern

@mlflow.trace(name="Get Similar Files by TOC")
def get_similar_files_by_toc(user_query: str, org_id: int, logger: logging.Logger) -> List[str]:
    """
    Check whether similar text is present in file_toc using SQL full-text search.
    If any similar text is found, return the corresponding file_ids.

    :param user_query: User input string.
    :param org_id: Organization ID.
    :param logger: Logger instance.
    :return: List of matching file IDs.
    """
    start_time = time.perf_counter()
    system_prompt = """ **Task**:  
Extract exactly one word representing a legal clause from the user's query.  
This word must be a standard clause term used in contracts/agreements (e.g., indemnification, arbitration).  

**Rules**:  
1. **Output Format**:  
   - Only one word (no sentences, explanations, or punctuation)  
   - Use lowercase  
   - Hyphenate multi-word terms (e.g., "force-majeure")  
   - NEVER include words like "clause", "section", or "provision"  

2. **Processing Logic**:  
   - Identify the core legal concept in the query  
   - Map it to standard contract clause terminology  
   - Ignore auxiliary words (e.g., "list down", "containing", "related to")  

3. **Examples**:  
   - Query: _"Show contracts with termination provisions"_ → **termination**  
   - Query: _"Find NDAs with non-compete clauses"_ → **non-compete**  
   - Query: _"Agreements mentioning force majeure"_ → **force-majeure**  
   - Query: _"List confidentiality sections"_ → **confidentiality**  

**Response Protocol**:  
- Output ONLY the extracted term (no prefix/suffix)  
- If no valid clause is found: output **null**  
"""
    user_prompt = f"Here is the user query: {user_query}. Please provide a concise and relevant response."
    try:
        user_query = open_ai_llm_call(system_prompt=system_prompt, user_prompt=user_prompt, model_name="gpt-4o-mini", temperature=0.1, function_name="get_similar_files_by_toc" ,logger=logger)
        # print(f"User query after LLM call for toc: {user_query}")
        logger.info(_log_message(f"User query after LLM call for toc: {user_query}", "get_similar_files_by_toc", module_name))
        if not user_query or user_query.lower() == "null":
            logger.info(_log_message("No valid clause found in user query.", "get_similar_files_by_toc", module_name))
            return []
        else:
            user_query = generate_clause_regex(user_query, logger)
            logger.info(_log_message(f"Generated regex for user query: {user_query}", "get_similar_files_by_toc", module_name))
            with DBConnectionManager(CONTRACT_GENERATION_DB_NAME, logger) as connection:
                logger.info(_log_message("Checking for similar files in the table of contents of files.", "get_similar_files_by_toc", module_name))
                with connection.cursor() as cursor:
                
                    query = f"""
                                SELECT DISTINCT gm.file_id
                                FROM generate_contract_markdown gm,
                                JSON_TABLE(
                                    gm.file_toc,
                                    '$.table_of_contents[*]'
                                    COLUMNS (
                                        section_name VARCHAR(255) PATH '$.section_name',
                                        sub_sections JSON PATH '$.sub_sections'
                                    )
                                ) AS toc
                                LEFT JOIN JSON_TABLE(
                                    toc.sub_sections,
                                    '$[*]' COLUMNS (
                                        sub_title VARCHAR(255) PATH '$.title'
                                    )
                                ) AS subs ON TRUE
                                WHERE gm.org_id = {org_id}
                                AND (
                                    REGEXP_LIKE(LOWER(toc.section_name), '{user_query}')
                                    OR REGEXP_LIKE(LOWER(subs.sub_title), '{user_query}')
                                );
                            """
                    cursor.execute(query)
                    
                    logger.info(_log_message(f"Executing query - {query}", "get_similar_files_by_toc", module_name))
                    matching_files = [row["file_id"] for row in cursor.fetchall()]

                    execution_time = time.perf_counter() - start_time
                    logger.info(_log_message(f"Found {len(matching_files)} similar files. Execution time: {execution_time:.4f} seconds.", "get_similar_files_by_toc", module_name))
                    return matching_files

    except Exception as e:
        logger.error(_log_message(f"Error in get_similar_files_by_toc: {str(e)}", "get_similar_files_by_toc", module_name))
        raise e 

@mlflow.trace(name="Rephrased Query")
def rephrase_search_text(search_text):

    system_prompt = """
        You are ChatGPT, an assistant that reformulates user queries into optimized legal-document retrieval phrases for keyword search, semantic search, and reranking.

        ## Task
        Given a user query, generate **exactly three distinct reformulated queries**:
        1. A **keyword search query** (compact; only core legal terms and their true synonyms).  
        2. A **semantic search query** (natural clause-level phrasing).  
        3. A **reranking query** (broader legal framing).  

        ## Guidelines
        - **Keyword search query**:  
        - Use only the **critical domain terms and variations** that actually appear in contracts.  
        - Do **not** add generic filler like “agreement” or “contract” unless required.  
        - Do **not conflate different contract types** (e.g., “sales contract” ≠ “purchase agreement”).  
        - Only expand terms when they are **true legal synonyms**, not just related concepts.  

        - **Semantic search query**:  
        - Express in **formal, clause-level legal phrasing**, focusing on obligations, rights, or restrictions.  

        - **Reranking query**:  
        - Broaden into a **legal context** (obligations, duties, compliance, remedies) while keeping the contract type or clause intact.  

        - Always preserve the **user’s exact intent**.  
        - Output must be **deterministic**: same input → same output.  
        - Queries should be **medium length** and target **clauses/sections**, not entire documents.  

        ## Output Format
        Return strictly as valid JSON:

        {
        "reformulated_queries": [
            "keyword_search_query",
            "semantic_search_query",
            "reranking_query"
        ]
        }

        ## Example 1
        ### Input user query:
        "find upsell or cross sell agreements"

        ### Output:
        {
        "reformulated_queries": [
            "upsell OR 'up-sell' OR 'add-on sale' OR 'incremental sale' OR 'upgrade sale' OR 'cross-sell' OR 'cross sell' OR 'cross selling' OR 'bundle sale' OR 'complementary sale'",
            "clauses addressing rights and obligations regarding upsell or cross-sell arrangements",
            "contract provisions governing upsell and cross-sell obligations, restrictions, and remedies"
        ]
        }

        ## Example 2
        ### Input user query:
        "list the sales contracts"

        ### Output:
        {
        "reformulated_queries": [
            "sales",
            "clauses outlining rights, duties, and obligations under sales contracts",
            "provisions within sales contracts governing transfer of goods, consideration, and remedies"
        ]
        }

        ## Example 3
        ### Input user query:
        "pull up the contracts having subleasing clause"

        ### Output:
        {
        "reformulated_queries": [
            "sublease OR sub-leasing OR subletting",
            "clauses addressing rights and restrictions related to subleasing of leased property",
            "lease or rental contract provisions governing subleasing, tenant obligations, and landlord approvals"
        ]
        }

        ## Example 4
        ### Input user query:
        "Show contracts that impose both late payment interest and allow the Seller to initiate arbitration if payment is overdue beyond 90 days"

        ### Output:
        {
        "reformulated_queries": [
            "late payment interest seller initiate arbitration overdue beyond '90 days'",
            "clauses imposing late payment interest and permitting seller-initiated arbitration for payments overdue beyond 90 days",
            "contract provisions governing late payment interest and seller rights to arbitration for payments overdue beyond 90 days"
        ]
        }

        ## Example 5
        ### Input user query:
        "What details are mentioned in article 9-11 having Xiamen Software Industry Investment & Development Co., Ltd as custodian?"
        
        ### Output:
        {
        "reformulated_queries": [
            "'article 9' OR 'article 10' OR 'article 11' AND 'Xiamen Software Industry Investment & Development Co., Ltd' AND custodian",
            "clauses within articles 9 to 11 mentioning Xiamen Software Industry Investment & Development Co., Ltd as custodian",
            "provisions in articles 9 through 11 referencing Xiamen Software Industry Investment & Development Co., Ltd in the role of custodian"
        ]
    """

    user_prompt = f"User Query: {search_text.lower()}"

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        temperature=0,
        top_p=1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"}
    )

    llm_answer = response.choices[0].message.content
    # Convert JSON string to Python dict
    try:
        reformulated_queries = json.loads(llm_answer)["reformulated_queries"]
    except json.JSONDecodeError:
        raise ValueError("Model did not return valid JSON. Content:", llm_answer)

    system_prompt_semantic = """
    You are ChatGPT, an assistant that reformulates user queries into a single-document perspective for semantic retrieval on legal text chunks.

    ## Task
    Rephrase the user query into a formal **clause-level phrasing**, as if it appears inside one contract, agreement, or legal document.  
    Always begin with the word **"Clause"** followed by the description.  
    Use **singular form** only — never "contracts" or "agreements" in plural.  
    Preserve the user's intent while normalizing into legal drafting style.  

    ## Output
    Only output the rephrased query. No explanations, no plurals, no extra commentary.

    ## Examples
    - Input: "give me the list of contracts where the delivery facility is located at 5281 carson avenue, las vegas, nv, 89102"  
    Output: "Clause specifying the delivery facility location at 5281 Carson Avenue, Las Vegas, NV, 89102."

    - Input: "pull up the contracts having subleasing clause"  
    Output: "Clause addressing subleasing rights and tenant obligations."

    - Input: "find upsell or cross sell agreements"  
    Output: "Clause governing upsell or cross-sell arrangements."

    - Input: "Show contracts that impose both late payment interest and allow the Seller to initiate arbitration if payment is overdue beyond 90 days"  
    Output: "Clause imposing late payment interest and granting the Seller rights to arbitration for overdue payments exceeding 90 days."

    - Input: "How many contracts have a force majeure clause?"
    Output: "Clause addressing force majeure events"

    - Input: "How many loan agreements are there?"
    Output: "Clause outlining the terms and conditions of a loan agreement."

    - Input: "List down all the loan agreements"
    Output: "Clause outlining the terms and conditions of a loan agreement."
    """



    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        temperature=0,  # Set temperature to 0 for deterministic output
        top_p=1,
        messages=[
            {"role": "system", "content": system_prompt_semantic},
            {"role": "user", "content": user_prompt},
        ],
    )

    semantic_query = response.choices[0].message.content

    return reformulated_queries[0], semantic_query, semantic_query
    # return reformulated_queries[0], reformulated_queries[1], reformulated_queries[2]


@mlflow.trace(name="Get Metadata")
def get_metadata(
    user_file_ids: List[str], user_id: str, org_id: str, logger
):
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
        
        with DBConnectionManager(config.CONTRACT_INTEL_DB, logger) as conn:
            with conn.cursor() as cursor:

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
                
                cursor.execute(present_file_ids_query)
                result = cursor.fetchall()

                structured_data: Dict[str, Any] = {}
                total_files = 0
                
                for row in result:
                    file_id = row['file_id']
                    file_name = row['file_name']
                    metadata_json = row['metadata']

                    if metadata_json:
                        try:
                            metadata_list = json.loads(metadata_json)
                            filtered_metadata = {item["title"]: item["value"] for item in metadata_list}
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