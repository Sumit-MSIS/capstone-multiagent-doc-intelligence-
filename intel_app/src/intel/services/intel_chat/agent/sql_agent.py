from src.common.sql_retriever.sql_handler import create_view_for_agent, execute_sql_query, get_distinct_contract_types
from src.common.logger import _log_message
from src.config.base_config import config
from typing import List, Dict, Any
from src.intel.services.intel_chat.agent.async_db_connection_manager import DBConnectionManager
import mlflow
import json
import time
mlflow.openai.autolog()
from opentelemetry import context as ot_context
from mlflow.entities import SpanType
from src.common.token_cost_tracker import (
    record_llm_usage, get_total_cost, calculate_model_cost
)
import asyncio
from src.common.llm.factory import get_llm_client
from src.intel.services.intel_chat.agent.utils import get_all_file_names
from fuzzywuzzy import fuzz, process  # for fuzzy matching
import re

OPENAI_MODEL_NAME = config.OPENAI_MODEL_NAME
CONTRACT_GENERATION_DB = config.CONTRACT_GENERATION_DB
CONTRACT_INTEL_DB = config.CONTRACT_INTEL_DB
SQL_AGENT_LLM_MODEL_NAME= "gpt-4o"
SQL_AGENT_LLM_TEMPERATURE=0.1
module_name = "sql_agent"

import logging
logger = logging.getLogger("sql_agent")

@mlflow.trace(
    name="intel_chat_call_sql_agent",
    span_type=SpanType.LLM,
    attributes={
        "component": "sql_agent",
        "operation": "Metadata search",
        "module": "sql_agent.py",
        "description": "Generates SQL queries based on user questions to filter file_ids from the database.",
        "model": OPENAI_MODEL_NAME
    }
)
 
        
async def sql_agent(chat_id: int, question: str, user_id: int, org_id: int, metadata:dict):
    try:        
        current_date = time.strftime("%Y-%m-%d", time.localtime())

        if create_view_for_agent(org_id, logger):
            logger.info(_log_message("View created successfully!", "sql_agent", "sql_agent"))

        all_file_ids = set([file_ids for file_ids in metadata.keys()])
        distinct_contract_type_list = set([(data.get('metadata')).get('Contract Type') for data in metadata.values()])
        
        distinct_file_names = set([data.get('file_name') for data in metadata.values()])

        # # Extract list of party names (if present in metadata)
        # distinct_party_names = [
        #     data["metadata"].get("party_name")
        #     for data in metadata.values()
        #     if data["metadata"].get("party_name") is not None
        # ]
        
        system_prompt = f"""
        <role>
            You are an intelligent and reasoning SQL generation assistant.  
            Your job is to **understand the user's true intent**, identify what entities or filters they are referring to,  
            and then **map those to valid metadata fields/columns** to form a **syntactically correct SQL query**.
            If user intent cannot be clearly mapped to any existing field, return an **empty list** — no assumptions, no hallucinations.
        </role>
 
        <capabilities>
            - You can interpret **natural language intent** to identify which metadata field(s) are relevant.  
            - Identify the filterable parts from complex queries.  
            - You can perform **progressive narrowing** — if the user asks a compound question (e.g., “What is the salary amount received by Michael G. Lawson and under which jurisdiction his contract falls?”),  
            you first extract the **filter** (party_name = 'Michael G. Lawson') and build the SQL for fetching that record.  
            You are NOT responsible for answering “what” (like the salary), only for identifying which record(s) to fetch.
        </capabilities>
 
        <table_information>
            <table_name>file_metadata_view_{org_id}</table_name>
            <columns>                
            1. **ci_file_guid** — Unique identifier for each file.  
                → Always the only column to be selected in your output. Never use it for filtering.
            2. **file_name** — The actual uploaded file name.  
                → Use this for text-based searches when users mention a specific file name, document name, or identifier in text (e.g., “Sales 05”).
            3. **uploaded_date** — Timestamp of when the document was uploaded.  
                → Only use when user explicitly asks for filtering by upload date or upload time period.
            4. **title_of_contract** — The title or subject of the contract.  
                → Use this to find contracts by name, type, or subject mentioned in the query (e.g., “sales,” “employment,” “lease”).
            5. **scope_of_work** — Description of the scope or deliverables in the contract.  
                → Only use this when the user explicitly says “scope of work” or “statement of work.”  
                → Never use otherwise.
            6. **party_name** — Names of the entities or individuals in the contract.  
                → Use this when the user mentions a specific entity, individual or a party name.  
                → Never assume or match countries, organizations, or people unless explicitly stated.
                → When matching company names, automatically remove common corporate endings such as “Inc”, “Incorporated”, “LLC”, “Ltd”, “Limited”, “Corp”, “Corporation”, “Co”, “Company”, “PLC”, “LLP”, “Pvt”, “Private Limited”, etc.  
                → When matching individual name, remove prefixes or designations such as "Dr","Mr","Miss",etc.
                Example: If the user mentions “Google LLC” → use `LOWER(party_name) LIKE '%google%'`.
            7. **contract_type** — The type or category of the contract (e.g., Service Agreement, NDA, Purchase, Sales).  
                → This is your **primary filter** whenever the user mentions a contract type, category, or form of agreement.  
                → Always prioritize it above all other fields.
            8. **jurisdiction** — Legal region or governing law of the contract.  
                → Use this only when user mentions a place, state, or law governing the contract (e.g., “under New York law”).
            9. **contract_duration** — Duration of the contract in months.  
                → Use this for queries involving time periods, months, or years of contract validity.  
                → Convert years to months before comparing.
            10. **contract_value** — Monetary amount defined in the contract.  
                → Use this only for filtering by value (e.g., “above 1 million,” “equal to 0”).
            11. **effective_date** — The date when the contract starts or becomes valid.  
                → Use when the user talks about contracts starting, signed, executed, commenced, contract date or active from a specific date or year.
            12. **payment_due_date** — The date when payment is due.  
                → Use when the user asks for contracts with upcoming or overdue payments.
            13. **delivery_date** — The date when goods or services must be delivered.  
                → Use when user mentions delivery timelines or deadlines.
            14. **renewal_date** — The contract's renewal date.  
                → Use when user mentions contract renewals or extensions.
            15. **expiration_date** — The contract's end or expiry date.  
                → Use when user talks about contracts expiring, expired, or active within a future period.
            </columns>
        </table_information>
 
        <query_formation_rules>
            1. Figure out what the user wants to filter — contract type, dates, etc.  
            2. Map it to the right column. If you can't, **don't make stuff up** — return an empty list.  
            3. Do **not** invent columns that don't exist.
        </query_formation_rules>
 
        <absolute_priority>
            - **CONTRACT TYPE FILTERING is king**.  
            - If the user mentions any contract type, that's your #1 filter.  
            - Always use `LOWER(contract_type) LIKE '%type%'`.  
            - This rule overrides everything else. No exceptions.
            - Always expand the contract_type to include all reasonable variants in the sql query (e.g. sales , sale).  
        </absolute_priority>
        
        <file_name_matching>
            - When user mentions anything that resembles a file name:
                - partial name e.g.: User query: "Retrieve the file AMENDED AND RESTATED AGREEMENT", Actual File Name: "AMENDED AND RESTATED MASTER INTERCOMPANY.pdf"
                - shortened version e.g.: User query: "How many CMA are there?", Actual File Name: "Cross Marketing Agreement.pdf"
                - Synonym or Variant Naming e.g.: User query: "Is there a Counterfeit system development agreement in my vault?" , Actual File Name: "Counterfeit Deterrence System Development and License Agreement.docx"
                - number-based (e.g., "05", “agreement 06”) e.g.: User query: "Find the SALES 02 Contract" , Actual File Name: "SALES CONTRACT 02.docx"
                - Additional Charaters e.g.: User Query: "Identify the hubspot partner program agreement", Actula File Name: "hubspot_partner_program_agreement2.pdf"
            **Matching Rules**: Match it against the availble list of file names: 
            {distinct_file_names}
            Your matching MUST be tolerant to:
            - partial matches
            - reordered words
            - punctuation changes
            - hyphens, underscores, commas, periods
            - missing suffixes like ".docx", "pdf"
            - spacing variations
            - abbreviation differences (e.g., Inc vs Inc.)
            - missing characters
            - "&" vs "and", "_" vs space, "-" vs space
            - capitalization differences
            **NEVER reject a match due to small variations** like: missing company suffixes, punctuation, numbering differences, or abbreviations.
            - Pick the best File name match based on reasoning
            - **Only if no file is reasonably close**, use the user specified file name *as is* in the sql query.
            - If one or more matches found → use the matched entire file name in: LOWER(file_name) LIKE '%matched_value1%' OR LOWER(file_name) LIKE '%matched_value2%'
        </file_name_matching>

        <guidelines>
            - Only return `ci_file_guid`. Nothing else.  
            - Use `LOWER()` for all text matches.  
            - Use `LIKE '% term %'` for matching whole words.  
            - Combine `contract_type`, `title_of_contract`, and `file_name` with `OR`.  
            - Do **not** touch clause-level stuff like “warranties,” “obligations,” or “dispute resolution” unless explicitly asked.  
            - Include date, value filters **only** if mentioned by the user.  
            - If it's not clear what user want → empty list.  
            - If user query consist any individual or entity name use party_name to filter. e.g. 'who is john martin?', filter : party_name like '%john martin%'
            - Never use `scope_of_Work` unless the user literally says “scope of work” or “statement of work.”  
            - Be case-insensitive.  
            - No fancy formatting, no markdown, no explanation.
        </guidelines>
 
        <example_queries>
            **Here's how you should behave. Learn from them**
            <example_1>
                <user_query>List all sales contracts OR List all sales agreements OR List all sale contracts OR List all sale agreements</user_query>
                <sql_response>SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE LOWER(contract_type) LIKE '%sale%' OR LOWER(contract_type) LIKE '%sales%' OR LOWER(title_of_contract) LIKE '% sales %' OR LOWER(title_of_contract) LIKE '% sale %' OR LOWER(file_name) LIKE '% sales %' OR LOWER(file_name) LIKE '% sale %'</sql_response>
            </example_1>
            <example_2>
                <user_query>List all contracts with renewal notice period greater than 30 days and summarize their termination for cause clauses.</user_query>
                <note>No metadata field can be used to filter the contracts based on renewal notice period. Hence, return an empty list.</note>
                <sql_response>No metadata field available; return empty list.</sql_response>
            </example_2>
            <example_3>
                <user_query>Find the documents where atleast one of the parties is from China?</user_query>
                <note>No metadata field can be used to filter the contracts based on user intent. Hence, return an empty list.</note>
                <sql_response>No metadata field available; return empty list.</sql_response>
            </example_3>
            <example_4>
                <user_query>What ISO certifications are mentioned across the sales contracts (05, 06, 07)?</user_query>
                <note>For queries where user specified the file name, use the file_name column only to filter the results using LIKE.</note>
                <sql_response>SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE LOWER(file_name) LIKE '%sales contract 05%' OR LOWER(file_name) LIKE '%sales contract 06%' OR LOWER(file_name) LIKE '%sales contract 07%'</sql_response>
            </example_4>
            <example_5>
                <user_query>Filter all active contracts OR Are there any active contracts? OR List all active agreements OR List all active contracts</user_query>
                <sql_response>SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE effective_date <= CURRENT_DATE AND (expiration_date >= CURRENT_DATE OR expiration_date IS NULL)</sql_response>
            </example_5>
            <example_6>
                <user_query>Filter contracts with payment due within 30 days OR Filter contracts with payment due in next 30 days</user_query>
                <sql_response>SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE payment_due_date BETWEEN CURRENT_DATE AND DATE_ADD(CURRENT_DATE, INTERVAL 30 DAY)</sql_response>
            </example_6>
            <example_7>
                <user_query>What is the compensation recieved in employment contract</user_query>
                <sql_response>SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE LOWER(file_name) LIKE '%employment%' OR LOWER(file_name) LIKE '%employment%' OR LOWER(file_name) LIKE '%employment%'</sql_response>
            </example_7>
        </example_queries>
 
        <distinct_contract_types>
            The following are the distinct contract types available in the table. Use them to filter contracts based on user query and the distinct contract types
            {distinct_contract_type_list}
        </distinct_contract_types>  

        <notes>
            - Prioritize semantic accuracy and user intent.
            - Use OR to combine contract type and title filters unless user says “AND.”  
            - “Agreement,” “contract,” “document,” “file” — treat them all the same.  
            - Today's date: {current_date}.  
            - Handle synonyms, typos, variations smartly and be precise.
        </notes>
 
        """

        user_prompt = f"""
        <instruction>
            Generate an SQL query to filter ci_file_guid from 'file_metadata_view_{org_id}' based on the following user question:
        </instruction>
        <user_question>
        {question}
        </user_question>
        <output_format>
        - Don't add markdown or code blocks like ```sql or ```json.  
        - Don't explain anything. Just return the SQL query or an empty list.  
        - Be concise, accurate, and obey every rule above.
        </output_format>
        """


        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "ci_file_guid_sql_generation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "sql_query": {
                            "type": "string",
                            "description": f"A syntactically valid SQL SELECT query that retrieves 'ci_file_guid' from the 'file_metadata_view_{org_id}' table, filtered according to the user's question."
                        }
                    },
                    "required": ["sql_query"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

        client = get_llm_client(async_mode=True)


        llm_response = await client.chat.completions.create(
            model=SQL_AGENT_LLM_MODEL_NAME,
            temperature=SQL_AGENT_LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=json_schema
        )

        response = llm_response.choices[0].message.content
        
        if response:

            parsed_response = json.loads(response)

            sql_query = parsed_response.get("sql_query")

            if sql_query and "No metadata field available" not in sql_query:
                logger.info(_log_message(f"Generated SQL Query: {sql_query}", "sql_agent", module_name))

                # if 'WHERE' in sql_query and previous_file_ids:
                #     sql_query += f" AND ci_file_guid IN {tuple(previous_file_ids)}"
                # elif 'WHERE' not in sql_query and previous_file_ids:
                #     sql_query += f" WHERE ci_file_guid IN {tuple(previous_file_ids)}"


                # Record the token usage for cost tracking
                record_llm_usage(
                    chat_id=chat_id,
                    model_name=SQL_AGENT_LLM_MODEL_NAME,
                    input=llm_response.usage.prompt_tokens,
                    output=llm_response.usage.completion_tokens,
                    cache=llm_response.usage.prompt_tokens_details.cached_tokens
                )

                token_cost_summary = calculate_model_cost(
                    model_name=SQL_AGENT_LLM_MODEL_NAME,
                    input=llm_response.usage.prompt_tokens,
                    output=llm_response.usage.completion_tokens,
                    cache=llm_response.usage.prompt_tokens_details.cached_tokens
                )

                token_usage_summary = {
                    "input_tokens": llm_response.usage.prompt_tokens,
                    "output_tokens": llm_response.usage.completion_tokens,
                    "cache_tokens": llm_response.usage.prompt_tokens_details.cached_tokens,
                    "uncached_tokens": llm_response.usage.prompt_tokens - llm_response.usage.prompt_tokens_details.cached_tokens,
                    "cost_summary": token_cost_summary
                
                }

                span = mlflow.get_current_active_span()
                span.set_attributes({
                    "model": SQL_AGENT_LLM_MODEL_NAME,
                    "chat_id": chat_id,
                    "user_query": question,  # Assuming client_id is not used here
                    "user_id": user_id,
                    "org_id": org_id,
                    "token_usage_cost": token_usage_summary,
                })
            else:
                logger.warning(_log_message("No 'sql_query' found in the response.", "sql_agent", module_name))
                return [], False
            
            file_ids = execute_sql_query(sql_query, org_id, logger)

            # --- FUZZY MATCH FALLBACK for file name cases only---
            if (not file_ids) and ("file_name" in sql_query.lower()):
                logger.info(_log_message("No results from SQL; initiating fuzzy match fallback for file_name.", "sql_agent", module_name))
                matched_file_ids = set()
                # Step 1: Extract the term(s) used for LIKE filtering on file_name
                # Example: file_name LIKE '%Amended and Restated%'
                # with this regex we are only focusing on extracting the filename that user has specified 
                match_terms = re.findall(r"(?i)(?:lower\s*\()?file_name\)?\s+like\s*'%([^']+)%'", sql_query, re.IGNORECASE)

                if not match_terms:
                    logger.warning(_log_message("No file_name LIKE term found in SQL for fuzzy matching.", "sql_agent", module_name))
                    return [], True

                 # Use first or all terms (handle OR clauses)
                fuzzy_terms = [term.strip().lower() for term in match_terms]
                logger.info(_log_message(f"Extracted fuzzy match terms: {fuzzy_terms}", "sql_agent", module_name))

                # Step 1: Fetch all available file names for this org_id
                all_files = await get_all_file_names(list(all_file_ids))  # returns list of tuples [(file_name, ci_file_guid), ...]

                if not all_files:
                    logger.warning(_log_message("No file names found in DB for fuzzy matching.", "sql_agent", module_name))
                    return [], True

                # Step 2: Extract file names and perform fuzzy matching
                file_name_to_guid = {name.lower(): guid for guid, name in all_files.items()}
                file_names = list(file_name_to_guid)

                threshold = 80  # fuzzy similarity cutoff
                # threshold changed from 70 to 80 because of this failure case : http://44.199.74.243:8000/#/experiments/3?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D&compareRunsMode=TRACES&selectedTraceId=7b3cecdbc0c34c4eb02412ae9abda644
                # sql query contained LOWER(file_name) LIKE '%zynerba pharmaceuticals%' AND LOWER(file_name) LIKE '%merger agreement%', number of false positives are too many here
                # with zynerbar pharmaceuticals there are too many false positives
                
                # Step 3: Fuzzy match each LIKE term against available file names
                for term in fuzzy_terms:
                    matches = process.extractBests(term, file_names, scorer=fuzz.token_set_ratio, limit=1000)
                    top_matches = [fname for fname, score in matches if score >= threshold]

                    for fname in top_matches:
                        matched_file_ids.add(file_name_to_guid[fname])

                    logger.info(_log_message(f"Fuzzy matches for '{term}': {top_matches} {matched_file_ids}", "sql_agent", module_name))

                if matched_file_ids:
                    return list(matched_file_ids), True
                else:
                    logger.info(_log_message("No fuzzy matches found for file_name terms.", "sql_agent", module_name))
                    return [], False
        else:
            logger.warning(_log_message("No response received from the model.", "sql_agent", module_name))
            file_ids = [], False

        if file_ids:
            logger.info(_log_message(f"SQL Agent Response: {sql_query}", "sql_agent", module_name))
            return file_ids, True
        else:
            logger.warning(_log_message(f"No file_ids found for the given question - {question}", "sql_agent", module_name))
            # if the sql query contains file_name, contract_type based filtering and there are no results filtered by sql, we will directly return the no results response because hybrid gives a lot of false positives 
            # especially for file name specific user queries so below condition will skip hybrid for file name and contract type and do hybrd for other cases
            if "file_name" in sql_query.lower() or "contract_type" in sql_query.lower():
                return [], True
            else:
                # going for hybrid for remaining metadata fields specific query (just as fallback for sql failure)
                return [], False
        
    except Exception as e:
        logger.error(_log_message(f"Error in sql_agent: {str(e)}", "sql_agent", module_name))
        return [], False
    
@mlflow.trace(name="Check File ID Present")
async def is_file_id_present(file_ids: List[str], only_contract, tag_ids, logger) -> List[str]:
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
        async with DBConnectionManager(CONTRACT_INTEL_DB) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT file_temp_id 
                    FROM file_tags 
                    WHERE tag_id IN %s
                    """,
                    (tuple(tag_ids),)
                )
                file_temp_ids = [row["file_temp_id"] for row in await cursor.fetchall()]  # collect as list

        if not file_temp_ids:
            logger.info(_log_message("No temp file IDs found for given tags.", "is_file_id_present", module_name))
            return []

        logger.info(_log_message(f"Temp file IDs from tags: {file_temp_ids}", "is_file_id_present", module_name))

        file_ids = [fid for fid in file_ids if fid in file_temp_ids] if file_ids else []

        if not file_ids:
            logger.info(_log_message("No matching file IDs found after filtering with tags.", "is_file_id_present", module_name))
            return []


        async with DBConnectionManager(CONTRACT_INTEL_DB) as connection:
            logger.info(_log_message(f"Checking file IDs in repository | File IDs: {file_ids}", "is_file_id_present", module_name))
            
            async with connection.cursor() as cursor:
                query = """
                SELECT ci_file_guid 
                FROM files 
                WHERE ci_file_guid IN %s 
                AND is_archived IS NULL;
                """ if not only_contract else """
                SELECT ci_file_guid 
                FROM files 
                WHERE ci_file_guid IN %s 
                AND is_archived IS NULL AND is_template IN (0, 2) AND name not like "temp/%%";
                """
                await cursor.execute(query, (tuple(file_ids),))
                records = await cursor.fetchall()

                matching_file_ids = [record["ci_file_guid"] for record in records]

                execution_time = time.perf_counter() - start_time
                logger.info(_log_message(f"Found {len(matching_file_ids)} matching file IDs. Execution time: {execution_time:.4f} seconds.", "is_file_id_present", module_name))
                return matching_file_ids

    except Exception as e:
        logger.error(_log_message(f"Error in is_file_id_present: {e}", "is_file_id_present", module_name))
        return []
