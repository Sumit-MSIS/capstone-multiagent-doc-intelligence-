import time
from src.config.base_config import config
from src.common.conceptual_search.retriever.hybrid_search_retrieval import get_context_from_pinecone
from src.common.conceptual_search.retriever.reranking import rerank_documents
import concurrent.futures
from itertools import islice
from typing import List, Dict
from src.common.logger import request_logger, _log_message, flush_all_logs, sanitize_log_stream_name
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.common.conceptual_search.retriever.sql_retriever.sql_agent import sql_agent_template
from src.common.sql_retriever.sql_agent import sql_agent
from src.common.conceptual_search.llm_call import open_ai_llm_call
import traceback
from src.common.conceptual_search.utils import is_file_id_present, get_similar_files
from src.common.conceptual_search.utils import get_metadata
import json
import mlflow
from collections import defaultdict
from src.common.llm_status_handler.status_handler import set_conceptual_search_results
from src.common.conceptual_search.utils import rephrase_search_text
from openai import OpenAI
import uuid
from src.common.llm.factory import get_llm_client
# mlflow.langchain.autolog()
mlflow.openai.autolog()
from opentelemetry import context as ot_context
import orjson   

# OpenAI API key (make sure to handle this securely)
PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_INDEX_NAME_CHUNK_SUMMARY = config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX
DOCUMENT_METADATA_INDEX = config.DOCUMENT_METADATA_INDEX
DOCUMENT_SUMMARY_INDEX = config.DOCUMENT_SUMMARY_INDEX
DOCUMENT_SUMMARY_AND_CHUNK_INDEX = config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX
OPENAI_MODEL_NAME = config.OPENAI_MODEL_NAME
module_name = "filter_contracts.py"

client = get_llm_client(async_mode=False)

def submit_with_context(executor, fn, *args, **kwargs):
    # Capture the current OpenTelemetry context (which includes the parent span)
    parent_ctx = ot_context.get_current()
    def wrapped():
        token = ot_context.attach(parent_ctx)
        try:
            return fn(*args, **kwargs)
        finally:
            ot_context.detach(token)
    return executor.submit(wrapped)



@mlflow.trace(name="Rephrase Contract Creation Search Text")
def rephrase_contract_creation_search_text(search_text):
    system_prompt = """
You are an expert assistant that rephrases user queries to optimize semantic search and template retrieval for contract drafting tasks.

Your objective is to transform the user’s request into a **precise, context-aware phrase** that clearly reflects the intended **contract type** and **purpose**, helping to match the most relevant contract templates (e.g., Lease Agreement, NDA, Service Contract, Loan Agreement, etc.).

**Guidelines:**

1. **Preserve the user’s drafting intent** — identify the primary type of contract they intend to create.
2. **Explicitly mention the contract or document type** in the rephrased output (e.g., “Lease Agreement”, “Service Agreement”).
3. **Emphasize the overarching contract**, not individual clauses (e.g., prioritize “Rental Agreement” over “Confidentiality Clause” unless only a clause is requested).
4. **Avoid transforming to unrelated contract types** (e.g., do not turn a rental with confidentiality terms into an NDA).
5. **Use declarative, formal phrasing** that resembles how contract templates are titled and filed.
6. **Do not phrase queries as questions** (e.g., avoid “Can you draft...”, “How do I write...”).
7. If the contract type is **unspecified or ambiguous**, use broad formal descriptors like “Business Agreement” or “Contractual Agreement”.
8. When multiple plausible contract types exist, **prioritize the most specific match** based on context (e.g., choose "Commercial Lease Agreement" over "Rental Agreement" if office space is involved).

**Examples:**

- User: "Create a rental agreement with the confidential terms"  
  → Rephrased: "a 'Rental Agreement' with confidentiality provisions"

- User: "Draft a confidentiality contract between two companies"  
  → Rephrased: "'Mutual Non-Disclosure Agreement (NDA)' between corporate entities"

- User: "I want to create a contract for renting out office space"  
  → Rephrased: "a commercial 'Lease Agreement' for office rental"

- User: "Make a contract for providing digital marketing services"  
  → Rephrased: "'Service Agreement' for digital marketing services"

- User: "Loan contract for $10,000 with repayment terms"  
  → Rephrased: "'Loan Agreement' specifying repayment terms for $10,000"

- User: "I need a template for buying and selling equipment"  
  → Rephrased: "'Sales Agreement' for purchase and sale of equipment"

Only output the rephrased query.
    """

    user_prompt = f"User Query: {search_text.lower()}"

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,  # Set temperature to 0 for deterministic output
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    llm_answer = response.choices[0].message.content
    return llm_answer


# Helper function to split list into batches of size n
@mlflow.trace(name="Batch List")
def batch_list(lst: List[Dict], batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# Wrapper for validation_call to handle one batch
@mlflow.trace(name="Process Batch")
def process_batch(batch, user_query, logger):
    try:
        return validation_call(batch, user_query, logger).get("file_ids", []) or []
    except Exception as e:
        logger.error(f"Validation call failed for batch: {e}")
        return []

@mlflow.trace(name="Search Contracts")
def search_contracts(org_id, user_id, user_query, tag_ids, only_contract, logger):
    """
    Searches for relevant contracts based on user query.

    :param search_text: The query text for searching contracts.
    :param logger: Logger instance.
    :return: Dictionary with a list of relevant file IDs.
    """
    try:
        start_time = time.time()
        logger.info(_log_message(f"Starting contract search for: {user_query}", "search_contracts", module_name))

        if not only_contract:
            try:
                logger.info(_log_message("Rephrasing search text for contract creation", "search_contracts", module_name))
                sql_file_ids = sql_agent_template(user_query, user_id, org_id, logger) or []
                if not sql_file_ids:
                    logger.info(_log_message(f"No SQL file IDs found for query: {user_query}", "search_contracts", module_name))
                    return {"success": True, "error": "", "data": {"fileids": []}}
                final_file_ids = is_file_id_present(sql_file_ids, only_contract, tag_ids, logger)
                logger.info(_log_message(f"Final Filtered File IDs: {final_file_ids}", "search_contracts", module_name))
                return {
                    "success": True,
                    "error": "",
                    "data": {"fileids": [{"id": file_id} for file_id in final_file_ids]}
                }

            except Exception as e:
                logger.error(_log_message(f"SQL Agent error: {e}", "fetch_sql_agent", module_name))
                return []
            
            
        else:
            keyword_search_query, semantic_search_query, reranking_query = rephrase_search_text(user_query)
            logger.info(_log_message(f"Keyword Search Query: {keyword_search_query}", "search_contracts", module_name))
            logger.info(_log_message(f"Semantic Search Query: {semantic_search_query}", "search_contracts", module_name))
            logger.info(_log_message(f"Reranking Query: {reranking_query}", "search_contracts", module_name))
            # search_text = user_query

            # logger.info(_log_message(f"Rephrased Search Text: {search_text}", 
            #                         "filter_relevant_file_ids", module_name))

            try:
                sql_file_ids = sql_agent(user_query, user_id, org_id, logger) or []
                sql_file_ids = is_file_id_present(sql_file_ids, only_contract, tag_ids, logger)
                logger.info(_log_message(f"Final Filtered SQL File IDs for conceptual search: {sql_file_ids}", "search_contracts", module_name))
                
            except Exception as e:
                    logger.error(_log_message(f"SQL Agent error: {e}", "fetch_sql_agent", module_name))
                    return []

            # if sql_file_ids and len(sql_file_ids) > 0:
            #     return {
            #     "success": True,
            #     "error": "",
            #     "data": {"fileids": [{"id": file_id} for file_id in sql_file_ids]}
            # }
            
            @mlflow.trace(name="Fetch Similar Files")
            def fetch_similar_files():
                try:
                    if len(user_query.split()) > 5:
                        return get_similar_files(user_query, org_id, logger) or []
                    return []
                except Exception as e:
                    logger.error(_log_message(f"Similarity search error: {e}", "fetch_similar_files", module_name))
                    return []
        

            @mlflow.trace(name="Fetch Pinecone")
            def fetch_pinecone():
                try:
                    pinecone_indexes = {
                        "contract_summary_and_chunk": DOCUMENT_SUMMARY_AND_CHUNK_INDEX
                    }

                    score_threshold = 0.30  # Adjust as needed
                    responses = {}

                    @mlflow.trace(name="Fetch Pinecone - Fetch Response")
                    def fetch_response(index_name, index_val, file_id=None, force_top_k=False):
                        # custom_filter = {"file_id": {"$in": [file_id]}} if file_id else {}
                        custom_filter = {
                            "$and": [
                                {"tag_ids": {"$in": tag_ids}},
                                {"file_id": {"$in": [file_id]}}
                            ]
                        } if tag_ids and file_id else {"tag_ids": {"$in": tag_ids}}

                        # If force_top_k=True, we always do top_k=80 (no filter)
                        top_k = 80 if (force_top_k or not file_id) else 10
                        file_id_list = [file_id] if file_id else []

                        return get_context_from_pinecone(
                            index_val, PINECONE_API_KEY, custom_filter, top_k,
                            keyword_search_query, semantic_search_query, reranking_query,
                            user_query, file_id_list, user_id, org_id, tag_ids, logger
                        )

                    with ThreadPoolExecutor() as executor:
                        futures = {}
                        for index_name, index_val in pinecone_indexes.items():
                            if sql_file_ids:
                                if len(sql_file_ids) <= 10:
                                    ### NEW LOGIC: First search with SQL IDs
                                    for fid in sql_file_ids:
                                        future = submit_with_context(executor, fetch_response, index_name, index_val, fid)
                                        futures[future] = f"{index_name}_{fid}"

                                    ### NEW LOGIC: Second search with top_k=80 without file_id
                                    future = submit_with_context(executor, fetch_response, index_name, index_val, None, True)
                                    futures[future] = f"{index_name}_extra"
                                else:
                                #     ### Existing logic: Only per-file_id calls
                                    for fid in sql_file_ids:
                                        future = submit_with_context(executor, fetch_response, index_name, index_val, fid)
                                        futures[future] = f"{index_name}_{fid}"
                            else:
                                # Single call when no file_ids
                                future = submit_with_context(executor, fetch_response, index_name, index_val, None)
                                futures[future] = index_name

                        for future in as_completed(futures):
                            name = futures[future]
                            try:
                                response = future.result()
                                if response:
                                    responses[name] = response
                                    logger.info(_log_message(f"Received response for {name}", "fetch_file_id", module_name))
                            except Exception as e:
                                logger.error(_log_message(f"Error fetching response for {name}: {e}", "fetch_file_id", module_name))

                    # Continue with your existing score-based filtering logic below...
                    preliminary_file_ids = set()
                    for response in responses.values():
                        if response and "matches" in response:
                            for match in response["matches"]:
                                if "metadata" in match and "file_id" in match["metadata"] and "score" in match:
                                    preliminary_file_ids.add(match["metadata"]["file_id"])

                    if not preliminary_file_ids:
                        logger.info(_log_message("No preliminary file IDs found in Pinecone responses.", "fetch_pinecone", module_name))
                        return sql_file_ids

                    filtered_preliminary_file_ids = is_file_id_present(list(preliminary_file_ids), only_contract, tag_ids, logger)
                    logger.info(_log_message(f"Final Filtered filtered_preliminary_file_ids for conceptual search: {filtered_preliminary_file_ids}", "search_contracts", module_name))
                    metadata_dict = get_metadata(filtered_preliminary_file_ids, user_id, org_id, logger)
                    
                    # Step 1: Collect all valid matches into a temporary dictionary grouped by file_id
                    matches_by_file_id = defaultdict(list)

                    # for response in responses.values():
                    #     if response and "matches" in response:
                    #         for match in response["matches"]:
                    #             metadata = match.get("metadata", {})
                    #             file_id = metadata.get("file_id")
                    #             score = match.get("score", 0)

                    #             logger.debug(_log_message(f"File ID: {file_id}, Score: {score}", "fetch_pinecone", module_name))

                    #             if (file_id in filtered_preliminary_file_ids and score >= score_threshold) or (file_id in sql_file_ids):
                    #                 matches_by_file_id[file_id].append(match)


                    for response in responses.values():
                        if response and "matches" in response:
                            for match in response["matches"]:
                                metadata = match.get("metadata", {})
                                file_id = metadata.get("file_id")
                                score = match.get("score", 0)

                                logger.debug(
                                    _log_message(
                                        f"File ID: {file_id}, Score: {score}", 
                                        "fetch_pinecone", 
                                        module_name
                                    )
                                )

                                if ((file_id in filtered_preliminary_file_ids and score >= score_threshold) 
                                    or (file_id in sql_file_ids)):

                                    # Ensure list exists
                                    if file_id not in matches_by_file_id:
                                        matches_by_file_id[file_id] = []

                                    # Avoid duplicates (based on match ID or chunk text)
                                    existing_ids = {m["id"] for m in matches_by_file_id[file_id]}
                                    if match["id"] not in existing_ids:
                                        matches_by_file_id[file_id].append(match)


                    preliminary_matches = []
                    seen_file_ids = set()

                    for file_id, matches in matches_by_file_id.items():
                        if file_id in seen_file_ids:
                            continue  # skip duplicate file_id
                        seen_file_ids.add(file_id)
                        # Sort and pick up to 3 matches
                        top_matches = sorted(matches, key=lambda x: x["score"], reverse=True)[:5]

                        # Extract file-level metadata once
                        file_metadata = metadata_dict.get(file_id, {})

                        # Collect chunks + scores
                        chunks = []
                        scores = []
                        file_name = ""

                        for match in top_matches:
                            metadata = match["metadata"]
                            file_name = metadata.get("file_name", file_name)  # pick any one (same across chunks)
                            chunks.append(metadata.get("text", ""))
                            scores.append(match["score"])

                        preliminary_matches.append({
                            "file_id": file_id,
                            "file_name": file_name,
                            "rag_score": max(scores) if scores else 0,  # max score among chunks
                            "chunks": chunks,                           # list of up to 3 chunks
                            "file_metadata": file_metadata
                        })


                    logger.info(_log_message(f"Preliminary Matches Before Sorting: {orjson.dumps(preliminary_matches).decode()}", "fetch_pinecone", module_name))
                    # sorted_preliminary_matches = sorted(
                    #     preliminary_matches,
                    #     key=lambda x: x["score"],
                    #     reverse=True
                    # )

                    # logger.info(_log_message(f"Preliminary Matches: {orjson.dumps(sorted_preliminary_matches).decode()}", "fetch_pinecone", module_name))

                    # sorted_file_ids = validation_call(sorted_preliminary_matches, user_query, logger).get("file_ids", []) or []


                    # Run validation in parallel
                    unique_file_ids = set()

                    with ThreadPoolExecutor() as executor:
                        futures = {
                            submit_with_context(executor, process_batch, batch, user_query, logger)
                            for batch in batch_list(preliminary_matches, 1)
                        }

                        for future in concurrent.futures.as_completed(futures):
                            result = future.result()
                            unique_file_ids.update(result)

                    # Final deduplicated and sorted list
                    sorted_file_ids = list(unique_file_ids)
                    
                    logger.info(_log_message(f"Sorted File IDs: {sorted_file_ids}", "fetch_pinecone", module_name))
                    return sorted_file_ids
                except Exception as e:
                    logger.error(_log_message(f"Pinecone retrieval error: {e}\n{traceback.format_exc()}", "search_csontract", module_name))
                    return []

            # Fetch data concurrently
            with ThreadPoolExecutor() as executor:
                # future_sql = submit_with_context(executor, fetch_sql_agent)
                future_similar_files = submit_with_context(executor, fetch_similar_files)
                future_pinecone = submit_with_context(executor, fetch_pinecone)

                # Collect results
                # sql_file_ids = future_sql.result() or []
                similar_file_ids = future_similar_files.result() or []
                pinecone_file_ids = future_pinecone.result() or []  # This is a sorted list

            # Merge results, removing duplicates
            logger.info(f"Types: similar_file_ids={type(similar_file_ids)},pinecone_file_ids={type(pinecone_file_ids)}")
            # logger.info(_log_message(f"SQL File IDs: {sql_file_ids}", "search_contracts", module_name))
            logger.info(_log_message(f"Similar File IDs: {similar_file_ids}", "search_contracts", module_name))
            logger.info(_log_message(f"Pinecone File IDs: {pinecone_file_ids}", "search_contracts", module_name))
            # all_file_ids = set(sql_file_ids) | set(similar_file_ids) | set(pinecone_file_ids)
            # all_file_ids = set(similar_file_ids) | set(pinecone_file_ids) | set(sql_file_ids)
            # all_file_ids = list(dict.fromkeys(similar_file_ids + pinecone_file_ids + sql_file_ids + toc_file_ids))  # Maintain order and remove duplicates
            # all_file_ids = list(dict.fromkeys(similar_file_ids + pinecone_file_ids + sql_file_ids))  # Maintain order and remove duplicates
            
            all_file_ids = list(dict.fromkeys(similar_file_ids + pinecone_file_ids))  # Maintain order and remove duplicates

            # all_file_ids = list(all_file_ids)  # Convert back to list

            logger.info(_log_message(f"Combined File IDs: {all_file_ids}", "search_contracts", module_name))

            if not all_file_ids:
                logger.info(_log_message(f"No Files Found for search text: {user_query}", "search_contracts", module_name))
                return {"success": True, "error": "", "data": {"fileids": []}}

            # Validate and filter file IDs
            final_file_ids = is_file_id_present(all_file_ids, only_contract, tag_ids, logger)
            logger.info(_log_message(f"Final Filtered File IDs: {final_file_ids}", "search_contracts", module_name))

            logger.info(_log_message(f"Total time taken: {time.time() - start_time:.2f} seconds", "search_contracts", module_name))

            return {
                "success": True,
                "error": "",
                "data": {"fileids": [{"id": file_id} for file_id in final_file_ids]}
            }
    except Exception as e:
        logger.error(_log_message(f"Error in search_contracts: {e}\n{traceback.format_exc()}", "search_contracts", module_name))
        return {"success": False, "error": str(e), "data": {"fileids": []}}


@mlflow.trace(name="Validation Call")
def validation_call(preliminary_matches, user_query, logger):
    logger.info(_log_message("Starting validation call with GPT-4o", "validation_call", module_name))
    if not preliminary_matches:
        logger.warning(_log_message("No preliminary matches found, returning empty file_ids", "validation_call", module_name))
        return {"file_ids": []}


    current_date = time.strftime("%Y-%m-%d", time.localtime())

    # system_prompt = """
    # You are ChatGPT, a large language model trained by OpenAI.
    # Knowledge cutoff: 2024-06
    # Current date: {current_date}

    # ### Role
    # You act as a **Contract Analysis Engine**.
    # Your task is to determine which files are **relevant** to a user's query, based on file metadata and content snippets.

    # ### Deterministic Evaluation Rules
    # Relevance must be judged with high precision and recall — do not miss relevant files, but never include false positives.
    
    # A file is **relevant** if:
    # 1. Contract Type Filtering (when user specifies a contract type):
    # - If the user explicitly specifies a contract type (e.g., "NDA", "Reseller Agreement", "Service Agreement"), then the file’s Contract Type field in metadata must either:
    #    a) Exactly match the requested type, OR
    #    b) Belong to the same type family:
    #       - This means the file’s Contract Type contains the requested type as a distinct substring 
    #         (e.g., "Service Agreement" matches "Merchant Services Agreement", "Management Services Agreement", 
    #         "Professional Services Agreement").
    #       - Recognized synonyms explicitly listed in the query also qualify as valid matches.

    # - **Do NOT include documents where the Contract Type differs significantly**, even if related clauses are mentioned in the text. **The metadata field takes precedence.**
    #     Examples:
    #     - Query = "NDA" → Match if Contract Type = "NDA" or "Non-Disclosure Agreement".
    #         Exclude "MSA" or "Distribution Agreement" even if they mention NDA clauses.
    #     - Query = "Service Agreement" → Match if Contract Type = "Service Agreement", "Merchant Services Agreement", "Management Services Agreement", or "Professional Services Agreement".
    #     - Query = "Reseller Agreement" → Match if Contract Type = "Reseller Agreement" or "Authorized Reseller Agreement", 
    #         but exclude "Distribution Agreement" unless explicitly listed as a synonym in the query.
 
    # 2. If the user does **not** specify a contract type, then **skip contract type filtering** and evaluate only based on **clauses, keywords, amounts, or conditions**.
    # 3. The file must contain all explicitly required clauses/keywords/conditions.
    # 4. It must meet any numerical thresholds or durations if specified.
 
    # ### Strictness rules:
    # - Contract Type strictness applies **only if type is explicitly in query**.
    # - When type is not in query → any contract type can qualify if the conditions are satisfied.
    # - Do not conflate contract types unless the query lists them.
    # - Synonyms/variations are allowed only for condition/keyword matching, not contract types.

    # ### Keyword/Clause Matching Rules
    # - Required clauses/keywords must appear **verbatim** in the document text OR match an explicitly defined synonym provided in the query.  
    # - Mentions in unrelated or generic contexts do **not** qualify. 
    # - No guessing: if the presence/absence of a condition is uncertain or ambiguous → treat as false.
    
    # ### Date Duration Calculation & Comparison Rules
    # When evaluating a user query that specifies a duration (in years, months, or days) and comparing it with contract dates (Effective Date and Expiration Date):
    # 1. Identify the Unit from the Query
    # - If the query mentions years, compute the duration in absolute years, including fractions.
    #     Example: 
    #     Query: "Which agreement has a contract duration of more than 3 years"
    #     Calculation: Start 2027-02-15, End 2030-08-15 → 3.5 years.
    #     Relevancy: "3.5 years is more than user expected threshold 3 years, hence condition is satisfied"
    # - If the query mentions months, compute the duration in calendar months.
    #     Example: 
    #     Query: "Find contracts that lasts atleast 42 months"
    #     Calculation: Start 2027-02-15, End 2030-08-15 → 40 months.
    #     Relevancy: "40 months is less than 42 months, hence condition is not satisfied"
    # - If the query mentions days, compute the duration in calendar days.
    #     Example: 
    #     Query: "Which contracts have a duration between 30 and 90 days?"
    #     Calculation: Start 2027-03-15, End 2027-04-14 → 30 days.
    #     Relevancy: "30 days lie between 30 and 90 days as specified in user query, hence condition is satisfied"
    # 2. Validation Requirement
    # - Always state both: 
    #     The computed duration (in the correct unit).
    #     Why it does or does not satisfy the user’s condition.
    
    # ### Processing Steps
    # For each file:
    # 1. Parse the query into:
    #    - Target contract type(s) (if specified).
    #    - Required clauses/keywords/conditions.
    #    - Numerical thresholds/durations.
    #    - Logical operators (AND/OR).
    # 2. Evaluate step by step:
    #    - If type specified → check ONLY if Contract Type in metadata matches. Ignore mentions in text.  
    #    - If type not specified → skip type filtering.  
    #    - Then check required clauses, keywords, thresholds.
    # 3. Decision rule:
    #    - Mark relevant = true only if (a) contract type matches when required, and (b) all conditions are satisfied.
    #    - If contract type is specified in the query and the file's Contract Type in metadata does not match → is_relevant_to_user_query = false, regardless of content mentions.
    
    # 4. Special Case – Vault-Wide Queries:
    #     - If the user query is about the number of files or listing all files (e.g., "how many files are there in my vault", "list all contracts"), then ignore contract type and condition filtering.  
    #     - In this case, mark all files as relevant (is_relevant_to_user_query = true).
    # ## Output Format
    # Return strictly in valid JSON:

    # {{"results": [
    #     {{
    #     "file_id": "file1",
    #     "document_type_match": true,
    #     "conditions_checked": {{
    #         "NDA clause": true,
    #         "minimum term 2 years": false
    #     }},
    #     "explanation": "File matches NDA type and includes NDA clause but is missing 2-year minimum term.",
    #     "is_relevant_to_user_query": false
    #     }},
    #     {{
    #     "file_id": "file2",
    #     "document_type_match": true,
    #     "conditions_checked": {{
    #         "contract value $0": true
    #     }},
    #     "explanation": "Contract has no specified value, which is treated as $0 per query condition.",
    #     "is_relevant_to_user_query": true
    #     }}
    # ]}}

    # ## Examples
    # Example 1:
    #   - Query: "Find all NDAs with at least 2 years duration"
    #   - Rule: Contract Type must be NDA. Must also contain "minimum term 2 years".
    #   - If file metadata says "Distribution Agreement" but text contains "2 years NDA clause" → NOT relevant.

    # Example 2:
    #   - Query: "Contracts mentioning arbitration clause"
    #   - Rule: No contract type specified, so ignore type. Only check presence of "arbitration clause".
    #   - If file metadata says "MSA" and text contains "arbitration clause" → relevant.

    # Example 3:
    #   - Query: "Reseller agreements with contract value > $1M"
    #   - Rule: Contract Type must be "Reseller Agreement". Condition: contract value must exceed $1,000,000.
    #   - If file type = Reseller Agreement but value = $900K → NOT relevant.
    #   - If file type = Reseller Agreement but value = $1M → NOT relevant

    # Example 4:
    #     - Query: "Summarise the sales contract"
    #     - Rule: Contract Type must be Sales. No conditions specified.
    #     - If file metadata says "Sales" → relevant.

    # Example 5:
    #     - Query: "Provide the delivery schedule of 'Sales Contract 01' contract"
    #     - Rule: Contract Type must be "Sales". Condition: file name must include as is "Sales Contract 01".
    #     - If file metadata says "Sales" and file name is "Sales Contract 01" → relevant.
    #     - If file name is "Sales Contract 02" → NOT relevant.
    #     - If file name is "SALES Agreement Template 01" → NOT relevant.

    # Example 6:
    #     - Query: "Select the contract which has 'SALES CONTRACT' in its name" OR "There are totally 9 contracts named as 'SALES CONTRACT'. Please check and list all the 9 files name"
    #     - Rule: No contract type specified, so ignore type. Condition: file name must include as is "SALES CONTRACT".
    #     - If file name is "SALES CONTRACT 01" → relevant.
    #     - If file name is "PURCHASE AGREEMENT 01" → NOT relevant.
    #     - If file name is "Sales Contract 02" → relevant (case insensitive match).
    #     - If file name is "SALES Demo 01" → NOT relevant.
    #     - If file name is "SALES Agreement Template 01" → NOT relevant.


    # Example 7:
    #     - Query: "which investment management contracts have provisions for compensation and expenses" OR "can you check my investment management contracts and come up with the best clause to handle compenstaion and expenses"
    #     - Rule: Contract Type must be "Investment" or its related type only. No conditions required".
    #     - If file type = "Investment Agreement" → relevant.
    #     - If file type = "Limited Partnership" → NOT relevant.
    #     - If file type = "MSA" → NOT relevant.
    
    # Example 8:
    #     - Query: "in the reseller contract involving litigation dynamics, who's responsible for customer support, repairs, and handling returns"
    #     - Rule: Contract Type must be Reseller Agreement (or synonym Reseller Contract). Must contain phrase "litigation dynamics". Responsibility clause is ignored at filtering stage.
    #     - If metadata = Reseller Agreement and text contains "litigation dynamics" → relevant.
    #     - If metadata = Reseller Contract and text contains "litigation dynamics" → relevant.
    #     - If metadata = Distribution Agreement and text contains "litigation dynamics" → NOT relevant.

    # Example 9:
    #     - Query: "Fetch all supplier contracts that include warranty clauses covering defect liability for more than 12 months"
    #     - Rule: Contract Type must be Supplier Contract (or synonym Supply Agreement). Must contain warranty clause with defect liability > 12 months.
    #     - If metadata = Supplier Contract and text contains "warranty clause with defect liability of 18 months" → relevant.
    #     - If metadata = Supply Agreement and text contains "warranty clause with defect liability of 15 months" → relevant.
    #     - If metadata = Supplier Contract and text contains "warranty clause with defect liability of 6 months" → NOT relevant.
    #     - If metadata = Vendor and Supply and text contains "warranty clause with defect liability of 18 months" → relevant (synonym match).

    # Example 10:
    #     - Query: "List all contracts in my vault" OR "How many files are there in my vault"
    #     - Rule: Vault-wide query, so ignore contract type and conditions. All files are relevant.
    #     - If metadata = any contract type → relevant.

    # Example 11:
    #     - Query: "List sales contract with contract duration of more than 3 years
    #     - Rule: Contract Type must be "Sales Agreement". Condition: contract duration must exceed 3 years.
    #     - If file type = Sales Agreement and contract duration is 4 years → relevant.
    #     - If file type = Sales Agreement and contract duration is exactly 3 years → NOT relevant.
    #     - If file type = Sales Agreement and contract duration is 2 years → NOT relevant

    # """

    # system_prompt = system_prompt.format(current_date=str(current_date))

    # prompt = (
    #     f"Retrieved Files:\n"
    #     f"---------------------\n"
    #     f"{json.dumps(preliminary_matches, indent=4)}\n\n"
    #     f"Task:\n"
    #     f"Evaluate each retrieved file against the user query step by step.\n"
    #     f"- If the query specifies a contract type (e.g., 'reseller agreement'), then check the file’s Contract Type in metadata. Only files with a matching Contract Type qualify.\n"
    #     f"- If the query does not specify a contract type, ignore type matching and check only by clauses, keywords, conditions, and numerical thresholds.\n"
    #     f"- For each file:\n"
    #     f"  1. If applicable, check strict Contract Type match in metadata (yes/no).\n"
    #     f"  2. For each required clause/keyword/condition, check if it is present (yes/no).\n"
    #     f"  3. For each numerical threshold/duration, check if it is satisfied (yes/no).\n"
    #     f"  - Apply strict comparison logic:\n"
    #     f"     - 'More than X' → strictly greater than X.\n"
    #     f"     - 'At least X' → greater than or equal to X.\n"
    #     f"     - 'Less than X' → strictly less than X.\n"
    #     f"     - 'Exactly X' → equal to X.\n"
    #     f"  4. Mark relevant = true only if (a) contract type matches when required, and (b) all required conditions are satisfied.\n\n"
    #     f"     - When contract type is specified: the Contract Type in metadata matches, AND all conditions are satisfied.\n"
    #     f"     - When no contract type is specified: ignore type and rely only on conditions.\n"
    #     f"     - **No guessing:** if the presence/absence of a condition is uncertain or ambiguous → treat as false.\n"
    #     f"Return the output strictly in valid JSON with this schema:\n\n"
    #     f"{{\n"
    #     f'  "results": [\n'
    #     f"    {{\n"
    #     f'      "file_id": "string",\n'
    #     f'      "document_type_match": true/false,\n'
    #     f'      "conditions_checked": {{ \"condition\": true/false, ... }},\n'
    #     f'      "explanation": "Short reason stating why file is relevant or not.",\n'
    #     f'      "is_relevant_to_user_query": true/false\n'
    #     f"    }}\n"
    #     f"  ]\n"
    #     f"}}\n\n"
    #     f"## Examples:\n"
    #     f"1. Query = 'NDAs with arbitration clause'\n"
    #     f"   - File type = NDA, text contains 'arbitration clause' → relevant.\n"
    #     f"   - File type = Service Agreement, text contains 'arbitration clause' → NOT relevant.\n\n"
    #     f"2. Query = 'Contracts with renewal term > 5 years'\n"
    #     f"   - No contract type specified, so ignore type. Only check duration.\n"
    #     f"   - File type = MSA, term = 10 years → relevant.\n"
    #     f"   - File type = NDA, term = 2 years → NOT relevant.\n"
    #     f"Check this document is relevant to user's query or not?\n\n"
    #     f"User Query:\n"
    #     f"---------------------\n"
    #     f"{user_query}\n\n\n\n"
    # )

    # system_prompt = "You are an advanced legal reasoning model specialized in analyzing any kind of legal document \n(contracts, agreements, policies, NDAs, MOUs, merger documents, sales contracts, supplier contracts, \nservice agreements, distribution contracts, etc.) and determining whether specific clauses, conditions, \nor legal concepts are present.\n\n---\n\n### Objectives\n1. Understand and interpret any user query related to legal clauses, compliance, risk, or contractual provisions.\n2. Analyze the given document text, metadata, and structure to determine if and how the condition or clause applies.\n3. Apply **legal equivalence reasoning** — even if the exact wording differs, determine if the meaning or effect is functionally the same.\n4. Identify the correct **document type** based on the text and metadata; do not confuse sales contracts, supplier contracts, or other contract types.\n5. Produce clear, consistent, and transparent JSON results describing your findings and rationale.\n\n---\n\n### Evaluation Process\nFor each document:\n1. Identify document type and purpose (from metadata, file name, or text).\n2. Examine the document to locate any relevant clauses or sections.\n3. Assess whether the condition or query is:\n   - **Explicitly satisfied** (clearly present),\n   - **Implicitly satisfied / equivalent** (legally or functionally the same),\n   - **Absent or contradicted**.\n4. Provide legal and logical reasoning for your determination.\n\n---\n\n### Output Format\nAlways respond strictly in this JSON format:\n\n{\n  \"results\": [\n    {\n      \"file_id\": \"<file_id>\",\n      \"document_type_match\": true | false,\n      \"conditions_checked\": {\n        \"<condition_name>\": true | false\n      },\n      \"explanation\": \"<Detailed reasoning including clause interpretation, presence, equivalence, or absence>\",\n      \"is_relevant_to_user_query\": true | false\n    }\n  ]\n}\n\n---\n\n### Reasoning Guidelines\n- Treat synonyms and equivalent legal expressions as matches (e.g., “terminate” ≈ “cancel,” “assign” ≈ “transfer”).\n- Interpret defined terms contextually (e.g., “Voting Power,” “Material Breach,” etc.).\n- If a clause or section serves the same *legal function* as the query (even if phrased differently), consider it equivalent and mark true.\n- **Do not classify a sales contract as a supplier contract.** Only mark `document_type_match = true` if the text and metadata clearly indicate a supplier contract.\n- If no sufficient evidence is found, mark false and explain why.\n- Always ensure your reasoning is transparent, precise, and legally sound.\n\n---\n\n### Example Queries\n- \"Does this agreement include a confidentiality clause?\"\n- \"Are voting rights proportional to shareholding?\"\n- \"Does this contract prohibit data transfer outside the EU?\"\n- \"Is there an indemnification obligation?\"\n- \"Does it specify governing law and jurisdiction?\"\n- \"Fetch all supplier contracts that include warranty clauses covering defect liability for more than 12 months.\"\n\n---\n\n### Inputs You Will Receive\nYou will receive:\n- file_id: unique identifier of the document.\n- file_metadata: metadata such as Contract Type, Parties, etc.\n- document_text: the full or partial text of the document.\n- user_query: the specific legal or compliance question to evaluate.\n\n---\n\n### Final Rules\n- Always analyze using legal interpretation and equivalence logic.\n- Do not guess or make unsupported claims — justify every “true” or “false.”\n- Always return a single, valid JSON object matching the schema above.\n- Avoid any extra commentary outside JSON."

    system_prompt = """
    You are an advanced legal reasoning model specialized in analyzing any kind of legal document (contracts, agreements, policies, NDAs, MOUs, merger documents, sales contracts, supplier contracts, service agreements, distribution contracts, etc.) and determining whether specific clauses, conditions, 
    or legal concepts are present.

    ---

    ### Objectives
    1. Understand and interpret any user query related to legal clauses, compliance, risk, or contractual provisions.
    2. Analyze the given document text, metadata, and structure to determine if and how the condition or clause applies.
    3. Apply **legal equivalence reasoning** — even if the exact wording differs, determine if the meaning or effect is functionally the same.
    4. Identify the correct **document type** based on the text and metadata; do not confuse sales contracts, supplier contracts, or other contract types.
    5. Produce clear, consistent, and transparent JSON results describing your findings and rationale.

    ---

    ### Evaluation Process
    For each document:
    1. Identify document type and purpose (from metadata, file name, or text).
    2. Examine the document to locate any relevant clauses or sections.
    3. Assess whether the condition or query is:
    - **Explicitly satisfied** (clearly present),
    - **Implicitly satisfied / equivalent** (legally or functionally the same),
    - **Absent or contradicted**.
    4. Provide legal and logical reasoning for your determination.

    ---

    ### Output Format
    Always respond strictly in this JSON format:

    {
    "results": [
        {
        "file_id": "<file_id>",
        "document_type_match": true | false,
        "conditions_checked": {
            "<condition_name>": true | false
        },
        "explanation": "<Detailed reasoning including clause interpretation, presence, equivalence, or absence>",
        "is_relevant_to_user_query": true | false
        }
    ]
    }

    ---

    ### Reasoning Guidelines
    - Treat synonyms and equivalent legal expressions as matches (e.g., “terminate” ≈ “cancel,” “assign” ≈ “transfer”).
    - Interpret defined terms contextually (e.g., “Voting Power,” “Material Breach,” etc.).
    - If a clause or section serves the same *legal function* as the query (even if phrased differently), consider it equivalent and mark true.
    - **Do not classify a sales contract as a supplier contract.** Only mark `document_type_match = true` if the text and metadata clearly indicate a supplier contract.
    - If no sufficient evidence is found, mark false and explain why.
    - Always ensure your reasoning is transparent, precise, and legally sound.

    ---

    ### Example Queries
    - "Does this agreement include a confidentiality clause?"
    - "Are voting rights proportional to shareholding?"
    - "Does this contract prohibit data transfer outside the EU?"
    - "Is there an indemnification obligation?"
    - "Does it specify governing law and jurisdiction?"
    - "Fetch all supplier contracts that include warranty clauses covering defect liability for more than 12 months."

    ---

    ### Inputs You Will Receive
    You will receive:
    - file_id: unique identifier of the document.
    - file_metadata: metadata such as Contract Type, Parties, etc.
    - document_text: the full or partial text of the document.
    - user_query: the specific legal or compliance question to evaluate.

    ---

    ### Final Rules
    - Always analyze using legal interpretation and equivalence logic.
    - Do not guess or make unsupported claims — justify every “true” or “false.”
    - Always return a single, valid JSON object matching the schema above.
    - Avoid any extra commentary outside JSON.
    """

    # prompt = f"### Retrieved File Context:\n{json.dumps(preliminary_matches, indent=4)}\n\n### Task:\nEvaluate the above retrieved file(s) to determine whether they are relevant to below user's query.\n\n###Today's Date:\{current_date}\n\n### User Query:\n{user_query}\n\nPerform your legal analysis now."

    prompt = f"""
    ### Retrieved File Context:
    {json.dumps(preliminary_matches, indent=4)}

    ---

    ### Task:
    Evaluate the above retrieved file(s) to determine whether they are relevant to below user's query.

    ---

    ###Today's Date: {current_date}
    
    ---

    ### User Query:
    {user_query}
    """
    # Call GPT-4o-mini
    try:
        response = client.chat.completions.create(
            # model="gpt-4o",
            model="gpt-4o-2024-08-06",
            temperature=0,
            max_completion_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
            
        )
        response_content = response.choices[0].message.content

        logger.info(_log_message(f"Validation response: {response_content}", "validation_call", module_name))

        # Extract JSON from the response
        response_json = json.loads(response_content)
        
        # Filter only relevant files (is_relevant_to_user_query == True)
        relevant_files = [
            item for item in response_json.get("results", [])
            if item.get("is_relevant_to_user_query", False)
        ]
        
        # Extract only the file_ids from relevant files
        file_ids = [item["file_id"] for item in relevant_files]
        
        # Return both file_ids and the filtered results with explanations
        return {
            "file_ids": file_ids,
            "results": relevant_files
        }

    except Exception as e:
        logger.error(_log_message(f"Error during validation call: {e}\n{traceback.format_exc()}", "validation_call", module_name))
        return {"file_ids": [], "results": []}


def _log_message(message: str, function_name: str, module_name: str) -> str:
    return f"[function={function_name} | module={module_name}] - {message}"


@mlflow.trace(name="Filter Relevant File IDs")
def filter_relevant_file_ids(event: dict, only_contract=True) -> dict:
    """Filters relevant file IDs based on the provided event parameters.

    Args:
        event (dict): Contains 'org_id', 'user_id', and 'search_text'.

    Returns:
        dict: The search result or error response.
    """
    org_id = event.get('org_id')
    user_id = event.get('user_id')
    search_text = event.get('search_text')
    search_id = event.get('search_id', None)
    tag_ids = event.get('tag_ids', [])

    if not all([org_id, user_id, search_text, tag_ids]):
        return {
            "success": False,
            "error": "Missing required parameters: org_id, user_id, or search_text",
            "data": {"fileids": []}
        }
    search_id = str(uuid.uuid4()) if search_id is None else search_id
    sanitized_query = sanitize_log_stream_name(search_text) # needed santization for queries like this : https://lawrato.com/civil-legal-advice/disputes-about-common-area-in-apartments-230952
    logger = request_logger(
        f"{search_id} | org_id-{org_id} | user_id-{user_id} | search_text-{sanitized_query[:30]}",
        str(config.CONCEPTUAL_SEARCH_LOG_DIR_NAME),
        "CONCEPTUAL_SEARCH"
    )

    try:
        logger.info(_log_message("##### CONCEPTUAL SEARCH START #####", 
                                  "filter_relevant_file_ids", module_name))
        logger.info(_log_message(f"Received Event: {event}", 
                                 "filter_relevant_file_ids", module_name))


        

        result = search_contracts(org_id, user_id, search_text, tag_ids, only_contract, logger)


        relevant_ids = [file['id'] for file in result.get('data', {}).get('fileids', [])]
        set_conceptual_search_results(search_id, relevant_ids, logger)
        logger.info(_log_message(f"Response Generated Successfully: {result}", 
                                 "filter_relevant_file_ids", module_name))

        return result

    except Exception as e:
        logger.error(_log_message(f"Error in filter_relevant_file_ids: {e}\n{traceback.format_exc()}", 
                                  "filter_relevant_file_ids", module_name))
        return {
            "success": False,
            "error": str(e),
            "data": {"fileids": []}
        }

    finally:
        logger.info(_log_message("##### CONCEPTUAL SEARCH END #####", 
                                "filter_relevant_file_ids", module_name))
        flush_all_logs(f"{search_id} | org_id-{org_id} | user_id-{user_id} | search_text-{search_text[:30]}", str(config.CONCEPTUAL_SEARCH_LOG_DIR_NAME), "CONCEPTUAL_SEARCH")
