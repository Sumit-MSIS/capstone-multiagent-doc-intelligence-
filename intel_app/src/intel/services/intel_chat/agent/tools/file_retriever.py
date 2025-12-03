import os
import mlflow
import asyncio
import textwrap
import json
from typing import List, AsyncIterator, Dict, Any
from agno.tools import tool
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team
from src.intel.services.intel_chat.agent.utils import retrieve_all_file_ids, filter_contracts, get_all_file_names
from src.intel.services.intel_chat.agent.sql_agent import sql_agent as sql_file_retriever
from src.intel.services.intel_chat.agent.pinecone_retriever.hybrid_search_retrieval import get_context_from_pinecone
from src.intel.services.intel_chat.agent.query_rephraser import rephrase_search_text
from src.intel.services.intel_chat.agent.pinecone_retriever.hybrid_search_retrieval import get_embeddings
from src.intel.services.intel_chat.agent.utils import get_metadata
from src.common.llm.factory import get_llm_client
from src.intel.services.intel_chat.agent.utils import get_metadata


MESSAGES = {
    "no_access": {
        "description": (
            "Prompt user he doesn't have access to any files in his vault. "
            "He can upload files to the vault or contact his administrator to gain access."
        ),
        "expected_output": (
            "A message indicating the user doesn't have access to any files in their vault. "
            "Either he can upload few files in vault and continue or he needs to contact administrator to gain access."
        )
    },
    "too_many_files": {
        "description": (
            "There are too many files found relevant to user query. Ask the user to narrow down the search."
        ),
        "expected_output": (
            "A message indicating there are too many files found relevant to user query in user's vault. "
            "Ask the user to provide more details to narrow down the search results."
        )
    },
    "no_relevant_info": {
        "description": (
            "No relevant information found in user's vault. Prompt user to provide more details or try another query."
        ),
        "expected_output": (
            "A message indicating no relevant information found in user's vault. "
            "Prompt user to provide more details or try another query."
        )
    }
}


@mlflow.trace(name="Hybrid Search Single File")
async def hybrid_single_file(
    file_id: str,
    keyword_search_query: str,
    semantic_search_query: str,
    org_id: int,
    tag_ids: List[str],
    dense_vectors,
    reranker_threshold,
    max_allowed: int
) -> List[dict]:
    """
    Now returns ONLY the raw matches for the file_id.
    Threshold logic is removed.
    """

    try:
        filters = {"file_id": {"$eq": file_id}}
        top_k = 10

        data = await get_context_from_pinecone(
            os.getenv("DOCUMENT_SUMMARY_AND_CHUNK_INDEX"),
            os.getenv("PINECONE_API_KEY"),
            filters,
            top_k,
            semantic_search_query,
            [file_id],
            org_id,
            tag_ids,
            keyword_search_query,
            dense_vectors,
            reranker_threshold
        )

        if not data or "matches" not in data:
            return []

        return data["matches"]

    except Exception:
        return []




@mlflow.trace(name="Hybrid search each file")
async def hybrid_file_retriever(
    user_query: str,
    file_guids: List[str],
    org_id: int,
    tag_ids: List[str],
    reranker_threshold: float = 0.5,
    max_allowed: int = 20
) -> List[str]:

    if not file_guids:
        return []
    
    keyword_search_query, semantic_search_query, reranking_query = await rephrase_search_text(user_query)
    dense_vectors = await get_embeddings(org_id, file_guids, semantic_search_query) or {}

    # Run per-file search → returns raw matches
    tasks = [
        hybrid_single_file(fid, keyword_search_query, semantic_search_query, org_id, tag_ids, dense_vectors, reranker_threshold, max_allowed)
        for fid in file_guids
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten matches
    all_matches = [m for r in results if isinstance(r, list) for m in r]

    # ----------------------------
    # Extract helper
    # ----------------------------
    def extract_unique_file_ids(match_list):
        seen = set()
        unique = []
        for m in match_list:
            fid = m.get("metadata", {}).get("file_id")
            if fid and fid not in seen:
                seen.add(fid)
                unique.append(fid)
        return unique

    # First pass
    unique_ids = extract_unique_file_ids(all_matches)
    if len(unique_ids) <= max_allowed:
        return unique_ids

    # ----------------------------
    # Threshold escalation
    # ----------------------------
    thresholds = [0.6, 0.7, 0.8, 0.9]
    last_unique_ids = []

    for threshold in thresholds:
        filtered = [m for m in all_matches if m.get("score", 0) >= threshold]
        unique_ids = extract_unique_file_ids(filtered)
        last_unique_ids = unique_ids
        if len(unique_ids) <= max_allowed:
            return unique_ids

    # too many → return threshold 0.9 result
    return last_unique_ids


@tool(
    name="get_files_names",
    description="Get file names for given file IDs.",
    show_result=False
)
@mlflow.trace(name="Get Files Names Tool")
async def get_files_names(session_state) -> Dict[str, str]:
    """Helper function to get file names for given file IDs."""
    file_names_dict = await get_all_file_names(session_state.get("file_guids", []))
    return file_names_dict
    

@mlflow.trace(name="Check Empty or All True")
async def check_empty_all_true(summary_json): # if not files or all files are True then return True else False
    try:
        for contract_type, files in summary_json["files"].items():
            if not files:  # Check if the files dictionary is empty
                print(f"Contract type '{contract_type}' has no files.")
                return True
            if not all(files.values()):  # Check if all values are True
                print(f"Contract type '{contract_type}' does not have all files set to True.")
                return False
        print("All contract types have all files set to True.")
        return True  # All contract types have all files set to True
    except Exception as e:
        print(f"Error checking empty or all true: {e}")
        return False

@mlflow.trace(name="Summary Query Classifier")
async def summary_query_classifier(user_query, coordinator_session_state, history):
    try:
        async_clent =  get_llm_client(async_mode=True)
        # Fetch all previous messages from history
        summary_query_classifier_prompt = """
            Your task is to classify the current user query based on the previous few conversation messages and a given list of contract types.

            You must output a strict JSON with the following 3 keys:
            - query_type → one of: "old_query" or "new_query"
            - contract_type → string (only one contract type, chosen from the provided list)
            - only_contract_type → boolean (True or False)

            Follow the rules:

            1. Determining query_type:
            - If the current query references, continues, depends on, or follows up on any previous summary-related question or any specific contract type as per below list, set:
            "query_type": "old_query"
            - If the current query does not refer to any earlier query, or introduces a new domain/topic or contract type not in the list below, or appears unrelated, set:
            "query_type": "new_query"
            - Previous conversation may include non-summary queries. Ignore them unless directly referenced.
            - If no previous messages exist, always return "new_query".

            2. Identifying contract_type:
            - You will receive a list of contract types.
            - Pick the first matching contract type found in the user’s current query.
            - Matching is simple phrase match, case insensitive.
            - No semantic inference needed.
            - If multiple contract types appear, choose the first one in the query order.
            - If none match, return an empty string.

            3. Determining only_contract_type:
            This field indicates how the user decides WHICH contract(s) should be summarized.

            Set only_contract_type = true when:
            - The contract is selected **only by contract type**, with NO additional filters, constraints, or conditions.
            Examples:
            "Summarize payment terms for my NDA agreements."
            "Yes, proceed with Sales Contract."
            "Summarize tax obligations clauses specified in MSA."
            "Summarize termination rights for Service Agreement."

            Set only_contract_type = false when:
            - The contract is selected by **contract type + additional filtering criteria** such as clauses, terms, properties, or constraints.
            Examples:
            "Summarize Sales contracts that have termination notice less than 3 days."
            "Go with NDA where liability cap exceeds $1M."
            "Summarize termination and renewal clauses for our Service Agreements that includes exclusivity clause."
            "Summarize the payment terms specified in Sales contracts that have a 3-day notice period."

            These conditions determine WHICH instances of the contract type should be summarized, so return false.

            4. Output format:
            Return only the JSON. Absolutely no extra text, explanation, or markdown.

            Example output:
            {"query_type":"old_query","contract_type":"NDA","only_contract_type":False}

            Examples of user queries and expected outputs:
            1. Summarize all contracts executed last year.
                User: Summarize all contracts executed last year.
                → {"query_type":"new_query","contract_type":"","only_contract_type":false}

                Assistant: I found 20 contracts from last year. Should I begin with Lease, Sales, or Vendor agreements?

                User: Begin with Lease contracts.
                → {"query_type":"old_query","contract_type":"Lease","only_contract_type":true}

                Assistant: Here is the summary of the first 5 Lease files. [summary_placeholder]
                Remaining Lease files found. Continue with remaining, or switch to Sales or Vendor?

                User: Switch to Sales agreements.
                → {"query_type":"old_query","contract_type":"Sales","only_contract_type":true}

                Assistant: Here is the summary of Sales agreements. [summary_placeholder]
                Should I move to Vendor next or complete remaining Lease?

                User: Show Vendor agreements that include arbitration clause.
                → {"query_type":"old_query","contract_type":"Vendor","only_contract_type":false}
            
                
            2. Summarize contracts with governing law New York.
                User: Summarize contracts with governing law New York.
                → {"query_type":"new_query","contract_type":"","only_contract_type":false}

                Assistant: I found 14 New-York governed files. Start with Service, Purchase, or Loan?

                User: Start with Service.
                → {"query_type":"old_query","contract_type":"Service","only_contract_type":true}

                Assistant: Service agreement summaries: [summary_placeholder]
                Remaining Service files available. Continue or switch to Purchase or Loan?

                User: Actually don’t do Service—go with Purchase agreements containing confidentiality clause.
                → {"query_type":"old_query","contract_type":"Purchase","only_contract_type":false}

                Assistant: Purchase agreements with confidentiality clause summarized: [summary_placeholder]
                Continue with all Purchase agreements or switch to Loan?

                User: Switch to Loan.
                → {"query_type":"old_query","contract_type":"Loan","only_contract_type":true}

                
            3. Summarize all Supplier contracts.
                User: Summarize all Supplier contracts.
                → {"query_type":"new_query","contract_type":"Supplier","only_contract_type":true}

                Assistant: Supplier summaries: [summary_placeholder]
                Should I continue with remaining Supplier files?

                User: Continue with remaining but sumarize only payment and termination terms.
                → {"query_type":"old_query","contract_type":"Supplier","only_contract_type":true}

                Assistant: Summary of payment and termination terms for Supplier contracts: [summary_placeholder]
                All Supplier files summarized. Any other contract types you want summarized?

            """
        # contract_types_list = "-".join([m for m in coordinator_session_state["files"].keys()])


        summary_query_classifier_user_prompt = f"""
        ### List of Contract Types:
        --- 
            {coordinator_session_state["files"].keys()}
        ---

        ### Current User Query: {user_query}

        """


        if history[0]["role"] == "system":
            history[0]["content"] = summary_query_classifier_prompt
        
        history.append({"role": "user", "content": summary_query_classifier_user_prompt})

        response = await async_clent.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            max_completion_tokens=500,
            temperature=0.1,
        )

        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error in Summary Query Classifier: {e}")
        return {"query_type":"new_query","contract_type":"","only_contract_type":False}


@mlflow.trace(name="SQL Call or Hybrid File Retriever")
async def sql_call(query: str, updated_file_ids: List[str], org_file_ids: List[str], metadata:dict, chat_id: int, user_id: int, org_id: int, tag_ids: List[str], max_allowed: int = 20) -> List[str]:
    sql_file_guids, is_sql_generated = await sql_file_retriever(chat_id, query, user_id, org_id, metadata)

    if sql_file_guids and is_sql_generated:
        return sql_file_guids
    elif is_sql_generated is False:
        hybrid_file_ids = await hybrid_file_retriever(
            query,
            updated_file_ids if updated_file_ids else org_file_ids,
            org_id,
            tag_ids,
            reranker_threshold=0.5,
            max_allowed=max_allowed
        )
        if hybrid_file_ids:
            return hybrid_file_ids
    return []



@mlflow.trace(name="File Search")
async def file_search(
    json_plan: dict,
    file_ids: List[str],
    chat_id: int,
    user_id: int,
    org_id: int,
    tag_ids: List[str],
    coordinator_session_state: Dict[str, Any],
    history: List[Dict[str, str]]
):
    """
    File search flow that separates retrieved vault file_ids from dynamically discovered ones.
    - org_file_ids: from vault
    - all_file_ids: dynamically updated via dependent/independent tasks
    """

    if not isinstance(json_plan, dict):
        return {"error": "Invalid input format. Expected a JSON object."}

    original_user_query = json_plan.get("original_user_query", "")
    tasks = json_plan.get("tasks", {})
    dependent_tasks = tasks.get("dependent_tasks", [])
    independent_tasks = tasks.get("independent_tasks", [])
    expected_output = json_plan.get("expected_output", "")
    tasks_to_skip = ['Web Search', 'Greeting', 'Comparison', 'Coordinated Analysis', 'Metadata Analysis', 'Summarization']


    #############################
    # 1. Check if any intent is of Summarization
    # 2. Check if file_ids selected or not
    # 3. If selected files, check if len(file_ids) <=6  -> return directly to summarize all files 
    # 4. If selected more than 6 files -> check the Json stoarge in session_state if empty update all files as per contract type elif all file_status is True then reset the Json storage else call **Summary Query Type** LLM to check if query is follow-up or a new query -> If follow-up call, update the file_status non-selected files as True Elif New Query reset the Json storage and update all files as per contract type
    # 5. If no selected files -> Proceed to normal flow for File Search
    # 6. If files found


    #############################

    # Step 1: If file_ids already provided
    if file_ids:
        ### ***If files are selected then that will be final set of files to process for that session_id****
        unique_file_ids = list(set(file_ids))
        if any(t.get("intent") == "Summarization" for t in dependent_tasks + independent_tasks):
            # Check if Json stoarge is empty or all True
            if await check_empty_all_true(coordinator_session_state):
                if len(unique_file_ids) <= 6:
                    coordinator_session_state["file_ids"] = unique_file_ids
                    coordinator_session_state["files_selected"] = True
                    ### Return only the Summarization Task with Selected Files

                    return {
                        "original_user_query": original_user_query,
                        "tasks": {
                            "dependent_tasks": dependent_tasks,
                            "independent_tasks": independent_tasks
                        },
                        "expected_output": expected_output,
                        "file_ids": unique_file_ids
                    }
                else:
                    # No new files as new Json query storage is created
                    print("More than 6 files selected, resetting JSON storage.")
                    coordinator_session_state["files"] = {}
                    coordinator_session_state["follow_up_count"] = 0
                    coordinator_session_state["files_matched"] = 0
                    # ---------------------------------------------------------------------
                    all_files_metadata = await get_metadata(unique_file_ids)
                    for file_id in unique_file_ids:
                        contract_type = all_files_metadata.get(file_id, {}).get("metadata", {}).get("Contract Type")
                        if contract_type not in coordinator_session_state["files"]:
                            coordinator_session_state["files"][contract_type] = {}
                        coordinator_session_state["files"][contract_type][file_id] = False
                        

                    coordinator_session_state["follow_up_count"] += 1
                    coordinator_session_state["files_matched"] = len(unique_file_ids)
                    coordinator_session_state["file_ids"] = unique_file_ids
                    coordinator_session_state["files_selected"] = True

                    return {
                        "original_user_query": original_user_query,
                        "tasks": {
                            "dependent_tasks": dependent_tasks,
                            "independent_tasks": independent_tasks
                        },
                        "expected_output": expected_output,
                        "file_ids": unique_file_ids
                    }
            
            else:
                # Update the follow_up_count +1 and files_matched to remaining files
                coordinator_session_state["follow_up_count"] += 1
                matched_files_count = 0
                for contract_type, files in coordinator_session_state["files"].items():
                    for file_id, status in files.items():
                        if status == False:
                            matched_files_count += 1
                coordinator_session_state["files_matched"] = matched_files_count
                coordinator_session_state["file_ids"] = unique_file_ids
                coordinator_session_state["files_selected"] = True
                
            #     query_classifier_json = await summary_query_classifier(original_user_query,coordinator_session_state,history)
            #     query_classifier_dict = json.loads(query_classifier_json)
            #     query_type, contract_type, only_contract_type = query_classifier_dict.get("query_type"), query_classifier_dict.get("contract_type"), query_classifier_dict.get("only_contract_type")

            #     if query_type == "old_query" and contract_type and only_contract_type:
            #         pass

            #     elif query_type == "old_query" and contract_type and not only_contract_type:
            #         pass

            #     elif query_type == "old_query" and not contract_type and not only_contract_type:
            #         pass

            #     elif query_type == "new_query":
            #         pass
                
        return {
            "original_user_query": original_user_query,
            "tasks": {
                "dependent_tasks": dependent_tasks,
                "independent_tasks": independent_tasks
            },
            "expected_output": expected_output,
            "file_ids": unique_file_ids
        }

    # Step 2: Retrieve from vault
    org_file_ids = await retrieve_all_file_ids(org_id=org_id, tag_ids=tag_ids)
    org_file_ids = list(set(org_file_ids or []))
    all_file_ids = set()  # maintain dynamically discovered IDs

    # Step 3: No access case
    if not org_file_ids:
        return {
            "original_user_query": original_user_query,
            "tasks": {
                "dependent_tasks": [{
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": MESSAGES["no_access"]["description"],
                    "sub_query": "",
                    "intent": "Clarification",
                    "file_ids": [],
                    "user_provided_context": "",
                    "user_provided_url": ""
                }],
                "independent_tasks": []
            },
            "expected_output": MESSAGES["no_access"]["expected_output"],
            "file_ids": []
        }

    # get metadata for the org , retrieved once - to be used at multiple places in sql agent
    metadata = await get_metadata(org_file_ids)
    metadata.pop('total_files', None) # metadata dictionary contains 'total_files' key 
    # Step 4: Handle dependent tasks
    if dependent_tasks:
        for task in dependent_tasks:
            intent = task.get("intent", "")
            if intent in tasks_to_skip:
                continue

            sub_query = task.get("sub_query", "")
            # Pass org_file_ids (from vault) + dynamically updated ones
            sql_file_ids = await sql_call(sub_query, list(all_file_ids), org_file_ids, metadata, chat_id, user_id, org_id, tag_ids, max_allowed=20)
            if sql_file_ids:
                all_file_ids.update(sql_file_ids)

        # No relevant info after dependent tasks (preserve original behavior)
        if not all_file_ids and any(task.get("intent", "") not in ["Web Search", "Greeting", "Coordinated Analysis", "Metadata Analysis", "Comparison", "Summarization"] for task in dependent_tasks):
            return {
                "original_user_query": original_user_query,
                "tasks": {
                    "dependent_tasks": [{
                        "task_id": "task_1",
                        "order_of_execution": 1,
                        "task_description": MESSAGES["no_relevant_info"]["description"],
                        "sub_query": "",
                        "intent": "Clarification",
                        "file_ids": [],
                        "user_provided_context": "",
                        "user_provided_url": ""
                    }],
                    "independent_tasks": []
                },
                "expected_output": MESSAGES["no_relevant_info"]["expected_output"],
                "file_ids": []
            }

        # Filter contracts after dependent tasks
        filtered_ids = await filter_contracts(list(all_file_ids), tag_ids) if tag_ids else list(all_file_ids)
        all_file_ids = set(filtered_ids)

    if len(all_file_ids) > 20:
        return {
            "original_user_query": original_user_query,
            "tasks": {
                "dependent_tasks": [{
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": MESSAGES["too_many_files"]["description"],
                    "sub_query": "",
                    "intent": "Clarification",
                    "file_ids": [],
                    "user_provided_context": "",
                    "user_provided_url": ""
                }],
                "independent_tasks": []
            },
            "expected_output": MESSAGES["too_many_files"]["expected_output"],
            "file_ids": []
        }

    # Step 5: Handle independent tasks
    if independent_tasks:
        # Take one snapshot so all tasks get the SAME input
        file_id_snapshot = list(all_file_ids)

        async def run_task(task):
            intent = task.get("intent", "")
            if intent in tasks_to_skip:
                return None  # skip

            sub_query = task.get("sub_query", "")
            return await sql_call(
                sub_query,
                file_id_snapshot,  # same dataset for all tasks
                org_file_ids,
                metadata,
                chat_id,
                user_id,
                org_id,
                tag_ids,
                max_allowed=20
            )

        # Execute concurrently (fail-safe mode)
        results = await asyncio.gather(
            *(run_task(t) for t in independent_tasks),
            return_exceptions=True
        )

        # Merge results after completion
        for sql_file_ids in results:
            if isinstance(sql_file_ids, Exception):
                # optional: log error here
                continue
            if sql_file_ids:
                all_file_ids.update(sql_file_ids)

        # No relevant info after dependent tasks (preserve original behavior)
        if not all_file_ids and any(task.get("intent", "") not in ["Web Search", "Greeting", "Coordinated Analysis", "Metadata Analysis", "Comparison"] for task in dependent_tasks):
            return {
                "original_user_query": original_user_query,
                "tasks": {
                    "dependent_tasks": [{
                        "task_id": "task_1",
                        "order_of_execution": 1,
                        "task_description": MESSAGES["no_relevant_info"]["description"],
                        "sub_query": "",
                        "intent": "Clarification",
                        "file_ids": [],
                        "user_provided_context": "",
                        "user_provided_url": ""
                    }],
                    "independent_tasks": []
                },
                "expected_output": MESSAGES["no_relevant_info"]["expected_output"],
                "file_ids": []
            }
    

    # Final filtering and return
    final_filtered_ids = await filter_contracts(list(all_file_ids), tag_ids) if tag_ids else list(all_file_ids)
    unique_final_ids = list(set(final_filtered_ids))

    if len(unique_final_ids) > 20:
        return {
            "original_user_query": original_user_query,
            "tasks": {
                "dependent_tasks": [{
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": MESSAGES["too_many_files"]["description"],
                    "sub_query": "",
                    "intent": "Clarification",
                    "file_ids": [],
                    "user_provided_context": "",
                    "user_provided_url": ""
                }],
                "independent_tasks": []
            },
            "expected_output": MESSAGES["too_many_files"]["expected_output"],
            "file_ids": []
        }


    # Files collected as per all sub-query retieval now update in session_state
    if any(t.get("intent") == "Summarization" for t in dependent_tasks + independent_tasks):
        if await check_empty_all_true(coordinator_session_state):
            if len(unique_final_ids) <= 6:
                coordinator_session_state["file_ids"] = unique_final_ids
                return {
                    "original_user_query": original_user_query,
                    "tasks": {
                        "dependent_tasks": dependent_tasks,
                        "independent_tasks": independent_tasks
                    },
                    "expected_output": expected_output,
                    "file_ids": unique_final_ids
                }
        
            else:
                # No new files as new Json query storage is created
                print("More than 6 files Found, Json is Empty,Updating all files.")
                # -------------------------------------------------------------------------------------
                coordinator_session_state["files"] = {}
                coordinator_session_state["follow_up_count"] = 0
                coordinator_session_state["files_matched"] = 0
                
                all_files_metadata = await get_metadata(unique_final_ids)
                for file_id in unique_final_ids:
                    contract_type = all_files_metadata.get(file_id, {}).get("metadata", {}).get("Contract Type")
                    if not contract_type:
                        continue
                    if contract_type not in coordinator_session_state["files"]:
                        coordinator_session_state["files"][contract_type] = {}
                    coordinator_session_state["files"][contract_type][file_id] = False

                coordinator_session_state["follow_up_count"] += 1
                coordinator_session_state["files_matched"] = len(unique_final_ids)
                coordinator_session_state["file_ids"] = unique_final_ids

                return {
                    "original_user_query": original_user_query,
                    "tasks": {
                        "dependent_tasks": dependent_tasks,
                        "independent_tasks": independent_tasks
                    },
                    "expected_output": expected_output,
                    "file_ids": unique_final_ids
                }
        
        else:
            query_classifier_json = await summary_query_classifier(original_user_query,coordinator_session_state,history)
            query_classifier_dict = json.loads(query_classifier_json)
            query_type, contract_type, only_contract_type = query_classifier_dict.get("query_type"), query_classifier_dict.get("contract_type"), query_classifier_dict.get("only_contract_type")
            all_files_metadata = await get_metadata(unique_final_ids)
            if query_type == "old_query" and contract_type and only_contract_type:
                # No Json update for files
                coordinator_session_state["follow_up_count"] += 1
                # Check matching files with False status files in Json
                matched_files_count = 0
                for file_id in unique_final_ids:
                    if file_id in coordinator_session_state["files"].get(contract_type, {}) and coordinator_session_state["files"][contract_type][file_id] == False:
                        matched_files_count += 1
                coordinator_session_state["files_matched"] = matched_files_count
                # If matched files found === 0
                coordinator_session_state["file_ids"] = unique_final_ids


            elif query_type == "old_query" and contract_type and not only_contract_type:
                # Update the Json files status for files not matched in unique_final_ids as True for that contract_type only
                matched_files_count = 0
                for file_id in coordinator_session_state["files"].get(contract_type, {}):
                    # if file_id in unique_final_ids:
                    #     # coordinator_session_state["files"][contract_type][file_id] = False
                    #     pass
                    # else:
                    #     coordinator_session_state["files"][contract_type][file_id] = True
                    #     matched_files_count += 1
                    if file_id not in unique_final_ids and coordinator_session_state["files"][contract_type][file_id] == False:
                        coordinator_session_state["files"][contract_type][file_id] = True
                    
                    elif file_id in unique_final_ids and coordinator_session_state["files"][contract_type][file_id] == False:
                        matched_files_count += 1

                coordinator_session_state["follow_up_count"] += 1
                coordinator_session_state["files_matched"] = matched_files_count
                coordinator_session_state["file_ids"] = unique_final_ids


            elif query_type == "old_query" and not contract_type and not only_contract_type:
                # Update the json for all contract types files status as True for non-matched files in unique_final_ids
                matched_files_count = 0
                for ctype in coordinator_session_state["files"].keys():
                    for file_id in coordinator_session_state["files"].get(ctype, {}):
                        if file_id not in unique_final_ids and coordinator_session_state["files"][contract_type][file_id] == False:
                            coordinator_session_state["files"][ctype][file_id] = True
                        
                        elif file_id in unique_final_ids and coordinator_session_state["files"][ctype][file_id] == False:
                            matched_files_count += 1

                coordinator_session_state["follow_up_count"] += 1
                coordinator_session_state["files_matched"] = matched_files_count
                coordinator_session_state["file_ids"] = unique_final_ids

            elif query_type == "new_query":
                # Reset the Json storage and update all files as per contract type
                print("New Query detected, resetting JSON storage.")
                coordinator_session_state["files"] = {}
                coordinator_session_state["follow_up_count"] = 0
                coordinator_session_state["files_matched"] = 0
                coordinator_session_state["file_ids"] = []
                # -------------------------------------------------------------------------
                for file_id in unique_final_ids:
                    contract_type = all_files_metadata.get(file_id, {}).get("metadata", {}).get("Contract Type")
                    if not contract_type:
                        continue
                    if contract_type not in coordinator_session_state["files"]:
                        coordinator_session_state["files"][contract_type] = {}
                    coordinator_session_state["files"][contract_type][file_id] = False

                coordinator_session_state["follow_up_count"] += 1
                coordinator_session_state["files_matched"] = len(unique_final_ids)
                coordinator_session_state["file_ids"] = unique_final_ids
            
            else:
                print("Error in Query Classifier LLM response, proceeding without JSON update.")
                coordinator_session_state["file_ids"] = unique_final_ids


        return {
            "original_user_query": original_user_query,
            "tasks": {
                "dependent_tasks": dependent_tasks,
                "independent_tasks": independent_tasks
            },
            "expected_output": expected_output,
            "file_ids": unique_final_ids
        }

    if not unique_final_ids and any(task.get("intent", "") not in ["Web Search", "Greeting", "Coordinated Analysis", "Metadata Analysis", "Comparison"] for task in dependent_tasks):
        return {
            "original_user_query": original_user_query,
            "tasks": {
                "dependent_tasks": [{
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": MESSAGES["no_relevant_info"]["description"],
                    "sub_query": "",
                    "intent": "Clarification",
                    "file_ids": [],
                    "user_provided_context": "",
                    "user_provided_url": ""
                }],
                "independent_tasks": []
            },
            "expected_output": MESSAGES["no_relevant_info"]["expected_output"],
            "file_ids": []
        }
    
    if not unique_final_ids and any(task.get("intent", "") not in ["Web Search", "Greeting", "Coordinated Analysis", "Metadata Analysis", "Comparison"] for task in independent_tasks):
        return {
            "original_user_query": original_user_query,
            "tasks": {
                "dependent_tasks": [{
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": MESSAGES["no_relevant_info"]["description"],
                    "sub_query": "",
                    "intent": "Clarification",
                    "file_ids": [],
                    "user_provided_context": "",
                    "user_provided_url": ""
                }],
                "independent_tasks": []
            },
            "expected_output": MESSAGES["no_relevant_info"]["expected_output"],
            "file_ids": []
        }
    return {
        "original_user_query": original_user_query,
        "tasks": {
            "dependent_tasks": dependent_tasks,
            "independent_tasks": independent_tasks
        },
        "expected_output": expected_output,
        "file_ids": unique_final_ids
    }
