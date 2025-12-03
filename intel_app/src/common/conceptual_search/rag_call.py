import time
import mlflow
from src.common.logger import _log_message
from src.config.base_config import config
from src.common.hybrid_retriever.hybrid_search_retrieval import get_context_from_pinecone
from src.common.conceptual_search.utils import rephrase_search_text
from concurrent.futures import ThreadPoolExecutor, as_completed
mlflow.openai.autolog()

MODULE_NAME = "lambda_function.py"
DOCUMENT_SUMMARY_AND_CHUNK_INDEX = config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX
PINECONE_API_KEY = config.PINECONE_API_KEY


@mlflow.trace(name="Call RAG")
def call_rag(user_id, org_id, question, file_ids, metadata_dict, logger):
    try:
        start_time = time.time()
        pinecone_index_name = DOCUMENT_SUMMARY_AND_CHUNK_INDEX
        pinecone_api_key = PINECONE_API_KEY
        top_k = 10

        # Helper to chunk file list
        @mlflow.trace(name="Call RAG - Chunk List")
        def chunk_list(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        # Function to retrieve context for a file
        @mlflow.trace(name="Call RAG - Retrieve Context")
        def retrieve_context(file_id):
            try:
                custom_filter = {"file_id": {"$eq": file_id}}
                keyword_query, semantic_query, reranking_query = rephrase_search_text(question)
                context_chunks = get_context_from_pinecone(
                    pinecone_index_name, pinecone_api_key, custom_filter,
                    top_k, semantic_query, file_id, user_id, org_id, logger=logger,keyword_search_query=keyword_query
                )
                return file_id, context_chunks.get("matches", [])
            except Exception as e:
                logger.error(_log_message(f"Error retrieving context for file {file_id}: {e}", "call_rag", MODULE_NAME))
                return file_id, []

        # Parallel retrieval using ThreadPoolExecutor
        all_matches = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_file = {executor.submit(retrieve_context, file_id): file_id for file_id in file_ids}
            for future in as_completed(future_to_file):
                file_id, matches = future.result()
                all_matches[file_id] = matches

        logger.info(_log_message(f"Total time for Pinecone retrieval: {time.time() - start_time:.2f} seconds", "call_rag", MODULE_NAME))

        final_chunks = []
        file_chunks = list(chunk_list(file_ids, 6))
        file_counter = 1
        total_files_found = metadata_dict.get("total_files", "N/A")

        for chunk_idx, chunk in enumerate(file_chunks):
            chunk_context = ["**CONTRACT CHUNKS RETRIEVED FROM RAG:**\n"] if chunk_idx == 0 else []

            for file_id in chunk:
                matches = all_matches.get(file_id, [])
                if not matches:
                    continue

                file_metadata_entry = metadata_dict.get(file_id, {})
                file_metadata = file_metadata_entry.get("metadata", {})
                file_name = file_metadata_entry.get("file_name", file_id)

                # Precompute the joined content
                contract_content = "\n".join([match["metadata"].get("text", "") for match in matches]).strip() or 'No content found.'

                structured_context = f"""
                ### File {file_counter}: {file_name}
                **Metadata:**
                - **File ID**: {file_id}
                - **File Name**: {file_name}
                - **Title**: {file_metadata.get("Title of the Contract", "N/A")}
                - **Scope of Work**: {file_metadata.get("Scope of Work", "N/A")}
                - **Parties Involved**: {file_metadata.get("Parties Involved", "N/A")}
                - **Contract Type**: {file_metadata.get("Contract Type", "N/A")}
                - **Jurisdiction**: {file_metadata.get("Jurisdiction", "N/A")}
                - **Contract Duration**: {file_metadata.get("Contract Duration", "N/A")}
                - **Has Recurring Payment**: {file_metadata.get("Has Recurring Payment", "N/A")}
                - **Contract Value**: {file_metadata.get("Contract Value", "N/A")}
                - **Effective Date**: {file_metadata.get("Effective Date", "N/A")}
                - **Termination Date**: {file_metadata.get("Termination Date", "N/A")}
                - **Renewal Date**: {file_metadata.get("Renewal Date", "N/A")}
                - **Expiration Date**: {file_metadata.get("Expiration Date", "N/A")}
                - **Term Date**: {file_metadata.get("Term Date", "N/A")}
                - **Payment Due Date**: {file_metadata.get("Payment Due Date", "N/A")}
                - **Delivery Date**: {file_metadata.get("Delivery Date", "N/A")}
                - **Version Control**: {file_metadata.get("Version Control", "N/A")}

                **Contract Content:**
                {contract_content}
                """
                chunk_context.append(structured_context.strip())
                file_counter += 1

            if chunk_context:
                if chunk_idx == 0:
                    chunk_context.append("\n\nTotal Contracts Files Found for the user query: " + str(total_files_found))
                final_chunks.append("\n\n".join(chunk_context))

        if not final_chunks:
            logger.info(_log_message("No contexts retrieved from any files", "call_rag", MODULE_NAME))
            return ""

        return final_chunks

    except Exception as e:
        logger.error(_log_message(f"Error in call_rag while context retrieval: {e}", "call_rag", MODULE_NAME))
        return ""
