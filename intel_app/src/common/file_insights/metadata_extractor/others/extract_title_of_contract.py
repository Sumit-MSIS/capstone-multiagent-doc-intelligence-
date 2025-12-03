from src.common.hybrid_retriever.hybrid_search_retrieval import get_context_from_pinecone, get_chunks_text
from src.common.logger import _log_message
from src.config.base_config import config
from src.common.file_insights.llm_call import dynamic_llm_call
import mlflow
mlflow.openai.autolog()

DOCUMENT_SUMMARY_AND_CHUNK_INDEX = config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX
PINECONE_API_KEY = config.PINECONE_API_KEY
MODULE_NAME = "extract_title_of_contract"

@mlflow.trace(name="Get Title of Contract")
def get_title_of_contract(user_id, org_id, file_id, tag_ids, logger):
    try:
        custom_filter = {"file_id": {"$eq": file_id}, "page_no": {"$in": [1,2,3,4,5]}}
        top_k = 10
        query = "Identify title of the contract, contract name?"
        context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, query, file_id, user_id, org_id, tag_ids, logger)
        context_text = get_chunks_text(context_chunks)    

        logger.info(_log_message(f"Context Retrived: {len(context_text)}","get_title_of_contract", MODULE_NAME))
        context_text = "\n\n---\n\n".join(context_text)

        system_prompt = """
        You are an expert assistant that strictly extracts the official title of a legal agreement or contract from the provided text. Your task is to identify and return the title exactly as it appears, without guessing, paraphrasing, or interpreting. If no clear, explicitly written title is found, return 'null'.
        """

        user_prompt = f"""
        Extract the official title of the legal contract or agreement from the provided text, defined as the concise label explicitly stated at the top of the document or in the introductory clauses, summarizing the purpose of the agreement (e.g., 'Service Agreement' or 'Purchase Contract').

        ### Instructions:
        - Identify the title from explicit references in the contract, such as the heading at the top of the document or in the introductory clauses (e.g., 'This Service Agreement is entered into...' yields 'Service Agreement').
        - Extract the title exactly as written, preserving capitalization, punctuation, and wording.
        - Do not infer, paraphrase, or modify the title in any way.
        - If no specific title is explicitly stated, return 'null'.
        - Ignore generic descriptions or references to the document type (e.g., 'this agreement' or 'the contract') unless a specific title is provided.
        - Return only the title or 'null' as a string in a JSON object, with no extra text or formatting.

        ### Examples:
        - 'Service Agreement\nThis agreement is made...' → 'Service Agreement'
        - 'PURCHASE CONTRACT BETWEEN ABC AND XYZ' → 'PURCHASE CONTRACT'
        - 'This agreement is entered into between...' → 'null'
        - 'NON-DISCLOSURE AGREEMENT (NDA)\n...' → 'NON-DISCLOSURE AGREEMENT (NDA)'
        - 'The contract governs the sale of goods...' → 'null'
        - 'Master Service Agreement, dated...' → 'Master Service Agreement'

        ### Context:
        {context_text}

        ### Output:
        - Provide a minimal valid JSON object: {{"Title of the Contract": "<title>"}}
        - No extra text, explanations, or markers like '```' or '```json'
        """

        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "contract_title_extraction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "Title of the Contract": {
                            "type": ["string", "null"],
                            "description": "The exact official title of the legal agreement or contract as explicitly stated in the document, preserving original formatting. If not explicitly stated, return null."
                        }
                    },
                    "required": ["Title of the Contract"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

        
        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
        logger.info(_log_message(f"LLM Response: {llm_response}","get_title_of_contract", MODULE_NAME))
        return llm_response
    except Exception as e:
        logger.error(_log_message(f"Error in get_title_of_contract: {str(e)}","get_title_of_contract", MODULE_NAME))
        return {"Title of the Contract": "null"}