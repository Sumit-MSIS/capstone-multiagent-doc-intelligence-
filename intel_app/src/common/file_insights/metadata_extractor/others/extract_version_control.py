from src.common.hybrid_retriever.hybrid_search_retrieval import get_context_from_pinecone, get_chunks_text
from src.common.logger import _log_message
from src.config.base_config import config
from src.common.file_insights.llm_call import dynamic_llm_call
import mlflow
from opentelemetry import context as ot_context

TRACKING_URI = config.MLFLOW_TRACKING_URI
ENV = config.MLFLOW_ENV

# mlflow.config.enable_async_logging()
mlflow.openai.autolog()

mlflow.set_tracking_uri(TRACKING_URI)
DOCUMENT_SUMMARY_AND_CHUNK_INDEX = config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX
PINECONE_API_KEY = config.PINECONE_API_KEY
MODULE_NAME = "extract_version_control"

@mlflow.trace(name="Metadata Others - Get Version Control")
def get_version_control(user_id,org_id,file_id,tag_ids, logger ):
    try:
        custom_filter = {"file_id": {"$eq": file_id}}
        top_k = 10
        query = "What is the version of this agreement?"
        context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, query, file_id, user_id, org_id, tag_ids, logger)
        context_text = get_chunks_text(context_chunks)        

        logger.info(_log_message(f"Context Retrieved: {len(context_text)}","get_version_control", MODULE_NAME))
        context_text = "\n\n---\n\n".join(context_text)

        system_prompt = """You are an expert assistant specialized in analyzing legal contracts to extract the version control information, defined as the version ID, contract number, or revision number used to track updates, amendments, or changes to the agreement. Your task is to identify and extract the version or contract number using contextual clues and legal phrasing, returning it in the specified JSON format."""

        user_prompt = f"""
        Extract the version ID, contract number, or revision number from the provided legal contract or agreement, used to track updates, amendments, or changes to the document.

        ### Instructions:
        - Identify the version ID, contract number, or revision number from explicit references in the contract, such as sections mentioning "Version," "Revision," "Contract No.," or similar identifiers (e.g., “This Agreement, Version 2.1, dated January 15, 2025” yields “2.1”).
        - Only extract specific identifiers explicitly stated in the contract.
        - Exclude vague or generic references (e.g., “this agreement” or “amended contract” without a specific version or number) unless accompanied by a clear identifier (e.g., “Amended Contract No. ABC123” yields “ABC123”).
        - Return the version or contract number as a concise string without extra formatting or text.
        - If no specific version ID, contract number, or revision number is explicitly stated, return "null".
        - Ignore references to unrelated numbers or identifiers not tied to version control (e.g., payment amounts, dates, or quantities).

        ### Examples:
        - “This Agreement, Version 3.0, dated March 1, 2025” → “3.0”
        - “Contract No. XY789 effective from June 2024” → “XY789”
        - “Revision 5 of the Master Service Agreement” → “5”
        - “The agreement is amended as of July 2025” → “null”
        - “Contract for services dated April 10, 2025” → “null”
        - “Version ID: 2025-REV-001” → “2025-REV-001”

        ### Context:
        {context_text}

        ### Output:
        - Provide a minimal valid JSON object: {{"Version Control": "<version or contract number or null>"}}
        - No extra text, explanations, or markers like '```' or '```json
        """

        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "version_control_extraction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "Version Control": {
                            "type": ["string", "null"],
                            "description": "The version ID, contract number, or revision number explicitly used to track updates or amendments to the contract, or null if not found."
                        }
                    },
                    "required": ["Version Control"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

        
        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
        logger.info(_log_message(f"LLM Response: {llm_response}","get_version_control", MODULE_NAME))
        return llm_response
    except Exception as e:
        logger.error(_log_message(f"Error in get_version_control: {str(e)}","get_version_control", MODULE_NAME))
        return None