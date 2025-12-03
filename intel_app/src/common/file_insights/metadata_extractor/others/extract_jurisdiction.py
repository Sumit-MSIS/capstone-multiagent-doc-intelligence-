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
MODULE_NAME = ""

@mlflow.trace(name="Metadata Others - Get Jurisdiction")
def get_jurisdiction(user_id, org_id, file_id, tag_ids, logger, regex_extracted_dates:str= ""):
    try:
        custom_filter = {"file_id": {"$eq": file_id}}
        top_k = 10
        keyword_search_query = """
        "jurisdiction"
        OR "legal authority"
        OR "governing law"
        OR "regime"
        OR "dispute resolution"
        OR "governed by"
        """
        semantic_query = "Identify the jurisdiction of this document, the legal authority or location governing the contract’s applicable laws and courts for dispute resolution."

        query = "What is the jurisdiction?"
        context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, semantic_query, file_id, user_id, org_id, tag_ids, logger,keyword_search_query)
        context_text = get_chunks_text(context_chunks)

        logger.info(_log_message(f"Context Retrieved: {len(context_text)}","get_jurisdiction", MODULE_NAME))
        context_text = "\n\n---\n\n".join(context_text)

        system_prompt = """You are an expert assistant specialized in analyzing legal contracts to extract the jurisdiction, defined as the legal authority or location governing the contract's applicable laws and courts. Your task is to identify the jurisdiction using legal phrasing and contextual clues, such as governing law clauses, and return it in the specified JSON format."""
        
        user_prompt = f"""
        Extract the jurisdiction from the provided legal contract or agreement, defined as the legal authority or location governing the contract’s applicable laws and courts for dispute resolution.

        ### Instructions:
        - Identify the jurisdiction from explicit clauses, such as “governed by,” “subject to,” or “construed in accordance with” (e.g., “This agreement is governed by the laws of New York” yields “New York”).
        - If no explicit governing law clause is found, infer the jurisdiction from phrases indicating the contract is valid, permissible, or enforceable “under <Place> law” (e.g., “enforceable under Texas law” yields “Texas”).
        - Return the jurisdiction name as a string, using the full form for clarity (e.g., “NY” or “New York State” becomes “New York”; “CA” becomes “California”; “UK” becomes “England and Wales” unless specified otherwise, such as “Scotland”).
        - If the jurisdiction is vague, ambiguous, or not explicitly stated (e.g., “applicable laws” or “local laws” without a location), return "null".
        - Ignore locations unrelated to the governing law, such as party addresses, place of signing, or performance (e.g., “signed in London” or “delivered in Florida”), unless explicitly tied to the governing law.

        ### Examples:
        - “Governed by the laws of NY” → “New York”
        - “Subject to California law” → “California”
        - “Construed under the laws of England” → “England and Wales”
        - “Enforceable under FL law” → “Florida”
        - “Governed by applicable federal laws” → “null”
        - “Signed in Tokyo under Japanese law” → “Japan”
        - “Interpreted under the laws of Ontario” → “Ontario”

        ### Context:
        {context_text}

        ### Output:
        - Provide a minimal valid JSON object: {{"Jurisdiction": "<jurisdiction string or null>"}}
        - No extra text, explanations, or markers like '```' or '```json
        """

        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "jurisdiction_extraction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "Jurisdiction": {
                            "type": ["string", "null"],
                            "description": "The jurisdiction governing the contract (e.g., 'New York', 'California', 'England and Wales'), or null if not explicitly or clearly stated."
                        }
                    },
                    "required": ["Jurisdiction"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

        
        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
        logger.info(_log_message(f"LLM Response: {llm_response}","get_jurisdiction", MODULE_NAME))
        return llm_response
    except Exception as e:
        logger.error(_log_message(f"Error in get_jurisdiction: {str(e)}","get_jurisdiction", MODULE_NAME))
        return None