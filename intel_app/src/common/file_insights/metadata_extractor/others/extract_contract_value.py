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
MODULE_NAME = "extract_contract_value"

@mlflow.trace(name="Metadata Others - Get Contract Value")
def get_contract_value(user_id,org_id,file_id, tag_ids, logger, regex_extracted_dates:str= "" ):
    try:
        custom_filter = {"file_id": {"$eq": file_id}}
        top_k = 10
        query = "What is the contract value?"
        context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, query, file_id, user_id, org_id, tag_ids, logger)
        context_text = get_chunks_text(context_chunks)

        logger.info(_log_message(f"Context Retrieved: {len(context_text)}","get_contract_value", MODULE_NAME))
        context_text = "\n\n---\n\n".join(context_text)

        system_prompt = """You are an expert assistant specialized in analyzing legal contracts to extract and calculate the total contract value, defined as the complete monetary amount of the contract. Your task is to identify an explicitly stated total value or compute it from clear components, such as installment payments, and return it as a float amount in the specified JSON format."""
        
        user_prompt = f"""
        Extract the total contract value from the provided legal contract or agreement, defined as the complete monetary amount of the contract, either explicitly stated or calculable from clear components like installment payments.

        ### Instructions:
        - Identify an explicit total contract value (e.g., "Total contract value is $500,000") or calculate it from clear components (e.g., "$10,000 monthly for 12 months" equals 120000).
        - Return the value as a float amount, excluding currency symbols (e.g., $, €) or words (e.g., dollars, euros).
        - If a calculated or stated value is non-integer, round to the nearest integer.
        - If the total value is not explicitly stated, not calculable, or vague (e.g., "reasonable costs" or "market rates"), return "null".
        - Ignore monetary amounts unrelated to the total contract value, such as deposits, penalties, or taxes, unless explicitly included in the total.

        ### Context:
        {context_text}

        ### Output:
        - Provide a minimal valid JSON object: {{"Contract Value": <total contract value or null>}}
        - No extra text, explanations, or markers like '```' or '```json
        """

        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "contract_value_extraction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "Contract Value": {
                            "type": ["number", "null"],
                            "description": "The total contract value as a float (rounded if needed), or null if it is not found or not calculable."
                        }
                    },
                    "required": ["Contract Value"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

        
        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
        logger.info(_log_message(f"LLM Response: {llm_response}","get_contract_value", MODULE_NAME))
        return llm_response
    except Exception as e:
        logger.error(_log_message(f"Error in get_contract_value: {str(e)}","get_contract_value", MODULE_NAME))
        return None