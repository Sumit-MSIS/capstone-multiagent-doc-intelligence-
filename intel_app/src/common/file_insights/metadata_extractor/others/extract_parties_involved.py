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
MODULE_NAME = "extract_parties_involved"

@mlflow.trace(name="Metadata Others - Get Parties Involved")
def get_parties_involved(user_id, org_id, file_id, tag_ids, logger):
    try:
        custom_filter = {"file_id": {"$eq": file_id}}
        top_k = 10
        query = "Identify individual person or Entities names present in the contract"
        context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, query, file_id, user_id, org_id, tag_ids, logger)
        context_text = get_chunks_text(context_chunks)

        logger.info(_log_message(f"Context Retrieved: {len(context_text)}","get_parties_involved", MODULE_NAME))
        context_text = "\n\n---\n\n".join(context_text)

        system_prompt = """You are an expert assistant specialized in analyzing legal contracts to extract the actual names of parties involved, defined as the specific individuals or entities explicitly named in the agreement. Your task is to identify these names using contextual clues and legal phrasing, excluding generic placeholders, and return them in the specified JSON format."""


        user_prompt = f"""
        Extract the actual names of the parties involved in the provided legal contract or agreement, defined as the specific individuals or entities explicitly named as signatories or participants in the agreement.

        ### Instructions:
        - Identify the names of parties from explicit references in the contract, such as in the preamble, signature blocks, or clauses identifying the agreement’s participants (e.g., “This agreement is entered into between BeOne Medicines USA, Inc. and John Smith” yields “BeOne Medicines USA, Inc., John Smith”).
        - Only extract real, specific names of individuals (e.g., “John Smith”) or entities (e.g., “BeOne Medicines USA, Inc.”, “Acme Corporation”).
        - Exclude generic placeholders or roles (e.g., “Company”, “Service Provider”, “HCP”, “Client”) unless accompanied by a specific name (e.g., “Acme Corporation, the Service Provider” yields “Acme Corporation”).
        - Return the names as a single string, with multiple parties separated by commas (e.g., “BeOne Medicines USA, Inc., John Smith”).
        - If no real party names are explicitly stated, return "null".
        - Ignore references to non-party entities, such as third-party beneficiaries or mentioned organizations not signing the contract, unless they are explicitly identified as parties to the agreement.

        ### Examples:
        - “This agreement between BeOne Medicines USA, Inc. and Jane Doe” → “BeOne Medicines USA, Inc., Jane Doe”
        - “Entered into by Acme Corporation and John Smith, MD” → “Acme Corporation, John Smith”
        - “Between the Company and the Consultant” → “null”
        - “Signed by XYZ Ltd. under California law” → “XYZ Ltd.”
        - “Agreement with HealthCare Provider (HCP)” → “null”
        - “Between MediTech Inc., Global Pharma, and Dr. Emily Chen” → “MediTech Inc., Global Pharma, Emily Chen”

        ### Context:
        {context_text}

        ### Output:
        - Provide a minimal valid JSON object: {{"Parties Involved": "<comma separated parties name or null>"}}
        - No extra text, explanations, or markers like '```' or '```json
        """

        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "parties_involved_extraction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "Parties Involved": {
                            "type": ["string", "null"],
                            "description": "Comma-separated names of real individuals or entities explicitly identified as parties to the contract, or null if no such names are found."
                        }
                    },
                    "required": ["Parties Involved"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

        
        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
        logger.info(_log_message(f"LLM Response: {llm_response}","get_parties_involved", MODULE_NAME))
        return llm_response
    except Exception as e:
        logger.error(_log_message(f"Error in get_parties_involved: {str(e)}","get_parties_involved", MODULE_NAME))
        return None