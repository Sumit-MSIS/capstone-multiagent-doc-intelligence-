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
MODULE_NAME = "extract_risk_mitigation_score"

@mlflow.trace(name="Metadata Others - Get Risk Mitigation Score")
def get_risk_mitigation_score(user_id,org_id,file_id,tag_ids, logger ):
    try:
        custom_filter = {"file_id": {"$eq": file_id}}
        top_k = 10
        query = "What is risk mitigation score?"
        context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, query, file_id, user_id, org_id, tag_ids, logger)
        context_text = get_chunks_text(context_chunks)        

        logger.info(_log_message(f"Context Retrieved: {len(context_text)}","get_risk_mitigation_score", MODULE_NAME))
        context_text = "\n\n---\n\n".join(context_text)

        system_prompt = """You are an expert assistant specialized in analyzing legal contracts to extract the risk mitigation score, defined as a numerical or descriptive evaluation of potential risks and their mitigation strategies. Your task is to identify the score using contextual clues and legal phrasing and return it in the specified JSON format."""
        
        user_prompt = f"""
        Extract the risk mitigation score from the provided legal contract or agreement, defined as a numerical or descriptive evaluation of potential risks and their mitigation strategies.

        ### Instructions:
        - Identify the risk mitigation score from explicit references in the contract, such as clauses or sections detailing risk assessment, mitigation strategies, or a specific score (e.g., “Risk mitigation score: 85%” yields “85%”; “Risk profile rated as Low” yields “Low”).
        - Return the score as a string, preserving its exact form (numerical, e.g., “85”, “75%”, or descriptive, e.g., “Low”, “High”) as stated in the contract.
        - If no risk mitigation score is explicitly stated or inferable, or if the reference is too vague (e.g., “risks are adequately managed” without a score), return "null".
        - Ignore references to unrelated metrics, such as financial performance scores or compliance ratings, unless explicitly tied to risk mitigation.

        ### Examples:
        - “Risk mitigation score: 85%” → “85%”
        - “Risk profile rated as Low” → “Low”
        - “Mitigation strategy effectiveness: High” → “High”
        - “Risk assessment score of 7.5/10” → “7.5/10”
        - “Risks are adequately managed” → “null”
        - “Financial risk score: 90” → “null”

        ### Context:
        {context_text}

        ### Output:
        - Provide a minimal valid JSON object: {{"Risk Mitigation Score": "<string or null>"}}
        - No extra text, explanations, or markers like '```' or '```json
        """

        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "risk_mitigation_score_extraction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "Risk Mitigation Score": {
                            "type": ["string", "null"],
                            "description": "The explicitly stated risk mitigation score from the contract, either numerical (e.g., '85%', '7.5/10') or descriptive (e.g., 'Low', 'High'), or null if not found or too vague."
                        }
                    },
                    "required": ["Risk Mitigation Score"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

        
        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
        logger.info(_log_message(f"LLM Response: {llm_response}","get_risk_mitigation_score", MODULE_NAME))
        return llm_response
    except Exception as e:
        logger.error(_log_message(f"Error in get_risk_mitigation_score: {str(e)}","get_risk_mitigation_score", MODULE_NAME))
        return None