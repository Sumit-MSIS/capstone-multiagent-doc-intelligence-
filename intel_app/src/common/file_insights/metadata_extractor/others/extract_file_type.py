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
MODULE_NAME = "extract_file_type"

@mlflow.trace(name="Metadata Others - Get File Type")
def get_file_type(user_id, org_id, file_id, logger):
    try:
        # custom_filter = {"file_id": {"$eq": file_id}, "page_no": {"$eq": 1}}
        # top_k = 10
        # query = ""
        # context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, query, file_id, user_id, org_id, logger)
        # context_text = get_chunks_text(context_chunks)

        # logger.info(_log_message(f"Context Retrieved: {len(context_text)}","", MODULE_NAME))

        # system_prompt = """You are an assistant that understands and extracts relevant dates from the legal agreement's and contract's context provided to you.
        #     You also provide the dates extractes in a specific format as per the instructions provided to you."""
        
        # user_prompt = f"""
        #     ###Context: {context_text}"""
        
        # llm_response = dynamic_llm_call(user_prompt, system_prompt, logger)
        # logger.info(_log_message(f"LLM Response: {llm_response}","", MODULE_NAME))
        
        return {"File Type": "Contract"}
    except Exception as e:
        logger.error(_log_message(f"Error in : {str(e)}","", MODULE_NAME))
        return None