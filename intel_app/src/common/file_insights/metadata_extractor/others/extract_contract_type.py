from src.common.hybrid_retriever.hybrid_search_retrieval import get_context_from_pinecone, get_chunks_text
from src.common.logger import _log_message
from src.config.base_config import config
from src.common.file_insights.llm_call import dynamic_llm_call
import mlflow
from openai import OpenAI
from opentelemetry import context as ot_context
import json
import time
from src.common.llm.factory import get_llm_client
TRACKING_URI = config.MLFLOW_TRACKING_URI
ENV = config.MLFLOW_ENV

# mlflow.config.enable_async_logging()
mlflow.openai.autolog()

mlflow.set_tracking_uri(TRACKING_URI)


openai_client = get_llm_client(async_mode=False)


DOCUMENT_SUMMARY_AND_CHUNK_INDEX = config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX
PINECONE_API_KEY = config.PINECONE_API_KEY
MODULE_NAME = "extract_contract_type"

@mlflow.trace(name="Metadata Others - Get Contract Type")
def get_contract_type(user_id, org_id, file_id, file_name, tag_ids, logger):
    try:
        custom_filter = {"file_id": {"$eq": file_id}, "page_no": {"$in": [1,2,3,4,5]}}
        top_k = 10
        query = "what is the contract Type?"
        context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, query, file_id, user_id, org_id, tag_ids, logger)
        context_text = get_chunks_text(context_chunks)

        logger.info(_log_message(f"Context Retrieved: {len(context_text)}","get_contract_type", MODULE_NAME))
        context_text = "\n\n---\n\n".join(context_text)

        system_prompt = """You are a legal expert specializing in contract analysis. Your task is to read the provided text and determine the type or nature of the contract based solely on its content. Do not rely on the file name or title; analyze only the actual text content."""
        
        user_prompt = f"""Identify the type of contract from the provided text.

        ### File Name: {file_name}
        
        ### File Content: 
        {context_text}

        ### Output:
        - Provide a minimal valid JSON object: {{"Contract Type": "<type of contract>"}}
        - No extra text, explanations, or markers like '```' OR '```json'."""



        json_schema = {"type": "json_object"}


        attempt = 0
        max_retries = 3
        while attempt < max_retries:
            attempt += 1

            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}],
                temperature=0,
                top_p=1,
                response_format=json_schema
            )
            response_content = response.choices[0].message.content.strip()
            logger.info(_log_message(f"LLM Response Attempt {attempt}: {response_content}","get_contract_type", MODULE_NAME))

            try:
                # Try parsing the JSON
                json_response = json.loads(response_content)


                import re

                raw_type = json_response.get("Contract Type", "")
                # Remove only standalone words "contract" or "agreement" (case-insensitive)
                cleaned_raw_type = re.sub(r"\b(contract|agreement)\b", "", raw_type, flags=re.IGNORECASE).strip()
                logger.info(_log_message(f"Cleaned Contract Type: {cleaned_raw_type}","get_contract_type", MODULE_NAME))
                json_response["Contract Type"] = cleaned_raw_type
                logger.info(_log_message(f"Parsed Contract Type: {json_response}","get_contract_type", MODULE_NAME))
                return json_response  # Exit after successful parse

            except json.JSONDecodeError as e:
                logger.error(_log_message(f"JSON decoding failed on attempt {attempt}: {str(e)}","get_contract_type", MODULE_NAME))
                if attempt < max_retries:
                    user_prompt += f"\n\nAttempt {attempt} failed with error: {str(e)}. Generate valid response."
                    time.sleep(1)  # Optional: wait 1 second before retrying
                else:
                    logger.error(_log_message("Max retries reached. Returning raw response.","get_contract_type", MODULE_NAME))
                    return {"Contract Type": "null"}  # Return raw if all retries fail

        return {"Contract Type": "null"}
    except Exception as e:
        logger.error(_log_message(f"Error in get_contract_type: {str(e)}","get_contract_type", MODULE_NAME))
        return None