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
MODULE_NAME = "extract_term_date"

@mlflow.trace(name="Metadata Dates - Get Term Date")
def get_term_date(user_id, org_id, file_id, tag_ids, logger, regex_extracted_dates: str = ""):
    try:
        custom_filter = {"file_id": {"$eq": file_id}}
        top_k = 15 # initially the topk was kept as 5 but in some cases the right chunk was not in top 5 so it missed the date, which is why we are keeping the value a little high

        query = "What is the term date of the agreement?"
        context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, query, file_id, user_id, org_id, tag_ids, logger)
        context_text = get_chunks_text(context_chunks)        

        logger.info(_log_message(f"Context Retrieved: {len(context_text)}","get_term_date", MODULE_NAME))

        context_text = "\n\n---\n\n".join(context_text)

        system_prompt = """You are a legal contract specialist specializing in identifying the start or commencement date of a contract, also referred to as the Term Date. You must use contextual clues and legal language to accurately extract only the date when the contract term begins. Ignore unrelated dates such as signature dates, effective dates, expiration dates, or date placeholders."""



        user_prompt = f"""
        ### Contract Context:
        ---
        {context_text}
        ---

        ### Task: Extract the Term Date (i.e., the date when the contract term begins) from the above contract in 'YYYY-MM-DD' format.

        ### Instructions:
            1. Identify the **Term Date**, which refers to the actual start or commencement of the contractual term.
            2. Look for specific legal phrasing, such as:
                - "The term shall commence on..."
                - "The agreement starts on..."
                - "The term of this agreement begins on..."
                - "Start date of the term is..."
                - "Commencement date of the term"

            3. Do **NOT** extract:
                - Signature date
                - Effective date (unless explicitly stated as the term start)
                - Termination, expiration, or renewal date

            4. Use any of the regex-extracted dates **only if** they clearly refer to the start of the contract term in the context.

            5. If no valid Term Date is found, or if the date is:
                - A placeholder (e.g., "________")
                - Incomplete (e.g., just the year "2022")
                - Ambiguous or missing

                Then return `"null"` as a string.

            6. Use regex-extracted dates ({regex_extracted_dates}) only if validated by the above contract's context as the explicit term date.


        ### Expected Output:
        {{"Term Date": <term start date in 'YYYY-MM-DD' format or null if no valid term date is found>}}

        Output should be a minimal json, DO NOT provide any extra words or markers '```' OR '```json'.
        """


        # json_schema = {
        #     "type": "json_schema",
        #     "json_schema": {
        #         "name": "term_date_extraction",
        #         "schema": {
        #             "type": "object",
        #             "properties": {
        #                 "Term Date": {
        #                     "type": ["string", "null"],
        #                     "pattern": r"^\d{4}-\d{2}-\d{2}$",
        #                     "description": "The term start date in 'YYYY-MM-DD' format, or null if ambiguous, missing, or incomplete."
        #                 }
        #             },
        #             "required": ["Term Date"],
        #             "additionalProperties": False
        #         },
        #         "strict": True
        #     }
        # }

        json_schema = {"type": "json_object"}
        
        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
        logger.info(_log_message(f"LLM Response: {llm_response}","get_term_date", MODULE_NAME))
        return llm_response
    except Exception as e:
        logger.error(_log_message(f"Error in get_term_date: {str(e)}","get_term_date", MODULE_NAME))
        return {"Term Date": "null"}