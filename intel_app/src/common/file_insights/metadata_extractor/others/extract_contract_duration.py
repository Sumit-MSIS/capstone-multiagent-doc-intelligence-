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
MODULE_NAME = "extract_contract_duration"

@mlflow.trace(name="Metadata Others - Get Contract Duration")
def get_contract_duration(user_id,org_id,file_id,tag_ids,logger ):
    try:
        custom_filter = {"file_id": {"$eq": file_id}}
        top_k = 10
        keyword_search_query = """
        "contract duration"
        OR "term"
        OR "time period"
        OR "Tenure"
        OR "For a period of"
        OR "Shall continue until"
        OR "Shall remain in force for"
        OR "Valid for"
        """
        semantic_query = "Identify the duration of this document, meaning the time period for which it is valid"

        # query = "what is the contract duration?"
        context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, semantic_query, file_id, user_id, org_id, tag_ids, logger,keyword_search_query)
        context_text = get_chunks_text(context_chunks)
      
        logger.info(_log_message(f"Context Retrieved: {len(context_text)} \n {context_text}","get_contract_duration", MODULE_NAME))
        context_text = "\n\n---\n\n".join(context_text)

        system_prompt = """You are an expert assistant specialized in analyzing legal contracts to extract the contract duration — the total time span for which the contract is valid. Use contextual clues and legal phrasing to identify the duration, which may be explicitly stated (e.g., "two years", "90 days","36 months") or inferred from specific start and end dates. If inferred, calculate the duration precisely in months. If the inferred durations is in years or days convert it into months, multiply by 12 to get months from years and divide by 30 to get months from days (round to 2 decimal place).  Return "null" if the duration is vague, incomplete, or cannot be confidently determined."""
        
        ## WHEN ASKING LLM TO CALCULATE THE MONTHS DURATION FROM DATES, IT IS NOT ABLE TO CORRECTLY DO BASIC MATHEMATICAL OPERATIONS 
        ## SOME FAILURE CASE : "Contract Duration": "38.87 months", dev_intel_app | "Reasoning": "The longest term is from December 1, 2016 through February 26, 2019. Calculating the duration: December 1, 2016 to February 26, 2019 is 2 years, 2 months, and 26 days. Converting to months: 2 years = 24 months, 2 months = 2 months, 26 days = 26/30 = 0.87 months. Total duration is 24 + 2 + 0.87 = 26.87 months." dev_intel_app | }
        ## THIS PROMPT WILL ONLY EXTRACT CONTRACT DURATION IF IT MENTIONED IN YEARS, MONTHS OR DAYS BUT IF DURATION NEEDS TO BE DERIVED FROM TWO DATES, WE WILL HANDLE THAT SEPRATELY IN update_contract_duation() in extract_metadata_parallely.py
        user_prompt = f"""
                
        ### Context:
        {context_text}
        ----

        Extract the contract duration from the provided legal contract or agreement. 
        The contract duration is the total time span for which the contract is valid, which may be explicitly stated (e.g., "two years," "90 days").
        - "term of [X years/months/days]"
        - "valid for a [X years/months.days] period"
        - "duration of the agreement"
        - "contract period"
        - "shall remain in force for"

        - If multiple references to duration exist (e.g., both a date range and an explicit term such as "3 years"):
            - Always prioritize the explicit numeric/textual term.
            - Only calculate based on start and end dates when NO explicit numeric/textual term is provided.
            - DO NOT mix or compare explicit terms with calculated ranges.
        - If an Initial Term is specified and renewal language exists but no fixed cumulative cap is stated, select the Initial Term (or other definite term) rather than assuming renewals will occur.
        - Normalize all results into **months**:
            - Years → multiply by 12. Example: "2 years" → "24 months"
            - Days → divide by 30.436 (round to 1 decimal place). Example : "110 days" → round(110/30.436,1) → "3.6 months"
            - Date ranges → compute days difference between start and end date →  total days divide by 30.436, round to 1 decimal places. 
                Example: Start date: "2016-01-01" End date: "2016-08-01" → total days = 213 days → round(213/30.436,1) = 7.0 months
        - If the duration is unclear, incomplete (e.g., only a start date), or uses placeholders like "__/__/____", return "null".
        
        --- 

        ### Strictness Rule:
        - The duration string must include: The length of the term **in months**        
        ---

        ### Output Format:
        - Return a minimal valid JSON object:
        {{
            "Contract Duration": "<duration in months, or null>",
            "Reasoning": "<brief explanation of how the duration was determined>"
        }}

        - If you cannot determine the duration and/or dates reliably, return:
        {{
            "Contract Duration": "null",
            "Reasoning": "No contract duration or explicit start and end dates found"
        }}

        Example 1:
        Context: "Term starting on February 1, 2016 for a three-year period."
        Contract duration Normalization : 3 Years = 3 * 12 = 36 months
        Output:
        {{
            "Contract Duration": "36 months",
            "Reasoning": "Term is starting on Feb 1, 2016 for a period of 3 years, 3 years are 36 months"
        }}

        Example 2:
        Context: "Agreement will be valid for a duration of 90 days"
        Contract duration Normalization : 90 days = 90 / 30.436 = 3 months
        Output:
        {{
            "Contract Duration": "3 months",
            "Reasoning": "Based on the context provided, the agreement remains valid for 90 days which makes it 3 months"
        }}

        """
        json_schema = {"type": "json_object"}
        
        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
        logger.info(_log_message(f"LLM Response: {llm_response}","get_contract_duration", MODULE_NAME))
        print(llm_response)
        return llm_response
    except Exception as e:
        logger.error(_log_message(f"Error in get_contract_duration: {str(e)}","get_contract_duration", MODULE_NAME))
        return None
    