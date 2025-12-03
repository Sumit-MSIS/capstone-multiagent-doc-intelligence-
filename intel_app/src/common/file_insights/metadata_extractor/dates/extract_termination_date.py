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
MODULE_NAME = "extract_termination_date"

@mlflow.trace(name="Metadata Dates - Get Termination Date")
def get_termination_date(user_id, org_id, file_id, tag_ids, logger, regex_extracted_dates:str = ""):
    try:
        custom_filter = {"file_id": {"$eq": file_id}}
        top_k = 15 # initially the topk was kept as 5 but in some cases the right chunk was not in top 5 so it missed the date, which is why we are keeping the value a little high
        query = "Identify the termination date of this document, meaning the explicit date on which the agreement ends early due to termination provisions (e.g., termination clauses, end of agreement, termination effective on), not expiration, renewal, or effective dates."
        keyword_search_query = """
        "termination date" 
        OR "this agreement shall terminate on" 
        OR "agreement terminates on" 
        OR "shall terminate effective" 
        OR "termination shall be effective on" 
        OR "agreement ends on" 
        OR "termination of this agreement" 
        OR "end of agreement" 
        OR "agreement shall end on" 
        OR "contract shall terminate on" 
        OR "shall be terminated on" 
        OR "terminated effective" 
        OR "terminate upon completion on" 
        OR "termination effective as of"
        """
        context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, query, file_id, user_id, org_id, tag_ids=tag_ids, logger=logger, keyword_search_query=keyword_search_query)
        context_text = get_chunks_text(context_chunks)        

        logger.info(_log_message(f"Context Retrieved: {len(context_text)}","get_termination_date", MODULE_NAME))

        context_text = "\n\n---\n\n".join(context_text)

        system_prompt = """
        You are ChatGPT, a large language model trained by OpenAI.  
        Knowledge cutoff: 2024-06  

        ## Personality & Style
        - Act as a precise, detail-oriented **legal document specialist**.  
        - Be professional, concise, and conservative: never guess or infer unstated dates.  
        - Always prioritize explicit legal wording over assumptions.  

        ## Role & Objective
        Your role is to analyze legal documents of any type (e.g., contracts, agreements, addendums, amendments, memoranda of understanding, leases, treaties, policies, or related instruments) to identify the document’s **termination date** — the date when the agreement formally ends early due to explicit termination provisions, as distinct from expiration, renewal, or effective dates.  

        ## Scope & Terminology
        Treat the following as candidate indicators when they clearly denote termination:  
        - “termination date”, “this agreement shall terminate on”, “agreement terminates on”  
        - “shall terminate effective”, “termination shall be effective on”  
        - “agreement ends on”, “end of agreement”  
        - “terminated upon completion on [date]”, “termination effective as of”  

        ## Decision Rules (Order of Preference)
        1. **Explicit Termination Clause Wins:** If the text explicitly provides a termination date, use that exact date.  
        2. **Mutual Termination Clauses:** If the contract specifies a fixed date for mutual termination, use that date.  
        3. **Event-driven Termination:** If the agreement ties termination to a specific event with a concrete date (e.g., “shall terminate upon completion on June 30, 2025”), extract that date.  
        4. **Relative Dates:** Only compute relative expressions (e.g., “90 days after notice”) if both reference dates are explicitly present and unambiguous. Otherwise → `null`.  
        5. **Conflict Handling:** If multiple termination dates exist, choose the one explicitly labeled as “Termination Date” or most clearly tied to termination.  
        6. **Exclusions:** Do not confuse expiration, renewal, survival, or effective dates with termination.  

        ## Handling Provided Regex Dates
        - You may reference regex-extracted dates from the context, but only validate and use them if the surrounding text explicitly ties them to termination.  
        - Ignore regex dates not supported by termination wording.  

        ## Formatting
        - Output **strict JSON** with exactly two keys:  
        - `"Termination Date"` → an ISO date (`YYYY-MM-DD`) or `null`.  
        - `"Reasoning"` → a brief explanation, citing the exact clause/phrase.  

        ## Constraints
        - Do not infer or invent dates (e.g., from today’s date, metadata, or unrelated sections).  
        - Do not normalize partial dates. If the day or month is missing → `null`.  
        - Convert valid written dates (e.g., “June 30, 2025”) into `YYYY-MM-DD`.  
        - Output only minimal JSON, without extra text, formatting, or code blocks.  
        """

        
        #######################################################################################################################################################################################################
        #######################################################################################################################################################################################################
        user_prompt = f"""
        ### Legal Document Context
        ---
        {context_text}
        ---

        ### Additional Data (optional)
        Regex-extracted dates: {regex_extracted_dates}

        ### Task
        Extract the document's **termination date** in `YYYY-MM-DD` format based only on explicit language in the context above.

        ### Instructions
        1. Prefer explicit termination clauses (e.g., “termination date”, “this agreement shall terminate on”, “agreement ends on”).  
        2. Accept mutual or event-driven termination dates only if tied to a clear calendar date.  
        3. For relative dates (e.g., “90 days after notice”), compute only if the base date is explicitly present and unambiguous. Otherwise → `null`.  
        4. Exclude expiration, renewal, survival, or effective dates unless explicitly tied to termination.  
        5. Return `null` if:  
        - No explicit termination date exists.  
        - The date is incomplete or placeholder-only.  
        - The clause is conditional or ambiguous.  

        ### Output
        {{"Termination Date": <YYYY-MM-DD or null>, "Reasoning": "<Brief justification by citing the exact clause or phrase that determines the termination date>"}}

        - Reasoning must:  
        (a) Reference the clause/phrase (≤20 words).  
        (b) Explain why it qualifies as a termination date (not expiration/renewal).  
        - If returning `null`, Reasoning must briefly explain why.

        ### Example Outputs
        - {{"Termination Date": "2025-06-30", "Reasoning": "Clause: 'This agreement shall terminate upon completion on June 30, 2025.' This is an explicit termination date."}}  
        - {{"Termination Date": null, "Reasoning": "Termination rights are described, but no fixed termination date is specified."}}  

        Output must be **minimal JSON only** — no markdown, no code fences, no extra text.
        """

        json_schema = {"type": "json_object"}

        
        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
        print("termniation date",llm_response)
        logger.info(_log_message(f"LLM Response: {llm_response}","get_termination_date", MODULE_NAME))
        return llm_response
    except Exception as e:
        logger.error(_log_message(f"Error in get_termination_date: {str(e)}","get_termination_date", MODULE_NAME))
        return {"Termination Date": "null"}