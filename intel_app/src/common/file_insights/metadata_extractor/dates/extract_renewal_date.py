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
MODULE_NAME = "extract_renewal_date"

@mlflow.trace(name="Metadata Dates - Get Renewal Date")
def get_renewal_date(user_id, org_id, file_id, tag_ids, logger, regex_extracted_dates: str = ""):
    try:
        custom_filter = {"file_id": {"$eq": file_id}}
        top_k = 15 # initially the topk was kept as 5 but in some cases the right chunk was not in top 5 so it missed the date, which is why we are keeping the value a little high

        query = "Identify the renewal or extension date of this document, meaning the date on which the agreement is set to automatically or explicitly renew, extend, or continue in effect, based on explicit contractual language."
        keyword_search_query = """
        "renewal date" 
        OR "shall automatically renew on" 
        OR "agreement renews effective" 
        OR "shall renew effective" 
        OR "this agreement shall renew on" 
        OR "renewal term begins" 
        OR "contract extends to" 
        OR "shall be extended starting from" 
        OR "extension date" 
        OR "shall automatically extend to" 
        OR "extended until" 
        OR "agreement is renewed as of" 
        OR "renewal period commences" keyword_search_query
        OR "renewal effective as of" 
        OR "renewal shall occur on"
        """

        context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, query, file_id, user_id, org_id, tag_ids=tag_ids, logger=logger, keyword_search_query=keyword_search_query)
        context_text = get_chunks_text(context_chunks)        

        logger.info(_log_message(f"Context Retrieved: {len(context_text)}","get_renewal_date", MODULE_NAME))
        context_text = "\n\n---\n\n".join(context_text)

        system_prompt = """
        You are ChatGPT, a large language model trained by OpenAI.  
        Knowledge cutoff: 2024-06  

        ## Personality & Style
        - Act as a precise, detail-oriented **legal document specialist**.  
        - Be professional, concise, and conservative: never guess or infer unstated dates.  
        - Always prioritize explicit legal wording over assumptions.  

        ## Role & Objective
        Your role is to analyze legal documents of any type (e.g., contracts, agreements, addendums, amendments, memoranda of understanding, leases, treaties, policies, or related instruments) to identify the document’s **renewal or extension date** — the date on which the contract automatically or explicitly renews or extends.

        ## Scope & Terminology
        Treat the following as candidate indicators when they clearly denote renewal/extension:  
        - “shall automatically renew on”, “renewal date is”, “agreement renews effective”  
        - “contract shall be extended starting from”, “term shall automatically extend to”, “extended until”  

        ## Decision Rules (Order of Preference)
        1. **Explicit Renewal Clause Wins:** If the text explicitly provides a renewal/extension date, use that exact date.  
        2. **Conditional/Relative Clauses:** If renewal is tied to an event (e.g., “shall renew 12 months after expiration”), extract the referenced event’s date **only if explicitly present and unambiguous**.  
        3. **Conflicts:** If multiple candidate dates exist, prefer the one explicitly labeled “renewal/extension date”; otherwise select the clause that most clearly states when the contract “shall renew/extend.”  
        4. **Placeholders/Incomplete:** If the date is blank, partial (e.g., only year), or a placeholder (e.g., “as of ________”), return `null`.  
        5. **Ambiguity:** If the renewal/extension date cannot be unambiguously determined, return `null`.  

        ## Handling Provided Regex Dates
        - You may reference regex-extracted dates from the context, but only validate and use them if the surrounding language explicitly ties them to renewal/extension.  
        - Ignore regex dates not justified by explicit contract wording.  

        ## Formatting
        - Output **strict JSON** with exactly two keys:  
        - `"Renewal Date"` → an ISO date (`YYYY-MM-DD`) or `null`.  
        - `"Reasoning"` → a brief explanation, citing the exact clause/phrase.  

        ## Constraints
        - Do not infer or invent dates (e.g., from today’s date, metadata, or unrelated sections).  
        - Do not normalize partial dates. If the day or month is missing → `null`.  
        - Convert valid written dates (e.g., “1 January 2026”, “Jan. 1, 2026”) into `YYYY-MM-DD`.  
        - Output only minimal JSON, without extra text, formatting, or code blocks.  
        """

        user_prompt = f"""
        ### Legal Document Context
        ---
        {context_text}
        ---

        ### Additional Data (optional)
        Regex-extracted dates: {regex_extracted_dates}

        ### Task
        Extract the document's **renewal or extension date** in `YYYY-MM-DD` format based only on explicit language in the context above.

        ### Instructions
        1. Prefer explicit “renewal/extension” clauses.  
        2. If renewal is tied to an event, compute the date only if the event’s date is explicitly stated and unambiguous.  
        3. Ignore placeholders or incomplete dates (e.g., “as of ________”, “as of __________, 2027”).  
        4. Validate regex-extracted dates only if explicitly tied to renewal/extension in the context.  
        5. If ambiguous or not clearly stated, return `null`.  

        ### Output
        {{"Renewal Date": <YYYY-MM-DD or null>, "Reasoning": "<Brief justification by citing the exact clause or phrase that determines the renewal/extension date>"}}

        Output must be minimal JSON only—no extra text, formatting, or code fences.
        """



        # json_schema = {
        #     "type": "json_schema",
        #     "json_schema": {
        #         "name": "renewal_date_extraction",
        #         "schema": {
        #             "type": "object",
        #             "properties": {
        #                 "Renewal Date": {
        #                     "type": ["string", "null"],
        #                     "pattern": r"^\d{4}-\d{2}-\d{2}$",
        #                     "description": "The renewal or extension date in 'YYYY-MM-DD' format, or null if not found or ambiguous."
        #                 }
        #             },
        #             "required": ["Renewal Date"],
        #             "additionalProperties": False
        #         },
        #         "strict": True
        #     }
        # }

        json_schema = {"type": "json_object"}
        
        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
        logger.info(_log_message(f"LLM Response: {llm_response}","get_renewal_date", MODULE_NAME))
        return llm_response
    except Exception as e:
        logger.error(_log_message(f"Error in get_renewal_date: {str(e)}","get_renewal_date", MODULE_NAME))
        return {"Renewal Date": "null"}