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
MODULE_NAME = "extract_expiration_date"

@mlflow.trace(name="Metadata Dates - Get Expiration Date")
def get_expiration_date(user_id,org_id,file_id,tag_ids, logger, regex_extracted_dates:str= ""):
    try:
        custom_filter = {"file_id": {"$eq": file_id}}
        top_k = 15 # initially the topk was kept as 5 but in some cases the right chunk was not in top 5 so it missed the date, which is why we are keeping the value a little high
        # the case which resulted in failure with topk = 5 is Grant Agreement_State of Oregon_Camas Valley.pdf raised by Prashanth
        semantic_query = "Identify the expiration date of this document, meaning the date on which it officially ends, expires, terminates, or ceases to be in effect."
        keyword_search_query = """
        "expiration date" 
        OR "shall expire on" 
        OR "expires on" 
        OR "terminates on" 
        OR "termination date" 
        OR "end of term" 
        OR "term shall end on" 
        OR "valid until" 
        OR "shall be in effect until" 
        OR "end date" 
        OR "contract ends on" 
        OR "termination of agreement" 
        OR "closing shall occur on" 
        OR "date of expiry" 
        OR "expiry date" 
        OR "until and including" 
        OR "valid through" 
        OR "date of termination"
        OR "duration of agreement"
        """

        
        context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, semantic_query, file_id, user_id, org_id, tag_ids=tag_ids, logger=logger, keyword_search_query=keyword_search_query)
        context_text = get_chunks_text(context_chunks)   

        logger.info(_log_message(f"Context Retrieved: {len(context_text)}","get_expiration_date", MODULE_NAME))
        context_text = "\n\n---\n\n".join(context_text)
        # logger.info(f"text for expiration {context_text}")
        system_prompt = """
        You are ChatGPT, a large language model trained by OpenAI.  
        Knowledge cutoff: 2024-06  

        ## Personality & Style
        - Act as a precise, detail-oriented **legal document specialist**.  
        - Be professional, concise, and conservative: never guess or infer unstated dates.  
        - Always prioritize explicit legal wording over assumptions.  

        ## Role & Objective
        Your role is to analyze legal documents of any type (e.g., contracts, agreements, addendums, amendments, memoranda of understanding, leases, treaties, policies, or related instruments) to identify the document’s **expiration date** — the date it officially ends, expires, terminates, or ceases to be in effect.  

        ## Scope & Terminology
        Treat the following as candidate indicators when they clearly denote expiration:  
        - “expiration date”, “shall expire on”, “expires on”, “contract ends on”, “agreement ends on”  
        - “termination date”, “terminates on”, “termination of agreement/contract”  
        - “end of term”, “term shall end on”, “end date”  
        - “valid until”, “valid through”, “shall be in effect until”  
        - “maturity date”, “closing shall occur on” (in sales/purchase contexts)  
  

        ## Decision Rules (Order of Preference)
        1. **Explicit Expiration Clause Wins:** If the text explicitly provides an expiration/termination/end date, use that exact date.  
        2. **Defined Term Sections:** If there is a “Term,” “Duration,” “Term of Agreement,” or similar section, use the end date defined there.  
        3. **Date Ranges:** If the contract states a range (e.g., “from Jan 1, 2020 to Dec 31, 2025”), extract the end date.  
        4. **Relative Dates:** Only compute relative expressions (e.g., “90 days after Closing”) if the referenced date is explicitly present and unambiguous in the provided context. Otherwise return `null`.
        - If the agreement defines expiration as a fixed duration from the Effective Date
            1. When duration is expressed in days:
            - Determine inclusivity based on the wording:
                **Exclusive wording** (e.g., “expires 365 days after the Effective Date”, “shall terminate one year after commencement”):
                    Exclude the start date.
                    Formula: Expiration = Effective Date + Duration.
                **Inclusive wording** (e.g., “expires within 365 days of the Effective Date”, “shall continue for 365 days beginning on the Effective Date”):
                    Include the start date.
                    Formula: Expiration = Effective Date + (Duration − 1).
            2. When duration is expressed in months or years:
            - Do not apply inclusive/exclusive adjustment.
            - Always use direct calendar addition.
            - Formula: Expiration = Effective Date + Duration (in months/years).
        - Duration plus maximum end date (cap)** → 
            - If the clause says the agreement ends after a duration but not later than a stated date:
            - The duration-based date is the actual expiration date.
            - The “no later than” date is not the expiration date. It only applies if the duration-based date would extend beyond it.
            - In other words:
                If (Effective Date + duration) ≤ cap → use the duration-based date.
                If (Effective Date + duration) > cap → use the cap date.
            ** Rule of precedence ** : The duration-based date governs unless it exceeds the cap.
        5. **Survival/Continuing Obligations:** Do not confuse survival/continuing obligations (e.g., “confidentiality survives 2 years”) with expiration. These extend duties after termination, but are not the contract’s expiration date.  
        6. **Placeholders/Incomplete Dates:** If the date is blank, partial (e.g., only a year), or a placeholder (e.g., “on ________”), return `null`.  
        7. **Ambiguity:** If the expiration date cannot be unambiguously determined, return `null`.  

        ## Handling Provided Regex Dates
        - You may reference regex-extracted dates from the context, but only accept a regex date as the expiration date if the surrounding language explicitly ties it to expiration/termination.  
        - Ignore regex dates not supported by the document wording.  

        ## Formatting
        - Output **strict JSON** with exactly two keys:  
        - `"Expiration Date"` → an ISO date (`YYYY-MM-DD`) or `null`.  
        - `"Reasoning"` → a brief explanation, citing the exact clause/phrase.  

        ## Constraints
        - Do not infer or invent dates (e.g., from today’s date, metadata, or unrelated clauses).  
        - Do not normalize partial dates. If the day or month is missing → `null`.  
        - Convert valid written dates (e.g., “31 December 2025”, “Dec. 31, 2025”) into `YYYY-MM-DD`.  
        - Output only minimal JSON, without extra text, formatting, or code blocks.  
        """
        

        #######################################################################################################################################################################################################

        # NEW PROMPT AS OF 3RD SEP 2025

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
        Extract the document's **expiration date** in `YYYY-MM-DD` format based only on explicit language in the context above.

        ### Instructions
        1. Prefer explicit expiration/termination clauses (e.g., “expiration date”, “shall expire on”, “terminates on”, “valid until”, “end of term”).  
        2. Check “Term”/“Duration” sections — use the end date stated.  
        3. If a date range is present (e.g., “from Jan 1, 2020 to Dec 31, 2025”), use the **end date**.  
        4. For relative dates (e.g., “90 days after Closing”), compute only if the referenced date is explicitly present and unambiguous. Otherwise, return `null`.  
        5. Do not confuse survival/continuing obligations (e.g., confidentiality survives) with expiration.  
        6. Return `null` if:  
        - No explicit expiration/maturity/termination date exists.  
        - The date is incomplete or placeholder-only.  
        - The date is conditional or ambiguous.  

        ### Date normalization
        - Always output `YYYY-MM-DD`.  
        - Convert formats like “December 31, 2025” to ISO.  
        - If only month/year is given → treat as incomplete → return `null`.  

        ### Use of regex_extracted_dates
        - Use regex-extracted dates only if tied to expiration/termination/maturity.  

        ### Output
        {{"Expiration Date": <YYYY-MM-DD or null>, "Reasoning": "<Brief justification citing the exact clause/phrase that determines expiration>"}}

        - Reasoning must:  
        (a) Quote or reference the clause/phrase (≤20 words).  
        (b) Explain why it qualifies as expiration.  
        - If returning `null`, Reasoning should explain why (e.g., “No explicit expiration date; only survival clause present”).  

        ### Example Outputs
        - {{"Expiration Date": "2025-12-31", "Reasoning": "Clause states: 'This Agreement shall expire on December 31, 2025.' This explicitly defines expiration."}}  
        - {{"Expiration Date": null, "Reasoning": "No explicit expiration date found; only survival obligations mentioned."}}  

        Output must be **minimal JSON only** — no markdown, no code fences, no extra text.
        """

        json_schema = {"type": "json_object"}
        
        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
      
    
        print("expiration date llm",llm_response)
        logger.info(_log_message(f"LLM Response: {llm_response}","get_expiration_date", MODULE_NAME))
        return llm_response
    except Exception as e:
        print("error in get_expiration_date",e)
        logger.error(_log_message(f"Error in get_expiration_date: {str(e)}","get_expiration_date", MODULE_NAME))
        return {"Expiration Date": "null"}