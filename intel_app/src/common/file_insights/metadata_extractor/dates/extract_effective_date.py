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
MODULE_NAME = "extract_effective_date"

@mlflow.trace(name="Metadata Dates - Get Effective Date")
def get_effective_date(user_id,org_id,file_id, tag_ids, logger, regex_extracted_dates:str= "" ):
    try:
        custom_filter = {"file_id": {"$eq": file_id}}
        top_k = 15 # initially the topk was kept as 5 but in some cases the right chunk was not in top 5 so it missed the date, which is why we are keeping the value a little high

        semantic_query = "Identify the effective date of this document, meaning the date on which it becomes valid, enforceable, signed, executed, or comes into force."

        keyword_search_query = """
        "effective date" 
        OR "effective as of" 
        OR "effective from" 
        OR "effective upon signing" 
        OR "effective upon execution" 
        OR "effective upon the last signature" 
        OR "commencement date" 
        OR "date of commencement" 
        OR "start date" 
        OR "term will start from"
        OR "commencement of" 
        OR "execution date" 
        OR "date of execution" 
        OR "execution date of agreement" 
        OR "execution date of memorandum" 
        OR "execution date of treaty" 
        OR "fully executed on" 
        OR "date of signature" 
        OR "signature date" 
        OR "signed on" 
        OR "date last signed" 
        OR "date of last signature" 
        OR "date first written above" 
        OR "valid from" 
        OR "enforceable from" 
        OR "coming into force" 
        OR "shall be effective as of" 
        OR "shall come into effect" 
        OR "document shall commence on"
        """

        context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, semantic_query, file_id, user_id, org_id, tag_ids=tag_ids, logger=logger, keyword_search_query=keyword_search_query)
        context_text = get_chunks_text(context_chunks)

        logger.info(_log_message(f"Context Retrieved: {len(context_text)}","get_effective_date", MODULE_NAME))
        context_text = "\n\n---\n\n".join(context_text)        

        system_prompt = """
        You are ChatGPT, a large language model trained by OpenAI.  
        Knowledge cutoff: 2024-06  

        ## Personality & Style
        - Act as a precise, detail-oriented **legal document specialist**.  
        - Be professional, concise, and conservative: never guess or infer unstated dates.  
        - Always prioritize explicit legal wording over assumptions.  

        ## Role & Objective
        Your role is to analyze legal documents of any type (e.g., contracts, agreements, addendums, amendments, memoranda of understanding, leases, treaties, policies, or related instruments) to identify the document’s **effective date** — the date it becomes valid, enforceable, or comes into force.  

        ## Scope & Terminology
        Treat the following as candidate indicators when they clearly denote effectiveness:  
        - “effective date”, “effective as of”, “effective from”, “effective upon [event]”  
        - “commencement date”, “date of commencement”, “start date”, “term will start from”  
        - “execution date”, “fully executed on” (only if explicitly tied to effectiveness)  
        - Signature indicators: “date of signature”, “signature date”, “date last signed”, “date first written above” (used only as fallback or when the text states effectiveness depends on them)  

        ## Decision Rules (Order of Preference)
        1. **Explicit Effective Clause Wins:** If the text explicitly provides an effective/commencement/start date, use that exact date.  
        2. **Conditional/Relative Clauses:** If effectiveness is tied to an event (e.g., “effective upon the last signature”), extract the referenced event’s date **only if explicitly present**.  
        3. **Execution/Signature as Fallback:** Use execution/signature-related dates only if no explicit effective/commencement/start date/term start date exists, or if the document specifies that effectiveness is tied to execution/signing/last signature.  
        4. **Conflicts:** If multiple candidate dates exist, prefer the one explicitly labeled “Effective Date”; otherwise select the clause that most clearly states when the document “shall be effective.”  
        5. **Placeholders/Incomplete:** If the date is blank, partial (e.g., only year), or a placeholder (e.g., “as of ________”), return `null`.  
        6. **Ambiguity:** If the effective date cannot be unambiguously determined, return `null`.  

        ## Handling Provided Regex Dates
        - You may reference regex-extracted dates from the context, but only validate and use them if the surrounding language explicitly ties them to effectiveness.  
        - Ignore regex dates that are not justified by explicit document wording.  

        ## Formatting
        - Output **strict JSON** with exactly two keys:  
        - `"Effective Date"` → an ISO date (`YYYY-MM-DD`) or `null`.  
        - `"Reasoning"` → a brief explanation, citing the exact clause/phrase.  

        ## Constraints
        - Do not infer or invent dates (e.g., from today’s date, metadata, or unrelated sections).  
        - Do not normalize partial dates. If the day or month is missing → `null`.  
        - Convert valid written dates (e.g., “1 January 2023”, “Jan. 1, 2023”) into `YYYY-MM-DD`.  
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
        Extract the document's **effective date** in `YYYY-MM-DD` format based only on explicit language in the context above.

        ### Instructions
        1. Prefer explicit “effective/commencement/start” clauses.  
        2. If effectiveness is tied to execution/signature/last signature, use that date **only if explicitly stated**.  
        3. Use execution/signature/“date first written above” **only as fallback** when no explicit effective/commencement/start date is present or when the clause ties effectiveness to them.  
        4. Ignore placeholders or incomplete dates (e.g., “as of ________”, “as of __________, 2019”).  
        5. Validate regex-extracted dates only if explicitly tied to the effective/commencement/start date in context.  
        6. If ambiguous or not clearly stated, return `null`.  

        ### Output
        {{"Effective Date": <YYYY-MM-DD or null>, "Reasoning": "<Brief justification by citing the exact clause or phrase that determines the effective date>"}}

        Output must be minimal JSON only—no extra text, formatting, or code fences.
        """



        

        # json_schema = {
        #     "type": "json_schema",
        #     "json_schema": {
        #         "name": "effective_date_extraction",
        #         "schema": {
        #             "type": "object",
        #             "properties": {
        #                 "Effective Date": {
        #                     "type": ["string", "null"],
        #                     "pattern": r"^\d{4}-\d{2}-\d{2}$",
        #                     "description": "The effective date in 'YYYY-MM-DD' format otherwise null if no valid date is found or it's ambiguous/incomplete."
        #                 }
        #             },
        #             "required": ["Effective Date"],
        #             "additionalProperties": False
        #         },
        #         "strict": True
        #     }
        # }

        json_schema = {"type": "json_object"}
        
        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
        logger.info(_log_message(f"LLM Response: {llm_response}","get_effective_date", MODULE_NAME))
        return llm_response
    except Exception as e:
        logger.error(_log_message(f"Error in get_effective_date: {str(e)}","get_effective_date", MODULE_NAME))
        return {"Effective Date": "null"}