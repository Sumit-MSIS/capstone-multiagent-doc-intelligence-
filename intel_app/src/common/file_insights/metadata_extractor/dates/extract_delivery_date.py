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
MODULE_NAME = "extract_delivery_date"

@mlflow.trace(name="Metadata Dates - Get Delivery Date")
def get_delivery_date(user_id,org_id,file_id, tag_ids, logger, regex_extracted_dates:str=""):
    try:
        custom_filter = {"file_id": {"$eq": file_id}}
        top_k = 15 # initially the topk was kept as 5 but in some cases the right chunk was not in top 5 so it missed the date, which is why we are keeping the value a little high

        semantic_query = """
        Identify the delivery date in this document, meaning the specific date on which goods, services, property, or contractual obligations must be delivered, completed, transferred, shipped, or performed.
        """
        keyword_search_query = """
        "delivery date"
        OR "to be delivered on"
        OR "scheduled delivery"
        OR "shall be delivered by"
        OR "delivery shall occur on"
        OR "must be delivered by"
        OR "to be provided by"
        OR "performance to be completed by"
        OR "completion date"
        OR "date of completion"
        OR "shipment date"
        OR "date of shipment"
        OR "dispatch date"
        OR "handover date"
        OR "vacant possession delivered"
        OR "possession shall be delivered"
        OR "closing date"
        OR "delivery deadline"
        OR "final delivery date"
        OR "date of handover"
        OR "goods shall be delivered"
        OR "services shall be completed by"
        """

        context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, semantic_query, file_id, user_id, org_id, tag_ids=tag_ids, logger=logger, keyword_search_query=keyword_search_query)
        context_text = get_chunks_text(context_chunks)

        logger.info(_log_message(f"Context Retrieved: {len(context_text)}","get_delivery_date", MODULE_NAME))
        context_text = "\n\n---\n\n".join(context_text)


        system_prompt = """
        You are ChatGPT, a large language model trained by OpenAI.  
        Knowledge cutoff: 2024-06  

        ## Personality & Style
        - Act as a precise, detail-oriented **legal document specialist**.  
        - Be professional, concise, and conservative: never guess or infer unstated dates.  
        - Always prioritize explicit legal wording over assumptions.  

        ## Role & Objective
        Your role is to analyze legal documents of any type (e.g., contracts, agreements, addendums, amendments, memoranda of understanding, leases, treaties, policies, or related instruments) to identify the document’s **delivery date** — the date when goods, services, property, or contractual obligations are scheduled to be delivered, completed, or performed.  

        ## Scope & Terminology
        Treat the following as candidate indicators when they clearly denote delivery or performance deadlines:  
        - “delivery date”, “to be delivered on”, “scheduled delivery”, “shall be delivered by”  
        - “shipment date”, “date of shipment”, “dispatch date”  
        - “completion date”, “performance to be completed by”, “must be provided by”  
        - “vacant possession delivered”, “handover date”, “closing date”  
        
        ## Contract-Type Differentiation
        # Sales, Procurement, Supply, Service, or Lease Agreements:
        - Delivery dates may also be referred to as closing date, completion date, handover date, ending date of performance obligations, or vacant possession date. Treat these as candidate delivery indicators when clearly tied to transfer, shipment, or completion of obligations.

        # Construction or Project Agreements:
        - Delivery may appear as completion date, substantial completion date, project handover date, final performance deadline.

        # Grant, Donation, Research Funding, or Academic Agreements:
        - Typically do not contain delivery dates. Only extract a delivery date if the text explicitly specifies a deliverable with a defined delivery or completion deadline. Otherwise return null.

        # Licensing, Intellectual Property, or Software Agreements:
        - Treat implementation completion date, go-live date, delivery of licensed materials/software as candidate delivery indicators.

        # Other Agreements:
        - Only extract delivery-related dates when the wording explicitly denotes delivery, shipment, handover, completion, or performance deadlines.

        ## Decision Rules (Order of Preference)
        1. **Explicit Delivery Clause:** If the text explicitly provides a delivery/shipment/completion/possession date, use that exact date.  
        2. **Ranges:** If delivery is specified as a range (e.g., “between Jan 1 and Jan 15, 2024”), select the **final deadline date** (here: Jan 15, 2024).  
        3. **Relative Clauses:** If the delivery is tied to another event (e.g., “30 days after Effective Date”), compute it **only if the referenced date is explicitly present** in the context. Otherwise return `null`.  
        4. **Ignore Unrelated Dates:** Do not confuse delivery with effective dates, renewal terms, survival clauses, or payment deadlines.  
        5. **Placeholders/Incomplete:** If the date is blank, partial (e.g., only month/year), or a placeholder (e.g., “to be determined”), return `null`.  
        6. **Ambiguity:** If no delivery/shipment/possession/completion date can be unambiguously determined, return `null`.  

        ## Handling Provided Regex Dates
        - You may reference regex-extracted dates from the context, but only validate and use them if the surrounding language explicitly ties them to delivery/shipment/completion.  
        - Ignore regex dates that are not justified by explicit document wording.  

        ## Formatting
        - Output **strict JSON** with exactly two keys:  
        - `"Delivery Date"` → an ISO date (`YYYY-MM-DD`) or `null`.  
        - `"Reasoning"` → a brief explanation, citing the exact clause/phrase.  

        ## Constraints
        - Do not infer or invent dates (e.g., from today’s date, metadata, or unrelated sections).  
        - Do not normalize partial dates. If the day or month is missing → `null`.  
        - Convert valid written dates (e.g., “1 January 2023”, “Jan. 1, 2023”) into `YYYY-MM-DD`.  
        - Output only minimal JSON, without extra text, formatting, or code blocks.  
        """


        #######################################################################################################################################################################################################

        # New prompt - 3rd september 2025

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
        Extract the document's **delivery date** in `YYYY-MM-DD` format based only on explicit language in the context above.

        ### Instructions
        1. Prefer explicit “delivery/shipment/completion/possession” clauses.  
        2. If a delivery is expressed as a range, return the **last date in the range**.  
        3. If delivery is tied to an event (e.g., “30 days after Effective Date”), compute it **only if the referenced date is explicitly stated** in context. Otherwise return `null`.  
        4. Ignore placeholders or incomplete dates (e.g., “to be delivered on ______”).  
        5. Validate regex-extracted dates only if explicitly tied to delivery/shipment/completion in context.  
        6. If ambiguous or not clearly stated, return `null`.  

        ### Output
        {{"Delivery Date": <YYYY-MM-DD or null>, "Reasoning": "<Brief justification by citing the exact clause or phrase that determines the delivery date>"}}

        Output must be minimal JSON only—no extra text, formatting, or code fences.
        """


        json_schema = {"type": "json_object"}

        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
       
        logger.info(_log_message(f"LLM Response: {llm_response}","get_delivery_date", MODULE_NAME))
        return llm_response
    except Exception as e:
        logger.error(_log_message(f"Error in get_delivery_date: {str(e)}","get_delivery_date", MODULE_NAME))
        return {"Delivery Date": "null"}
    