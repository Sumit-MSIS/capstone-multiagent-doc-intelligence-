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
MODULE_NAME = "extract_scope_of_work"

@mlflow.trace(name="Metadata Others - Get Scope of Work")
def get_scope_of_work(user_id,org_id,file_id,tag_ids, logger):
    try:
        custom_filter = {"file_id": {"$eq": file_id}}
        top_k = 10
        semantic_query = "Identify the scope of work in this document, meaning the specific tasks, deliverables, and timelines explicitly outlined in the agreement."

        keyword_search_query = """
        "scope of work"
        OR "statement of work"
        OR "SOW"
        OR "services to be performed"
        OR "services include"
        OR "contractor shall"
        OR "consultant shall"
        OR "provider will"
        OR "deliverables"
        OR "tasks include"
        OR "work to be performed"
        OR "obligations"
        OR "responsibilities"
        OR "project scope"
        OR "the following services"
        OR "shall provide"
        OR "shall perform"
        OR "shall deliver"
        """

        context_chunks = get_context_from_pinecone(
            DOCUMENT_SUMMARY_AND_CHUNK_INDEX,
            PINECONE_API_KEY,
            custom_filter,
            top_k,
            semantic_query,
            file_id,
            user_id,
            org_id,
            tag_ids=tag_ids, 
            logger=logger,
            keyword_search_query=keyword_search_query
        )

        context_text = get_chunks_text(context_chunks)
        logger.info(_log_message(f"Context Retrieved: {len(context_text)}","get_scope_of_work", MODULE_NAME))
        context_text = "\n\n---\n\n".join(context_text)

        system_prompt = """
        You are ChatGPT, a large language model trained by OpenAI.  
        Knowledge cutoff: 2024-06  

        ## Personality & Style
        - Act as a precise, detail-oriented **legal document specialist**.  
        - Be professional, concise, and conservative: never infer tasks not explicitly stated.  

        ## Role & Objective
        Your role is to analyze legal contracts and agreements to extract the **Scope of Work** — defined as the specific tasks, deliverables, and timelines explicitly outlined in the agreement.  

        ## Scope & Terminology
        Treat the following as candidate indicators:  
        - Sections explicitly labeled “Scope of Work” or “Statement of Work”  
        - Phrases such as “The Contractor shall…”, “Provider will deliver…”, “Consultant shall perform…”  
        - References to deliverables, milestones, or timelines tied to performance.  

        ## Decision Rules (Order of Preference)
        1. **Explicit Scope Wins:** If a scope of work section is present, extract the listed tasks/deliverables/timelines verbatim in concise summary form.  
        2. **Other Clauses:** If scattered across multiple sections (e.g., “Services to be performed” + “Deliverables”), combine them into a single summary string.  
        3. **General Purpose Fallback:** If no specific tasks/deliverables/timelines are stated, return a minimal purpose summary (e.g., “General consulting services agreement” or “Ongoing advisory support agreement”).  
        4. **Exclusions:** Ignore vague or generic language (e.g., “provide services as needed”) unless paired with specific details.  
        5. **Secondary Activities:** Exclude incidental or ancillary activities unless explicitly included in scope.  
        6. **Ambiguity:** If scope cannot be identified, return a general-purpose fallback as above.  

        ## Formatting
        - Output **strict JSON** with exactly one key:  
        - `"Scope of Work"` → concise string summarizing explicit tasks, deliverables, and timelines OR a fallback purpose summary.  

        ## Constraints
        - Do not invent or infer details.  
        - Keep the summary short and precise (1–2 lines max).  
        - No extra formatting, bullet points, or explanations.  
        """

        user_prompt = f"""
        ### Legal Document Context
        ---
        {context_text}
        ---

        ### Task
        Extract the document’s **Scope of Work**, defined as the specific tasks, deliverables, and timelines explicitly outlined in the agreement.

        ### Instructions
        1. Prefer explicit “Scope of Work” or “Statement of Work” sections.  
        2. If multiple scattered clauses describe tasks/deliverables/timelines, consolidate into one concise summary string.  
        3. If only generic references exist (e.g., “consulting services”), output a fallback summary like “General consulting services agreement.”  
        4. Ignore incidental or vague references not central to the contract.  

        ### Output
        {{"Scope of Work": "<concise summary string>"}}

        Output must be minimal JSON only—no extra text, formatting, or code fences.
        """


        json_schema = {"type": "json_object"}


        # json_schema = {
        #     "type": "json_schema",
        #     "json_schema": {
        #         "name": "scope_of_work_extraction",
        #         "schema": {
        #             "type": "object",
        #             "properties": {
        #                 "Scope of Work": {
        #                     "type": "string",
        #                     "description": "A concise summary of the scope of work, including specific tasks, deliverables, and/or timelines. If none are found, a 1–2 line summary of the contract's general purpose."
        #                 }
        #             },
        #             "required": ["Scope of Work"],
        #             "additionalProperties": False
        #         },
        #         "strict": True
        #     }
        # }

        
        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
        logger.info(_log_message(f"LLM Response: {llm_response}","get_scope_of_work", MODULE_NAME))
        return llm_response
    except Exception as e:
        logger.error(_log_message(f"Error in get_scope_of_work: {str(e)}","get_scope_of_work", MODULE_NAME))
        return None