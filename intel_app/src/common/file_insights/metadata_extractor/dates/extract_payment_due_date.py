from src.common.hybrid_retriever.hybrid_search_retrieval import get_context_from_pinecone, get_chunks_text
from src.common.logger import _log_message
from src.config.base_config import config
from src.common.file_insights.llm_call import dynamic_llm_call, payment_due_date_validatior
from datetime import datetime
import mlflow
from opentelemetry import context as ot_context

TRACKING_URI = config.MLFLOW_TRACKING_URI
ENV = config.MLFLOW_ENV

# mlflow.config.enable_async_logging()
mlflow.openai.autolog()

mlflow.set_tracking_uri(TRACKING_URI)

DOCUMENT_SUMMARY_AND_CHUNK_INDEX = config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX
PINECONE_API_KEY = config.PINECONE_API_KEY
MODULE_NAME = "extract_payment_due_date"

@mlflow.trace(name="Metadata Dates - Get Payment Due Date")
def get_payment_due_date(user_id, org_id, file_id, tag_ids, logger, effective_date ,expiration_date, regex_extracted_dates: str = ""):
    try:
        custom_filter = {"file_id": {"$eq": file_id}}
        top_k = 15 # initially the topk was kept as 5 but in some cases the right chunk was not in top 5 so it missed the date, which is why we are keeping the value a little high

        semantic_query = "Identify the next payment due date in this document, meaning the specific date on which a payment obligation must be made, based on the stated payment schedule (monthly, quarterly, annually, or fixed), and ensuring it falls within the contract's valid term before expiration."

        keyword_search_query = """query
        "payment due on"  
        OR "due on the"  
        OR "installment due"  
        OR "monthly payment"  
        OR "quarterly payment"  
        OR "annual payment"  
        OR "semi-annual payment"  
        OR "weekly payment"  
        OR "rent due"  
        OR "fees due"  
        OR "recurring payment"  
        OR "shall pay on"  
        OR "payments shall be made"  
        OR "installments payable"  
        OR "payment schedule"  
        OR "due date"  
        """     
        context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, semantic_query, file_id, user_id, org_id, tag_ids=tag_ids, logger=logger, keyword_search_query=keyword_search_query)
        context_text = get_chunks_text(context_chunks)        
        current_date = datetime.now().strftime("%Y-%m-%d")
        logger.info(_log_message(f"Context Retrieved: {len(context_text)}","get_payment_due_date", MODULE_NAME))
        context_text = "\n\n---\n\n".join(context_text)

        system_prompt = """
        You are ChatGPT, a large language model trained by OpenAI.  
        Knowledge cutoff: 2024-06  

        ## Personality & Style
        - Act as a precise, detail-oriented **legal document specialist**.  
        - Be professional, concise, and conservative: never guess or infer unstated terms or dates.  
        - Always prioritize explicit legal wording over assumptions.  

        ## Role & Objective
        Your role is to analyze legal documents of any type (e.g., contracts, agreements, addendums, amendments, memoranda of understanding, leases, policies, etc.) to extract structured **payment-related information**, focusing on **recurring payment schedules** and the **next immediate payment due date**.

        ## Core Rule (STRICT)
        - The **Effective Date always overrides today's date**.
        - If Effective Date > today → use Effective Date as the base.
        - Never calculate or return a Payment Due Date earlier than the Effective Date.  

        ## Scope & Terminology
        Treat the following as valid candidate indicators when tied to payment terms:  
        - Recurrence: “monthly”, “quarterly”, “annually”, “semi-annual”, “weekly”,"biweekly" 
        - Fixed due dates: “payment due on [specific date] of each month”, “due on the first of every quarter”, “annual payment due on [date]” ,“ Nth business day of each month”,“On or before the Nth calendar day of each month”
        - Explicit due language: “payment due on”, “installment due”, “rent due on”, “fees due by”, “payments shall be made”  

        ## Decision Rules
        1. **Recurring Schedule Recognition:** Identify explicit recurring terms (monthly/quarterly/annual) and validate they fall before or on the expiration date.  
        2. **Effective Date Rule (STRICT & Highest Priority):** 
                - If current Date < Effective Date → always use Effective Date as the base date.
                - If current Date ≥ Effective Date → use current Date as the base date.
                - Never calculate or return a Payment Due Date that is earlier than the Effective Date. 
        3. **Fixed Dates:**
            - For “on or before Nth day of each month”:
            - If Effective Date ≤ N → that month’s Nth day.
            - If Effective Date > N → next month’s Nth day.
        4.  **Business Day Clauses:**
            - Count Mon–Fri only.
            - If Nth business day falls on weekend → roll forward to Monday.
        4. **General Recurrence Rule:** 
            - If Effective Date is before the scheduled due day of that month → first payment is that month.
            - If Effective Date is after the due day → roll to next month.
        5. **Expiration Check:**  
            - If expiration date is before the base date → recurring flag = false, Payment Due Date = null.  
            - If expiration date is missing or ambiguous → recurring flag = false.  
        6. **Relative Dates:** Only compute (e.g., “15 days after invoice”) if both reference and base date are explicitly given and unambiguous. Otherwise → null.  
        7. **Invalid/Incomplete Dates:** Return null if placeholder (“________”), incomplete (“December 2023”), or outside valid contract period.    
        8. **Ambiguity:** If next due date cannot be determined unambiguously, return null.
        
        ## Handling Regex Dates
        - Use regex-extracted dates only if surrounding language explicitly ties them to **payment obligations**.  
        - Ignore regex dates without context.

        ## FormattingIncomplete Dates:
        - Output strict JSON with exactly two keys:  
        - `"flag"` → true/false (whether recurring payments are explicitly supported and valid).  
        - `"Payment Due Date"` → ISO date (YYYY-MM-DD) or null.  

        ## Constraints
        - Never use today's date if Effective Date of the contract is in the future.
        - Never output a Payment Due Date earlier than the Effective Date.
        - Do not infer from unrelated dates.  
        - Do not invent recurring patterns.  
        - Do not output anything except minimal JSON.  
        """
        
        user_prompt = f"""
        ### Legal Document Context
        ---
        {context_text}
        ---

        ### Additional Data (optional)
        Regex-extracted dates: {regex_extracted_dates}

        **Effective Date : {effective_date}**
        **Expiration Date : {expiration_date}**

        ### Task
        Extract the document’s **next immediate payment due date**.  
        - Apply the **Effective Date Rule first**:
        - If Effective Date > today ({current_date}) → use Effective Date as the base date.
        - If Effective Date is null -> "Payment Due Date" is `null`
        - Otherwise, use today as the base date.
        - Never return a Payment Due Date earlier than the Effective Date.
        - Ensure the due date falls between Effective Date ({effective_date}) and Expiration Date ({expiration_date}).
        - For recurring schedules, compute the first valid occurrence strictly after the base date.
        - For business day clauses, skip weekends and roll forward if needed.

        ### Instructions
        #### Part A: Recurring Payments
        1. Identify recurring terms: “monthly”, “quarterly”, “annually”, etc., or phrases like “15th of each month”, “quarterly on Jan 1, Apr 1, Jul 1, Oct 1”.  
        2. Validate that payments fall before or on the expiration date.  
        3. Set `"flag": true` if recurring payments are explicitly specified and valid; otherwise `"flag": false`.  

        #### Part B: Compute Next Payment Due Date
        1. Always apply Effective Date first. Base date = max(today, Effective Date).
        2. For fixed-day clauses → compute the next valid day after base date.
        3. For recurring terms:  
            - Monthly → compute next monthly date.  
            - Quarterly → compute next quarter date.  
            - Annual → compute next annual date.
        4. For “on or before Nth” → if Effective Date ≤ N, same month; else next month.
        5. For “Nth business day” → count Mon–Fri, roll forward if weekend.  
        6. Use regex dates only if tied to payment context and within the contract duration.  
        7. Return `"Payment Due Date": null` if:  
            - No valid payment due date is found.  
            - Fffective Date is null.
            - The date is incomplete or a placeholder.  
            - The date is ambiguous, conditional, or past expiration.  
        8. If expiration date is before {current_date} or invalid → `"flag": false`, `"Payment Due Date": null`.  

        ### Output
        {{"flag": <true/false>, "Payment Due Date": <YYYY-MM-DD or null>}}

        Output must be minimal JSON only — no markdown, no code fences, no extra text.
        """

        json_schema = {"type": "json_object"}

        llm_response = dynamic_llm_call(user_prompt, system_prompt, json_schema, logger)
        logger.info(_log_message(f"LLM Response: {llm_response}","get_payment_due_date", MODULE_NAME))
        if isinstance(llm_response, dict):
            payment_due_date = llm_response.get("Payment Due Date", None)
            payment_due_date = None if payment_due_date == "null" else payment_due_date
            flag_value = "Yes" if llm_response.get("flag", None) else "No"
            flag_value, payment_due_date = payment_due_date_validatior(flag_value, payment_due_date, expiration_date, current_date, file_id, user_id, org_id, logger)
        return flag_value, payment_due_date
    except Exception as e:
        logger.error(_log_message(f"Error in get_payment_due_date: {str(e)}","get_payment_due_date", MODULE_NAME))
        return "No", None