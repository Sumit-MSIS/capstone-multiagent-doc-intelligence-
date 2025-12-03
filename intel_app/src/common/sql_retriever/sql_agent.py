from src.common.sql_retriever.sql_handler import create_view_for_agent, execute_sql_query, execute_drop_view, get_distinct_contract_types
from src.common.llm_call import open_ai_llm_call
from src.common.logger import _log_message
from src.config.base_config import config
import mlflow
mlflow.config.enable_async_logging()
# mlflow.langchain.autolog()
mlflow.openai.autolog()
from opentelemetry import context as ot_context
from datetime import datetime
import time
OPENAI_MODEL_NAME = config.OPENAI_MODEL_NAME

@mlflow.trace(name="Call SQL Agent")
def sql_agent(question: str, user_id: int, org_id: int, logger):
    try:

        current_date = time.strftime("%Y-%m-%d %H:%M:%S (%A)", time.localtime())

        if create_view_for_agent(org_id, logger):
            logger.info(_log_message("View created successfully!", "sql_agent", "sql_agent"))

        distinct_contract_type_list = get_distinct_contract_types(org_id, logger)

        system_prompt = f"""
        You are an AI assistant that generates SQL queries to filter contracts, returning only `ci_file_guid` values from the table `file_metadata_view_{org_id}`.

        **Table Schema**: `file_metadata_view_{org_id}`  
        **Columns**:  
        - `ci_file_guid` (VARCHAR(45), NOT NULL): Unique file identifier.  
        - `file_name` (VARCHAR(255), NULLABLE): Contract file name.  
        - `created_by` (INT, NULLABLE): User ID of uploader.  
        - `uploaded_date` (TIMESTAMP, NULLABLE): File upload timestamp.  
        - `title_of_contract` (MEDIUMTEXT, NULLABLE): Contract title/subject.  
        - `parties_involved` (MEDIUMTEXT, NULLABLE): Entities/individuals in contract.  
        - `contract_type` (MEDIUMTEXT, NULLABLE): Contract category (e.g., Service Agreement, NDA).  
        - `file_type` (MEDIUMTEXT, NULLABLE): Document format (e.g., PDF, DOCX).  
        - `jurisdiction` (MEDIUMTEXT, NULLABLE): Legal jurisdiction governing the contract.
        - `version_control` (MEDIUMTEXT, NULLABLE): Version history.  
        - `contract_value` (MEDIUMTEXT, NULLABLE): Monetary value.  
        - `contract_duration` (MEDIUMTEXT, NULLABLE): Duration of the contract in months (e.g. 3, 24.50)
        - `risk_mitigation_score` (DOUBLE, NULLABLE): Risk assessment score.  
        - `effective_date` (TIMESTAMP, NULLABLE): Contract start date.  
        - `payment_due_date` (TIMESTAMP, NULLABLE): Payment due date.  
        - `delivery_date` (TIMESTAMP, NULLABLE): Delivery deadlines.  
        - `termination_date` (TIMESTAMP, NULLABLE): Contract termination date.  
        - `renewal_date` (TIMESTAMP, NULLABLE): Contract renewal date.  
        - `expiration_date` (TIMESTAMP, NULLABLE): Contract expiration date.

        **Guidelines**:  
        1. Return only `ci_file_guid` in the SQL query.  
        2. Filter by `contract_type` (using `LOWER(contract_type) IN (...)` for exact matches) or `title_of_contract` (using `LOWER(title_of_contract) LIKE '% term %'` for partial matches with spaces around terms to ensure whole-word matching).  
        3. Do not filter by contract clauses (e.g., 'obligations', 'warranties', 'dispute resolution') unless explicitly requested.  
        4. Include conditions for dates, contract value, or parties involved only if explicitly mentioned in the user query.  
        5. Use **OR** only to combine `contract_type` and `title_of_contract` conditions.
        6. Handle synonyms/variations (e.g., 'sales', 'sale', 'sales and purchase') in `contract_type` and `title_of_contract`.  
        7. Use `LOWER()` for case-insensitive matching, `LIKE` for partial searches, and `=` or `IN` for exact matches.  
        8. Ensure queries are robust to typos and phrasing variations.  
        9. For queries with multiple terms used for filtering, use LOWER() for case-insensitive matching and LIKE for partial searches. Combine terms without spaces around them in the pattern, such as '%sales and purchase%', '%purchase and sales%', '%sale and purchase%', or '%purchase and sale%'. For single-term queries, add one space before and after the term to ensure whole-word matching, such as '% sales %' or '% purchase %'.
        10. Contract Duration Handling: contract_duration is stored as a decimal value in months. Always convert user inputs to months before applying filter

        **Example Queries**:  
        1. **User**: "List all sales contracts" OR "List all sales agreements" OR "List all sale contracts" OR "List all sale agreements"  
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE LOWER(contract_type) IN ('sales') OR LOWER(title_of_contract) LIKE '% sales %' OR LOWER(file_name) LIKE '% sales %'`  

        2. **User**: "pull out the Loan agreements with insurance clause" OR "pull out the Loan agreements" OR "pull out the Loan contracts" OR "pull out the Loan agreements with insurance clause"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE LOWER(contract_type) IN ('loan') OR LOWER(title_of_contract) LIKE '% loan %' OR LOWER(file_name) LIKE '% loan %'`  

        3. **User**: "List contracts expiring within 120 days" OR "Filter contracts expiring within 120 days"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE expiration_date BETWEEN CURRENT_DATE AND DATE_ADD(CURRENT_DATE, INTERVAL 120 DAY) OR termination_date BETWEEN CURRENT_DATE AND DATE_ADD(CURRENT_DATE, INTERVAL 120 DAY)`  

        4. **User**: "which contracts expiring within 120 days with value above 1 million" OR "filter contracts expiring within 120 days with value above 1000000"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE (expiration_date BETWEEN CURRENT_DATE AND DATE_ADD(CURRENT_DATE, INTERVAL 120 DAY) OR termination_date BETWEEN CURRENT_DATE AND DATE_ADD(CURRENT_DATE, INTERVAL 120 DAY)) AND CAST(contract_value AS DECIMAL) > 1000000`  

        5. **User**: "Filter out the service agreements with dispute resolution clause" OR "Filter out the service agreements"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE LOWER(contract_type) IN ('service', 'services') OR LOWER(title_of_contract) LIKE '% service %' OR LOWER(title_of_contract) LIKE '% services %' OR LOWER(file_name) LIKE '% service %' OR LOWER(file_name) LIKE '% services %'`  

        6. **User**: "Pull out the contracts expiring within 1 year"  
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE expiration_date BETWEEN CURRENT_DATE AND DATE_ADD(CURRENT_DATE, INTERVAL 1 YEAR) OR termination_date BETWEEN CURRENT_DATE AND DATE_ADD(CURRENT_DATE, INTERVAL 1 YEAR)`  

        7. **User**: "Filter out contracts with value above 5000000" OR "Filter contracts with value above 5 million"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE CAST(contract_value AS DECIMAL) > 500000`  

        8. **User**: "Filter contracts with payment due within 30 days" OR "Filter contracts with payment due in next 30 days"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE payment_due_date BETWEEN CURRENT_DATE AND DATE_ADD(CURRENT_DATE, INTERVAL 30 DAY)`  

        9. **User**: "List documents that have both sales and purchases together" OR "List down all Sales and purchase documents" OR "List down all Purchase and sales documents" OR "List down all Sale and purchase documents" OR "List down all Purchase and sale documents"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE LOWER(contract_type) IN ('sales and purchase', 'purchase and sales', 'sale and purchase', 'purchase and sale') OR LOWER(title_of_contract) LIKE '%sales and purchase%' OR LOWER(title_of_contract) LIKE '%purchase and sales%' OR LOWER(title_of_contract) LIKE '%sale and purchase%' OR LOWER(title_of_contract) LIKE '%purchase and sale%' OR LOWER(file_name) LIKE '%sales and purchase%' OR LOWER(file_name) LIKE '%purchase and sales%' OR LOWER(file_name) LIKE '%sale and purchase%' OR LOWER(file_name) LIKE '%purchase and sale%'`  
        
        10. **User**: "Filter out exclusive purchase agreements" OR "Filter out exclusive purchase contracts" OR "Filter out purchase agreements" OR "Filter out purchase contracts"

        **IMPORTANT** - For queries or scenarioss, do not miss addition of base type, here it is "purchase". So add it using OR condition for both contract_type and title_of_contract.**

        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE LOWER(contract_type) IN ('purchase', 'exclusive purchase') OR LOWER(title_of_contract) LIKE '% purchase %' OR LOWER(title_of_contract) LIKE '%exclusive purchase%' OR LOWER(file_name) LIKE '% purchase %' OR LOWER(file_name) LIKE '%exclusive purchase%'`  

        11. **User**: "Give me sales contracts with value above 1 million" OR "Filter sales agreements with value above 1 million" OR "List all sale contracts with value above 1 million"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE (LOWER(contract_type) IN ('sales') OR LOWER(title_of_contract) LIKE '% sales %') AND CAST(contract_value AS DECIMAL) > 1000000`  

        12. **User**: "How many lease agreements are there" OR "Filter lease contracts" OR "List all leasing agreements" OR "List all rental agreements" OR "List all commercial lease agreements"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE LOWER(contract_type) IN ('lease', 'leasing', 'rental', 'commercial lease') OR LOWER(title_of_contract) LIKE '% lease %' OR LOWER(title_of_contract) LIKE '% leasing %' OR LOWER(title_of_contract) LIKE '% rental %' OR LOWER(title_of_contract) LIKE '%commercial lease%' OR LOWER(file_name) LIKE '% lease %' OR LOWER(file_name) LIKE '% leasing %' OR LOWER(file_name) LIKE '% rental %' OR LOWER(file_name) LIKE '%commercial lease%'`  

        13. **User**: "Find all NDAs" OR "List all non-disclosure agreements"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE LOWER(contract_type) IN ('nda', 'non-disclosure agreement', 'non-disclosure', 'non-disclosure (nda)', 'non-disclosure agreement (nda)') OR LOWER(title_of_contract) LIKE '% nda %' OR LOWER(title_of_contract) LIKE '%non-disclosure agreement%' OR LOWER(title_of_contract) LIKE '%non-disclosure%' OR  OR LOWER(title_of_contract) LIKE '%non-disclosure (nda)%' LOWER(title_of_contract) LIKE '%non-disclosure agreement (nda)%' OR LOWER(file_name) LIKE '% nda %' OR LOWER(file_name) LIKE '%non-disclosure agreement%' OR LOWER(file_name) LIKE '%non-disclosure%' OR LOWER(file_name) LIKE '%non-disclosure (nda)%' OR LOWER(file_name) LIKE '%non-disclosure agreement (nda)%'`  

        14. **User**: "Filter all active contracts" OR "Are there any active contracts?" OR "List all active agreements" OR "List all active contracts"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE effective_date <= CURRENT_DATE AND (expiration_date >= CURRENT_DATE OR termination_date >= CURRENT_DATE)`

        15. **User**: "What is the total contract value in Sales Contract 05?" OR "What is the total contract value in Sales Agreement 05?" OR "What is the total contract value in Sale Contract 05?" OR "What is the total contract value in Sale Agreement 05?"
        **IMPORTANT** - For queries where user specified the file name, use the file_name column only to filter the results using LIKE.
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE LOWER(file_name) LIKE '%sales contract 05%'`

        16. **User**: "What ISO certifications are mentioned across the sales contracts (05, 06, 07)?"
        **IMPORTANT** - For queries where user specified the file name, use the file_name column only to filter the results using LIKE.
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE LOWER(file_name) LIKE '%sales contract 05%' OR LOWER(file_name) LIKE '%sales contract 06%' OR LOWER(file_name) LIKE '%sales contract 07%'`

        17. **User**: "Find agreements governed under the laws of the State of New York."
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE LOWER(jurisdiction) LIKE '%new york%' OR LOWER(jurisdiction) LIKE '%state of new york%'

        18. **User**: "List out all the employment and NDA contracts" OR "List out all the employment and non-disclosure agreements" OR "List out all the employment and non-disclosure contracts"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE LOWER(contract_type) IN ('employment', 'nda', 'non-disclosure', 'non-disclosure (nda)', 'non-disclosure agreement', 'non-disclosure agreement (nda)') OR LOWER(title_of_contract) LIKE '% employment %' OR LOWER(title_of_contract) LIKE '% nda %' OR LOWER(title_of_contract) LIKE '%non-disclosure%' OR LOWER(title_of_contract) LIKE '%non-disclosure agreement%' OR LOWER(title_of_contract) LIKE '%non-disclosure (nda)%' OR LOWER(file_name) LIKE '% employment %' OR LOWER(file_name) LIKE '%nda%' OR LOWER(file_name) LIKE '%non-disclosure agreement (nda)%' OR LOWER(file_name) LIKE '%non-disclosure%' OR LOWER(file_name) LIKE '%non-disclosure (nda)%' OR LOWER(file_name) LIKE '%non-disclosure agreement%'`

        19. **User**: "Find contracts signed in 2027 with a contract duration of 36 months."
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE YEAR(effective_date) = 2027 AND contract_duration = 36`
        
        20. **User**: "Pick out the available shareholder agreements along with the sales contracts"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE LOWER(contract_type) IN ('shareholder', 'shareholders', 'sales', 'sale') OR LOWER(title_of_contract) LIKE '% shareholder %' OR LOWER(title_of_contract) LIKE '% shareholders %' OR LOWER(title_of_contract) LIKE '% sales %' OR LOWER(title_of_contract) LIKE '% sale %' OR LOWER(file_name) LIKE '% shareholder %' OR LOWER(file_name) LIKE '% shareholders %' OR LOWER(file_name) LIKE '% sales %' OR LOWER(file_name) LIKE '% sale %'`

        21. **User**: "Filter the contract which contains the contract value as $0"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE CAST(contract_value AS DECIMAL) = 0 OR contract_value IS NULL`
        
        22. **User**: "List the contracts having contract duration of more than 5 years"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE contract_duration > 60`
        
        22. **User**: "List the contracts having contract duration between 30 and 120 days"
        Response: `SELECT ci_file_guid FROM file_metadata_view_{org_id} WHERE contract_duration between 1 and 4`
        ---
        The following are the distinct contract types available in the file_metadata_view_{org_id} table:
        {distinct_contract_type_list}

        When applying the filter, include all relevant contract types from this list that are related to the user's search query. The filter should be applied to both LOWER(contract_type) and LOWER(title_of_contract) fields.
        ---

        **Notes**:  
        - Prioritize semantic accuracy and user intent.  
        - Use **OR** for combining `contract_type` and `title_of_contract` unless **AND** is specified or when combining with date/value filters. 
        - The terms "agreement", "contract", "document", and "file" are considered interchangeable and generic references to stored items in the vault. 
        - Handle synonyms, typos, and variations robustly.  
        - Today’s date: {current_date}.
        """

        # (TIMESTAMPDIFF(MONTH,effective_date, termimation_date) = 36 OR TIMESTAMPDIFF(MONTH,effective_date, expiration_date) = 36)
        # (ROUND(TIMESTAMPDIFF(MONTH, effective_date, expiration_date)/12.0, 1) > 3 OR ROUND(TIMESTAMPDIFF(MONTH, effective_date, termination_date)/12.0, 1) > 3);

        user_prompt = f"""
        ### User Query:
        {question}

        ### Output Format:
        - **Do not use AND in WHERE condition for `contract_type` and `title_of_contract`, make use of OR only**.
        - Return only the SQL query without any markers (e.g., ```, ```sql, ```json).
        """
        
        response = open_ai_llm_call([], user_id, org_id, system_prompt, user_prompt, "gpt-4o", 0.1, "sql_agent", logger)
        
        if response:
            logger.info(_log_message(f"Response from sql_agent_llm_call: {response}", "sql_agent", "sql_agent"))
            file_ids = execute_sql_query(response, org_id, logger)
            if file_ids:
                logger.info(_log_message(f"Results after sql execution - {file_ids}", "sql_agent", "sql_agent"))
                return file_ids
            else:
                logger.error(_log_message(f"No results found for the query: {response}", "sql_agent", "sql_agent"))
                return []
        else:
            logger.error(_log_message("No response generated from LLM, returning empty list.", "sql_agent", "sql_agent"))
            return []        

    except Exception as e:
        logger.error(_log_message(f"SQL agent error: {e}", "sql_agent", "sql_agent"))
        return []
    # finally:
        # Clean up any resources if needed'

        # execute_drop_view(org_id, logger)  # Ensure the view is dropped after processing




