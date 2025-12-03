from src.common.conceptual_search.retriever.sql_retriever.sql_handler import create_view_for_agent, execute_sql_query, execute_drop_view, get_distinct_contract_types
from src.common.llm_call import open_ai_llm_call
from src.common.logger import _log_message
from src.config.base_config import config
import mlflow
import time
mlflow.config.enable_async_logging()
# mlflow.langchain.autolog()
mlflow.openai.autolog()
from opentelemetry import context as ot_context
from datetime import datetime
OPENAI_MODEL_NAME = config.OPENAI_MODEL_NAME

@mlflow.trace(name="Call SQL Agent")
def sql_agent_template(question: str, user_id: int, org_id: int, logger):
    try:

        current_date = time.strftime("%Y-%m-%d %H:%M:%S (%A)", time.localtime())

        if create_view_for_agent(org_id, logger):
            logger.info(_log_message("View created successfully!", "sql_agent", "sql_agent"))

        distinct_contract_type_list = get_distinct_contract_types(org_id, logger)
       
        system_prompt = f"""
        You are an AI assistant that generates SQL queries to recommend contract templates based on user queries. The queries should return `ci_file_guid` values from the table `file_metadata_view_{org_id}` and be robust to handle spelling mistakes, inconsistent phrasing, synonyms, and word order variations (e.g., 'purchase and sale' vs. 'sale and purchase').

        Below is the table schema:

        **Table Name**: `file_metadata_view_{org_id}`

        **Columns**:
        - `ci_file_guid` (VARCHAR(45), NOT NULL): Unique file identifier.
        - `file_name` (VARCHAR(255), NULLABLE): Name of the contract file.
        - `created_by` (INT, NULLABLE): User ID of the uploader.
        - `uploaded_date` (TIMESTAMP, NULLABLE): File upload timestamp.
        - `title_of_contract` (MEDIUMTEXT, NULLABLE): Contract title or subject.
        - `scope_of_work` (MEDIUMTEXT, NULLABLE): Description of contract scope.
        - `parties_involved` (MEDIUMTEXT, NULLABLE): Entities/individuals in the contract.
        - `contract_type` (MEDIUMTEXT, NULLABLE): Contract category (e.g., Service Agreement, NDA).
        - `file_type` (MEDIUMTEXT, NULLABLE): Document format (e.g., PDF, DOCX).
        - `jurisdiction` (MEDIUMTEXT, NULLABLE): Legal jurisdiction of the contract.
        - `version_control` (MEDIUMTEXT, NULLABLE): Version history of the contract.
        - `contract_duration` (MEDIUMTEXT, NULLABLE): Duration of the contract.
        - `contract_value` (MEDIUMTEXT, NULLABLE): Monetary value of the contract.
        - `risk_mitigation_score` (DOUBLE, NULLABLE): Risk assessment score.
        - `effective_date` (TIMESTAMP, NULLABLE): Contract start date.
        - `term_date` (TIMESTAMP, NULLABLE): Contract end date.
        - `payment_due_date` (TIMESTAMP, NULLABLE): Payment due date.
        - `delivery_date` (TIMESTAMP, NULLABLE): Delivery deadlines.
        - `termination_date` (TIMESTAMP, NULLABLE): Contract termination date.
        - `renewal_date` (TIMESTAMP, NULLABLE): Contract renewal date.
        - `expiration_date` (TIMESTAMP, NULLABLE): Contract expiration date.

        ### Guidelines:
        1. Your SQL query should **only return 'ci_file_guid'** values.
        2. If filtering is required based on user input, apply appropriate **WHERE** conditions.
        3. Use **`LIKE`** for partial text searches and **`=`** or **`IN`** for exact matches. Always convert text to lowercase using `LOWER()` for case-insensitive matching.
        4. For text-based searches:
        - Handle spelling mistakes or inconsistencies by using broader `LIKE` patterns (e.g., include singular/plural forms like 'sale' and 'sales', or common typos like 'slae' → 'sale', 'resollution' → 'resolution').
        - For phrases involving 'and' (e.g., 'purchase and sale'), include both word orders (e.g., 'purchase and sale' and 'sale and purchase') unless the user explicitly restricts to one order.
        - Included all relevant synonyms or variations in the `LOWER(contract_type)` and `LOWER(title_of_contract)` conditions to ensure comprehensive matching. For example if user is asking for consultancy contract include 'consultancy', 'consulting', 'consultant', 'consultation' in `LOWER(contract_type)` and `LOWER(title_of_contract)` conditions.
        - Account for synonyms or related terms (e.g., 'sales' vs. 'sale', 'agreement' vs. 'contract', 'subletting' vs. 'subleasing').
        - For `LIKE` queries in `title_of_contract`, use proper spacing (e.g., `'% sale %'`, `'% sales %'`) to match standalone words or phrases, but also include specific phrases like `'%sales agreement%'` or `'%sales contract%'` to ensure precision.
        5. For date-based filters like 'within X days/months/years':
        - Convert time-based ranges into `BETWEEN CURRENT_DATE AND DATE_ADD(CURRENT_DATE, INTERVAL X UNIT)` SQL format.
        - Handle the following cases:
            - 'within X days' → `INTERVAL X DAY`
            - 'within X months' → `INTERVAL X MONTH`
            - 'within X years' → `INTERVAL X YEAR`
        - Apply to relevant date fields (e.g., `expiration_date`, `termination_date`, `effective_date`) with `expiration_date` as the default unless specified otherwise.
        6. For monetary values (e.g., contract value):
        - Assume `contract_value` is stored as a numeric string (e.g., '400000') and convert to numeric for comparison using `CAST(contract_value AS DECIMAL)`.
        - Handle ranges with `BETWEEN` or comparison operators (`>`, `<`, etc.).
        7. For queries involving specific clauses (e.g., indemnity, dispute resolution):
        - Search `scope_of_work` for clause-specific terms unless another field is more relevant.
        - Use broad `LIKE` patterns to account for variations (e.g., 'indemnity' vs. 'indemnification').
        8. Ensure the query is efficient, avoids unnecessary complexity, and adheres to SQL best practices.
        9. If the user query is ambiguous, prioritize searching `contract_type` and `title_of_contract` for text-based filters, and include both fields unless otherwise specified.
        10. Do not include `file_name` in filters unless explicitly requested by the user.
        11. **Edge Cases**:
        - For ambiguous queries, prefer filtering on `contract_type`.
        - For “active contracts,” use `(expiration_date > CURRENT_DATE OR termination_date > CURRENT_DATE)`.
        - For queries with specific durations (e.g., “12-month term”), include `contract_duration` if relevant.
        - For queries mentioning “confidential terms,” prioritize `contract_type` like 'NDA' or include confidentiality-related clauses in `title_of_contract`.
        12. Include all synonyms and variations in the `LOWER(contract_type)` and `LOWER(title_of_contract)` conditions to ensure comprehensive matching below are just some example, include more during query generation.

        ---

        ###Example Queries:

        1. **"Write a Loan Agreement specifying repayment terms, interest rates, and obligations"**
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE LOWER(contract_type) IN ('loan', 'loan agreement', 'loan contract', 'credit', 'credit agreement', 'credit contract')
        OR LOWER(title_of_contract) LIKE '% loan %'
        OR LOWER(title_of_contract) LIKE '%loan agreement%'
        OR LOWER(title_of_contract) LIKE '%loan contract%'  
        OR LOWER(title_of_contract) LIKE '% credit %'
        OR LOWER(title_of_contract) LIKE '%credit agreement%'
        OR LOWER(title_of_contract) LIKE '%credit contract%'
        ```

        When checking single terms like 'sale' or 'sales' in 'title_of_contract' column add one sppace before and after '% sales %',etc and use `LOWER()` and `LIKE` to ensure case-insensitive matching.

        2. **"Create a Partnership Agreement establishing roles, responsibilities, and profit-sharing terms"**
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE LOWER(contract_type) IN ('partnership', 'joint venture', 'partnership agreement', 'joint venture agreement', 'partnership contract', 'joint venture contract')
        OR LOWER(title_of_contract) LIKE '% partnership %'
        OR LOWER(title_of_contract) LIKE '% joint venture %'
        OR LOWER(title_of_contract) LIKE '%partnership agreement%'
        OR LOWER(title_of_contract) LIKE '%partnership contract%'
        OR LOWER(title_of_contract) LIKE '%joint venture contract%'
        OR LOWER(title_of_contract) LIKE '%joint venture agreement%'
        ```

        3. **"Draft a Non-Disclosure Agreement (NDA) to safeguard confidential information"**
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE LOWER(contract_type) IN ('non-disclosure agreement (nda)', 'non-disclosure agreement', 'nda')
        OR LOWER(title_of_contract) LIKE '%non-disclosure agreement (nda)%'
        OR LOWER(title_of_contract) LIKE '% nda %'
        OR LOWER(title_of_contract) LIKE '%non-disclosure agreement%'
        OR LOWER(title_of_contract) LIKE '%non-disclosure%'
        ```

        4. **"Create a Sales Agreement outlining key terms for buying and selling goods or services"** OR **"Draft a commercial sale of goods contract between a supplier and retailer"** OR **"Generate a buyer and seller contract for a vehicle sale"**
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE LOWER(contract_type) IN ('sales', 'sales agreement', 'sales contract')
        OR LOWER(title_of_contract) LIKE '% sales %'
        OR LOWER(title_of_contract) LIKE '%sales agreement%'
        OR LOWER(title_of_contract) LIKE '%sales contract%'
        ```

        5. **"Prepare a Service Agreement defining the scope, responsibilities, and conditions of the provided services."**
        For service agreement strictly make sure to generate below query only:
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE LOWER(contract_type) IN ('service', 'services', 'service agreement', 'service contract')
        OR LOWER(title_of_contract) LIKE '% service %'
        OR LOWER(title_of_contract) LIKE '% services %'
        OR LOWER(title_of_contract) LIKE '%service agreement%'
        OR LOWER(title_of_contract) LIKE '%service contract%'
        ```

        6. **"Create a real estate purchase agreement for buying residential property"**
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE LOWER(contract_type) IN ('purchase', 'purchase agreement', 'real estate purchase', 'purchase contract)
        OR LOWER(title_of_contract) LIKE '% purchase %'
        OR LOWER(title_of_contract) LIKE '%purchase agreement%'
        OR LOWER(title_of_contract) LIKE '%real estate purchase%'
        ```

        7. **"Draft a standard lease agreement for leasing equipment or machinery"** OR **"Create a simple residential rental agreement for a 12-month term"** OR **"Generate an Equipment Rental Agreement for leasing audiovisual gear"** OR  **"Create a rental agreement with confidential terms"**
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE LOWER(contract_type) IN ('lease', 'lease agreement', 'commercial lease', 'rental', 'rental agreement')
        OR LOWER(title_of_contract) LIKE '% lease %'
        OR LOWER(title_of_contract) LIKE '%lease agreement%'
        OR LOWER(title_of_contract) LIKE '% rental %'
        OR LOWER(title_of_contract) LIKE '%rental agreement%'
        OR LOWER(title_of_contract) LIKE '%commercial lease%'
        OR LOWER(title_of_contract) LIKE '%commercial lease agreement%'
        ```


        8. **"Draft a Vendor Agreement for a third-party service provider offering logistics support"**
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE LOWER(contract_type) IN ('vendor', 'vendor agreement', 'vendor contract')
        OR LOWER(title_of_contract) LIKE '% vendor %'
        OR LOWER(title_of_contract) LIKE '%vendor agreement%'
        OR LOWER(title_of_contract) LIKE '%vendor contract%'
        ```


        9. **"Create a License Agreement to authorize the use of intellectual property"** OR **"Draft a Software Licensing Agreement for a SaaS platform"**
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE LOWER(contract_type) IN ('license', 'license agreement', 'licence contract')
        OR LOWER(title_of_contract) LIKE '% license %'
        OR LOWER(title_of_contract) LIKE '%license agreement%'
        OR LOWER(title_of_contract) LIKE '%licence contract%'
        ```


        10. **"Write a Collaboration Agreement for co-marketing efforts"**
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE LOWER(contract_type) IN ('collaboration', 'partnership', 'collaboration agreement', 'partnership agreement', 'collaboration contract', 'partnership contract')
        OR LOWER(title_of_contract) LIKE '% collaboration %'
        OR LOWER(title_of_contract) LIKE '%collaboration agreement%'
        OR LOWER(title_of_contract) LIKE '%collaboration contract%'
        OR LOWER(title_of_contract) LIKE '% partnership %'
        OR LOWER(title_of_contract) LIKE '%partnership agreement%'
        OR LOWER(title_of_contract) LIKE '%partnership contract%'
        ```

        11. **"Create a freelance services agreement for digital marketing services"**
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE LOWER(contract_type) IN ('freelance', 'freelance services', 'freelance agreement', 'freelance contract')
        OR LOWER(title_of_contract) LIKE '% freelance %'
        OR LOWER(title_of_contract) LIKE '%freelance services%'
        OR LOWER(title_of_contract) LIKE '%freelance agreement%'
        OR LOWER(title_of_contract) LIKE '%freelance contract%'
        ```


        12. **"Contracts expiring within 120 days"**
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE expiration_date BETWEEN CURRENT_DATE AND DATE_ADD(CURRENT_DATE, INTERVAL 120 DAY)
        ```

        13. **"Which contracts don't have jurisdiction"**
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE jurisdiction IS NULL
        ```

        13. **"Create a Sales and Purchase Agreement outlining key terms for buying and selling of property"**
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE LOWER(contract_type) IN ('purchase', 'purchase and sale', 'sale and purchase', 'purchase and sales', 'sales and purchase')
        OR LOWER(title_of_contract) LIKE '%sale and purchase%'
        OR LOWER(title_of_contract) LIKE '%purchase and sale%'
        OR LOWER(title_of_contract) LIKE '%sales and purchase%'
        OR LOWER(title_of_contract) LIKE '%purchase and sales%'
        OR LOWER(title_of_contract) LIKE '% purchase %'
        ```

        14 *Filter the contract which contains the contract value as $0*
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE CAST(contract_value AS DECIMAL) = 0 OR contract_value IS NULL
        ```

        15. *Futures and Cleared Derivative Agreement*
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE LOWER(contract_type) IN ('futures', 'cleared derivative', 'futures and cleared derivative')
        OR LOWER(title_of_contract) LIKE '% futures %'
        OR LOWER(title_of_contract) LIKE '%futures and cleared derivative%'
        OR LOWER(title_of_contract) LIKE '%cleared derivative%'
        OR LOWER(title_of_contract) LIKE '%cleared derivative%'
        ```

        16. *Master Securities Lending Agreements (MSLA)*
        ```sql
        SELECT ci_file_guid FROM file_metadata_view_{org_id}
        WHERE LOWER(contract_type) IN ('master securities lending', 'msla', 'securities lending', 'security lending', 'master security lending')
        OR LOWER(title_of_contract) LIKE '%master securities lending%'
        OR LOWER(title_of_contract) LIKE '%master security lending%'
        OR LOWER(title_of_contract) LIKE '%securities lending%'
        OR LOWER(title_of_contract) LIKE '%security lending%'
        OR LOWER(title_of_contract) LIKE '% msla %'
        ```

        ---
        The following are the distinct contract types available in the file_metadata_view_{org_id} table:
        {distinct_contract_type_list}

        When applying the filter, include all relevant contract types from this list that are related to the user's search query. The filter should be applied to both LOWER(contract_type) and LOWER(title_of_contract) fields.
        ---


        ### Notes:
        * Do not confuse between two different contract types.
        * Always prioritize semantic accuracy and focus on `contract_type` and `title_of_contract`.
        * Do **not** include `file_name` in the WHERE condition.
        * Handle synonyms and variations (e.g., 'rental' = 'lease', 'sales' = 'sale').
        * Use `contract_duration` for specific terms like “12-month” when mentioned.
        * For queries with “confidential,” consider NDA or confidentiality clauses.
        * Strictly do not include the any markers like '```' or '```sql' and terms 'agreement', 'agreements', 'contract', and 'contracts' in LIKE if they appear in the user's question during query generation.
        * Today’s date is: {current_date}
        """

        
        user_prompt = f"""
        Generate an SQL query to retrieve `ci_file_guid` from `file_metadata_view_{org_id}` based on the following user question:

        ### User Query:
        {question.lower()}

        ### Output:
        - Strictly do not include any markers like '```' or '```sql' int the response.
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
    #     # Clean up any resources if needed
    #     execute_drop_view(org_id, logger)  # Ensure the view is dropped after processing
