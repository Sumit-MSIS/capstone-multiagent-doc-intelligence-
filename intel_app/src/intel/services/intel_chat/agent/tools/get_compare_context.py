import os
import json
import asyncio
import mlflow
from src.intel.services.intel_chat.agent.pinecone_retriever.hybrid_search_retrieval import get_pinecone_index_async, get_embeddings, get_context_from_pinecone
from src.intel.services.intel_chat.agent.utils import get_page_count, get_metadata, get_all_file_names, read_summary_json
from src.intel.services.intel_chat.utils import get_metadata, get_file_names
read_summary_json
import asyncio
from pinecone import Pinecone
from src.common.llm.factory import get_llm_client
import re
from src.intel.services.intel_chat.agent.query_rephraser import rephrase_search_text
async_client = get_llm_client(async_mode=True)

@mlflow.trace(name="Escape Markdown Filename")
async def escape_markdown_filename(file_name: str) -> str:
    """
    Escapes special Markdown characters in a file name so it renders correctly.
    """
    escape_chars = r"[\[\]\(\)\\]"
    return re.sub(escape_chars, lambda m: "\\" + m.group(0), file_name)



@mlflow.trace(name="Retreive Entire Chunks - Pinecone")
async def retrieve_entire_chunks(user_id: int, org_id: int, file_id: str) -> str:
    """Fetches and reconstructs an entire document from Pinecone vector chunks."""
    try:
        index = await get_pinecone_index_async(os.getenv("DOCUMENT_SUMMARY_AND_CHUNK_INDEX"), os.getenv("PINECONE_API_KEY"))
        logical_partition = f"org_id_{org_id}#"
        loop = asyncio.get_running_loop()

        # Fetch first chunk to determine total number of chunks
        initial_id = f"{org_id}#{file_id}#1"
        data = await loop.run_in_executor(
            None,
            lambda: index.fetch(ids=[initial_id], namespace=logical_partition)
        )

        # Extract chunk count safely
        chunks_value = next(
            (int(vector['metadata'].get('chunks_count', 0)) for vector in data.get('vectors', {}).values()), 0
        )

        if chunks_value == 0:
            print(f"No chunks found for file_id {file_id}.")
            # logger.warning(_log_message(f"No chunks found for file_id {file_id}.", 'retrieve_entire_chunks', module_name))
            return ""

        # Generate all chunk IDs
        chunk_ids = [f"{org_id}#{file_id}#{i+1}" for i in range(chunks_value)]

        # Fetch in batches to respect rate limits
        whole_document_dict = {}
        batch_size = 200

        for i in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[i:i + batch_size]
            data_batch = await loop.run_in_executor(
                None,
                lambda batch=batch_ids: index.fetch(ids=batch, namespace=logical_partition)
            )

            for vector_id, vector in data_batch.get('vectors', {}).items():
                chunk_number = int(vector_id.rsplit('#', 1)[-1])  # Safer split
                whole_document_dict[chunk_number] = vector['metadata'].get('text', '')

            # Respect rate limits (delay only if more requests remain)
            if i + batch_size < len(chunk_ids):
                await asyncio.sleep(1)

        # Reassemble document from sorted chunks
        return ' '.join(whole_document_dict[i] for i in sorted(whole_document_dict) if whole_document_dict[i]), len(whole_document_dict)

    except Exception as e:
        print(f"Error in retrieve_entire_chunks: {e}")
        # logger.error(_log_message(f"Error in retrieve_entire_chunks: {e}", 'retrieve_entire_chunks', module_name))
        return ""


@mlflow.trace(name="Generate Compare Context For File")
async def get_compare_context_for_file(user_query:str, user_id:int, org_id:int, file_id:str, tag_ids, file_info,  file_metadata, focus_list):
    try:
        # docs = ""
        page_count = await get_page_count(file_id)

        if page_count > 10:
            print(f"Generating compare using JSON method for file_id {file_id} with {page_count} pages.")
            return await generate_compare_json_method(user_query, user_id, org_id, file_id, tag_ids, file_info, file_metadata, focus_list)
        
        # Retrieve the entire document
        docs, _ = await retrieve_entire_chunks(user_id, org_id, file_id)
        doc_length = len(docs)
        if doc_length == 0:
            return "Unable to retrieve document", file_info.get(file_id)

        # if focus_list:
        #     context_set = set()
        #     # Run all focus parallel and collect context from get_context_from_pinecone for that file_id
        #     for focus in focus_list:
        #         dense_vectors = get_embeddings(org_id,file_id,focus)
        #         custom_filter = {
        #             "$and": [
        #                 {"tag_ids": {"$in": tag_ids}},
        #                 {"file_id": {"$eq": file_id}}
        #             ]
        #         } if tag_ids else {"file_id": {"$eq": file_id}}
        #         keyword, semantic, _ = await rephrase_search_text(focus)

        #         context_chunks = get_context_from_pinecone(
        #             pinecone_dense_index_name=os.getenv("DOCUMENT_SUMMARY_AND_CHUNK_INDEX"),
        #             pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        #             custom_filter=custom_filter,
        #             top_k=5,
        #             query=semantic,
        #             file_id=file_id,
        #             org_id=org_id,
        #             tag_ids=tag_ids,
        #             keyword_search_query=keyword,
        #             dense_vectors=dense_vectors,
        #         )
        #         matches = context_chunks.get("matches", [])
        #         for match in matches:
        #             context_set.add(match["metadata"]["text"])

        #     docs += "\n\n".join(list(context_set))

        # else:
            # page_count = await get_page_count(file_id)

            # if page_count > 10:
            #     print(f"Generating summary using JSON method for file_id {file_id} with {page_count} pages.")
            #     return await generate_summary_json_method(user_query, user_id, org_id, file_id, tag_ids, file_info, file_metadata)
            
            # # Retrieve the entire document
            # docs_, _ = await retrieve_entire_chunks(user_id, org_id, file_id)
            # doc_length = len(docs)
            # if doc_length == 0:
            #     return "Unable to retrieve document", file_info.get(file_id)

        file_name = file_info.get(file_id, "")
        cleaned_file_name = await escape_markdown_filename(file_name)

        # File hyperlink marker
        file_hyperlink_start = f"<<< FILE START: [{cleaned_file_name}](/preview?fileId={file_id}) >>>\n\n"
        file_hyperlink_end = f"\n\n<<< FILE END: [{cleaned_file_name}](/preview?fileId={file_id}) >>>\n\n"

        # if agent_call:
        #     return combined_summaries, file_info.get(file_id)
                
        get_summary_system_prompt = """
            You are a Legal AI assistant specializing in extracting key legal details from contracts and agreements with accuracy, precision, and clause-level references.

            ## Core Guidelines (Strict Extraction Only)
            - DO NOT SUMMARIZE the document.
            - Extract **every key detail**, clause-by-clause where applicable.
            - For each key detail:
                1. **Quote the exact clause text** (full clause if necessary).
                2. **Provide a clause citation / reference** (e.g., clause number, heading, or location).
                3. **Give a short interpretation** explaining what the clause means in simple terms.
            - Maintain accuracy, neutrality, and professionalism.
            - Do not skip any important obligations, rights, restrictions, conditions, or exceptions.
            - No assumptions or opinions.

            ## Extract at least the following categories (if present):
            - Parties, dates, duration, renewal terms.
            - Scope of work / services, deliverables, standards, obligations.
            - Payment terms: fees, schedule, taxes, penalties, late fees, invoices.
            - Intellectual property rights, licenses, confidentiality, data security.
            - Liability, indemnification, warranties, disclaimers.
            - Termination rights, triggers, post-termination duties, penalties.
            - Governing law, dispute resolution, arbitration.
            - Force majeure, amendments, notices, subcontracting, assignment.
            - Any restrictive covenants (non-compete, non-solicit, exclusivity).
            - Any other clause containing material legal obligations.

            ## Format Requirements
            - Output must be organized by **extracted key point**, each with:
                - **Clause Citation**
                - **Exact Clause Text**
                - **Interpretation**
            - No summary paragraphs.
            - No filler, no boilerplate, no placeholders.
            - Do not repeat the document headings directly unless needed for the clause reference.
        """
        get_summary_user_prompt = f"""
            ---
            {file_hyperlink_start}
            {docs}
            {file_hyperlink_end}
            ---

            ## Extraction Instructions
            - Begin the response with the file name as a Markdown hyperlink heading in this format:  
            **[file_name](/preview?fileId=file_id)**

        """

        initial_messages = [
            {"role": "system", "content": get_summary_system_prompt},
            {"role": "user", "content": get_summary_user_prompt}
        ]
        res = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=initial_messages,
            max_completion_tokens=5000 ### nEEDS TO BE ADJUSTED FOR MULTIPLE FILES
        )
        summary_results = res.choices[0].message.content
        return summary_results, f"[{cleaned_file_name}](/preview?fileId={file_id})"
        
    except Exception as e:
        print(f"Error in generate_summary_for_file: {e}")
        return "Unable to generate summary", f"[{cleaned_file_name}](/preview?fileId={file_id})"

@mlflow.trace(name="Generate Compare JSON Method")
async def generate_compare_json_method(user_query:str, user_id:int, org_id:int, file_id:str, tag_ids, file_info, file_metadata, focus_list):
    try:
        contract_type = file_metadata.get(file_id,{}).get("metadata",{}).get("Contract Type",None)
        data = read_summary_json()
        key_points = data.get(contract_type, [])

        if not key_points:
            print(f"No key points found for contract type: {contract_type}. Using default key points.")
            return await generate_compare_hyde_method(user_query, user_id, org_id, file_id, tag_ids, file_info, file_metadata, focus_list)
        
        key_points.append({"title": "User Query","semantic_query": user_query, "keyword_search": user_query, "description": "Ignore this section unless specified any specific focus in user query."})
        custom_filter = {
            "$and": [
                {"tag_ids": {"$in": tag_ids}},
                {"file_id": {"$eq": file_id}}
            ]
        } if tag_ids else {"file_id": {"$eq": file_id}}

        @mlflow.trace(name="Generate Summary JSON - Fetch Context Chunks")
        async def fetch_context_chunks(
            pinecone_index_name,
            pinecone_api_key,
            custom_filter,
            file_id,
            user_id,
            org_id,
            queries,
            tag_ids=None,
            top_k=3
        ):
            """
            Run multiple Pinecone queries (keyword + semantic) async and return unique chunks.
            """

            @mlflow.trace(name="Generate Summary JSON - Run Pinecone Query")
            async def run_query(semantic_query,keywords):
                dense_vectors = await get_embeddings(org_id,file_id,semantic_query)
                # start_time = time.time()
                context_chunks = await get_context_from_pinecone(
                    pinecone_index_name,
                    pinecone_api_key,
                    custom_filter,
                    top_k,
                    semantic_query,
                    file_id,
                    org_id,
                    tag_ids,
                    keywords,
                    dense_vectors,
                )
                matches = context_chunks.get("matches", [])
                return [match["metadata"]["text"] for match in matches]

            # Run all queries concurrently
            if focus_list:
                for focus in focus_list:
                    keyword, semantic, _ = await rephrase_search_text(focus)
                    queries.append({"title": "","semantic_query": semantic, "keyword_search": keyword, "description": ""})

            results = await asyncio.gather(*(run_query(q["semantic_query"], q["keyword_search"]) for q in queries))

            # Flatten & deduplicate
            all_chunks = [chunk for result in results for chunk in result]
            unique_chunks = list(dict.fromkeys(all_chunks))

            return unique_chunks
    
        unique_chunks = await fetch_context_chunks(
            pinecone_index_name=os.getenv("DOCUMENT_SUMMARY_AND_CHUNK_INDEX"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            custom_filter=custom_filter,
            file_id=file_id,
            user_id=user_id,
            org_id=org_id,
            queries=key_points,
            tag_ids=tag_ids,
            top_k=10
        )

        for chunk in unique_chunks:
            print(f"Chunk: {chunk}\nLength: {len(chunk)}\n")
        
        combined_chunks = "\n\n".join(unique_chunks)
        file_name = file_info.get(file_id, "")
        cleaned_file_name = await escape_markdown_filename(file_name)

        # File hyperlink marker
        file_hyperlink_start = f"<<< FILE START: [{cleaned_file_name}](/preview?fileId={file_id}) >>>\n\n"
        file_hyperlink_end = f"\n\n<<< FILE END: [{cleaned_file_name}](/preview?fileId={file_id}) >>>\n\n"
        all_sections = [f"{t['title']}: {t['description']}" for t in key_points]
        sections_str = "\n".join([f"- {title}" for title in all_sections])

        get_summary_system_prompt = f"""
            You are a Legal AI assistant specializing in extracting key legal details from {contract_type} contracts and agreements with clause-level accuracy, without summarizing any part of the document.

            ## Core Requirement (STRICT)
            - Do **NOT** summarize.
            - Extract **all key details**, including but not limited to the reference key-points provided below.
            - The details must come directly from the document. Do not infer or assume.

            ### Reference Key-Points (for guidance only; DO NOT limit extraction to these):
            {sections_str}

            You must also extract **any additional important legal content even if not listed in the reference key-points**.

            ## For Each Extracted Point:
            You must produce the following three components:

            1. **Clause Citation**  
            - Exact clause number, section title, heading, paragraph identifier, or any textual locator available.

            2. **Exact Clause Text**  
            - Quote the full clause or the full exact portion necessary to preserve meaning.  
            - Do not rewrite or compress the clause.

            3. **Interpretation**  
            - Provide a concise explanation in plain language describing what the clause means or obligates.  
            - No opinions, no speculation—only clarifying the meaning.

            ## Output Format Requirements
            - Organize output by extracted key points.
            - For each point, include:
            - **Clause Citation**
            - **Exact Clause Text**
            - **Interpretation**

            ## Prohibited
            - No summaries.
            - No introductory paragraphs.
            - No closing statements.
            - No filler text.
            - No assumptions.
            - No format constraints unless explicitly requested by the user.
        """

        get_summary_user_prompt = f"""
            ---
            {file_hyperlink_start}
            {combined_chunks}
            {file_hyperlink_end}
            ---

            ## Extraction Instructions
            - The response must be valid, user-friendly, and formatted in Markdown, fully compatible with react-markdown.
            - Extract information **one contract at a time**.
            - Begin the response with the file name as a Markdown hyperlink heading in this format:  
            **[file_name](/preview?fileId=file_id)**
        """


        initial_messages = [
            {"role": "system", "content": get_summary_system_prompt},
            {"role": "user", "content": get_summary_user_prompt}
        ]

        res = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=initial_messages,
            max_completion_tokens=5000 ### nEEDS TO BE ADJUSTED FOR MULTIPLE FILES
        )
        summary_results = res.choices[0].message.content
        return summary_results, f"[{cleaned_file_name}](/preview?fileId={file_id})"

    except Exception as e:
        print(f"Error in generate_compare_json_method: {e}")
        return "Unable to generate compare", f"[{cleaned_file_name}](/preview?fileId={file_id})"



@mlflow.trace(name="Generate Compare Hyde Method")
async def generate_compare_hyde_method(user_query:str, user_id:int, org_id:int, file_id:str, tag_ids,file_info,  file_metadata, focus_list):
    try:
        contract_type = file_metadata.get(file_id,{}).get("metadata",{}).get("Contract Type",None)
        
        @mlflow.trace(name="Generate Compare HYDE - Generate Hypothetical Queries")
        async def generate_hypothetical_queries(user_query: str, contract_type: str) -> str:
            """
            Generate key points, keyword queries, and semantic queries
            based on the user query and contract type.
            Returns JSON only.
            """
            try:
                # system_prompt = f"""
                # You are an expert legal contract analyst.
                # Your role is to generate a structured set of the most important clauses or sections that are typically found in contracts.

                # Instructions:
                # 1. Always output in valid JSON format.
                # 2. Respect the user request strictly:
                # - If the user specifies a number (e.g., 2, 10, 20), return exactly that many clauses.
                # - If the user specifies a particular clause or focus (e.g., "termination clause"), return only that clause.
                # - If nothing is specified, default to 8 clauses.
                # 3. Select the most critical and useful sections typically present in the given contract type.
                # 4. For each clause, provide:
                # - "title": Clear name of the clause.
                # - "keyword_query": Boolean/search-operator-style keyword query.
                # - "semantic_query": Natural language query describing what to look for in context.
                # 5. Do not summarise the document text. Instead, return the **standard important clauses** a contract analyst would extract for that contract type.
                # """

                system_prompt =  f"""
                You are an expert legal contract analyst.
                Your role is to generate a structured set of the most important clauses or sections 
                that must be extracted from contracts, and also flag missing or weak clauses.

                Instructions:
                1. Always output in valid JSON format.
                2. Respect the user request strictly:
                - If the user specifies a number (e.g., 2, 10, 20), return exactly that many clauses.
                - If the user specifies a particular clause or focus (e.g., "termination clause"), return only that clause.
                - If nothing is specified, default to all key points below.
                3. Select the most critical and useful sections listed below.
                4. For each clause, provide:
                - "title": Clear name of the clause.
                - "description": Guidance on what information must be captured in that clause.
                - "keyword_query": Boolean/search-operator-style keyword query.
                - "semantic_query": Natural language query describing what to look for in context.

                Mandatory Coverage - Key Points
                ====================================

                1. Basic Details
                - Contract Name & Type:
                * Capture the formal name of the contract and its category (e.g., Sales Agreement, Lease).
                - Parties Involved:
                * Capture full legal names, entity type (Corp, LLC, Trust), and roles (Seller/Buyer, Lessor/Lessee).
                * Must also handle implicit forms: “between X and Y,” “entered into by…”.
                - Effective Date:
                * Start date of the contract (execution, signature, commencement).
                * Must capture implicit “shall commence on…” or “effective upon signing.”
                - Expiration Date / Closing Date:
                * Explicit end date or closing date.
                * Must detect implicit “valid until,” “end of term,” or “shall expire.”
                - Contract Duration / Term Period:
                * Total period in months/years.
                * Capture formulas like “99 months from commencement” and calculate end.
                - Renewal / Extension Terms:
                * Clauses about automatic renewal, evergreen, rollover.
                * Must catch implicit wording: “successive one-year periods,” “shall continue unless terminated.”

                2. Scope of Agreement
                - Purpose / Scope of Services or Goods:
                * Define subject matter of the contract.
                - Service Levels (SLA) if applicable:
                * Capture obligations like uptime, quality standards, penalties.

                3. Financial Terms
                - Contract Value / Purchase Price:
                * Must capture explicit total value, or implicit formulas.
                - Payment Due Dates:
                * Capture explicit deadlines and implicit timing (“within 30 days,” “45 days after invoice”).
                - Payment Terms:
                * Deposits, milestone payments, interest, penalties, late fees.
                - Payment Methods:
                * Wire, cheque, escrow, ACH, etc.
                - Taxes / Withholding Obligations:
                * Which party bears VAT, sales tax, duties, or withholding.

                4. Obligations and Responsibilities
                - Core Duties of Each Party:
                * Capture what each side must perform.
                - Operational / Reporting Obligations:
                * Regular reporting, audits, notices.
                - Restrictions on Use:
                * Limitations on IP, data, or resources.

                5. Confidentiality & Intellectual Property
                - Confidentiality Obligations:
                * Scope, exclusions, duration.
                - Data Protection Standards:
                * Explicit compliance (GDPR, HIPAA, CCPA).
                - Ownership of Deliverables / IP Rights:
                * Who owns work product, source code, designs.
                - Licenses Granted or Restricted:
                * Software, patents, trade secrets.

                6. Liability & Risk Allocation
                - Limitation of Liability:
                * Caps, exclusions, disclaimers.
                - Indemnification Obligations:
                * Who indemnifies whom, for what (IP infringement, third-party claims).
                - Risk of Loss:
                * Which party bears risk before delivery/closing.
                - Warranties / Disclaimers:
                * Express, implied, “as-is.”

                7. Termination
                - Termination Date / Events:
                * Explicit termination dates or event-based triggers.
                - Termination Provisions:
                * Triggers: breach, insolvency, convenience, regulatory issues.
                - Notice & Cure Periods:
                * Timeframes for notice and remedy.
                - Consequences of Termination:
                * Payments due, obligations surviving, return of materials.

                8. Governing Law & Disputes
                - Governing Law & Jurisdiction:
                * Capture state/country of governing law.
                - Dispute Resolution:
                * Arbitration, mediation, forum selection, fee-shifting.

                9. Red-Flag Clauses (High-Risk Indicators — handled through reasoning and examples)
                Red-flag clauses are provisions that unfairly shift risk, reduce transparency, or create legal or commercial disadvantage 
                for one party. Do NOT rely on fixed keywords. Instead, reason about clause intent and generate dynamic queries that 
                can detect such risky language.

                Your goal:
                - Identify patterns or clause intents that typically disadvantage one party.
                - For each detected risk theme, create one query that could help locate such language in the document.
                - The number of red-flag queries can vary (typically 5–10), depending on what risks are relevant to this contract type.

                Few-shot guidance — examples of what counts as a red flag:

                Example 1 – Automatic Renewal without Notice:
                - Pattern: contract renews automatically unless notice is given.
                - Example semantic_query: "Does the contract renew automatically without explicit consent?"

                Example 2 – Unilateral Control or Sole Discretion:
                - Pattern: one party can amend, terminate, or make decisions without mutual consent.
                - Example semantic_query: "Can one party make changes without the other party’s approval?"

                Example 3 – Unlimited Liability / Broad Indemnity:
                - Pattern: unlimited liability, indemnification for all losses, or missing liability caps.
                - Example semantic_query: "Does the contract impose unlimited liability or one-sided indemnification?"

                Example 4 – Restrictive or Exclusive Dealing:
                - Pattern: exclusivity, non-competition, or inability to engage with others.
                - Example semantic_query: "Does the contract restrict a party from working with competitors?"

                Example 5 – Unreasonable Termination / Cure Periods:
                - Pattern: immediate termination or short notice (≤10 days).
                - Example semantic_query: "Are there termination rights with very short notice or cure periods?"

                Example 6 – Other Risky Terms:
                - Pattern: perpetual confidentiality, foreign jurisdiction, penalty interest rates, data-sharing without consent.
                - Example semantic_query: "Does the contract contain any other terms that impose hidden or unfair risk?"

                Output rule:
                Include red-flag queries under titles like “Red Flag – Auto Renewal,” “Red Flag – Liability Cap,” etc.
                Always include at least one red-flag query even if none are obvious.

                10. Other Important Clauses (contract-type specific)
                - Subcontracting / Sub-licensing restrictions.
                - Non-compete / Non-solicitation obligations.
                - Insurance Requirements:
                * Fire, liability, builder’s risk, worker’s comp (esp. leases).
                - Environmental Compliance:
                * Hazardous waste, legal obligations.
                - Credit Line / Collateral / Guarantees:
                * Esp. loan or Letter of Credit agreements.

                """

                hyde_prompt = f"""
                Summarise the most important clauses for the following contract type:

                User Query: "{user_query}"
                Contract Type: "{contract_type}"

                Return strictly in this JSON structure:
                {{
                "contract_type": "{contract_type}",
                "queries": [
                    {{
                    "title": "...",
                    "keyword_query": "...",
                    "semantic_query": "..."
                    }}
                ]
                }}
                """

                hyde_response = await async_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": hyde_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_completion_tokens=5000
                )

                # Extract content safely
                response_content = hyde_response.choices[0].message.content.strip()
                print(f"Raw Hypothetical Queries Response: {response_content}")

                # Ensure valid JSON output
                try:
                    parsed_json = json.loads(response_content)
                    print(
                        f"Generated Hypothetical Queries: {json.dumps(parsed_json, indent=4)}"
                    )
                    return parsed_json
                except json.JSONDecodeError:
                    print("Failed to parse JSON from response.")
                    return json.dumps(
                        {"error": "Invalid JSON returned", "raw_output": response_content},
                        indent=2,
                    )

            except Exception as e:
                print(f"Error in generate_hypothetical_queries: {e}")
                return json.dumps({"error": str(e)}, indent=2)


        key_points_json = await generate_hypothetical_queries(user_query, contract_type)
        key_points = key_points_json.get("queries", [])
        if not key_points:
            print(f"No key points found for contract type: {contract_type}.")
            # logger.info(_log_message(f"No key points found for contract type: {contract_type}", 'get_summary', module_name))
            return "Unable to generate summary." , file_info.get(file_id)
        
        custom_filter = {
            "$and": [
                {"tag_ids": {"$in": tag_ids}},
                {"file_id": {"$eq": file_id}}
            ]
        } if tag_ids else {"file_id": {"$eq": file_id}}

        @mlflow.trace(name="Generate Summary HYDE - Fetch Context Chunks")
        async def fetch_context_chunks(
            pinecone_index_name,
            pinecone_api_key,
            custom_filter,
            file_id,
            user_id,
            org_id,
            queries,
            tag_ids=None,
            top_k=3
        ):
            """
            Run multiple Pinecone queries (keyword + semantic) async and return unique chunks.
            """

            @mlflow.trace(name="Generate Summary JSON - Run Pinecone Query")
            async def run_query(semantic_query,keywords):
                dense_vectors = await get_embeddings(org_id,file_id,semantic_query)
                # start_time = time.time()
                context_chunks = await get_context_from_pinecone(
                    pinecone_index_name,
                    pinecone_api_key,
                    custom_filter,
                    top_k,
                    semantic_query,
                    file_id,
                    org_id,
                    tag_ids,
                    keywords,
                    dense_vectors,
                )
                matches = context_chunks.get("matches", [])
                return [match["metadata"]["text"] for match in matches]

            for focus in focus_list:
                keyword, semantic, _ = await rephrase_search_text(focus)
                queries.append({"title": "","semantic_query": semantic, "keyword_query": keyword, "description": ""})
            # Run all queries concurrently
            results = await asyncio.gather(*(run_query(q["semantic_query"], q["keyword_query"]) for q in queries))

            # Flatten & deduplicate
            all_chunks = [chunk for result in results for chunk in result]
            unique_chunks = list(dict.fromkeys(all_chunks))

            return unique_chunks
    
        unique_chunks = await fetch_context_chunks(
            pinecone_index_name=os.getenv("DOCUMENT_SUMMARY_AND_CHUNK_INDEX"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            custom_filter=custom_filter,
            file_id=file_id,
            user_id=user_id,
            org_id=org_id,
            queries=key_points,
            tag_ids=tag_ids,
            top_k=10
        )

        for chunk in unique_chunks:
            print(f"Chunk: {chunk}\nLength: {len(chunk)}\n")
        
        combined_chunks = "\n\n".join(unique_chunks)
        file_name = file_info.get(file_id, "")
        cleaned_file_name = await escape_markdown_filename(file_name)

        # File hyperlink marker
        file_hyperlink_start = f"<<< FILE START: [{cleaned_file_name}](/preview?fileId={file_id}) >>>\n\n"
        file_hyperlink_end = f"\n\n<<< FILE END: [{cleaned_file_name}](/preview?fileId={file_id}) >>>\n\n"
        
        hyde_system_prompt = """
            You are a senior legal contract analyst specializing in clause-level extraction from contracts and agreements.

            Your task is to read the retrieved contract chunks and produce a unified extraction of ALL key details — NOT a summary.

            ## Core Requirement (Strict)
            - Do NOT summarize.
            - Extract **every material clause**, obligation, right, restriction, term, condition, exception, and definition.
            - For each extracted point, include:
                1. **Clause Citation** — clause number, heading, paragraph label, or identifiable locator.
                2. **Exact Clause Text** — quote the full clause or the portion required to preserve complete meaning.
                3. **Interpretation** — a concise explanation of the clause in plain, neutral English.

            ## What to Extract (Non-Exhaustive)
            Extract all key legal and commercial content, including but not limited to:
            - Parties, roles, contract type, dates, duration, renewal.
            - Scope, deliverables, duties, standards, milestones.
            - Pricing, financial obligations, payment terms, late fees, taxes, billing, credits.
            - Confidentiality, IP rights, licenses, ownership, restrictions.
            - Liability limits, indemnification, warranties, disclaimers.
            - Termination triggers, notice periods, survival obligations.
            - Governing law, jurisdiction, dispute resolution, arbitration.
            - Data protection, privacy, security.
            - Force majeure, amendments, assignment, subcontracting, notices.
            - Compliance requirements, representations, restrictive covenants.
            - Any additional material obligations or effects not listed above.

            ## Output Format
            - Organize output as a list of extracted points.
            - For each point:
                - **Clause Citation**
                - **Exact Clause Text**
                - **Interpretation**
            - No narrative flow or linking sentences.
            - No introductory or closing remarks.
            - No filler text.
            - If a detail is missing, state: **"Not specified in retrieved text."**

            ## Prohibited
            - No summarization of any kind.
            - No paraphrasing instead of quoting.
            - No assumptions, no invented details.
            - No narrative paragraphs.

        """

        hyde_final_prompt = f"""
        {file_hyperlink_start}
        {combined_chunks}
        {file_hyperlink_end}

        ---
        Contract Type: {contract_type}

        ---
        ## Extraction Instructions
        - The response must be valid, user-friendly, and formatted in Markdown, fully compatible with react-markdown.
        - Begin the response with the file name as a Markdown hyperlink heading:  
        **[file_name](/preview?fileId=file_id)**
        - Extract all key details from the retrieved chunks using clause citations, exact clause text, and interpretations.
        - Do NOT summarize.
        - Do NOT add boilerplate, filler, or closing notes.
        ---

        """


        print(f"Length of final prompt: {len(hyde_final_prompt)} characters")

        # print(f"Final Summary: {summary}")

        initial_messages = [
            {"role": "system", "content": hyde_system_prompt},
            {"role": "user", "content": hyde_final_prompt}
        ]

        res = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=initial_messages,
            max_completion_tokens=5000
        )
        summary_results = res.choices[0].message.content
        return summary_results, f"[{cleaned_file_name}](/preview?fileId={file_id})"

    except Exception as e:
        print(f"Error in generate_compare_hyde_method: {e}")
        return "Unable to generate compare", f"[{cleaned_file_name}](/preview?fileId={file_id})"