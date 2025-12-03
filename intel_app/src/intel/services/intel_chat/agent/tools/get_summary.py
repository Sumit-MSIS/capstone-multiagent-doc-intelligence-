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


@mlflow.trace(name="Generate Summary for File")
async def generate_summary_for_file(user_query:str, user_id:int, org_id:int, file_id:str, tag_ids, file_info,  file_metadata, focus_list):
    try:
        # docs = ""
        page_count = await get_page_count(file_id)

        if page_count > 10:
            print(f"Generating summary using JSON method for file_id {file_id} with {page_count} pages.")
            return await generate_summary_json_method(user_query, user_id, org_id, file_id, tag_ids, file_info, file_metadata, focus_list)
        
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
                

        get_summary_user_prompt = f"""
            ---
            {file_hyperlink_start}
            {docs}
            {file_hyperlink_end}
            ---

            ## Summarization Instructions
            - Ensure the response is valid, user-friendly, and formatted in Markdown, fully compatible with the react-markdown library.
            - Begin the response with the file name as a Markdown hyperlink heading in the format:  
            **[file_name](/preview?fileId=file_id)**
            - End the summary naturally without adding any boilerplate or extra notes.

            ##User Query:
            {user_query}

        """

        get_summary_system_prompt = """
            You are a Legal AI assistant specializing in summarizing contracts and agreements with clarity, precision, and professionalism.

            Guidelines:
            - Generate a complete summary of the document, reflecting all content from start to end.
            - Each key point(or critical details) should be explained with enough context so the reader clearly understands its meaning and practical implication, without requiring further interpretation.
            - Ensure the summary includes all critical details such as:
                - Parties, important dates, duration of contract
                - Scope of services , obligations, any service standards, deliverables
                - Financial terms (payments, fees, penalties, schedule, payment structure, taxes, tax details, invoices, loan, other financial details), conditions related to payments, conditions for delayed payments, any interest or late fees
                - Roles, responsibilities of each party, any limits, any cooperation requirements, or specific duties
                - Intellectual property rights, confidentiality clauses, data protection, privacy details, methods for data protection, any licenses or usage rights
                - Liability, indemnification, warranties, disclaimers, risk allocations related details
                - Termination conditions , post-termination obligations, any renewal terms, conditions for termination or renewal, any penalties or fees related to termination or renewal
                - Governing law, dispute resolution, mechanism for resolving disputes, arbitration details, jurisdiction
                - Force majeure, amendment procedures, notice requirements, any clauses about changes to the agreement, how notices should be given
                - Any other significant clauses (e.g., subcontracting, non-compete, non-solicit, )
            - Maintain accuracy, neutrality, and professionalism (no assumptions or opinions).
            - If no specific format is requested, summarize in a clear, reader-friendly style using short paragraphs with bullet points where useful.
            - Avoid using exact heading or section titles from document. Avoid keywords like 'Article', 'Section', etc. as summary headings.
            - If the user requests a specific format (e.g., table, JSON, bullet points), strictly follow that format without omitting any critical details.
            - Do not add placeholders, filler commentary, or extra notes.
            - If the document is not a legal contract or agreement, provide a concise summary of the main points and key information in the document.

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
        return summary_results, file_info.get(file_id)
        
    except Exception as e:
        print(f"Error in generate_summary_for_file: {e}")
        return "Unable to generate summary", file_info.get(file_id)


@mlflow.trace(name="Generate Summary JSON Method")
async def generate_summary_json_method(user_query:str, user_id:int, org_id:int, file_id:str, tag_ids, file_info, file_metadata, focus_list):
    try:
        contract_type = file_metadata.get(file_id,{}).get("metadata",{}).get("Contract Type",None)
        data = read_summary_json()
        key_points = data.get(contract_type, [])

        if not key_points:
            print(f"No key points found for contract type: {contract_type}. Using default key points.")
            return await generate_summary_hyde_method(user_query, user_id, org_id, file_id, tag_ids, file_info, file_metadata, focus_list)
        
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

        get_summary_user_prompt = f"""
            ---
            {file_hyperlink_start}
            {combined_chunks}
            {file_hyperlink_end}
            ---

            ## Summarization Instructions
            - Ensure the response is valid, user-friendly, and formatted in Markdown, fully compatible with the react-markdown library.
            - Summarize one contract at a time.
            - Begin the response with the file name as a Markdown hyperlink heading in the format:  
            **[file_name](/preview?fileId=file_id)**
            - End the summary naturally without adding any boilerplate or extra notes.

            ##User Query:
            {user_query}

        """

        get_summary_system_prompt = f"""
            You are a Legal AI assistant specializing in summarizing {contract_type} contracts and agreements with clarity, precision, and professionalism.

            ### Key-points Section:
                {sections_str}

            ### Guidelines:

            1. **Introductory Paragraph**
            - Begin with a short 3-4 lines introduction in natural legal contract language.  
            - Include:
                * Effective Date
                * Parties involved (names/roles if available)
                * Purpose of the Agreement  
            - This paragraph should read like the opening of a legal contract.
            - Do not give this paragraph a heading or title.

            2. **Body of the Summary**
            - Structure the summary into **Sections** using the provided key-points (rephrase headings slightly for readability).  
            - Do not number or bullet-point the section headings—just use them as clear bolded titles.  
            - Within each section, present details in a combination of **short paragraphs and bullet-points**.  
            - Provide enough context so that each point is understandable on its own, without requiring further interpretation.
            - Clearly cite the relevant section or clause numbers whenever possible, so the user knows where the information comes from. 
            - If a section has no relevant information in the document, simply omit that section from the summary.
            - If the document contains important clauses or terms not covered by the predefined key-points, create a new section with an appropriate title reflecting the content of these uncovered details.

            3. **Ending**
            - End the summary in a natural flow after the last section.  
            - Do not add notes, opinions, or personal conclusions.  

            ### Special Formatting Rules:
            - If the user requests a **specific format (JSON, Table)** → produce the summary in that format, but keep the above structure intact.  
            - If the user requests a **concise format (e.g., 5 points, 7 points)** → compress the sections into the requested number of bullet-points while still covering all important details.  

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
        return summary_results, file_info.get(file_id)

    except Exception as e:
        print(f"Error in generate_summary_json_method: {e}")
        return "Unable to generate summary", file_info.get(file_id)



@mlflow.trace(name="Generate Summary Hyde Method")
async def generate_summary_hyde_method(user_query:str, user_id:int, org_id:int, file_id:str, tag_ids,file_info,  file_metadata, focus_list):
    try:
        contract_type = file_metadata.get(file_id,{}).get("metadata",{}).get("Contract Type",None)
        
        @mlflow.trace(name="Generate Summary HYDE - Generate Hypothetical Queries")
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
            You are a senior legal contract analyst and summarizer.
            Your task is to read the retrieved contract chunks and produce one unified, well-structured professional summary.
            
            Follow these rules:

            1. Write in clear, formal, lawyer-style English that is also readable for business users.
            2. The output should be a list or bullet format for each key-points not a continuous full summary.
            3. Cover all essential legal and commercial elements in logical flow:
            - Parties and contract type
            - Effective and expiration dates, duration, renewal
            - Scope and purpose
            - Financial details (value, payment terms, late fees, taxes)
            - Obligations and rights of each party
            - Confidentiality or IP clauses
            - Liability, indemnification, risk allocation
            - Termination and notice periods
            - Governing law and dispute resolution
            - Any critical red-flag clauses or potential risks

            4. Maintain a professional narrative flow — each clause or topic should naturally transition to the next.
            5. If any data is missing, write “not specified” instead of hallucinating.
            6. Aim for about 12–15 sentences (or 2–3 short paragraphs).

            Your goal is to produce a summary that feels like it was written by a contract manager for an executive briefing — concise, accurate, and insight-rich.
        """

        hyde_final_prompt = f"""
        {file_hyperlink_start}
        {combined_chunks}
        {file_hyperlink_end}

        ---
        Contract Type: {contract_type}

        ---
        ## Summarization Instructions
            - Ensure the response is valid, user-friendly, and formatted in Markdown, fully compatible with the react-markdown library.
            - Begin the response with the file name as a Markdown hyperlink heading in the format:  
            **[file_name](/preview?fileId=file_id)**
            - End the summary naturally without adding any boilerplate or extra notes.

        ---
        ## User Query: {user_query}
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
        return summary_results, file_info.get(file_id)

    except Exception as e:
        print(f"Error in generate_summary_hyde_method: {e}")
        return "Unable to generate summary", file_info.get(file_id)

