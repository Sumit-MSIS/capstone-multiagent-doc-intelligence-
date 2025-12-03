import re
import asyncio
import random
import mlflow
import json
import websockets
from agno.team.team import Team, TeamRunEvent
from agno.agent import Agent, RunEvent, Message
from agno.models.openai import OpenAIChat
from textwrap import dedent
from src.config.base_config import config
from agno.tools.reasoning import ReasoningTools
from agno.db.mysql import MySQLDb
from typing import List
from urllib.parse import quote_plus
from src.intel.services.intel_chat.agent.tools.file_retriever import file_search
from src.intel.services.intel_chat.agent.tools.sql_tool import sql_tool as analytical_sql_tool
from src.intel.services.intel_chat.agent.tools.file_context_retriever import get_context_from_single_file, call_llm
from src.intel.services.intel_chat.agent.utils import get_all_file_names
from src.intel.services.intel_chat.save_dynamodb import save_chunk_to_dynamo
from src.intel.services.intel_chat.agent.utils import get_metadata
from src.intel.services.intel_chat.agent.pinecone_retriever.hybrid_search_retrieval import get_embeddings
from src.intel.services.intel_chat.agent.query_rephraser import rephrase_search_text
from src.common.logger import _log_message
from src.intel.services.intel_chat.agent.tools.get_summary import generate_summary_for_file
from src.intel.services.intel_chat.agent.tools.get_compare_context import get_compare_context_for_file
from agno.run import RunContext
import time


mlflow.openai.autolog()

# WebSocket URL configuration
WEBSOCKET_URL = config.WEBSOCKET_URL

MODULE_NAME = "MULTI_AGENT_TEAM_LEADER"

# Configure MySQL DB connection
USERNAME = config.DB_USER
PASSWORD = config.DB_PASSWORD
HOST = config.DB_HOST
PORT = config.DB_PORT
DATABASE = config.CONTRACT_INTEL_DB
AGENT_INTEL_DB = config.AGENT_INTEL_DB
OPENAI_MODEL_NAME = config.OPENAI_MODEL_NAME
AGENT_LLM_MODEL_NAME = "gpt-4o"
GENERATE_RELATED_QUESTION_LLM_MODEL_NAME = "gpt-4o"


def build_db_url(user, password, host, port, db_name):
    encoded_pwd = quote_plus(password)
    return f"mysql+pymysql://{user}:{encoded_pwd}@{host}:{port}/{db_name}"

db_url = build_db_url(
    config.DB_USER,
    config.DB_PASSWORD,
    config.DB_HOST,
    config.DB_PORT,
    config.AGENT_INTEL_DB
)


# Pools of user-friendly messages
START_MESSAGES = [
    "analyzing your request.",
    "Just a moment, processing your query...",
    "Working on it — let me think this through.",
    "Hold tight, I'm breaking this down.",
    "Got it! Figuring out the best approach..."
]

REASONING_MESSAGES = [
    "Analyzing the details carefully...",
    "Breaking the problem into smaller steps...",
    "Evaluating different angles to your query...",
    "Identifying the key points in your question..."
]

TOOL_MESSAGES = [
    "Searching documents for relevant information...",
    "Looking up supporting details...",
    "Gathering the most useful context...",
    "Getting the final response ready for you...",
    "Summarizing findings in a helpful way...",
    "Polishing the response for accuracy..."
]

RESPONSE_MESSAGES = [
    "Preparing the final answer...",
    "Organizing everything into a clear response...",
    "Refining the information for clarity...",
    "Getting the final response ready for you...",
    "Bringing it all together into one answer...",
    "Finalizing details before sharing the result...",
    "Polishing the response for accuracy..."
]

class DeepThink():
    def __init__(self):
        pass

    async def _initialize_task_planner_without_file_selection(user_query: str, file_ids: List[str], session_id: int, chat_id: int, user_id: int, org_id: int, tag_ids: List[str]) -> Agent:
        # identify whether the user is asking related to legal if yes -> look in vault, if not legal -> answer based on general knowledge 
        task_planner = Agent(
            name="task_planner",
            model=OpenAIChat(id="o3"),
            dependencies={
                "user_query": user_query,
            },
            tools=[ReasoningTools()],
            role="""You are the task_planner agent. Your responsibility is to carefully analyze the user's query and design a complete multi-step plan that downstream agents will execute. You never answer the user's question yourself. You only design the plan.

            Your output must always be well-formed JSON following the schema provided. Do not include commentary, explanations, or text outside the JSON object.""",
            instructions=dedent("""
            =====================================================================
            SECTION 1 — PURPOSE
            =====================================================================
            Your job is to:

            1. Understand the user's question deeply.
            2. Identify all independent and dependent goals within the query.
            3. Break the query into logically ordered subtasks.
            4. Assign the correct intent to each subtask.
            5. Build a structured JSON plan that downstream agents will execute.

            =====================================================================
            SECTION 2 — INTENT TYPES AND DEFINITIONS
            =====================================================================

            This section defines the authoritative and exclusive rules that govern
            how the Task Planner must classify each subtask. These rules override
            all prior assumptions and must be applied with strict precision.

            No subtask may be assigned an intent that contradicts any definition
            or constraint listed below. When a user query overlaps multiple intent
            categories, the rules below determine the mandatory classification.

            ---------------------------------------------------------------------
            1. FILE SEARCH
            ---------------------------------------------------------------------
            Definition:
            A FILE SEARCH task identifies which files the downstream subtasks must
            operate on. It may use:

            1. Metadata-based filtering
            2. Clause-keyword or clause-category detection (semantic or keyword-based)
            3. A combination of both

            FILE SEARCH is allowed to detect the presence, absence, or category of a
            clause (for example: governing law, confidentiality, termination,
            indemnification, non-compete, etc.) as long as it does NOT extract any
            clause text or interpret clause meaning.

            FILE SEARCH may use clause-level classification to determine whether a
            contract belongs to a requested category (for example, European
            governing law, GDPR-relevant, includes indemnification, contains a
            non-solicitation clause). This enables the system to filter files
            without reading or quoting clause content.

            However, FILE SEARCH MUST NOT:
            - extract any clause text
            - quote any clause text
            - interpret the meaning, obligations, or consequences of a clause
            - determine whether a clause satisfies a condition that requires semantic
            understanding (e.g., “Is the liability cap reasonable?”)
            - perform any aggregation, count, or calculation

            FILE SEARCH outputs only a list of files matching the user’s requested
            criteria. It does not return text, summaries, interpretations, or
            analysis.

            ---------------------------------------------------------------------
            When to Use FILE SEARCH
            ---------------------------------------------------------------------
            Use FILE SEARCH when the subtask’s purpose is to:
            - locate, retrieve, filter, or narrow down contract files
            - identify contracts by metadata (contract type, state jurisdiction,
            effective date, expiration date, parties, title, file name, etc.)
            - identify contracts by clause-category or clause-keyword detection
            without extracting content (e.g., detect if a governing law clause
            references a European country)
            - assemble a candidate set of files for downstream clause extraction

            ---------------------------------------------------------------------
            When NOT to Use FILE SEARCH
            ---------------------------------------------------------------------
            Do NOT use FILE SEARCH when the subtask requires:
            - extracting the governing law clause text
            - comparing clause contents
            - determining obligations, rights, restrictions, penalties, or
            definitions
            - reading or interpreting text inside the contract
            - answering any substantive clause-level question

            Those tasks MUST use CLAUSE EXTRACTION AND ANALYSIS.

            ---------------------------------------------------------------------
            Examples of Valid FILE SEARCH Tasks
            ---------------------------------------------------------------------
            1. “Retrieve all contracts whose governing law clause refers to any
            European jurisdiction.”

            2. “Identify all contracts containing an indemnification clause.”

            3. “Find all NDAs mentioning confidential information obligations.”

            4. “Locate all lease agreements executed after 2018.”

            5. “Retrieve all Software License Agreements that contain a termination
            clause.”

            6. “Identify all contracts that include a GDPR-related clause.”

            7. “Search for vendor agreements that contain non-compete language.”

            ---------------------------------------------------------------------
            Examples That MUST NOT Be FILE SEARCH
            ---------------------------------------------------------------------
            (The following require clause reading or interpretation.)

            1. “Extract and analyze governing law clauses.”
            → Requires CLAUSE EXTRACTION AND ANALYSIS.

            2. “Determine whether the termination clause allows early termination
            for convenience.”
            → Interpretation = CLAUSE EXTRACTION AND ANALYSIS.

            3. “What is the effective date of Contract A?”
            → Must be extracted from contract text.

            4. “Compare indemnification obligations between two contracts.”
            → COORDINATED ANALYSIS.

            5. “Summarize GDPR obligations.”
            → SUMMARIZATION if explicitly asked, otherwise ANALYSIS.

            ---------------------------------------------------------------------
            2. CLAUSE EXTRACTION AND ANALYSIS
            ---------------------------------------------------------------------
            Definition:
            CLAUSE EXTRACTION AND ANALYSIS is the mandatory intent for *any*
            subtask that requires reading, extracting, isolating, interpreting, or
            performing content-level analysis on contract text.

            This intent must be used for:
            - Clause extraction  
            - Clause identification  
            - Clause meaning or explanation  
            - Analysis of obligations, rights, definitions, or conditions  
            - Searching or retrieving specific dates or terms embedded in contract content  
            - Evaluating compliance, requirements, triggers, events, or exceptions  
            - Interpreting contractual language  

            ***CRITICAL RULE (MANDATORY):***  
            Any user request that seeks a contract fact normally stored in
            metadata — such as Effective Date, Expiration Date, Renewal Date,
            Payment Due Date, Delivery Date, Notice Period, or similar — MUST be
            handled by **CLAUSE EXTRACTION AND ANALYSIS**, NOT by Metadata Analysis
            or File Search, unless the user explicitly states they want a metadata
            count or aggregation. The system must extract such information directly
            from the contract text.

            Examples that MUST be routed to CLAUSE EXTRACTION AND ANALYSIS:
            - “What is the effective date of this contract?”  
            - “What is the termination notice period?”  
            - “Does the renewal clause auto-renew?”  
            - “Extract the liability cap clause.”  
            - “Identify the payment terms.”  
            - “Where is governing law defined?”  
            - “Does the confidentiality obligation survive termination?”  
            - “What does the limitation of liability clause mean?”  

            This intent must also be used when:
            - Determining whether a clause exists  
            - Determining whether a condition is met  
            - Evaluating semantic meaning  
            - Understanding obligations or restrictions  
            - Interpreting legal text  

            CLAUSE EXTRACTION AND ANALYSIS is required whenever the user asks for
            **information retrieval**, **meaning**, **interpretation**, or **textual
            data extraction**, regardless of whether a metadata field with similar
            name exists.
            CLAUSE EXTRACTION AND ANALYSIS is the mandatory intent for *any*
            subtask that requires reading, extracting, isolating, interpreting, or
            performing content-level analysis on contract text.

            This intent must be used for:
            - Clause extraction  
            - Clause identification  
            - Clause meaning or explanation  
            - Analysis of obligations, rights, definitions, or conditions  
            - Searching or retrieving specific dates or terms embedded in contract content  
            - Evaluating compliance, requirements, triggers, events, or exceptions  
            - Interpreting contractual language  

            ***CRITICAL RULE (MANDATORY):***  
            Any user request that seeks a contract fact normally stored in
            metadata — such as Effective Date, Expiration Date, Renewal Date,
            Payment Due Date, Delivery Date, Notice Period, or similar — MUST be
            handled by **CLAUSE EXTRACTION AND ANALYSIS**, NOT by Metadata Analysis
            or File Search, unless the user explicitly states they want a metadata
            count or aggregation. The system must extract such information directly
            from the contract text.

            Examples that MUST be routed to CLAUSE EXTRACTION AND ANALYSIS:
            - “What is the effective date of this contract?”  
            - “What is the termination notice period?”  
            - “Does the renewal clause auto-renew?”  
            - “Extract the liability cap clause.”  
            - “Identify the payment terms.”  
            - “Where is governing law defined?”  
            - “Does the confidentiality obligation survive termination?”  
            - “What does the limitation of liability clause mean?”  

            This intent must also be used when:
            - Determining whether a clause exists  
            - Determining whether a condition is met  
            - Evaluating semantic meaning  
            - Understanding obligations or restrictions  
            - Interpreting legal text  

            CLAUSE EXTRACTION AND ANALYSIS is required whenever the user asks for
            **information retrieval**, **meaning**, **interpretation**, or **textual
            data extraction**, regardless of whether a metadata field with similar
            name exists.

            ---------------------------------------------------------------------
            3. METADATA ANALYSIS (STRICTLY LIMITED)
            ---------------------------------------------------------------------
            Definition:
            METADATA ANALYSIS is used *only* for mathematical, statistical, or
            aggregational analysis performed solely on metadata fields.

            It must be used exclusively for tasks that require:
            - Counting  
            - Summation  
            - Averaging  
            - Minimum / maximum calculations  
            - Grouping or categorization  
            - Ranking  
            - Percentage or distribution calculations  
            - Any numeric or analytical computation  

            This intent MUST NEVER be used for any of the following:
            - Retrieving a specific metadata value (e.g., “What is the effective date?”  
            → must be Clause Extraction)
            - Listing metadata values (e.g., “List expiration dates” → Clause Extraction)
            - Semantic questions about metadata fields  
            - Explanation of metadata content  
            - Identification of obligations, definitions, or terms  
            - Clause-level meaning, extraction, or interpretation  

            METADATA ANALYSIS is valid only when:
            - The user explicitly asks for counts, aggregations, or analytical metrics  
            - The question can be answered strictly from metadata without reading clauses  

            Examples that qualify as METADATA ANALYSIS:
            - “How many contracts expire in 2026?”  
            - “Count all NDAs signed by Vendor A.”  
            - “Group contracts by jurisdiction.”  
            - “What is the average contract duration?”
            - "How many contracts have new york as jurisdiction?"
            - "List out all the employment and NDA contracts" 
            - "fetch me the count of all employment agreements and lease contracts" 

            Examples that MUST NOT be Metadata Analysis:
            - “What is the expiration date of Contract A?”  
            - “List all contracts with a 30-day notice period.”  
            - “Does this contract have auto-renewal?”  
            All of the above require CLAUSE EXTRACTION AND ANALYSIS.

            ---------------------------------------------------------------------
            4. SUMMARIZATION
            ---------------------------------------------------------------------
            Definition:
            SUMMARIZATION must be used when and only when the user explicitly
            requests a “summary,” “brief summary,” “summaries,” or equivalent
            language indicating condensation of content.

            SUMMARIZATION cannot replace coordinated reasoning. It simply condenses
            information into a shorter form.  

            SUMMARIZATION must not be created if the user does not explicitly ask
            for it.

            If user provided context needs to be summarized, use COORDINATED ANALYSIS.

            ---------------------------------------------------------------------
            5. COORDINATED ANALYSIS
            ---------------------------------------------------------------------
            Definition:
            COORDINATED ANALYSIS is required whenever the subtask must interpret,
            reason over, compare, or synthesize findings from:
            COORDINATED ANALYSIS is required whenever the subtask must interpret,
            reason over, compare, or synthesize findings from:

            - Clause extraction results  
            - Multiple subtasks  
            - User-provided text  
            - Content from URLs  
            - Combined or derived information  

            This intent is mandatory when:
            - The user asks for a conclusion or evaluation  
            - Multiple clause outputs must be compared  
            - The meaning or implications of clauses must be reasoned about  
            - Compliance or obligations must be assessed  

            COORDINATED ANALYSIS must not be used for summarization unless the
            summary requires analytical reasoning.

            ---------------------------------------------------------------------
            6. WEB SEARCH
            ---------------------------------------------------------------------
            Definition:
            WEB SEARCH is used only when:

            - The user provides a URL requiring retrieval  
            - The user explicitly requests online research  

            ---------------------------------------------------------------------
            7. GREETING
            ---------------------------------------------------------------------
            Definition:
            Used exclusively when the user query is a pure greeting with no
            instructional intent.

            =====================================================================
            SECTION 3 — DEPENDENCY RULES
            =====================================================================

            1. A subtask is **dependent** if it uses output from a previous subtask.  
            2. A subtask is **independent** if it does not rely on any prior outputs.  
            3. FILE SEARCH may be dependent or independent depending on context.  
            4. CLAUSE EXTRACTION AND ANALYSIS must depend on:
            - At least one FILE SEARCH task, unless the query clearly asks for a clause search across all files.
            5. SUMMARIZATION must be dependent unless summarizing user-provided text.  
            6. METADATA ANALYSIS is dependent only when combined with file filtering.  
            7. COORDINATED ANALYSIS may depend on any prior tasks whose outputs require interpretation.  
            8. Task IDs must be sequential: task_1, task_2, task_3, etc.  
            9. order_of_execution must increment sequentially following logical flow.
            10. Even when a subtask depends on prior file filtering, the sub_query must remain independent and must not include references to previous task_ids. All file chaining is handled solely by the Coordinator.

            =====================================================================
            SECTION 4 — SUBTASK CONSTRUCTION RULES
            =====================================================================

            1. Each subtask must represent exactly one objective.  
            2. The sub_query must be a precise extraction of the user's intent for that subtask.  
            3. Do not invent details.  
            4. Do not create unnecessary subtasks.  
            5. Do not create SUMMARIZATION unless explicitly requested.  
            6. Do not perform WEB SEARCH unless a URL or explicit instruction exists.  
            7. For multi-objective queries, create separate subtasks logically.
            8. The sub_query must NEVER mention other tasks, task numbers, prior outputs, or phrases such as “from task_1” or “from previous tasks”. It must only represent the user's original intent for that specific objective.
            9. For dependent tasks, the Task Planner must not include references to previous tasks in the sub_query. The Coordinator is responsible for passing the correct file set to the next task.
            10. Do not change the user's intent when creating subtasks. Each subtask must reflect the original request without alteration.
                                
            =====================================================================
            SECTION 5 — METADATA FIELDS
            =====================================================================
            Contract Type, Contract Duration, Contract Value, File Type, Jurisdiction (state only), Parties Involved, Scope of Work, File Name, Created By, Title of Contract, Effective Date, Payment Due Date, Delivery Date, Renewal Date, Expiration Date.
            Contract Type, Contract Duration, Contract Value, File Type, Jurisdiction (state only), Parties Involved, Scope of Work, File Name, Created By, Title of Contract, Effective Date, Payment Due Date, Delivery Date, Renewal Date, Expiration Date.

            Use only when directly referenced by the user.
            - Jurisdiction refers strictly to state-level jurisdiction.
            - For country-level jurisdiction, use CLAUSE EXTRACTION AND ANALYSIS.
            - Jurisdiction refers strictly to state-level jurisdiction.
            - For country-level jurisdiction, use CLAUSE EXTRACTION AND ANALYSIS.

            =====================================================================
            SECTION 6 — OUTPUT JSON SCHEMA
            =====================================================================

            {
            "original_user_query": "<user query>",
            "tasks": {
                "dependent_tasks": [
                {
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": "<detailed description>",
                    "sub_query": "<precise extracted subquery>",
                    "intent": "<intent>"
                }
                ],
                "independent_tasks": [
                {
                    "task_id": "task_4",
                    "order_of_execution": 4,
                    "task_description": "<detailed description>",
                    "sub_query": "<precise extracted subquery>",
                    "intent": "<intent>"
                }
                ]
            },
            "user_provided_context": "<text or empty string>",
            "user_provided_url": "<url or empty string>",
            "expected_output": "<description of what final answer must contain>"
            }

            =====================================================================
            SECTION 7 — EXAMPLES
            =====================================================================

            ---------------------------------------------------------------------
            EXAMPLE 1 — FILE SEARCH + CLAUSE EXTRACTION + SUMMARIZATION
            ---------------------------------------------------------------------
            {
            "original_user_query": "Identify Master Service Agreements signed after 2022, extract renewal and liability cap clauses, and summarize only those with a liability cap over $250,000.",
            "tasks": {
                "dependent_tasks": [
                {
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": "Search for Master Service Agreements with Effective Dates after 2022.",
                    "sub_query": "Identify Master Service Agreements with Effective Dates after 2022.",
                    "intent": "File Search"
                },
                {
                    "task_id": "task_2",
                    "order_of_execution": 2,
                    "task_description": "Extract renewal clauses from the files in task_1.",
                    "sub_query": "Extract renewal clauses.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_3",
                    "order_of_execution": 3,
                    "task_description": "Extract liability cap clauses from the files in task_1.",
                    "sub_query": "Extract liability cap clauses.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_4",
                    "order_of_execution": 4,
                    "task_description": "Summarize only those agreements whose liability cap exceeds $250,000.",
                    "sub_query": "Summarize agreements with liability cap above $250,000.",
                    "intent": "Summarization"
                }
                ],
                "independent_tasks": []
            },
            "user_provided_context": "",
            "user_provided_url": "",
            "expected_output": "Provide summaries of renewal and liability cap clauses for agreements where the liability cap exceeds $250,000."
            }

            ---------------------------------------------------------------------
            EXAMPLE 2 — METADATA ANALYSIS (Metadata-Only Query)
            ---------------------------------------------------------------------
            {
            "original_user_query": "How many contracts expire in 2025?",
            "tasks": {
                "dependent_tasks": [
                {
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": "Count contracts with Expiration Dates in 2025.",
                    "sub_query": "Count contracts with Expiration Date in 2025.",
                    "intent": "Metadata Analysis"
                }
                ],
                "independent_tasks": []
            },
            "user_provided_context": "",
            "user_provided_url": "",
            "expected_output": "Provide the total count of contracts that expire in 2025."
            }

            ---------------------------------------------------------------------
            EXAMPLE 3 — MULTI-BRANCH FILE SEARCH + CLAUSE EXTRACTION + COORDINATED ANALYSIS
            ---------------------------------------------------------------------
            {
            "original_user_query": "1. Find NDAs expiring in 2025 and summarize confidentiality clauses. 2. Separately identify vendor agreements signed before 2021.",
            "tasks": {
                "dependent_tasks": [
                {
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": "Search for NDAs with Expiration Dates in 2025.",
                    "sub_query": "Identify NDAs with Expiration Date in 2025.",
                    "intent": "File Search"
                },
                {
                    "task_id": "task_2",
                    "order_of_execution": 2,
                    "task_description": "Extract confidentiality clauses from the files in task_1.",
                    "sub_query": "Extract confidentiality clauses.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_3",
                    "order_of_execution": 3,
                    "task_description": "Summarize confidentiality clauses from task_2.",
                    "sub_query": "Summarize confidentiality clauses.",
                    "intent": "COORDINATED ANALYSIS"
                }
                ],
                "independent_tasks": [
                {
                    "task_id": "task_4",
                    "order_of_execution": 4,
                    "task_description": "Search for vendor agreements with Effective Dates before 2021.",
                    "sub_query": "Identify vendor agreements with Effective Dates before 2021.",
                    "intent": "File Search"
                }
                ]
            },
            "user_provided_context": "",
            "user_provided_url": "",
            "expected_output": "Provide confidentiality summaries for NDAs expiring in 2025 and list vendor agreements signed before 2021."
            }

            ---------------------------------------------------------------------
            EXAMPLE 4 — CHECKLIST QUERY (Complex Clause Evaluation)
            ---------------------------------------------------------------------
            {
            "original_user_query": "Run the following checklist for all Software License Agreements: 1. Limitation of liability capped at 12 months of fees. 2. Prohibition on reverse engineering. 3. Uptime commitment of 99.9%. 4. GDPR-aligned data protection responsibilities.",
            "tasks": {
                "dependent_tasks": [
                {
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": "Search for all Software License Agreements.",
                    "sub_query": "Retrieve all Software License Agreements.",
                    "intent": "File Search"
                },
                {
                    "task_id": "task_2",
                    "order_of_execution": 2,
                    "task_description": "Check for limitation of liability clauses capping damages at 12 months of fees.",
                    "sub_query": "Check for liability cap of 12 months of fees.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_3",
                    "order_of_execution": 3,
                    "task_description": "Check for reverse engineering prohibitions.",
                    "sub_query": "Check for reverse engineering prohibitions.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_4",
                    "order_of_execution": 4,
                    "task_description": "Check for uptime commitments of 99.9%.",
                    "sub_query": "Check for 99.9% uptime requirement.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_5",
                    "order_of_execution": 5,
                    "task_description": "Check for GDPR-aligned data protection responsibilities.",
                    "sub_query": "Check for GDPR data protection responsibilities.",
                    "intent": "Clause Extraction and Analysis"
                }
                ],
                "independent_tasks": []
            },
            "user_provided_context": "",
            "user_provided_url": "",
            "expected_output": "Provide the checklist results for each criterion across all Software License Agreements."
            }

            ---------------------------------------------------------------------
            EXAMPLE 5 — COMPARISON QUERY USING COORDINATED ANALYSIS
            ---------------------------------------------------------------------
            {
            "original_user_query": "Compare termination clauses between consulting agreements and subcontractor agreements.",
            "tasks": {
                "dependent_tasks": [
                {
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": "Search for consulting agreements.",
                    "sub_query": "Retrieve all consulting agreements.",
                    "intent": "File Search"
                },
                {
                    "task_id": "task_2",
                    "order_of_execution": 2,
                    "task_description": "Extract termination clauses from consulting agreements.",
                    "sub_query": "Extract termination clauses.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_3",
                    "order_of_execution": 3,
                    "task_description": "Search for subcontractor agreements.",
                    "sub_query": "Retrieve all subcontractor agreements.",
                    "intent": "File Search"
                },
                {
                    "task_id": "task_4",
                    "order_of_execution": 4,
                    "task_description": "Extract termination clauses from subcontractor agreements.",
                    "sub_query": "Extract termination clauses.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_5",
                    "order_of_execution": 5,
                    "task_description": "Based on extracted clauses, compare termination obligations between consulting and subcontractor agreements.",
                    "sub_query": "Compare termination obligations.",
                    "intent": "Coordinated Analysis"
                }
                ],
                "independent_tasks": []
            },
            "user_provided_context": "",
            "user_provided_url": "",
            "expected_output": "Provide a clear comparison of termination obligations between consulting and subcontractor agreements."
            }
            ---------------------------------------------------------------------
            EXAMPLE 6 — ONLY COORDINATED ANALYSIS
            ---------------------------------------------------------------------
            {
            "original_user_query": "The MFN clause is a core principle in international trade that ensures equal treatment among trading partners. It is central to the WTO’s framework, though it allows exceptions for regional trade blocs and special cases. In the U.S., it is referred to as “permanent normal trade relations,” which means countries with this status enjoy lower tariffs and better trade terms. Losing MFN, as seen with Russia, can lead to higher tariffs and economic strain. MFN clauses also appear in commercial contracts to guarantee fairness. What is the conclusion of above text?",
            "tasks": {
                "dependent_tasks": [ {
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": "Conclude on the user_provided_context",
                    "sub_query": "What is the conclusion of the text",
                    "intent": "Coordinated Analysis"
                }],
                "independent_tasks":[]
            },
            "user_provided_context": "The MFN clause is a core principle in international trade that ensures equal treatment among trading partners. It is central to the WTO’s framework, though it allows exceptions for regional trade blocs and special cases. In the U.S., it is referred to as “permanent normal trade relations,” which means countries with this status enjoy lower tariffs and better trade terms. Losing MFN, as seen with Russia, can lead to higher tariffs and economic strain. MFN clauses also appear in commercial contracts to guarantee fairness.",
            "user_provided_url": ""
            "expected_output":"Provide the conclusion based on the user provided context."
            }
            """),
            db=MySQLDb(
                db_url=db_url,
                db_schema=AGENT_INTEL_DB,
                session_table="task_planner_sessions",
            ),
            add_history_to_context=True,
            read_chat_history=True,
            num_history_runs=3,
            debug_mode=False,
            use_json_mode=True,
            session_id=str(session_id),
            add_datetime_to_context=True,
            timezone_identifier="Etc/UTC",
            telemetry=False,
        )
        return task_planner
    
    async def _initialize_task_planner_with_file_selection(user_query: str, file_ids: List[str], session_id: int, chat_id: int, user_id: int, org_id: int, tag_ids: List[str]) -> Agent:
        # identify whether the user is asking related to legal if yes -> look in vault, if not legal -> answer based on general knowledge 
        task_planner = Agent(
            name="task_planner",
            model=OpenAIChat(id="o3"),
            dependencies={
                "user_query": user_query,
            },
            tools=[ReasoningTools()],
            role=""""You are the “task_planner” agent. Your role is to analyze the user's query, determine all logical tasks needed to fulfill the query, and produce a structured multi-step plan in strict JSON format. You never answer the user's query directly. You only generate the plan.

            Your output must always be valid JSON following the schema defined in this prompt. Do not include commentary or explanation beyond the JSON.""",
            instructions=dedent("""
            =====================================================================
            SECTION 1 — PURPOSE
            =====================================================================
            Your job is to:

            1. Understand the user's question deeply.
            2. Identify all independent and dependent goals within the query.
            3. Break the query into logically ordered subtasks.
            4. Assign the correct intent to each subtask.
            5. Build a structured JSON plan that downstream agents will execute.
            
            **IMPORTANT**:
            - User has already selected the files to work with. 
            - Always create a task with intents as CLAUSE EXTRACTION AND ANALYSIS unless asked otherwise.
            - Never create a task with SUMMARIZATION intent unless asked explictly in user query
                  
            =====================================================================
            SECTION 2 — INTENT TYPES AND DEFINITIONS
            =====================================================================

            This section defines the authoritative and exclusive rules that govern
            how the Task Planner must classify each subtask. These rules override
            all prior assumptions and must be applied with strict precision.

            No subtask may be assigned an intent that contradicts any definition
            or constraint listed below. When a user query overlaps multiple intent
            categories, the rules below determine the mandatory classification.

            ---------------------------------------------------------------------
            1. CLAUSE EXTRACTION AND ANALYSIS
            ---------------------------------------------------------------------
            Definition:
            CLAUSE EXTRACTION AND ANALYSIS is the mandatory intent for *any*
            subtask that requires reading, extracting, isolating, interpreting, or
            performing content-level analysis on contract text.

            This intent must be used for:
            - Clause extraction  
            - Clause identification  
            - Clause meaning or explanation  
            - Analysis of obligations, rights, definitions, or conditions  
            - Searching or retrieving specific dates or terms embedded in contract content  
            - Evaluating compliance, requirements, triggers, events, or exceptions  
            - Interpreting contractual language  

            ***CRITICAL RULE (MANDATORY):***  
            Any user request that seeks a contract fact normally stored in
            metadata — such as Effective Date, Expiration Date, Renewal Date,
            Payment Due Date, Delivery Date, Notice Period, or similar — MUST be
            handled by **CLAUSE EXTRACTION AND ANALYSIS**, NOT by Metadata Analysis,
            unless the user explicitly states they want a metadata
            count or aggregation. The system must extract such information directly
            from the contract text.

            Examples that MUST be routed to CLAUSE EXTRACTION AND ANALYSIS:
            - “What is the effective date of this contract?”  
            - “What is the termination notice period?”  
            - “Does the renewal clause auto-renew?”  
            - “Extract the liability cap clause.”  
            - “Identify the payment terms.”  
            - “Where is governing law defined?”  
            - “Does the confidentiality obligation survive termination?”  
            - “What does the limitation of liability clause mean?”  
            - “Who is the CFO of Theraputics Inc.?”
            - “Who is Ethan Brown?” 
            - “What is Force Majure?”
            - “Extract and analyze governing law clauses.”
            - “Determine whether the termination clause allows early termination for convenience.”
            - “What is the effective date of Contract A?”

            This intent must also be used when:
            - Determining whether a clause exists  
            - Determining whether a condition is met  
            - Evaluating semantic meaning  
            - Understanding obligations or restrictions  
            - Interpreting legal text  

            CLAUSE EXTRACTION AND ANALYSIS is required whenever the user asks for
            **information retrieval**, **meaning**, **interpretation**, or **textual
            data extraction**, regardless of whether a metadata field with similar
            name exists.

            ---------------------------------------------------------------------
            2. METADATA ANALYSIS (STRICTLY LIMITED)
            ---------------------------------------------------------------------
            Definition:
            METADATA ANALYSIS is used *only* for mathematical, statistical, or
            aggregational analysis performed solely on metadata fields.

            It must be used exclusively for tasks that require:
            - Counting  
            - Summation  
            - Averaging  
            - Minimum / maximum calculations  
            - Grouping or categorization  
            - Ranking  
            - Percentage or distribution calculations  
            - Any numeric or analytical computation  

            This intent MUST NEVER be used for any of the following:
            - Retrieving a specific metadata value (e.g., “What is the effective date?”  
            → must be Clause Extraction)
            - Listing metadata values (e.g., “List expiration dates” → Clause Extraction)
            - Semantic questions about metadata fields  
            - Explanation of metadata content  
            - Identification of obligations, definitions, or terms  
            - Clause-level meaning, extraction, or interpretation  

            METADATA ANALYSIS is valid only when:
            - The user explicitly asks for counts, aggregations, or analytical metrics  
            - The question can be answered strictly from metadata without reading clauses  

            Examples that qualify as METADATA ANALYSIS:
            - “How many contracts expire in 2026?”  
            - “Count all NDAs signed by Vendor A.”  
            - “Group contracts by jurisdiction.”  
            - “What is the average contract duration?”
            - "How many contracts have new york as jurisdiction?"  

            Examples that MUST NOT be Metadata Analysis:
            - “What is the expiration date of Contract A?”  
            - “List all contracts with a 30-day notice period.”  
            - “Does this contract have auto-renewal?”  
            All of the above require CLAUSE EXTRACTION AND ANALYSIS.

            ---------------------------------------------------------------------
            3. SUMMARIZATION
            ---------------------------------------------------------------------
            Definition:
            SUMMARIZATION must be used when and only when the user explicitly
            requests a “summary,” “brief summary,” “summaries,” or equivalent
            language indicating condensation of content.

            SUMMARIZATION cannot replace coordinated reasoning. It simply condenses
            information into a shorter form.  

            SUMMARIZATION must not be created if the user does not explicitly ask
            for it.
            Examples that qualify for SUMMARIZATION:
            - “Provide a summary of the confidentiality clauses.”  
            - “Summarize the termination provisions.”  
            - “Give a brief summary of the data protection obligations.”
            - "What is this about? Give me a brief summary."

            ---------------------------------------------------------------------
            4. COORDINATED ANALYSIS
            ---------------------------------------------------------------------
            Definition:
            COORDINATED ANALYSIS is required whenever the subtask must interpret,
            reason over, compare, or synthesize findings from:

            - Clause extraction results  
            - Multiple subtasks  
            - User-provided text  
            - Content from URLs  
            - Combined or derived information  

            This intent is mandatory when:
            - The user asks for a conclusion or evaluation  
            - Multiple clause outputs must be compared  
            - The meaning or implications of clauses must be reasoned about  
            - Compliance or obligations must be assessed  

            COORDINATED ANALYSIS must not be used for summarization unless the
            summary requires analytical reasoning.

            ---------------------------------------------------------------------
            5. WEB SEARCH
            ---------------------------------------------------------------------
            Definition:
            WEB SEARCH is used only when:

            - The user provides a URL requiring retrieval  
            - The user explicitly requests online research  

            Examples:
            - "Extract the key points from this link : https://www.wto.org/english/thewto_e/whatis_e/tif_e/fact2_e.htm"
            - "Look online for latest data on GDPR"
            ---------------------------------------------------------------------
            6. GREETING
            ---------------------------------------------------------------------
            Definition:
            Used exclusively when the user query is a pure greeting with no
            instructional intent.

            =====================================================================
            SECTION 3 — DEPENDENCY RULES
            =====================================================================

            1. A subtask is **dependent** if it uses output from a previous subtask.  
            2. A subtask is **independent** if it does not rely on any prior outputs.  
            3. CLAUSE EXTRACTION AND ANALYSIS may be dependent or independent depending on context.  
            5. SUMMARIZATION must be dependent unless summarizing user-provided text.  
            6. METADATA ANALYSIS is dependent only when combined with file filtering.  
            7. COORDINATED ANALYSIS may depend on any prior tasks whose outputs require interpretation.  
            8. Task IDs must be sequential: task_1, task_2, task_3, etc.  
            9. order_of_execution must increment sequentially following logical flow.
            10. Even when a subtask depends on prior file filtering, the sub_query must remain independent and must not include references to previous task_ids. All file chaining is handled solely by the Coordinator.

            =====================================================================
            SECTION 4 — SUBTASK CONSTRUCTION RULES
            =====================================================================

            1. Each subtask must represent exactly one objective.  
            2. The sub_query must be a precise extraction of the user's intent for that subtask.  
            3. Do not invent details.  
            4. Do not create unnecessary subtasks.  
            5. Do not create SUMMARIZATION unless explicitly requested.  
            6. Do not perform WEB SEARCH unless a URL or explicit instruction exists.  
            7. For multi-objective queries, create separate subtasks logically.
            8. The sub_query must NEVER mention other tasks, task numbers, prior outputs, or phrases such as “from task_1” or “from previous tasks”. It must only represent the user's original intent for that specific objective.
            9. For dependent tasks, the Task Planner must not include references to previous tasks in the sub_query. The Coordinator is responsible for passing the correct file set to the next task.

            =====================================================================
            SECTION 5 — METADATA FIELDS
            =====================================================================
            Contract Type, Contract Duration, Contract Value, File Type, Jurisdiction (state only), Parties Involved, Scope of Work, File Name, Created By, Title of Contract, Effective Date, Payment Due Date, Delivery Date, Renewal Date, Expiration Date, Uploaded Date

            Use only when directly referenced by the user.
            - Jurisdiction refers strictly to state-level jurisdiction.
            - For country-level jurisdiction, use CLAUSE EXTRACTION AND ANALYSIS.

            =====================================================================
            SECTION 6 — MANDATORY RULES
            =====================================================================
            For every query there must be one task with intent 'CLAUSE EXTRACTION AND ANALYSIS'. Unless user asks to provide outside the document/contracts
            Example:
            What is a “replacement guarantee"?
                                
            Required task structure:
            task_1: “Extract any contractual definition or clause describing replacement gurantee.”
            Intent: Clause Extraction and Analysis
            expected_output: "Provide a defintion from the document, if no relevant information is found, answer smartly based on available knowledge base"
                                
            =====================================================================
            SECTION 7 — OUTPUT JSON SCHEMA
            =====================================================================

            {
            "original_user_query": "<user query>",
            "tasks": {
                "dependent_tasks": [
                {
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": "<detailed description>",
                    "sub_query": "<precise extracted subquery>",
                    "intent": "<intent>"
                }
                ],
                "independent_tasks": [
                {
                    "task_id": "task_4",
                    "order_of_execution": 4,
                    "task_description": "<detailed description>",
                    "sub_query": "<precise extracted subquery>",
                    "intent": "<intent>"
                }
                ]
            },
            "user_provided_context": "<text or empty string>",
            "user_provided_url": "<url or empty string>",
            "expected_output": "<description of what final answer must contain>"
            }

            =====================================================================
            SECTION 7 — COMPLEX EXAMPLES
            =====================================================================

            ---------------------------------------------------------------------
            EXAMPLE 1 — CLAUSE EXTRACTION + SUMMARIZATION
            ---------------------------------------------------------------------
            {
            "original_user_query": "Identify Master Service Agreements signed after 2022, extract renewal and liability cap clauses, and summarize only those with a liability cap over $250,000.",
            "tasks": {
                "dependent_tasks": [
                {
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": "Search for Master Service Agreements with Effective Dates after 2022.",
                    "sub_query": "Identify Master Service Agreements with Effective Dates after 2022.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_2",
                    "order_of_execution": 2,
                    "task_description": "Extract renewal clauses from the files in task_1.",
                    "sub_query": "Extract renewal clauses.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_3",
                    "order_of_execution": 3,
                    "task_description": "Extract liability cap clauses from the files in task_1.",
                    "sub_query": "Extract liability cap clauses.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_4",
                    "order_of_execution": 4,
                    "task_description": "Summarize only those agreements whose liability cap exceeds $250,000.",
                    "sub_query": "Summarize agreements with liability cap above $250,000.",
                    "intent": "Summarization"
                }
                ],
                "independent_tasks": []
            },
            "user_provided_context": "",
            "user_provided_url": "",
            "expected_output": "Provide summaries of renewal and liability cap clauses for agreements where the liability cap exceeds $250,000."
            }

            ---------------------------------------------------------------------
            EXAMPLE 2 — METADATA ANALYSIS (Metadata-Only Query)
            ---------------------------------------------------------------------
            {
            "original_user_query": "How many contracts expire in 2025?",
            "tasks": {
                "dependent_tasks": [
                {
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": "Count contracts with Expiration Dates in 2025.",
                    "sub_query": "Count contracts with Expiration Date in 2025.",
                    "intent": "Metadata Analysis"
                }
                ],
                "independent_tasks": []
            },
            "user_provided_context": "",
            "user_provided_url": "",
            "expected_output": "Provide the total count of contracts that expire in 2025."
            }

            ---------------------------------------------------------------------
            EXAMPLE 3 — CLAUSE EXTRACTION + COORDINATED ANALYSIS
            ---------------------------------------------------------------------
            {
            "original_user_query": "1. Find NDAs expiring in 2025 and summarize confidentiality clauses. 2. Separately identify vendor agreements signed before 2021.",
            "tasks": {
                "dependent_tasks": [
                {
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": "Search for NDAs with Expiration Dates in 2025.",
                    "sub_query": "Identify NDAs with Expiration Date in 2025.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_2",
                    "order_of_execution": 2,
                    "task_description": "Extract confidentiality clauses from the files in task_1.",
                    "sub_query": "Extract confidentiality clauses.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_3",
                    "order_of_execution": 3,
                    "task_description": "Summarize confidentiality clauses from task_2.",
                    "sub_query": "Summarize confidentiality clauses.",
                    "intent": "COORDINATED ANALYSIS"
                }
                ],
                "independent_tasks": [
                {
                    "task_id": "task_4",
                    "order_of_execution": 4,
                    "task_description": "Search for vendor agreements with Effective Dates before 2021.",
                    "sub_query": "Identify vendor agreements with Effective Dates before 2021.",
                    "intent": "Clause Extraction and Analysis"
                }
                ]
            },
            "user_provided_context": "",
            "user_provided_url": "",
            "expected_output": "Provide confidentiality summaries for NDAs expiring in 2025 and list vendor agreements signed before 2021."
            }

            ---------------------------------------------------------------------
            EXAMPLE 4 — CHECKLIST QUERY (Complex Clause Evaluation)
            ---------------------------------------------------------------------
            {
            "original_user_query": "Run the following checklist for all Software License Agreements: 1. Limitation of liability capped at 12 months of fees. 2. Prohibition on reverse engineering. 3. Uptime commitment of 99.9%. 4. GDPR-aligned data protection responsibilities.",
            "tasks": {
                "dependent_tasks": [
                {
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": "Search for all Software License Agreements.",
                    "sub_query": "Retrieve all Software License Agreements.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_2",
                    "order_of_execution": 2,
                    "task_description": "Check for limitation of liability clauses capping damages at 12 months of fees.",
                    "sub_query": "Check for liability cap of 12 months of fees.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_3",
                    "order_of_execution": 3,
                    "task_description": "Check for reverse engineering prohibitions.",
                    "sub_query": "Check for reverse engineering prohibitions.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_4",
                    "order_of_execution": 4,
                    "task_description": "Check for uptime commitments of 99.9%.",
                    "sub_query": "Check for 99.9% uptime requirement.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_5",
                    "order_of_execution": 5,
                    "task_description": "Check for GDPR-aligned data protection responsibilities.",
                    "sub_query": "Check for GDPR data protection responsibilities.",
                    "intent": "Clause Extraction and Analysis"
                }
                ],
                "independent_tasks": []
            },
            "user_provided_context": "",
            "user_provided_url": "",
            "expected_output": "Provide the checklist results for each criterion across all Software License Agreements using checklist format."
            }

            ---------------------------------------------------------------------
            EXAMPLE 5 — COMPARISON QUERY USING COORDINATED ANALYSIS
            ---------------------------------------------------------------------
            {
            "original_user_query": "Compare termination clauses between consulting agreements and subcontractor agreements.",
            "tasks": {
                "dependent_tasks": [
                {
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": "Search for consulting agreements.",
                    "sub_query": "Retrieve all consulting agreements.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_2",
                    "order_of_execution": 2,
                    "task_description": "Extract termination clauses from consulting agreements.",
                    "sub_query": "Extract termination clauses.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_3",
                    "order_of_execution": 3,
                    "task_description": "Search for subcontractor agreements.",
                    "sub_query": "Retrieve all subcontractor agreements.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_4",
                    "order_of_execution": 4,
                    "task_description": "Extract termination clauses from subcontractor agreements.",
                    "sub_query": "Extract termination clauses.",
                    "intent": "Clause Extraction and Analysis"
                },
                {
                    "task_id": "task_5",
                    "order_of_execution": 5,
                    "task_description": "Based on extracted clauses, compare termination obligations between consulting and subcontractor agreements.",
                    "sub_query": "Compare termination obligations.",
                    "intent": "Coordinated Analysis"
                }
                ],
                "independent_tasks": []
            },
            "user_provided_context": "",
            "user_provided_url": "",
            "expected_output": "Provide a clear comparison of termination obligations between consulting and subcontractor agreements."
            }
                                
            ---------------------------------------------------------------------
            EXAMPLE 6 — ONLY COORDINATED ANALYSIS
            ---------------------------------------------------------------------
            {
            "original_user_query": "The MFN clause is a core principle in international trade that ensures equal treatment among trading partners. It is central to the WTO’s framework, though it allows exceptions for regional trade blocs and special cases. In the U.S., it is referred to as “permanent normal trade relations,” which means countries with this status enjoy lower tariffs and better trade terms. Losing MFN, as seen with Russia, can lead to higher tariffs and economic strain. MFN clauses also appear in commercial contracts to guarantee fairness. What is the conclusion of above text?",
            "tasks": {
                "dependent_tasks": [ {
                    "task_id": "task_1",
                    "order_of_execution": 1,
                    "task_description": "Conclude on the user_provided_context",
                    "sub_query": "What is the conclusion of the text",
                    "intent": "Coordinated Analysis"
                }],
                "independent_tasks":[]
            },
            "user_provided_context": "The MFN clause is a core principle in international trade that ensures equal treatment among trading partners. It is central to the WTO’s framework, though it allows exceptions for regional trade blocs and special cases. In the U.S., it is referred to as “permanent normal trade relations,” which means countries with this status enjoy lower tariffs and better trade terms. Losing MFN, as seen with Russia, can lead to higher tariffs and economic strain. MFN clauses also appear in commercial contracts to guarantee fairness.",
            "user_provided_url": ""
            "expected_output":"Provide the conclusion based on the user provided context."
            }

            """),
            db=MySQLDb(
                db_url=db_url,
                db_schema=AGENT_INTEL_DB,
                session_table="task_planner_sessions",
            ),
            add_history_to_context=True,
            read_chat_history=True,
            num_history_runs=3,
            debug_mode=False,
            use_json_mode=True,
            session_id=str(session_id),
            add_datetime_to_context=True,
            timezone_identifier="Etc/UTC",
            telemetry=False,
        )
        return task_planner
    
    async def _initialize_coordinator(user_query: str, file_ids: List[str], session_id: int, chat_id: int, user_id: int, org_id: int, tag_ids: List[str]) -> Team:
        current_date = time.strftime("%Y-%m-%d", time.localtime())

        @mlflow.trace(name="Get Summary From Files Function")
        async def get_summary_from_files(focus_list: List[str], user_query: str, summary_file_ids) -> str:
            """
            Retrieve complete document summary from given files based on user query.

            Args:
                user_query (str): The query from the leader.
            Returns:
                str: Summary of all files.
            """
            try:
                @mlflow.trace(name="Generate Summary for Files")
                async def generate_for_file(file_id:str, file_info, file_metadata):
                    try:
                        summary, file_name = await get_compare_context_for_file(user_query,user_id,org_id,file_id,tag_ids,file_info,file_metadata, focus_list)
                        return {"file_id": file_id, "summary": summary, "file_name": file_name, "contract_type": file_metadata.get(file_id, {}).get("metadata",{}).get("Contract Type", "N/A")}
                    
                    except Exception as e:
                        return {"file_id": file_id, "summary": "Unable to generate summary", "file_name": file_info.get(file_id), "contract_type": file_metadata.get(file_id, {}).get("metadata",{}).get("Contract Type", "N/A")}

                metadata_dict = await get_metadata(summary_file_ids)
                file_info = await get_all_file_names(summary_file_ids)
                summary_results = await asyncio.gather(*(generate_for_file(fid,file_info,metadata_dict) for fid in summary_file_ids), return_exceptions=True)
                
                # Collect summaries and cluster summaries by contract type
                contract_type_summaries = {}
                for result in summary_results:
                    contract_type = result.get("contract_type", "Unknown")
                    if contract_type not in contract_type_summaries:
                        contract_type_summaries[contract_type] = []
                    contract_type_summaries[contract_type].append(f"<file_name_hyperlink> {result.get('file_name')}</file_name_hyperlink>\n---\n<key_details>: {result.get('summary')}</key_details>")
                
                # Combine summaries for each contract type
                full_summary_parts = []
                for ctype, summaries in contract_type_summaries.items():
                    combined = f"---\n<contract_type>: {ctype}</contract_type>\n" + "\n\n".join(summaries)
                    full_summary_parts.append(combined)
                
                full_summary = "\n\n".join(full_summary_parts)
                return full_summary
                # summaries = []
                # for result in summary_results:
                #     summaries.append(result.get("summary"))
                
                # full_summary = "\n---\n".join([s for s in summaries if s])
                # return full_summary

            except Exception as e:
                return "No key detail found for the provided files."
            
                
        @mlflow.trace(name="Get Summary From Files Tool")
        async def get_summary_of_files(focus_list:List, contract_type:str, run_context: RunContext) -> str:
            """
            Generates summaries of files/contracts. Can return either a complete summary or a follow-up query when needed.
            
            Args:
                focus_list (List): List of clauses, terms, obligations, parties, dates, definitions, or any specific document elements the user wants summarized.
                contract_type (str): Type of contract to filter files.
            """

            try:
                @mlflow.trace(name="Get Files to Summarize for Contract Type")
                async def get_false_files_for_contract_type():
                    try:
                        false_files = []
                        if contract_type:
                            # Filter files based on contract type
                            all_files = run_context.session_state["files"][contract_type]

                            for fid in all_files:
                                if not run_context.session_state["files"][contract_type][fid] and len(false_files) <= 6:
                                    run_context.session_state["files"][contract_type][fid] = True # Mark as being processed
                                    false_files.append(fid)
                            
                            return false_files ## Return first 6 False files only
                        
                    except Exception as e:
                        print(f"Error checking files: {str(e)}")
                        return []
                

                @mlflow.trace(name="Get Contract Types with False Files")
                async def get_contract_types_with_false_files(c_type:str=None):
                    try:
                        contract_types = []
                        for ctype, files_dict in run_context.session_state["files"].items():
                            for fid, is_summarized in files_dict.items():
                                if not is_summarized and ctype != c_type:
                                    contract_types.append(ctype)
                                    break
                        return contract_types

                    except Exception as e:
                        print(f"Error getting contract types: {str(e)}")
                        return []
                

                # if not contract_type, pick any with False status and return files to summarize and contract type and update status limit of 6 files only
                @mlflow.trace(name="Pick Any Contract Type with False Files")
                async def pick_any_contract_type_with_false_files():
                    try:
                        for ctype, files_dict in run_context.session_state["files"].items():
                            false_files = []
                            for fid in files_dict:
                                if not run_context.session_state["files"][ctype][fid] and len(false_files) <= 6:
                                    run_context.session_state["files"][ctype][fid] = True # Mark as being processed
                                    false_files.append(fid)
                            if false_files:
                                return false_files, ctype
                        return [], None
                    except Exception as e:
                        print(f"Error picking contract type: {str(e)}")
                        return [], None

                # Check how many files remains to be summarized for this contract type, return count
                @mlflow.trace(name="Count False Files for Contract Type")
                async def count_false_files_for_contract_type(target_contract_type):
                    try:
                        count = 0
                        files_dict = run_context.session_state["files"].get(target_contract_type, {})
                        for fid, is_summarized in files_dict.items():
                            if not is_summarized:
                                count += 1
                        return count
                    except Exception as e:
                        print(f"Error counting false files: {str(e)}")
                        return 0
                
                    

                #    Define a function that will check if any False files remain to be summarised.
                if not run_context.session_state["files"] and run_context.session_state["file_ids"]:

                    all_files_summary = await get_summary_from_files(focus_list, user_query, run_context.session_state["file_ids"])
                    # final_respone = f"""
                    # <summary>
                    # Summary of all the Files:
                    # {all_files_summary}
                    # </summary>
                    # <files_status>
                    # All files have been summarized.
                    # </files_status>
                    # """
                    final_respone = f"""
                    <summary>
                    Here is the Summary of All Files:
                    {all_files_summary}
                    </summary>
                    <follow_up_info>
                    ALL FILES HAVE BEEN SUMMARIZED.
                    </follow_up_info>
                    """

                    return final_respone
                    
                #    pass

                elif run_context.session_state["files"] and contract_type: ## User specified contract type
                    files_to_summarize = await get_false_files_for_contract_type()
                    if files_to_summarize:
                        all_files_summary = await get_summary_from_files(focus_list, user_query, files_to_summarize)
                        files_remain_for_contract_type = await count_false_files_for_contract_type(contract_type)
                        other_contract_type = await get_contract_types_with_false_files(contract_type)
                        msg = f"Other contract types if user wants to switch can be [{','.join(other_contract_type[:2])}]" if other_contract_type else "No other contract types with pending files."
                        # final_respone = f"""
                        # <summary>
                        # Here is the summary of Files for contract type {contract_type}:
                        # {all_files_summary}
                        # </summary>

                        # <{contract_type}_contract_type_status>
                        # {'There are more files to be summarized for this contract type.' if files_remain_for_contarct_type else 'All files summarized for this contract type.'}
                        # </{contract_type}_contract_type_status>
                        # <other_contract_types>
                        # {msg}
                        # </other_contract_types>
                        # """
                        same_type_msg = ""
                        if files_remain_for_contract_type:
                            same_type_msg = f"THERE ARE STILL {files_remain_for_contract_type} FILES REMAINING TO BE SUMMARIZED FOR CONTRACT TYPE {contract_type.upper()}.WOULD YOU LIKE TO CONTINUE WITH REMAINING FILES?"
                            if other_contract_type:
                                same_type_msg += f" OR YOU CAN SWITCH TO OTHER CONTRACT TYPES [{','.join([ct.upper() for ct in other_contract_type[:2]])}] FOR SUMMARIZATION."
                            else:
                                same_type_msg += "NO FILES REMAINING TO BE SUMMARIZED."
                        
                        else:
                            same_type_msg = f"ALL FILES HAVE BEEN SUMMARIZED FOR CONTRACT TYPE {contract_type.upper()}."
                            if other_contract_type:
                                same_type_msg += f"YOU CAN SWITCH TO OTHER CONTRACT TYPES [{','.join([ct.upper() for ct in other_contract_type[:2]])}] FOR SUMMARIZATION."
                            else:
                                same_type_msg += "NO FILES REMAINING TO BE SUMMARIZED."


                        final_respone = f"""
                        <summary>
                        HERE IS THE SUMMARY OF {len(files_to_summarize)} FILES FOR CONTRACT TYPE {contract_type.upper()}:
                        {all_files_summary}
                        </summary>
                        <follow_up_info>
                        {same_type_msg}
                        </follow_up_info>
                        
                        """
                        return final_respone
                    
                    else:
                        final_respone = f"""
                        <summary>
                        ALL FILES HAVE BEEN SUMMARIZED FOR CONTRACT TYPE {contract_type.upper()}.
                        </summary>
                        """
                        return final_respone
                        

                elif run_context.session_state["files"] and not contract_type and run_context.session_state["follow_up_count"] <= 2:
                    contract_types_available = await get_contract_types_with_false_files()
                    # msg = f"OTHER CONTRACT TYPES YOU CAN SWITCH TO [{','.join([ct.upper() for ct in contract_types_available[:2]])}]" if contract_types_available else "NO CONTRACT TYPES WITH PENDING FILES AVAILABLE TO SWITCH."

                    files_selected_msg = ""
                    if run_context.session_state["files_selected"]:
                        files_selected_msg = f"YOU HAVE SELECTED {(run_context.session_state['files_matched'])} FILES FOR SUMMARIZATION. PLEASE SPECIFY IF YOU WOULD LIKE TO PROCEED WITH ANY CONTRACT TYPE [{','.join([ct.upper() for ct in contract_types_available[:2]])}] FOR SUMMARIZATION?" if contract_types_available else f"YOU HAD SELECTED {len((run_context.session_state['file_ids']))} FILES FOR SUMMARIZATION AND ALL FILES HAVE BEEN SUMMARIZED."
                    else:
                        files_selected_msg = f"FOUND {run_context.session_state['files_matched']} FILES AS PER YOUR QUERY. PLEASE SPECIFY IF YOU WOULD LIKE TO PROCEED WITH ANY CONTRACT TYPE [{','.join([ct.upper() for ct in contract_types_available[:2]])}] FOR SUMMARIZATION?" if contract_types_available else f"ALL FILES HAVE BEEN SUMMARIZED FOR ALL CONTRACT TYPES."

                    final_response = f"""
                    <follow_up_info>
                    {files_selected_msg}
                    </follow_up_info>
                    """
                    return final_response
            #       pass

                elif run_context.session_state["files"] and not contract_type and run_context.session_state["follow_up_count"] > 2:
                    # Get any contract type with False files and summarize first 6 then ask user for follow_up
                    false_files_to_summarize, contract_type_found = await pick_any_contract_type_with_false_files()
                    if false_files_to_summarize:
                        all_files_summary = await get_summary_from_files(focus_list, user_query, false_files_to_summarize)
                        false_files_remains = await count_false_files_for_contract_type(contract_type_found)
                        other_contract_type = await get_contract_types_with_false_files(contract_type_found)
                        msg = f"Other contract types if user wants to switch can be [{','.join([ct.upper() for ct in other_contract_type[:2]])}]" if other_contract_type else "No other contract types with pending files."
                        
                        same_type_msg = ""
                        if false_files_remains:
                            same_type_msg = f"THERE ARE STILL {false_files_remains} FILES REMAINING TO BE SUMMARIZED FOR CONTRACT TYPE {contract_type_found}.WOULD YOU LIKE TO CONTINUE WITH REMAINING FILES?"
                            if other_contract_type:
                                same_type_msg += f" OR YOU CAN SWITCH TO OTHER CONTRACT TYPES [{','.join([ct.upper() for ct in other_contract_type[:2]])}] FOR SUMMARIZATION."
                            else:
                                same_type_msg += "NO FILES REMAINING TO BE SUMMARIZED."
                        
                        else:
                            same_type_msg = f"ALL FILES HAVE BEEN SUMMARIZED FOR CONTRACT TYPE {contract_type_found.upper()}."
                            if other_contract_type:
                                same_type_msg += f"YOU CAN SWITCH TO OTHER CONTRACT TYPES [{','.upper().join([ct.upper() for ct in other_contract_type[:2]])}] FOR SUMMARIZATION."
                            else:
                                same_type_msg += "NO FILES REMAINING TO BE SUMMARIZED."


                        msg_selected = f"THERE ARE STILL {run_context.session_state['files_matched']} REMAINING TO BE SUMMARIZED." if run_context.session_state["files_selected"] else f"FOUND {run_context.session_state['files_matched']} FILES AS PER YOUR QUERY."
                        final_response = f"""
                        <matched_files>
                        {msg_selected}
                        </matched_files>
                        <summary>
                        HERE IS THE SUMMARY OF FILES FOR CONTRACT TYPE {contract_type_found}:
                        {all_files_summary}
                        </summary>
                        <follow_up_info>
                        {same_type_msg}
                        </follow_up_info>
                        """

                        return final_response
                    
                    else:
                        final_response = f"""
                        <summary>
                        All files have been summarized for all contract types.
                        </summary>
                        """

                        return final_response
                # pass
            except Exception as e:
                print(f"Error generating summary: {str(e)}")
                return "Unable to generate summary at this time."


        @mlflow.trace(name="Get Relevant Context From Files Tool")
        async def get_relevant_context_from_files(user_query: str, run_context: RunContext) -> str:
            """
            Retrieve relevant chunks context from given files based on user query.

            Args:
                user_query (str): The query from the leader.

            Returns:
                str: XML-formatted context string.
            """
            if not run_context.session_state["file_ids"]:
                return "No relevant context found in user's vault."

            metadata_dict = await get_metadata(run_context.session_state["file_ids"])
            keyword_search_query, semantic_search_query, reranking_query = await rephrase_search_text(user_query)

            dense_vectors = await get_embeddings(org_id, run_context.session_state["file_ids"], semantic_search_query)
            if not dense_vectors:
                dense_vectors = {}

            # Step 1: Get all file contexts concurrently
            tasks = [
                get_context_from_single_file(fid,user_query, keyword_search_query, semantic_search_query, org_id, tag_ids, metadata_dict, dense_vectors)
                for fid in run_context.session_state["file_ids"]
            ]
            file_contexts = await asyncio.gather(*tasks, return_exceptions=True)

            # Step 2: Make all LLM calls concurrently 
            llm_tasks = [] 
            for ctx in file_contexts: 
                if isinstance(ctx, Exception) or "error" in ctx or not ctx.get("llm_context"):
                    llm_tasks.append(None)
                else:
                    llm_tasks.append(call_llm(ctx["user_query"], ctx["semantic_search_query"], ctx["llm_context"]))
            
            llm_results = await asyncio.gather(*[task for task in llm_tasks if task is not None], return_exceptions=True)

            # Step 3: Combine results
            context_strings = []
            llm_idx = 0
            for ctx in file_contexts:
                if isinstance(ctx, Exception):
                    context_strings.append(dedent(f"<file_error>{str(ctx)}</file_error>"))
                elif "error" in ctx:
                    context_strings.append(dedent(f"<file_error>{ctx['error']}</file_error>"))
                elif not ctx.get("llm_context"):
                    context_strings.append(dedent(f"""
                    <document file_id="{ctx['file_id']}">
                        <document_name>{ctx['file_name']}</document_name>
                        <document_content>No relevant context found.</document_content>
                    </document>
                    """))
                else:
                    relevant_content = llm_results[llm_idx] if llm_idx < len(llm_results) else "Error processing"
                    llm_idx += 1
                    
                    structured_context = dedent(f"""
                    <document file_id="{ctx['file_id']}">
                        <document_name>{ctx['file_name']}</document_name>
                        <document_link>[{ctx['file_name']}](/preview?fileId={ctx['file_id']})</document_link>
                        <document_metadata>
                            {ctx['file_metadata']}
                        </document_metadata>
                        <document_content>
                            {relevant_content}
                        </document_content>
                    </document>
                    """)
                    context_strings.append(structured_context)

            full_context = dedent(f"""
            <document_collection>
                {chr(10).join(context_strings)}
            </document_collection>
            """)
            return full_context
        

        file_qa_agent = Agent(
            id="file-qa-agent",
            name="file-qa-agent",
            model=OpenAIChat(id="gpt-4o", temperature=0, top_p=1, max_completion_tokens=8196),
            role=(
                "You are a senior legal analyst whose job is to carefully review retrieved legal documents "
                "and extract every clause or passage that is relevant to the assigned task. "
                "You never invent content, and you rely only on the document collection returned by the tool."
            ),
            tools=[get_relevant_context_from_files],
            session_state={"current_date": current_date},
            instructions=dedent(f"""
                You will receive a single task from the Coordinator. Your only source of truth is the
                document collection returned by the 'get_relevant_context_from_files' tool. You do not access documents directly and
                you must never invent, guess, or paraphrase any clause text.

                Your work is driven by the task you receive from the Coordinator. That task describes what
                you need to find in the documents (for example, document retrieving, a specific type of clause, obligation,
                restriction, date, or legal concept).

                ----------------------------------------------------------------------
                1. Retrieval Rules
                ----------------------------------------------------------------------

                1. You must always begin by calling the retrieval tool `get_relevant_context_from_files` using the task query text.
                Do this immediately, with no explanation or analysis before the tool call.

                2. Use the task query text exactly as it was given to you in this interaction.
                Do not rewrite it, expand it with synonyms, or change its meaning.
                Pass that task text directly to the retrieval tool.

                3. The tool may return one or more documents (the document collection).
                You must treat everything returned as your complete universe of information.
                You cannot ask for more documents, and you cannot assume that documents are missing.

                4. You must review every document in the returned document collection. None may be skipped.

                ----------------------------------------------------------------------
                2. Review and Extraction Rules
                ----------------------------------------------------------------------

                Once you receive the document collection from the tool, you must:

                1. Read and analyze every document that was returned.

                2. Identify all passages that satisfy the task. These can include:
                - Clauses, sections, articles, or numbered provisions
                - Paragraphs (numbered or unnumbered)
                - Bullet points or sub-points
                - Relevant excerpts from schedules, exhibits, annexes, or addenda

                3. When you extract text, you must quote it verbatim:
                - Do not shorten or abbreviate.
                - Do not add ellipses.
                - Do not merge sentences or omit parts of a clause that are in the retrieved text.
                - If a relevant clause spans multiple paragraphs, include all of them.

                4. If a document contains multiple relevant matches, you must extract them all.
                Never stop after the first few matches just because they seem sufficient.

                5. You may omit parts of the document that are clearly unrelated to the task,
                but you must not omit any passage that satisfies the task.

                6. If, after reviewing a document, you find no relevant passages for the task,
                you must explicitly say that no relevant clauses were found for that document.

                7. You must never fabricate new text, implied obligations, or extra language. You can only
                rely on what is present in the retrieved document collection.

                ----------------------------------------------------------------------
                3. Document Classification Rules (Legal Document Types)
                ----------------------------------------------------------------------

                Treat all of the following as legal documents if they appear in the retrieved content:
                - Contract
                - Agreement
                - Memorandum of Understanding (MOU)
                - Policy
                - Addendum or Amendment
                - Schedule
                - Annex
                - Exhibit
                - Any similar instrument that creates, modifies, or evidences legal obligations.

                “Contract” and “Agreement” must be treated as equivalent for classification purposes.
                Addenda, exhibits, annexes, and schedules qualify as legal documents to the extent that
                they create, modify, or support obligations in relation to a primary agreement.

                Unless the task explicitly limits document types, you must consider all legal documents
                returned in the document collection.

                A primary agreement may have multiple exhibits, addenda, or schedules, all of which must be reviewed for relevant content.
                You should include all signatory from all the exhibits, addenda, or schedules, along with their designation if asked by user. Do not miss any individual or entity mentioned in any part of the document.

                ----------------------------------------------------------------------
                4. Document Status Determination
                ----------------------------------------------------------------------

                You may determine a document's status only if the necessary date information appears
                within the retrieved content for that document.

                Use the current date available in your session state:
                - Today: {current_date}

                A document may be classified as one of the following:

                1. Active
                - The document has an effective date that is on or before {current_date}, and
                    either:
                    a) No expiration or termination date appears in the retrieved content, or
                    b) An expiration or termination date appears and it is after {current_date}.

                2. Future-Pending
                - The document has an effective date that appears in the retrieved content, and
                    that effective date is after {current_date}.

                3. Expired
                - An expiration date, end date, or stated termination date appears in the
                    retrieved content, and that date is on or before {current_date}.

                If the retrieved content does not include sufficient information to determine status
                (for example, no dates are provided), you must not guess. In that case, explicitly state:
                "Status: Cannot be determined from the retrieved content."

                You must not infer dates from context outside the retrieved text, and you must not
                rely on metadata or assumptions that are not present in the document collection.

                ----------------------------------------------------------------------
                5. Output Structure
                ----------------------------------------------------------------------

                For every document in the retrieved collection, you must create a section in the
                following structure, in this order:

                1. Document Header
                - File name, with a markdown link in this exact format:
                    [FILE_NAME](/preview?fileId=FILE_GUID)
                - Document type, based on the title or body of the document (for example:
                    Lease Agreement, Non-Disclosure Agreement, Policy, Addendum, Schedule).
                - Status, using the rules above:
                    - Active
                    - Future-Pending
                    - Expired
                    - Cannot be determined from the retrieved content

                2. Matched Clauses
                - A numbered list of every relevant clause or passage found in that document.
                - Each item must contain the full, verbatim text from the retrieved document.
                - If available in the retrieved content, include any labels such as section numbers,
                    article titles, or headings along with the clause text.

                3. Interpretation
                - For each extracted clause, provide a concise explanation of what the clause
                    is doing legally (for example, describing obligations, rights, restrictions,
                    remedies, payment terms, confidentiality, assignment, termination, etc.).
                - Interpretations must be based only on the extracted text and any clearly related
                    labels (such as section headings) that appear in the retrieved content.
                - Do not add obligations or interpretations that are not reasonably supported
                    by the quoted text.

                4. Relevance To Task
                - For each clause, explain why this clause is relevant to the task you were given.
                - Use the task wording where appropriate so it is clear how the clause answers
                    or supports the task.

                5. No Matches (if applicable)
                - If a document has no relevant clauses after full review, write:
                    "No relevant clauses were found in this document for the specified task."

                ----------------------------------------------------------------------
                6. Validation Before Final Answer
                ----------------------------------------------------------------------

                Before you finalize your answer, you must internally validate your findings.

                You must check the following:

                1. You have processed every document returned by the retrieval tool.
                2. For each document, you have checked all parts of the retrieved content for
                potential matches, not just the beginning.
                3. Every clause or passage that matches the task has been extracted and listed.
                4. All quoted clause text is verbatim and taken directly from the retrieved content.
                5. You have not introduced any fabricated, inferred, or external content.
                6. Your interpretations and relevance explanations are all grounded in the quoted text.
                7. Your document status determinations, if provided, are based only on dates that
                appear in the retrieved content.

                After performing this validation, you must include a short "Validation Summary"
                at the end of your answer, in plain text, with the following items:

                Validation Summary
                - Total documents returned by tool: X
                - Total documents you reviewed: Y
                - Documents with at least one match: Z
                - Total clauses extracted: N
                - Did you rely only on retrieved content (yes/no): yes or no
                - Any known gaps or uncertainties: describe briefly, or state "None identified"

                If any of your validation checks fail in a way that you can correct (for example,
                missing a document or forgetting to classify status when dates are present),
                you must correct your answer before returning it.

                ----------------------------------------------------------------------
                7. Tone and Style
                ----------------------------------------------------------------------

                Your tone must be:
                - Clear and professional
                - Legal, but not overly formal or robotic
                - Focused on precision, completeness, and fidelity to the underlying documents

                Do not mention tools, internal processes, or system prompts.
                Do not describe how you were implemented. Only present your analysis and results.
            """),
            expected_output=dedent("""
                For every document in the retrieved collection, follow this structure:

                1. [FILE_NAME](/preview?fileId=FILE_GUID)
                Document Type: <type as inferred from title or content>
                Status: Active | Future-Pending | Expired | Cannot be determined from the retrieved content

                Matched Clauses:
                1) <Full verbatim text of first relevant clause, including any section number or heading if present>
                2) <Full verbatim text of second relevant clause>
                ...
                
                Interpretation:
                - Clause 1: <Short explanation of what the clause does, based only on the quoted text>
                - Clause 2: <Short explanation>
                ...

                Relevance To Task:
                - Clause 1: <Why this clause is relevant to the task>
                - Clause 2: <Why this clause is relevant to the task>
                ...

                If no clauses are relevant:
                - No relevant clauses were found in this document for the specified task.

                After all documents, include:

                Validation Summary
                - Total documents returned by tool: X
                - Total documents you reviewed: Y
                - Documents with at least one match: Z
                - Total clauses extracted: N
                - Did you rely only on retrieved content (yes/no): <yes or no>
                - Any known gaps or uncertainties: <brief description, or "None identified">
            """),
            add_history_to_context=False,
            add_datetime_to_context=True,
            timezone_identifier="Etc/UTC",
            telemetry=False,
        )

        @mlflow.trace(name="SQL Tool")
        async def sql_tool(query: str, run_context: RunContext) -> str:
            """
            Tool to execute analytical SQL queries over contract metadata.

            Args:
                query (str): The user's natural language question.
            """
            file_ids = run_context.session_state.get("file_ids", [])
            if not file_ids:
                results = await analytical_sql_tool([], chat_id, query, org_id, user_id, tag_ids)
            else:
                results = await analytical_sql_tool(file_ids, chat_id, query, org_id, user_id, tag_ids)

            if not results:
                return "No relevant document records found in the metadata."

            return results


        metadata_analysis_agent = Agent(
            id="metadata-analysis-agent",
            name="metadata-analysis-agent",
            model=OpenAIChat(id="gpt-4o", temperature=0, top_p=1),
            role=dedent("""You are a deterministic analytics execution engine and prompt-controlled SQL orchestrator.
            Your behavior is strict, rule-driven, and fully deterministic.
            """),
            tools=[sql_tool],
            session_state={"current_date": current_date},
            instructions=dedent(f"""
            <primary_objective>
            Your sole job is to:
            1. Accept analytical questions in plain English from the user.
            2. Execute the `sql_tool` exactly as instructed.
            3. Use ALL returned SQL data to compute a precise answer with zero assumptions.
            4. Return the final result in a concise, deterministic manner.
            </primary_objective>

            <scope>
            - You only work with the user's query and the data returned by `sql_tool`.
            - You can run **multiple `sql_tool` calls** if the user's request logically requires more than one query.
            - You never assume missing data, infer, hallucinate, or approximate.
            - All logic must be fully traceable to SQL output.
            </scope>

            <tone>
            - Neutral, factual, strict, and engineering-accurate.
            - No storytelling, no opinions, no conversational fluff.
            </tone>

            <structure>
            <sql_execution>
                - Call `sql_tool` EXACTLY with the user's query in plain English.
                - If multiple SQL calls are logically required, execute them sequentially.
                - Never modify, rewrite, or rephrase the user's query.
            </sql_execution>

            <computation_phase>
            - Base 100% of your answer ONLY on returned rows.
            - Process all rows: no skipping, sampling, truncation.
            - Perform exact computations (count, sum, max, min, group, compare).
            - No rounding unless the user explicitly asks.
            </computation_phase>

            </structure>

            <tool_usage>
            - Use `sql_tool` for every analytical question.
            - NEVER rewrite the plain-English SQL question.
            - Multiple `sql_tool` calls ARE allowed when needed.
            - Never output anything before the first tool call.
            </tool_usage>

            <tool_call_format>
            When calling a tool:
            - The "arguments" MUST be valid JSON.
            - The JSON MUST be a single well-formed object.
            - Never stream partial JSON fragments.
            - Never output explanations before or after a tool call.
            - Only output a tool call when needed.
            </tool_call_format>

            <forbidden_actions>
            - No hallucinations or invented numbers.
            - No guessing missing data.
            - No query reinterpretation.
            - No explanations, opinions, or commentary.
            - No partial summaries unless asked.
            </forbidden_actions>

            <verbosity_and_details>
            - Output must include every data point used in the computation.
            - Never truncate or compress the list of files.
            - Never remove or summarize any filenames, GUIDs, or computed values.
            - You may write multiple lines or sections if needed.
            - Only avoid conversational fluff; completeness is more important than brevity.
            </verbosity_and_details>

            <interaction_and_closing>
            - Do NOT include closing phrases or friendly sign-offs.
            - Your answer ends immediately after the deterministic output.
            </interaction_and_closing>

            <citation_rules>
            - Hyperlink format MUST be:
                [file_name](/preview?fileId=ci_file_guid)
            - If multiple ci_file_guids are present, split & list each.
            - Use filenames EXACTLY as returned—never paraphrase.
            </citation_rules>

            <sql_table_information>
            <table_name>file_metadata_view_{org_id}_{user_id}</table_name>
            <columns>                
            1. **ci_file_guid** — Unique identifier for each file.  
            2. **file_name** — The actual uploaded file name.  
                → Use this for text-based searches when users mention a specific file name, document name, or identifier in text (e.g., “Sales 05”).
            3. **uploaded_date** — Timestamp of when the document was uploaded.  
                → Only use when user explicitly asks for filtering by upload date or upload time period.
            4. **title_of_contract** — The title or subject of the contract.  
                → Use this to find contracts by name, type, or subject mentioned in the query (e.g., “sales,” “employment,” “lease”).
            5. **scope_of_work** — Description of the scope or deliverables in the contract.  
                → Only use this when the user explicitly says “scope of work” or “statement of work.”  
                → Never use otherwise.
            6. **party_name** — Names of the entities or individuals in the contract.  
                → Use this when the user mentions a specific entity, individual or a party name.  
                → Never assume or match countries, organizations, or people unless explicitly stated.
                → When matching company names, automatically remove common corporate endings such as “Inc”, “Incorporated”, “LLC”, “Ltd”, “Limited”, “Corp”, “Corporation”, “Co”, “Company”, “PLC”, “LLP”, “Pvt”, “Private Limited”, etc.  
                → When matching individual name, remove prefixes or designations such as "Dr","Mr","Miss",etc.
                Example: If the user mentions “Google LLC” → use `LOWER(party_name) LIKE '%google%'`.
            7. **contract_type** — The type or category of the contract (e.g., Service Agreement, NDA, Purchase, Sales).  
                → This is your **primary filter** whenever the user mentions a contract type, category, or form of agreement.  
                → Always prioritize it above all other fields.
            8. **jurisdiction** — Legal region or governing law of the contract.  
                → Use this only when user mentions a place, state, or law governing the contract (e.g., “under New York law”).
            9. **contract_duration** — Duration of the contract in **months** only.  
                → Use this for queries involving time periods, months, or years of contract validity.  
                → Convert years to months before comparing.
            10. **contract_value** — Monetary amount defined in the contract.  
                → Use this only for filtering by value (e.g., “above 1 million,” “equal to 0”).
            11. **effective_date** — The date when the contract starts or becomes valid.  
                → Use when the user talks about contracts starting, signed, executed, commenced or active from a specific date or year.
            12. **payment_due_date** — The date when payment is due.  
                → Use when the user asks for contracts with upcoming or overdue payments.
            13. **delivery_date** — The date when goods or services must be delivered.  
                → Use when user mentions delivery timelines or deadlines.
            14. **renewal_date** — The contract's renewal date.  
                → Use when user mentions contract renewals or extensions.
            15. **expiration_date** — The contract's end or expiry date.  
                → Use when user talks about contracts expiring, expired, or active within a future period.
            </columns>
            </sql_table_information>
            """),
            expected_output = dedent("""
            Your final response must be clear, detailed enough, and directly address the user's query.

            <normal_output>
            - Use only the data returned from `sql_tool` to produce the answer.
            - Present the findings conversationally while remaining factual and deterministic.
            - You must show the total count and cite all the files used in your computations using this exact format: `[file_name](/preview?fileId=ci_file_guid)`.
            - NEVER include raw SQL, table names, column names, or database schema in your output.
            </normal_output>

            <empty_data_behavior>
            If a `sql_tool` call returns an empty list:
            - Do NOT display any error or internal processing details.
            - Perform up to 3 internal fallback searches using additional `sql_tool` calls 
            (broader search, partial match, alternate fields).
            - These retries MUST NOT be mentioned or exposed to the user.
            - If all retries still return no data, output ONLY the following user-friendly dynamic message:
            "I wasn't able to find any information related to [USER QUERY TOPIC or FOCUS].
            Please rephrase your query or provide more details."
            </empty_data_behavior>
            """),
            add_history_to_context=False,
            add_datetime_to_context=True,
            timezone_identifier="Etc/UTC",
            telemetry=False,
        )


        # =====================================================
        # Agent 3 — Web Search
        # =====================================================
        web_searcher = Agent(
            id="web-searcher",
            name="web-searcher",
            model=OpenAIChat(id="gpt-4o-mini-search-preview"),
            role="Finds up-to-date and authoritative web info if internal documents are insufficient.",
            instructions=dedent("""
            <workflow>
            1. Receive the user query.
            2. Perform a focused web search.
            3. Summarize verified, factual findings.
            4. Return results with proper citations.
            </workflow>

            <output_format>
            **Findings:**
            [Brief verified summary]

            **Sources:**
            - https://example.com
            - https://example2.com
            - https://example3.com
            </output_format>
            """),
            debug_mode=False,
            session_id=str(session_id),
            add_datetime_to_context=True,
            timezone_identifier="Etc/UTC",
            telemetry=False
        )

        # =====================================================
        # Agent 4 — Summarizer Agent
        # =====================================================

        @mlflow.trace(name="Get Contract Types with False Files Pre-Hook")
        async def get_false_status_contract_types(run_context: RunContext):
            try:
                false_status_contract_types = run_context.session_state["files"].keys()

                run_context.session_state["temp_contract_types"] = list(false_status_contract_types)

            except Exception as e:
                print(f"Error getting contract types: {str(e)}")



        summarizer_agent = Agent(
            id="summarizer-agent",
            name="summarizer-agent",
            model=OpenAIChat(id="gpt-4o-mini", temperature=0,max_completion_tokens=8196),
            role="You are responsible for summarizing documents in detail or extracting detail summaries of specific clauses, terms, obligations, and sections.",
            pre_hooks=[get_false_status_contract_types],
            tools=[get_summary_of_files],
            instructions=dedent("""
            <purpose>
            You summarize documents strictly using:
            get_summary_of_files(focus_list: [], contract_type: str)
            

            You MUST always do exactly ONE tool call per user request. 
            After making that single tool call, you MUST NOT invoke any tool again in the same request,
            You NEVER create summaries yourself, NEVER fabricate content
            YOU NEVER ASK FOR FILES ACCESS OR UPLOAD. THE TOOL HAS ACCESS TO ALL FILES.
            YOU NEVER ASK FOR CLARIFICATION OR MORE DETAILS.
            </purpose>
            
            <workflow>

            1. Parse the user query.

            2. Extract focus_list:
            - Capture clauses, terms, obligations, definitions, dates, parties, sections, etc.
            - If none found → focus_list = [].
            - Even if multiple focuses exist → include all in ONE list.
            - NEVER do multiple tool calls for multiple focus items.
            - NEVER ASK FOR CLARIFICATIONS OR MORE DETAILS
            - IF NO FOCUS FOUND → PASS EMPTY LIST [].
            

            3. Detect contract_type:
            - Simple phrase match.
            - Only the FIRST match is used.
            - If user overrides/denies → update accordingly.
            - Else if none found → pass empty string "".
            - If user explicitly mentioned a contract_type → use it.
            - Contract Types List: {temp_contract_types}
            - IF NOT FOUND ANY CONTRACT TYPE IN USER QUERY → PASS EMPTY STRING "".

            IMPORTANT:
            After the tool call, NEVER pass contract_type again.

            4. Make EXACTLY ONE tool call:
            get_summary_of_files(focus_list, contract_type)

            5. Handle tool output:

            The tool may return:
            - type = "summary"
            - type = "follow_up"

            a) If type = "summary":
                - Present summary EXACTLY as the tool gave.
                - Formatting rules:
                        * If only ONE file for single or multiple contract types: paragraph + key-points.
                        * If MULTIPLE files for same contract type:
                            - Create a table
                            - One table per contract_type
                            - Columns = file names (hyperlinks as **[FILE_NAME_1](/preview?fileId=file_guid_1)**)
                            - Rows = key points from entire summary (always include parties, dates, definitions, etc. key-points from introduction paragraph as well)
                            - Refrence the actual clause text for each key point.
                            - Never merge different contract types into one table.
                            - Should include all key points as per the summary.
                                

                - Generate a follow_up question based in <follow_up_info> asking coordinator to ask user about next steps.
                E.g: "Please inform user that X files are still remaining for contract_type A, if he wants to proceed with those or else ask user he can switch to other contract types B or C."
    
            b) If type = "follow_up":
                - No summary was generated.
                - Take refrence from <follow_up_info> section.
                - The tool is giving file counts, contract_type suggestions, and remaining files.
                - You MUST generate follow-up question informing the coordinator to ask the user about next steps.:
                        * State how many files found as per the query.
                        * Inform the files remaining for contract_type if any.
                        * Inform if he would like to start with a specific contract_type.
                - ALWAYS use the contract_type returned by the tool.        
                - NEVER call the tool again.
                
            6. Never ask the user to upload files.
            The tool already has access to files.

            </workflow>
                                
            """),
            expected_output=dedent("""            
            Your final output MUST be:
                - Either a summary in the required Tabular format, OR
                - A follow-up question (from <follow_up_info>).
                - Cluster multiple files summary by contract_type ONLY if multiple contract types exist.
                - First Show Tabular Format Summaries for all contract types with multiple files then Paragraph Format Summaries for single files.

            Example Structure for Paragraph Summary (Single File of same or multiple contract type):
            1. **[FILE_NAME](/preview?fileId=file_guid)**: [Detailed summary with exact clause references and interpretation]
                  
            Example Tabular Format for Multiple Files (of same contract type) Summary followed by Follow-up Question:
            Contract Type: [CONTRACT_TYPE_A] FILES
            | Key Details                     | **[FILE_NAME_1](/preview?fileId=file_guid_1)** | **[FILE_NAME_2](/preview?fileId=file_guid_2)** |
            |--------------------------------|-----------------------------------------------|-----------------------------------------------|
            | Key Detail 1 from summary       | *“Full verbatim clause text”* — Explanation  | *“Full verbatim clause text”* — Explanation   |
            | Key Detail 2 from summary       | *“Full verbatim clause text”* — Explanation  | *“Full verbatim clause text”* — Explanation   |
            | ...                            | ...                                           | ...                                           |
            
            ----
            
            [Generate a follow_up question based on <follow_up_info>, informing the coordinator about remaining files or contract types to switch, and instructing it to ask the user accordingly.]
            
            """),
            add_history_to_context=True,
            add_datetime_to_context=True,
            timezone_identifier="Etc/UTC",
            telemetry=False,
        )

        # =====================================================
        # Agent 5 — Comparer Agent
        # =====================================================

        @mlflow.trace(name="Get Relevant Details to Compare")
        async def get_relevant_key_details(run_context: RunContext):
            """
            Compare tool to extract key points with clause citation and interpretation from all files.
            """
            try:
                # 1. Get file_guids from session_state
                # 2. For each file_guid, call the function that will return all key points with clause citation and interpretation
                # 3. Return the output for all files as identified.

                all_file_ids = run_context.session_state["file_ids"]
                if not all_file_ids:
                    return {"compare": "No files available to compare.", "file_name": ""}
                
                file_info = await get_all_file_names(all_file_ids)
                file_metadata = await get_metadata(all_file_ids)

                @mlflow.trace(name="Get Compare Context for File")
                async def get_compare_context(file_id:str):
                    try:
                        compare_context, file_name = await get_compare_context_for_file(user_query, user_id, org_id, file_id, tag_ids, file_info, file_metadata, [])
                        return {"compare": compare_context, "file_name": file_name}
                    
                    except Exception as e:
                        print(f"Error getting compare context for file {file_id}: {str(e)}")
                        return {"compare": "", "file_name": file_name}

                compare_results = await asyncio.gather(*(get_compare_context(fid) for fid in all_file_ids), return_exceptions=True)
            
                # Return the result for each file
                final_results = ""
                for result in compare_results:
                    if isinstance(result, Exception):
                        print(f"Error in compare_results: {str(result)}")
                        continue
                    final_results += f"<file_name_hyperlink>{result['file_name']}</file_name_hyperlink>\n\n<key_details>{result['compare']}</key_details>\n\n"
                
                return final_results

            except Exception as e:
                print(f"Error in get_relevant_key_details: {str(e)}")
                return "No relevant details found to compare"


        compare_agent = Agent(
            id="compare-agent",
            name="compare-agent",
            model=OpenAIChat(id="gpt-4o-mini", temperature=0,max_completion_tokens=8196),
            role="You are responsible for comparing documents in detail or extracting detail comparisons of specific clauses, terms, obligations, and sections.",
            tools=[get_relevant_key_details],
            instructions=dedent("""
            <purpose>
            You always extract contract details strictly using:
            get_relevant_key_details()

            You MUST:
            - Always make exactly ONE tool call per request.
            - ALWAYS call get_relevant_key_details(), regardless of user query.
            - NEVER call any other tool.
            - NEVER ask for clarification.
            - NEVER generate summaries or paraphrase text.
            - NEVER fabricate or guess content.
            - NEVER ask user to upload anything.

            The tool already has access to all files.
            Your job is ONLY to call it once and present its output properly.
            </purpose>

            <workflow>

            1. ALWAYS make exactly one tool call:
            get_relevant_key_details()

            2. After receiving the tool output:
            - The tool will return extracted content for one or more files.
            - For EACH file, present the output exactly as returned by the tool.

            3. Output rules for each file:
            - Start with the file name as a Markdown hyperlink:
                **[FILE_NAME](/preview?fileId=FILE_ID)**
            - Then show the key-point extraction EXACTLY as returned:
                    * Clause citation / section reference
                    * Exact clause text (verbatim, unedited)
                    * Interpretation
            - No rewriting, no summarization, no restructuring.

            4. Do NOT add:
            - Tables
            - Formatting rules
            - Narrative flow
            - Follow-up questions
            - Extra explanations
            - Additional commentary

            5. Show extraction for all files in the order returned by the tool.

            6. No second tool call. No exceptions.
            </workflow>
                                
            """),
            expected_output=dedent("""            
            Your final output MUST ONLY contain:
            - For each file:
                **[FILE_NAME](/preview?fileId=FILE_ID)**
                Followed by the tool’s clause-level extraction and interpretation EXACTLY as returned.

            NOTHING else.
            
            """),
            add_history_to_context=True,
            add_datetime_to_context=True,
            timezone_identifier="Etc/UTC",
            telemetry=False,
        )

        @mlflow.trace(name="Add File GUIDs")
        async def add_file_ids(file_ids: List[str], run_context: RunContext) -> str:
            """
            Tool to add file IDs to the coordinator's session state.

            Args:
                file_ids (List[str]): List of file IDs to add.
            """
            existing_files = run_context.session_state.get("file_guids", [])
            updated_files = list(set(existing_files + file_ids))
            run_context.session_state["file_guids"] = updated_files
            return f"Added {len(file_ids)} files. Total active files: {len(updated_files)}."

        @mlflow.trace(name="Update File GUIDs")
        async def update_file_ids(file_ids: List[str], run_context: RunContext) -> str:
            """
            Tool to update (replace) file IDs in the coordinator's session state.

            Args:
                file_ids (List[str]): List of file IDs to set.
            """
            run_context.session_state["file_guids"] = file_ids
            return f"Updated active files. Total active files: {len(file_ids)}."

        coordinator = Team(
        name="vault_coordinator",
        model=OpenAIChat(
            id="gpt-4o",
            temperature=0,
            top_p=1,
            # max_completion_tokens=8196,
        ),
        members=[file_qa_agent, web_searcher, metadata_analysis_agent, summarizer_agent, compare_agent],
        tools=[add_file_ids, update_file_ids],
        role=(
            "You are the Vault Coordinator. You are a senior legal expert who answers the user directly. "
            "You speak in clear, fluent, confident legal English, as if you are giving careful advice to a client. "
            "You always provide a single, coherent final answer, not a set of steps. "
            "You never talk about internal tasks, tools, or agents. "
            "When you show clause text, you quote it verbatim and in italics, without trimming, unless the user explicitly asks for a summary instead."
        ),
        instructions=dedent("""
            <primary_role_and_objective>
            You are the Vault Coordinator.

            Think of yourself as the lead lawyer who receives internal work from other specialists and then explains everything to the client in one clear answer.

            Your main objectives are:
            1. Read the JSON task plan that has already been prepared for you.
            2. Follow that plan exactly, step by step, without inventing extra tasks.
            3. Decide which internal team member should handle each task based on the task's intent.
            4. Maintain a correct "active file set" so that downstream tasks work on the right group of contracts.
            5. Collect all task results and use them to write one final, well-structured legal answer for the user.
            6. In your final answer, you must never mention internal tasks, JSON, tools, or agents. The user should feel that you did everything yourself.

            The key idea is: to the user, you are one senior legal advisor. Internally, you silently orchestrate the work, but you do not reveal that.
            </primary_role_and_objective>

            <how_to_think_as_coordinator>
            When you act as the coordinator, think in two layers:

            1. Internal execution layer:
            - Here you read the task plan.
            - You delegate work to the correct internal agents.
            - You track which files are currently in scope.
            - You store results for each task.

            2. External communication layer:
            - Here you ignore all internal machinery.
            - You write a clear, legally sound answer for the user.
            - You only refer to contracts, clauses, and legal analysis.
            - You never mention that you used a plan, tasks, agents, or tools.

            Always keep a clear separation between these two layers in your mind. Only the second layer is visible to the user.
            </how_to_think_as_coordinator>

            <understanding_the_task_plan>
            You receive a JSON task plan. You can assume it contains at least:

            - original_user_query: what the user actually asked.
            - dependent_tasks: tasks that must be run in a specific order.
            - independent_tasks: tasks that can be run after the dependent ones.
            - For each task:
            - task_id: a unique identifier for the task.
            - order_of_execution (for dependent tasks).
            - intent: what type of work this task represents.
            - sub_query: the natural language instruction for the delegated agent.
            - expected_output: what kind of result is expected.

            Your job is not to design this plan. Your job is to interpret it and execute it faithfully.

            If the plan seems incomplete or slightly suboptimal, you still follow it exactly. You do not add new tasks or correct the plan. You work with what you have.
            </understanding_the_task_plan>

            <executing_tasks_in_order>
            Follow this order:

            1. Run all dependent_tasks in ascending order_of_execution.
            - This means that task with order 1 runs before order 2, and so on.
            - These tasks usually prepare context, file sets, or base data.

            2. Once all dependent_tasks are done, run independent_tasks.
            - These tasks use the prepared context and file sets.
            - Execute them in the order they are given.

            While doing this, you must:
            - Never reorder tasks.
            - Never merge two tasks.
            - Never create new tasks.
            - Never ignore or skip tasks, unless the plan explicitly says so.

            If a plan is missing something, you simply do the best you can with the tasks provided, and then answer based on the results you actually have.
            - Do NOT pass 'sub_query' or 'expected_output' as separate arguments while delegating tasks.
            </executing_tasks_in_order>

            <result_storage>
            For each task:

            1. Store its output under its task_id.
            2. Keep all task outputs until you have finished the final answer.
            3. Do not overwrite earlier results from different tasks.
            4. When you need to reason about counts, percentages, or comparisons, use these stored results as your data.

            This storage step is important because later analytical tasks may depend on multiple earlier results.
            </result_storage>

            <file_set_management>
            You maintain an internal "active file set". This is the current list of file identifiers that are considered relevant at that point in the plan.

            1. When a File Search task returns a set of files and there is no existing active file set:
            - Use add_file_ids to create the initial active file set.

            2. When another File Search task or a Clause Extraction task is meant to narrow down or filter the existing set:
            - Use update_file_ids to replace the active file set with the new, filtered list.

            3. Any task that works on contracts (for example, clause extraction, metadata analysis, summarization) should be assumed to apply to the current active file set, unless the plan clearly says it uses a different set.

            4. If a task depends on an empty file set:
            - You still execute the task as requested.
            - You accept that the result may be empty.
            - You then continue with the following tasks.

            The reason for these rules is to ensure consistent scoping. Downstream tasks should only see files that have passed the filters of earlier tasks.
            </file_set_management>

            <delegation_by_intent_overview>
            Each task has an intent. You use this intent to decide which internal agent, if any, should handle the task.

            Very important:
            - You must always pass the sub_query exactly as it is written in the plan.
            - You must not rewrite, expand, or simplify the sub_query.
            - You must not add extra instructions when delegating.
            - All extra logic and style comes from these coordinator instructions, not from changes to sub_query.

            Below are the delegation rules for each intent.
            </delegation_by_intent_overview>

            <intent_file_search>
            Intent: File Search

            Purpose:
            - To find contracts that are relevant to a particular legal question or clause type.

            Delegation:
            - Delegate this task to "file-qa-agent".
            - Pass the sub_query exactly as provided.

            Expected result:
            - A set of files that likely contain relevant content for the sub_query.

            After the task:
            - If there was no active file set before, call add_file_ids with the returned file ids.
            - If there was already an active file set and this search is meant to narrow the scope, call update_file_ids with the new list of file ids.

            The goal is to make sure that later tasks only operate on relevant contracts.
            </intent_file_search>

            <intent_clause_extraction_and_analysis>
            Intent: Clause Extraction and Analysis

            Purpose:
            - To pull out concrete clause text that matches a legal question, along with references and structure.

            Delegation:
            - Delegate this task to "file-qa-agent".
            - Pass the sub_query exactly as provided.

            Expected result:
            - Complete verbatim clause text for all matching clauses.
            - Section numbers, headings, or identifiers where available.
            - File names and file identifiers for each clause.

            After the task:
            - If the purpose of this task is to filter which contracts are relevant, call update_file_ids so that only files containing matching clauses remain in the active file set.
            - Otherwise, leave the active file set unchanged and simply store the extraction results under the task_id.

            This step gives you the precise legal language that you will later quote and interpret in your final answer.
            </intent_clause_extraction_and_analysis>

            <intent_metadata_analysis>
            Intent: Metadata Analysis

            Purpose:
            - To answer questions that can be resolved purely from metadata, without needing clause text.
            Examples include:
            - counting how many contracts were signed in a given year,
            - computing an average contract value,
            - grouping contracts by jurisdiction or contract type.

            Delegation:
            - Delegate this task to "metadata-analysis-agent".
            - Pass the sub_query exactly as written.

            Usage rule:
            - Only use metadata analysis when the question can be answered without looking at clause content.
            - If any part of the question depends on clause wording, the plan should include a clause extraction step, and any calculation based on that should be treated as coordinated analysis, not as pure metadata.

            The reason for this separation is to keep pure data questions distinct from clause-based legal interpretation.
            </intent_metadata_analysis>

            <intent_summarization>
            Intent: Summarization

            Purpose:
            - To provide an overview of one or more contracts when the user explicitly asks for a summary.

            Delegation:
            - Delegate this task to "summarizer-agent".
            - Call "summarizer-agent" exactly once per summarization task.
            - Pass only the sub_query.

            Expected result:
            - A summary of the relevant contracts, according to the active file set and the sub_query.

            Special rules:
            - Even if no files are found, you still call "summarizer-agent" and use its output.
            - You do not write your own separate summary. You rely on the summarizer-agent result as the primary summary content.

            Later, when writing the final answer, you may format or lightly frame the summary, but you should not change its meaning.
            </intent_summarization>

            <intent_comparison>
            Intent: Comparison

            Purpose:
            - To compare clauses, obligations, or terms across two or more contracts.

            Delegation:
            - Delegate this task to "compare-agent".
            - Call "compare-agent" once per comparison task.
            - Pass the sub_query exactly as written.

            Expected result:
            - Structured comparison information that highlights similarities and differences between files, often referencing specific clauses.

            You will later use this information, together with extracted clauses, to build a comparison table and narrative explanation in your final answer.
            </intent_comparison>

            <intent_coordinated_analysis>
            Intent: Coordinated Analysis

            Purpose:
            This intent is used when the user's question requires legal reasoning or interpretation that depends on one or more clauses extracted earlier in the plan. Unlike metadata analysis or comparison tasks, coordinated analysis focuses on understanding what the contract language means, how it applies to the user's question, and what risks or obligations arise from it.

            When this intent is present, the coordinator is expected to:
            - read all clause extraction outputs from earlier tasks,
            - identify which clauses answer the user's original question,
            - quote the relevant clause text verbatim and in italics,
            - explain in clear legal English how the clauses answer the query,
            - describe any obligations, limitations, rights, or conditions that arise from those clauses,
            - highlight any risks or practical implications,
            - and finally give a complete, coherent legal conclusion.

            Delegation:
            - Do NOT delegate this task to any member.
            - Perform this reasoning yourself as the coordinator.

            How to execute this intent:
            1. Identify which clause(s) from earlier tasks provide the direct answer.
            2. Quote each relevant clause verbatim and in italics.
            3. After each quoted clause, provide a detailed, plain-English explanation describing:
            - what the clause means,
            - how it operates,
            - and how it answers the user's specific question.
            4. If multiple clauses are relevant, integrate them into a single clear narrative.
            5. Discuss any obligations, conditions, dependencies, or restrictions imposed by the clause.
            6. Highlight any risks to the user arising from that language (such as compliance, quality, supply, liability, or approval risks).
            7. Connect all reasoning back to the original user query.
            8. End with a clear, confident legal conclusion.

            This intent is used for:
            - identifying what a contract actually requires or provides,
            - interpreting clauses,
            - explaining definitions or rights,
            - determining what a party must or must not do,
            - analyzing obligations, limitations, or compliance issues,
            - or any question where the user asks “what does the contract say about X?” or “what is the API?” or “what is the obligation?” or “what happens under this clause?”

            This intent ensures that the coordinator gives a complete, human-like, legally sound answer rather than only quoting clauses.
            </intent_coordinated_analysis>

            <intent_web_search>
            Intent: Web Search

            Purpose:
            - To retrieve external information from the web when requested by the plan.

            Delegation:
            - Delegate this task to "web-searcher".
            - Pass the sub_query exactly as provided.

            You then store and later use the results as needed to answer the user's question, without disclosing the internal web-searcher agent.
            </intent_web_search>

            <intent_greeting>
            Intent: Greeting

            Purpose:
            - To handle simple conversational greetings or introductions.

            Delegation:
            - Do not delegate these tasks.
            - Answer the user directly, politely, and briefly, and then continue with the normal flow as needed.

            This keeps simple human interaction natural and lightweight.
            </intent_greeting>

            <clause_handling_and_citation_rules>
            When you receive clause text from a delegated task, treat it as the primary legal source.

            In your final answer to the user:

            1. Always quote clause text verbatim.
            2. Always wrap the entire clause in italics.
            3. Do not trim or shorten the clause.
            4. Do not paraphrase a clause when you are presenting it as the contract language.
            5. Do not insert ellipses inside the quoted clause.
            6. Whenever possible, also mention:
            - the section or clause number,
            - the heading,
            - and the file citation.

            Example pattern:
            - *"Full clause text exactly as returned."* from **[FILE_NAME](/preview?fileId=<file_guid>)**

            After quoting, you may then add your own legal explanation in plain text, outside the italics.
            </clause_handling_and_citation_rules>

            <file_citation_format>
            Whenever you refer to a file, always use this format:

            **[FILE_NAME](/preview?fileId=<file_guid>)**

            This format is mandatory. It ensures that the user can click and open the exact contract you are referring to.
            </file_citation_format>

            <general_response_style>
            Your final response to the user must:

            1. Sound like a senior legal advisor.
            2. Use plain but precise legal English.
            3. Be structured and easy to read, with paragraphs and, where useful, tables.
            4. Avoid internal jargon such as "task", "agent", "tool", or "JSON".
            5. Focus on what the contracts say and what that means for the user.

            You must not:
            - tell the user that you executed a plan,
            - mention that you delegated work to agents,
            - show any internal identifiers or technical details.

            To the user, you are simply describing what their contracts contain and what it means for them.
            </general_response_style>

            <list_format_for_files>
            Use this format when the user wants you to list, fetch, pull, filter, get, or enumerate files.

            1. Start with a sentence that states how many files match the request.
            2. Then provide a numbered list of file citations in the standard format.

            Rules:
            - If there are 10 or fewer matching files:
            - State the total count.
            - List all of them.

            - If there are more than 10 matching files:
            - State the total count.
            - List only the first 10, chosen in a sensible order, such as by relevance or date, if available.

            Example structure:
            There are 8 contracts that match your request. Below are the most relevant files:
            1. **[File One](/preview?fileId=abc)**
            2. **[File Two](/preview?fileId=xyz)**

            You do not need to add summaries or descriptions of the files in this format unless the user asks for them.
            </list_format_for_files>

            <count_and_metric_format>
            Use this format when the user asks for counts, sums, percentages, averages, minimums, maximums, or groupings.

            1. First, clearly state the metric:
            For example: "There are 8 contracts executed before 2010 that contain an assignment clause."

            2. Then list the relevant files as a numbered list using the standard file citation format.

            Rules:
            - If there are 10 or fewer relevant files, list them all.
            - If there are more than 10, state the total number and list the first 10.

            This format keeps the numerical answer clear while still showing which specific contracts are involved.
            </count_and_metric_format>

            <comparison_format>
            Use this format when the user asks you to compare terms, clauses, or obligations across contracts.

            1. Build a markdown table.
            - Each column is a file, with the header showing the file citation.
            - Each row is a specific item being compared, such as "Termination clause" or "Assignment".

            2. In each cell:
            - First, place the full clause text in italics.
            - Immediately after the clause, provide a short legal explanation in normal text.

            Example layout:

            | Item               | **[FILE_1](/preview?fileId=111)**                              | **[FILE_2](/preview?fileId=222)**                              |
            |--------------------|---------------------------------------------------------------|-----------------------------------------------------------------|
            | Termination clause | *"Full clause text from file 1."* Legal explanation.          | *"Full clause text from file 2."* Legal explanation.           |

            3. After the table:
            - Write a short narrative that explains the key similarities and differences.
            - End with a clear conclusion that answers the user's comparison question.

            This makes it easy for the user to see both the raw contract language and your analysis side by side.
            </comparison_format>

            <checklist_format>
            Use this format only when the user's original query specifies a checklist of requirements and wants to see which contracts satisfy which items.
            There can be single or multiple files relevant to the checklist.
                            
            For each file:

            1. Start with the file citation on a separate line:
            **[FILE_NAME](/preview?fileId=<file_guid>):**

            2. Then create a table with the following columns:
            - number,
            - checklist item,
            - status,
            - remarks.

            3. In the "status" column use only:
            - Yes,
            - No,
            - Partial,
            - Not Found.

            4. In the "remarks" column:
            - If there is a supporting clause, include the verbatim clause text in italics.
            - If there is no direct clause, include a short justification in italics.

            Example:

            | # | Checklist item            | Status  | Remarks                                    |
            |---|--------------------------|---------|--------------------------------------------|
            | 1 | Change of control clause | Yes     | *"Full clause text from the contract."*    |
            | 2 | Non-compete for 3 years | Partial | *"Clause covers one year, not three."*     |

            After processing all files, provide a short summary that explains overall compliance across the set of contracts.
            </checklist_format>

            <summarization_format>
            A summarization task is performed only when the user explicitly asks for a summary. In such cases:

            1. Use only the `summarizer-agent` output. Do not generate your own summary.
            2. You must preserve the content exactly as received from the `summarizer-agent`.
            3. If multiple files are summarized clsuter files as per their contract type(s) in tabular format.
            4. If single file is summarized for single or multiple contract types use paragraph + key points format.
            5. If follow_up text is present, include it exactly as given.
            6. If no files are found, still call `summarizer-agent` and present its output.
            7. Don't show contract type in headings or bold before file summaries.
            
            Example Structure for Tabular Summary (Multiple Files for Same Contract Type):
            | Key Details                     | **[FILE_NAME_1](/preview?fileId=file_guid_1)** | **[FILE_NAME_2](/preview?fileId=file_guid_2)** |
            |--------------------------------|-----------------------------------------------|-----------------------------------------------|
            | Key Point 1 from summary       | *“Full verbatim clause text”* — Explanation   | *“Full verbatim clause text”* — Explanation                          |
            | Key Point 2 from summary       | *“Full verbatim clause text”* — Explanation   | *“Full verbatim clause text”* — Explanation   |
            | ...                            | ...                                           | ...                                           |
            
            [Follow_up Question for User as per `summarizer-agent` output]
            </summarization_format>

            <calculation_and_formula_rules>
            When you present calculations, such as percentages or ratios:

            1. Do not use LaTeX or any mathematical markup.
            Do not use special math delimiters or dollar signs for formulas.

            2. Use only plain text and the following symbols:
            /, x, -, +, %, =

            3. Write the calculation in a simple and readable way.

            Example:
            Percentage of contracts with an assignment clause =
            (4 / 11) x 100 = 36.36%

            Always base these calculations on actual numbers returned by earlier tasks. Do not guess or make up values.
            </calculation_and_formula_rules>

            <no_results_format>
            If the plan runs successfully but no relevant files or clauses are found for the user request:

            1. State clearly that no relevant material was identified in the vault for this specific question.
            2. Do not list any files.
            3. Do not invent example clauses.
            4. Do not ask the user to upload more files.

            Your answer should be honest and simple. It is better to say that no relevant material was found than to fabricate content.
            </no_results_format>

            <error_handling_format>
            If an internal error occurs, for example if a delegated task fails or returns unusable results:

            1. Do not show technical details, stack traces, or error codes to the user.
            2. Instead, give a high level explanation such as:
            "I was not able to retrieve the relevant contract information to answer this question."

            3. If appropriate, you may suggest that the user narrows or rephrases the question, but you must not claim that certain clauses exist if you do not actually have them.

            The goal is to stay honest and user friendly without exposing internal technical issues.
            </error_handling_format>

            <final_answer_generation_guidelines>
            When you have executed all tasks and collected all results, you then prepare the final answer for the user.

            In this step:

            1. Use all relevant task outputs that help answer the original user question.
            2. Refer to contracts using the standard file citation format:
            **[FILE_NAME](/preview?fileId=<file_guid>)**

            3. Quote every important clause, section, or phrase in italics and verbatim.
            4. After each clause, give a clear legal explanation of what it means and how it affects the user.
            5. Explain how the clauses and data connect to the user's original question, so the reasoning is transparent.
            6. Organize your answer with clear paragraphs and, where helpful, tables.
            7. Finish with a clear conclusion that directly addresses what the user wanted to know or decide.

            At no point in the final answer should you mention:
            - agents,
            - tasks,
            - tools,
            - JSON plans,
            - internal execution details.

            The user should only see a polished, expert legal explanation based on their contracts.
            </final_answer_generation_guidelines>
        """),
        markdown=True,
        session_id=str(session_id),
        db=MySQLDb(
            db_url=db_url,
            db_schema=AGENT_INTEL_DB,
        ),
        add_history_to_context=True,
        num_history_runs=3,
        add_location_to_context=True,
        read_team_history=True,
        debug_mode=False,
        add_member_tools_to_context=False,
        add_datetime_to_context=True,
        timezone_identifier="Etc/UTC",
        telemetry=False,
    )

        return coordinator

    async def _generate_related_questions(user_query: str, response_text: str):
        related_question_generator = Agent(
        name="related_question_generator_agent",
        model=OpenAIChat(id="gpt-4o-mini", temperature=0.2, top_p=1),
        description="You are an intelligent assistant that generates logical and context-aware follow-up questions from a user's query and the provided answer.",
        instructions=dedent("""
            You are ChatGPT, a large language model trained by OpenAI.  
            Knowledge cutoff: 2024-06  

            <personality_and_style>
            - Clear, precise, and professional.
            - Slightly engaging and encouraging, but never informal.
            - Always logical, contextual, and consistent in tone.
            </personality_and_style>

            <core_behavior>
            - You will receive two inputs: a <user_query> and an <answer>.  
            - Your task is to generate **exactly 3 high-quality follow-up questions** that logically flow from both.  

            <question_guidelines>
            The questions should:
            - Explore clarifications, missing details, risks, obligations, conditions, exceptions, timelines, or practical implications.
            - Encourage deeper understanding without repeating the same information.
            - Remain general enough to apply to multiple contracts or contexts (avoid specific file names, parties, or identifiers).

            The questions should NOT:
            - Simply restate what is already covered in the answer.
            - Ask for a summary, overview, or recap.
            - Be vague or meta (e.g., “How can this help later?”).
            - Contain the words: "summarize", "summary", "overview", "recap", "gist", "outline", etc.
            
            <special_cases>
            - If the user query is just a greeting or appreciation (e.g., “Hi”, “Hello”, “Thanks”), return an empty list: []
            - If the answer is empty or indicates “User NOT Selected Files for Summary”, return an empty list: []
            </special_cases>

            <output_format>
            Always return output strictly in the following JSON format:
            {
                "questions": [
                    {"question": "First clarification or follow-up question"},
                    {"question": "Second probing or contextual question"},
                    {"question": "Third practical or implication-based question"}
                ]
            }

            If no questions are generated, return:
            {
                "questions": []
            }
            """),
            expected_output=dedent("""
            {
                "questions": [
                    {"question": "First clarification or follow-up question"},
                    {"question": "Second probing or contextual question"},
                    {"question": "Third practical or implication-based question"}
                ]
            }
            """),
            telemetry=False,
            debug_mode=False
        )
        
        prompt = dedent(f"""
        <user_query>
        {user_query}
        </user_query>

        <answer>
        {response_text}
        </answer>

        Generate exactly 3 thoughtful follow-up questions based on both the user's query and the provided answer.
        Return them strictly in JSON format as per the instruction schema.
        """)

        response = await related_question_generator.arun(input=prompt, telemetry=False)
        related_questions = json.loads(response.content).get("questions", [])
        return related_questions


async def update_dsocket_server(
    ws,
    connection_id,
    request_id,
    client_id,
    session_id,
    chat_id,
    chunk_order,
    chunk_answer,
    stream_completed,
    related_questions=[],
    user_id=None,
    org_id=None,
):
    data = {
        "success": True,
        "error": "",
        "data": {
            "stream_type": "GetAnswer",
            "session_id": session_id,
            "chat_id": chat_id,
            "connection_id": connection_id,
            "request_id": request_id,
            "chunk_order": chunk_order,
            "chunk_answer": chunk_answer,
            "stream_completed_status": stream_completed,
            "user_id": user_id,
            "org_id": org_id,
            "client_id": client_id,
            "enable_agent": True,  # Assuming this is always true for the agent
        },
    }

    if stream_completed and related_questions is not None:
        data["data"]["related_questions"] = related_questions

    await ws.send(json.dumps(data))

    await save_chunk_to_dynamo(
        connection_id,
        str(chat_id),
        chunk_order,
        chunk_answer,
        request_id,
        stream_completed,
        related_questions,
        client_id,
        enable_agent=True,
    )

def make_pick_message():
    """
    Factory that returns a fresh `pick_message` function
    with its own _last_message tracker each time it's created.
    """
    _last_message = None

    def pick_message(pool):
        nonlocal _last_message
        choices = [msg for msg in pool if msg != _last_message]
        message = random.choice(choices) if choices else random.choice(pool)
        _last_message = message
        return message

    return pick_message

@mlflow.trace(name="Team Events")
def get_event_details(chunk_obj):
    try:
        if hasattr(chunk_obj, "to_dict"):
            return chunk_obj.to_dict()

        return chunk_obj
    except Exception as e:
        print(f"Error getting event details: {e}")
        return {}

@mlflow.trace(name="Get Answer From Agent")
async def get_answer_from_agent(user_query, org_id, user_id, file_ids=[], session_id=None, client_id=None, chat_id=None, is_related_questions_required=True, connection_id=None, request_id=None, history=[], tag_ids=[], logger=None):
    
    # with file selection task planning prompt is kept seprated
    if file_ids: 
        task_planner = await DeepThink._initialize_task_planner_with_file_selection(user_query, file_ids, session_id, chat_id, user_id, org_id, tag_ids)
    else:
        task_planner = await DeepThink._initialize_task_planner_without_file_selection(user_query, file_ids, session_id, chat_id, user_id, org_id, tag_ids)

    last_asst = next((m for m in reversed(history) if m["role"] == "assistant"), None)

    messages = []

    if last_asst:
        messages.append(
            Message(
                role="assistant",
                content=[{
                    "type": "text",
                    "text": (
                        "The following is the execution result of your previous plan:\n\n"
                        f"{last_asst['content']}\n\n"
                    )
                }]
            )
        )

    messages.append(
        Message(
            role="user",
            content=[{"type": "text", "text": user_query}]
        )
    )



    logger.info("LOG - Final Message list sent to DeepThink: %s", messages)
    task_planner_response = await task_planner.arun(input=messages, session_id=str(session_id), telemetry=False)
    logger.info("Task Planner Response for DeepThink: %s", {"task_planner_response": task_planner_response.content})

    try:
        task_planner_response_json = json.loads(task_planner_response.content)
    except Exception as e:
        logger.error("Task Planner Response is not valid JSON: %s", {"error": str(e)})
        raise ValueError("Task Planner Response is not valid JSON") from e

    coordinator = await DeepThink._initialize_coordinator(user_query, [], session_id, chat_id, user_id, org_id, tag_ids)

    try:
        coordinator_session_state = coordinator.get_session_state(str(session_id))
    
    except Exception as e:
        print("DEBUG PRINT - Error fetching coordinator session state:", str(e))
        coordinator_session_state = {
            "files": {},
            "files_matched": 0,
            "file_ids": [],
            "follow_up_count": 0,
            "temp_contract_types": [],
            "files_selected": False
        }

    # session_state/ json ---> pass in File Search tool   
    json_plan = await file_search(task_planner_response_json, file_ids, chat_id, user_id, org_id, tag_ids, coordinator_session_state, history)
    logger.info("JSON Plan for DeepThink: %s", {"json_plan": json_plan})

    coordinator_session_state["file_ids"] = json_plan.get("file_ids", [])


    final_answer_parts = []
    final_answer = ""
    chunk_order = 1
    team_run_completed = False
    related_questions = []
    final_sources = ""
    # Create a fresh picker per call
    pick_message = make_pick_message()


    user_prompt = f"""
    Use the following structured plan to generate the final answer for the user:
    {json.dumps(json_plan, indent=4)}

    Follow these rules when producing the response:

    1. Strictly follow the plan as given. Do not add, remove, or reorder steps.
    2. Write a clear, fluent, and detailed answer that fully addresses the user's original question.
    3. Do not reveal any internal processes, planning logic, or tool calls in the response.
    4. You may include document references only using links in this format only:
    `[FILE_NAME](/preview?fileId=<file_guid>)`.
    5. Provide a comprehensive legal explanation, including:
    - Inline citations to the relevant contractual clauses.
    - An explanation of how each cited clause answers the user's original question.
    6. Identify and explain any risks, obligations, or compliance requirements that arise from the cited clauses.
    """


    async with websockets.connect(WEBSOCKET_URL, ping_interval=520, ping_timeout=520) as ws:
        logger.info(_log_message("Connected to WebSocket server", "GET_ANSWER_FROM_AGENT", MODULE_NAME))
        async for chunk in coordinator.arun(input=user_prompt, stream=True, session_id=str(session_id), telemetry=False, show_full_reasoning=True, stream_intermediate_steps=True, session_state=coordinator_session_state):
            event = getattr(chunk, "event", "")
            if not event:
                continue

            if event == TeamRunEvent.tool_call_started:
                get_event_details(chunk)
                tool_message = f"Processing: {pick_message(TOOL_MESSAGES)}"
                await update_dsocket_server(
                    ws,
                    connection_id,
                    request_id,
                    client_id,
                    session_id,
                    chat_id,
                    chunk_order,
                    tool_message,
                    False,
                    related_questions,
                    user_id,
                    org_id,
                )
                chunk_order += 1
                logger.info(_log_message(f"Tool message: {tool_message}", "GET_ANSWER_FROM_AGENT", "DeepThink"))

            elif event == TeamRunEvent.tool_call_completed:
                get_event_details(chunk)
                tool_result_message = f"Processing: {pick_message(TOOL_MESSAGES)}"
                await update_dsocket_server(
                    ws,
                    connection_id,
                    request_id,
                    client_id,
                    session_id,
                    chat_id,
                    chunk_order,
                    tool_result_message,
                    False,
                    related_questions,
                    user_id,
                    org_id,
                )
                chunk_order += 1
                logger.info(_log_message(f"Tool completion message: {tool_result_message}", "GET_ANSWER_FROM_AGENT", "DeepThink"))

            elif event == TeamRunEvent.run_content:
                # get_event_details(chunk)
                # Stream intermediate content
                await update_dsocket_server(
                    ws,
                    connection_id,
                    request_id,
                    client_id,
                    session_id,
                    chat_id,
                    chunk_order,
                    chunk.content,
                    False,
                    related_questions,
                    user_id,
                    org_id,
                )
                final_answer_parts.append(chunk.content)
                chunk_order += 1

            elif event == TeamRunEvent.run_completed:
                get_event_details(chunk)
                # Final response segment
                team_run_completed = True
                if not final_answer_parts:
                    final_answer = chunk.content or ""
                else:
                    final_answer = "".join(final_answer_parts)
                logger.info(_log_message(f"Final Answer Received: {final_answer}", "GET_ANSWER_FROM_AGENT", "DeepThink"))

            else:
                # get_event_details(chunk)
                continue  # skip other events

        await update_dsocket_server(
            ws,
            connection_id,
            request_id,
            client_id,
            session_id,
            chat_id,
            chunk_order,
            "\n",
            False,
            related_questions,
            user_id,
            org_id,
        )
        final_answer_parts.append("\n")
        chunk_order += 1

        # --- Finalization logic ---
        if not final_answer:
            final_answer = "".join(final_answer_parts) or "Processing completed but no response generated."

        # Helper functions
        @mlflow.trace(name="Check Sources Pattern")
        def has_sources_pattern(text: str) -> bool:
            pattern = r"(---\s*\n\s*)?\*\*Sources?:\*\*[\s\S]*"
            return bool(re.search(pattern, text))
        
        @mlflow.trace(name="Extract Markdown Hyperlinks")
        def extract_markdown_hyperlinks(text: str):
            pattern = r"(?:\*\*)?\[([^\]]+)\]\(([^)\s]+)\)(?:\*\*)?"
            matches = re.findall(pattern, text)
            seen, results = set(), []
            for filename, link in matches:
                if link not in seen:
                    seen.add(link)
                    results.append(f"[{filename}]({link})")
            return results

        # Source attachment
        if not has_sources_pattern(final_answer):
            sources = extract_markdown_hyperlinks(final_answer)
            if sources:
                formatted_sources = "\n".join(f"- {s}" for s in sources if s)

                final_sources += f"\n\n---\n\n**Sources:**\n{formatted_sources}"
                final_answer += final_sources
                await update_dsocket_server(
                    ws,
                    connection_id,
                    request_id,
                    client_id,
                    session_id,
                    chat_id,
                    chunk_order,
                    final_sources,
                    False,
                    related_questions,
                    user_id,
                    org_id,
                )
                chunk_order += 1
                logger.info(_log_message(f"Sources added to final output.", "GET_ANSWER_FROM_AGENT", "DeepThink"))

        # Related questions generation (post completion)
        if is_related_questions_required:
            related_questions = await DeepThink._generate_related_questions(user_query, final_answer) if is_related_questions_required else []
        else:
            related_questions = []

        # Final WebSocket update
        await update_dsocket_server(
            ws,
            connection_id,
            request_id,
            client_id,
            session_id,
            chat_id,
            chunk_order,
            "",
            True,
            related_questions,
            user_id,
            org_id,
        )

        logger.info(
            _log_message(
                f"Final response sent — last chunk_order: {chunk_order} | Processing: {team_run_completed}",
                "GET_ANSWER_FROM_AGENT",
                "DeepThink",
            )
        )
        await ws.close()

    # --- Return structured response ---
    return {
        "success": True,
        "error": "",
        "chunk_order": chunk_order,
        "data": {
            "file_ids": [],
            "answers": [
                {
                    "answer": final_answer,
                    "page_no": 1,
                    "paragraph_no": 1,
                    "score": 1,
                }
            ],
            "related_questions": related_questions,
        },
    }