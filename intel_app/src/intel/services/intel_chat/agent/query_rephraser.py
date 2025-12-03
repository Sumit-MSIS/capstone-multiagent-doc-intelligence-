
from openai import AsyncOpenAI
import json
import os
import mlflow

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

module_name = "utils.py"
module_name = "query_rephraser"

def _log_message(message: str, function_name: str, module_name: str) -> str:
    return f"[function={function_name} | module={module_name}] - {message}"


@mlflow.trace(name="Rephrase Search Text")
async def rephrase_search_text(search_text: str):
    system_prompt = """
    You are ChatGPT, an assistant that reformulates user queries into optimized legal-document retrieval phrases.

    ## Task
    Given a user query, generate **two reformulated outputs** in valid JSON:
    1. **Keyword Search Query** — compact, Boolean-friendly expression for keyword or hybrid search engines.
    2. **Semantic Search Query** — rich, clause-level phrasing for semantic embedding or reranking.

    ## Guidelines

    ### 1. Keyword Search Query
    - Use **Boolean operators** (`AND`, `OR`) with quoted multiword terms.
    - Include key **legal concepts, jurisdictions, or parties** from the query.
    - Add **contract language synonyms** (e.g., “governing law” OR “choice of law”).
    - Avoid filler like “find”, “show”, “documents”, unless meaningful.
    - Optimize for both keyword and semantic overlap (hybrid search).
    - Avoid generic terms like "agreement", "contract".

    ### 2. Semantic Search Query
    - Always start with **"Clause"** if referring to a clause, or **"Provision"** if broader.
    - Write in **formal contract style**, describing legal purpose or effect.
    - Reflect intent (obligation, right, restriction, condition, jurisdiction, etc.).
    - Include relevant context (e.g., agreement type, state, party).
    - Avoid generic or conversational phrasing.

    ## Output Format
    Output strictly as valid JSON (no markdown, no code fences):
    {
        "keyword_search_query": "string",
        "semantic_search_query": "string"
    }

    ## Examples

    ### Example 1
    User Query: "find upsell or cross sell agreements"
    {
        "keyword_search_query": "upsell OR 'up-sell' OR 'cross-sell' OR 'cross sell' OR 'bundle sale' OR 'complementary sale'",
        "semantic_search_query": "Clause outlining terms and conditions for upselling or cross-selling products or services."
    }

    ### Example 2
    User Query: "pull up the contracts having subleasing clause"
    {
        "keyword_search_query": "sublease OR sub-leasing OR subletting",
        "semantic_search_query": "Clause permitting or governing subleasing or subletting of leased property."
    }

    ### Example 3
    User Query: "Which account control agreement is governed by Texas laws"
    {
        "keyword_search_query": "'account control agreement' AND ('Texas law' OR 'State of Texas' OR 'laws of Texas' OR 'Texas jurisdiction' OR 'choice of law Texas' OR 'governing law Texas')",
        "semantic_search_query": "Clause specifying that the account control agreement is governed by and construed in accordance with the laws of the State of Texas."
    }

    ### Example 4
    User Query: "show confidentiality clause about employee data"
    {
        "keyword_search_query": "confidentiality OR 'non-disclosure' AND ('employee data' OR 'personnel information' OR 'HR data')",
        "semantic_search_query": "Clause defining confidentiality obligations regarding employee or personnel data."
    }

    ### Example 5
    User Query: "Get me the contract between Wells Fargo and Interwoven"
    {
        "keyword_search_query": "'Wells Fargo' AND 'Interwoven'",
        "semantic_search_query": "Provision identifying a contract between Wells Fargo and Interwoven."
    }

    ### Example 6
    User Query: "Find contract governed by the laws of Delaware"
    {
        "keyword_search_query": "('Delaware law' OR 'State of Delaware' OR 'laws of Delaware' OR 'choice of law Delaware' OR 'governing law Delaware')",
        "semantic_search_query": "Clause specifying that the contract is governed by and construed under the laws of the State of Delaware."
    }

    ### Example 7
    User Query: "who is john martin?"
    {
        "keyword_search_query": "'John Martin'",
        "semantic_search_query": "Clause mentioning John Martin"
    }
    """

    user_prompt = f"User Query: {search_text.strip()}"

    response = await client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        temperature=0,
        top_p=1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    llm_answer = response.choices[0].message.content

    try:
        parsed = json.loads(llm_answer)
        keyword_query = parsed["keyword_search_query"]
        semantic_query = parsed["semantic_search_query"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Invalid JSON from model: {llm_answer}") from e

    return keyword_query, semantic_query, semantic_query
