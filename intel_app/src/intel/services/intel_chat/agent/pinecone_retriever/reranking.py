from together import Together, AsyncTogether
import os
import mlflow
mlflow.openai.autolog()

MODULE_NAME = "reranking.py"

@mlflow.trace(name="Async Rerank Documents")
async def rerank_documents_async(query: str, pinecone_results):
    """
    Async reranking using Together AI.
    Preserves EXACT same functionality and output of your existing sync version.
    """

    function_name = "rerank_documents_async"

    try:
        matches = pinecone_results
        if not matches:
            return pinecone_results

        texts, index_to_match = [], {}

        # Collect text + index mapping
        for idx, match in enumerate(matches):
            text = match["metadata"].get("text", "")
            if text:
                texts.append(text)
                index_to_match[idx] = match

        if not texts:
            return pinecone_results

        # SAME behavior as before
        top_n = 5

        client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"))

        # -- ASYNC rerank call --
        response = await client.rerank.create(
            model="Salesforce/Llama-Rank-V1",
            query=query,
            documents=texts,
            top_n=top_n
        )

        # Sort by relevance
        reranked_results = sorted(
            response.results,
            key=lambda x: x.relevance_score,
            reverse=True
        )

        # Map scores back to Pinecone results
        final_matches = []
        for item in reranked_results:
            idx = item.index
            match = index_to_match.get(idx)
            if match:
                match["score"] = item.relevance_score
                final_matches.append(match)

        final_matches.sort(key=lambda x: x["score"], reverse=True)

        return {"matches": final_matches}

    except Exception as e:
        error = f"Error in {MODULE_NAME}::{function_name} - {str(e)}"
        raise Exception(error)



@mlflow.trace(name="Rerank Documents")
def rerank_documents(query: str, pinecone_results):
    """Rerank Pinecone results using Together AI's relevance model."""
    function_name = "rerank_documents"

    try:
        matches = pinecone_results # .get("matches", [])
        if not matches:
            return pinecone_results

        texts, index_to_match = [], {}

        for idx, match in enumerate(matches):
            text = match["metadata"].get("text", "")
            if text:
                texts.append(text)
                index_to_match[idx] = match

        if not texts:
            return pinecone_results

        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        # top_n = max(1, int(len(texts) * 0.75)) if len(texts) > 5 else len(texts)
        top_n = 7
        # top_n = 3


        response = client.rerank.create(
            model="Salesforce/Llama-Rank-V1",
            query=query,
            documents=texts,
            top_n=top_n
        )


        # Sort reranked results by relevance score
        reranked_results = sorted(response.results, key=lambda x: x.relevance_score, reverse=True)


        # Map reranked indexes back to original Pinecone matches
        final_matches = []
        for item in reranked_results:
            idx = item.index
            match = index_to_match.get(idx)
            if match:
                match["score"] = item.relevance_score  # Updating score from reranker
                final_matches.append(match)

        # pinecone_results["matches"] = final_matches
        # Sort final matches by rerank_score (descending)
        final_matches.sort(key=lambda x: x["score"], reverse=True)
        
        response = {
            "matches": final_matches
        }


        return response

    except Exception as e:
        error = f"Error in {MODULE_NAME}::{function_name} - {str(e)}"
        raise Exception(error)

        