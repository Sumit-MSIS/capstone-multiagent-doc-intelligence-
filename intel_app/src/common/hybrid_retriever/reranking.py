from together import Together
from src.config.base_config import config
from src.common.logger import _log_message
from pinecone import Pinecone
import mlflow
mlflow.openai.autolog()

MODULE_NAME = "reranking.py"

@mlflow.trace(name="Rerank Documents")
def rerank_documents(query: str, pinecone_results, logger):
    """Rerank Pinecone results using Together AI's relevance model."""
    function_name = "rerank_documents"

    try:
        matches = pinecone_results # .get("matches", [])
        if not matches:
            logger.warning(_log_message(f"No matches found for query: '{query}'", function_name, MODULE_NAME))
            return pinecone_results

        texts, index_to_match = [], {}

        for idx, match in enumerate(matches):
            text = match["metadata"].get("text", "")
            if text:
                texts.append(text)
                index_to_match[idx] = match

        if not texts:
            logger.warning(_log_message("No valid texts available for reranking.", function_name, MODULE_NAME))
            return pinecone_results

        client = Together(api_key=config.TOGETHER_API_KEY)
        top_n = max(1, int(len(texts) * 0.75)) if len(texts) > 5 else len(texts)
        # top_n = 3

        logger.debug(_log_message(f"Sending {len(texts)} texts for reranking.", function_name, MODULE_NAME))

        response = client.rerank.create(
            model="Salesforce/Llama-Rank-V1",
            query=query,
            documents=texts,
            top_n=top_n
        )

        logger.debug(_log_message("Together AI response received.", function_name, MODULE_NAME))

        # Sort reranked results by relevance score
        reranked_results = sorted(response.results, key=lambda x: x.relevance_score, reverse=True)

        logger.debug(_log_message("Reranked results sorted successfully.", function_name, MODULE_NAME))

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

        logger.info(_log_message(f"Successfully reranked {len(final_matches)} results for query: '{query}'.", function_name, MODULE_NAME))

        return response

    except Exception as e:
        logger.error(_log_message(f"Error in {function_name}: {e}", function_name, MODULE_NAME))
        return {
           "matches": pinecone_results
        }
    

@mlflow.trace(name="Pinecone Rerank Documents")
def pinecone_rerank_documents(query: str,inference_model, pinecone_api_key, pinecone_results, logger):
    """Rerank Pinecone results using Pinecone's cohere"""
    function_name = "pinecone_rerank_documents"

    try:
        pinecone_client = Pinecone(api_key=pinecone_api_key)
        # Prepare docs for reranker
        results_for_reranker = []
        id_lookup = {}

        # below step is required because rerankers need flattened input does not take nested inputs
        for idx, match in enumerate(pinecone_results):
            text = match.get("metadata", {}).get("text", "")
            if text:
                doc = {
                    "id": match["id"],         # keep id for mapping later
                    "chunk_text": text         # rerank will use this
                }
                results_for_reranker.append(doc)
                id_lookup[match["id"]] = match  # store original match by id

        if not results_for_reranker:
            logger.warning(_log_message("No valid texts available for reranking.", function_name, MODULE_NAME))
            return pinecone_results

        logger.debug(_log_message(f"Sending {len(results_for_reranker)} docs for reranking.", function_name, MODULE_NAME))


        top_n = max(1, int(len(results_for_reranker) * 0.75))  # Ensure at least one result
        # print(top_n)
        # top_n = 3

        logger.debug(_log_message(f"Reranking top n chunks are {top_n}.", function_name, MODULE_NAME))

        if inference_model == "cohere-rerank-3.5":
            # cohere does not support parameters parameter, max token limit is 40k 
            ranked_results = pinecone_client.inference.rerank(
            model=inference_model,
            query=query,
            documents=results_for_reranker, # documents = takes list of dictionaries as input [{},{}]
            top_n=top_n,
            rank_fields=["chunk_text"],
            return_documents=True
        )
        else:
            ranked_results = pinecone_client.inference.rerank(
                model=inference_model,
                query=query,
                documents=results_for_reranker, # documents = takes list of dictionaries as input [{},{}]
                top_n=top_n,
                rank_fields=["chunk_text"],
                return_documents=True,
                parameters={
                    "truncate": "END" 
                }
            )
        logger.info(_log_message(f"received response from cohere {ranked_results}","pinecone_rerank_documents",MODULE_NAME))
        # Build final matches sorted by reranker score
        final_matches = []
        for item in ranked_results.data:  # the .data holds reranked docs
            doc_id = item.document["id"]
            if doc_id in id_lookup:
                match_copy = id_lookup[doc_id].copy()
                match_copy["score"] = item.score  # Add reranker score
                final_matches.append(match_copy)

        # Sort final matches by rerank_score (descending)
        final_matches.sort(key=lambda x: x["score"], reverse=True)

        response = {
            "matches": final_matches
        }

        logger.info(_log_message(f"Successfully reranked {len(final_matches)} results for query: '{query}'.", function_name, MODULE_NAME))

        return response

    except Exception as e:
        logger.error(_log_message(f"Error when Reranking with {inference_model}: {e}", function_name, MODULE_NAME))
        return {
           "matches": pinecone_results
        }