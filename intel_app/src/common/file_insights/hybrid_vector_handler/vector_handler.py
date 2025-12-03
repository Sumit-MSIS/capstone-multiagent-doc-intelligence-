import time
from openai import OpenAI
from pinecone import Pinecone
from src.config.base_config import config
from src.common.logger import _log_message
import mlflow
from concurrent.futures import ThreadPoolExecutor
from src.common.db_connection_manager import DBConnectionManager
from src.common.file_insights.hybrid_vector_handler.bm25_encoder import CustomBM25Encoder
from opentelemetry import context as ot_context
import requests
from requests.exceptions import RequestException
import json
import threading
from src.common.llm.factory import get_llm_client

mlflow.openai.autolog()

module_name = "vector_handler.py"
CONTEXTUAL_RAG_DB = config.CONTEXTUAL_RAG_DB
BM25_CORPUS_DB_TABLE = config.BM25_CORPUS_DB_TABLE
GET_BM25_ORG_AVGDL_URL = config.GET_BM25_ORG_AVGDL_URL
# Initialize OpenAI client
OPENAI_API_KEY = config.OPENAI_API_KEY

client = get_llm_client(async_mode=False)


@mlflow.trace(name="Generate Single Embedding")
def generate_single_embedding(text):
    """Generate embedding for a single text chunk using OpenAI directly."""
    try:

        response = client.embeddings.create(
            model=config.OPENAI_EMBEDING_MODEL_NAME,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"Error generating embedding: {e}")


@mlflow.trace(name="Generate Embeddings")
def generate_embeddings(avgdl_value, org_id, chunks, logger):
    """Generate dense and sparse embeddings for text chunks in parallel."""
    try:
        logger.info(_log_message(
            f"Processing {len(chunks)} chunks in parallel",
            "generate_embeddings", module_name))
        

        bm25_encoder = CustomBM25Encoder()
        sparse_vectors = bm25_encoder.custom_encode_documents(avgdl_value, chunks)

        # Generate dense embeddings in parallel
        def map_with_context(executor: ThreadPoolExecutor, fn, iterable):
            """Execute a function in parallel across multiple inputs, preserving OpenTelemetry context."""
            parent_ctx = ot_context.get_current()

            def wrapped(item):
                token = ot_context.attach(parent_ctx)
                try:
                    return fn(item)
                finally:
                    ot_context.detach(token)

            return list(executor.map(wrapped, iterable))
        
        # Use ThreadPoolExecutor to generate embeddings in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on rate limits
            logger.info(_log_message(
                f"Starting parallel embedding generation with {executor._max_workers} workers",
                "generate_embeddings", module_name))
            
            dense_vectors = map_with_context(executor, generate_single_embedding, chunks)
        
        logger.info(_log_message(
            f"Successfully generated embeddings for all {len(chunks)} chunks in parallel",
            "generate_embeddings", module_name))
        
        return dense_vectors, sparse_vectors
        
    except Exception as e:
        logger.error(_log_message(
            f"Error generating embeddings: {str(e)}",
            "generate_embeddings", module_name))
        raise RuntimeError(f"Error generating embeddings: {e}")
    

@mlflow.trace(name="Upsert and Check Vector Count")
def upsert_vectors(index, vectors, namespace, logger, batch_size=100):
    """Upsert vectors into Pinecone in batches and ensure they are added correctly."""
    try:
        initial_count = index.describe_index_stats().get('namespaces', {}).get(namespace, {}).get('vector_count', 0)
        logger.info(_log_message(
            f"[{namespace}] Initial vector count: {initial_count}", 
            "upsert_vectors", module_name))

        total_vectors = len(vectors)
        logger.info(_log_message(
            f"[{namespace}] Total vectors to upsert: {total_vectors}", 
            "upsert_vectors", module_name))

        for i in range(0, total_vectors, batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            logger.info(_log_message(
                f"[{namespace}] Upserted batch {i//batch_size + 1} with {len(batch)} vectors.", 
                "upsert_vectors", module_name))
            time.sleep(0.25)

        # Retry to verify upsert success
        expected_min = initial_count + 1
        for attempt in range(1, 11):
            time.sleep(5)
            current_count = index.describe_index_stats().get('namespaces', {}).get(namespace, {}).get('vector_count', 0)
            logger.info(_log_message(
                f"[{namespace}] Retry {attempt}: Current = {current_count}, Expected >= {expected_min}", 
                "upsert_vectors", module_name))
            if current_count >= expected_min:
                break

        logger.info(_log_message(
            f"[{namespace}] Vectors upserted successfully. Final count: {current_count}", 
            "upsert_vectors", module_name))
        return
    except Exception as e:
        logger.error(_log_message(
            f"[{namespace}] Error during upsert: {str(e)}", 
            "upsert_vectors", module_name))
        raise

@mlflow.trace(name="Upsert Text Chunks")
def upsert_text_chunks(chunks, file_id, user_id, org_id, file_name, avgdl_value, tag_ids, logger):
    """Process and upsert text chunks into Pinecone."""
    try:
        namespace = f"org_id_{org_id}#"
        total_no_of_chunks = len(chunks)
        logger.info(_log_message(
            f"[{namespace}] Starting upsertion of {total_no_of_chunks} chunks from file {file_name} (file_id: {file_id}) (tag_ids: {tag_ids})",
            "upsert_text_chunks", module_name))

        # Generate embeddings individually
        dense_vectors, sparse_vectors = generate_embeddings(avgdl_value, org_id, chunks, logger)

        def map_with_context(executor: ThreadPoolExecutor, fn, iterable):
            """Execute a function in parallel across multiple inputs, preserving OpenTelemetry context."""
            parent_ctx = ot_context.get_current()

            def wrapped(item):
                token = ot_context.attach(parent_ctx)
                try:
                    return fn(item)
                finally:
                    ot_context.detach(token)

            return list(executor.map(wrapped, iterable))

        @mlflow.trace(name="Build Vectors")
        def build_vector(i, vector_type):
            base_metadata = {
                "file_id": file_id,
                "file_name": file_name,
                "user_id": user_id,
                "org_id": org_id,
                "text": chunks[i],
                "chunks_count": total_no_of_chunks,
                "char_length": len(chunks[i]),
                "tag_ids": tag_ids,
                "page_no": i + 1
            }

            if vector_type not in ["sparse", "dense"]:
                return ValueError(f"Incorrect Vector Type: {vector_type}")
            
            if vector_type == "sparse":
                sparse = sparse_vectors[i]
                if not sparse or "indices" not in sparse or "values" not in sparse or len(sparse["indices"]) == 0:
                    logger.error(_log_message(
                        f"[org_id: {org_id}, file_id: {file_id}] Sparse vector missing or malformed for chunk {i+1} | Spare Vector: {sparse}",
                        "build_vector", module_name))
                    raise ValueError(f"Sparse vector missing or malformed for chunk {i+1} | Spare Vector: {sparse}")
                return {
                    "id": f"{org_id}#{file_id}#{i + 1}",
                    "values": [],  # Required field even for sparse-only vectors
                    "sparse_values": sparse,
                    "metadata": base_metadata
                }

            if vector_type == "dense":
                return {
                    "id": f"{org_id}#{file_id}#{i + 1}",
                    "values": dense_vectors[i],
                    "metadata": base_metadata
                }

        # Create sparse and dense vectors
        with ThreadPoolExecutor() as executor:
            sparse_list = list(filter(None, map_with_context(executor, lambda i: build_vector(i, "sparse"), range(total_no_of_chunks))))
            dense_list = list(filter(None, map_with_context(executor, lambda i: build_vector(i, "dense"), range(total_no_of_chunks))))

        

        # Upsert both sparse and dense vectors
        if sparse_list and dense_list:
            sparse_list = [v for v in sparse_list if v and "spare_values" in v and v["sparse_values"] and len(v["sparse_values"].get("indices", [])) > 0]
            dense_list = [v for v in dense_list if v]
            sparse_index = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX_SPARSE)
            logger.info(_log_message(
                f"[{namespace}] Starting Sparse upsertion of {total_no_of_chunks} chunks from file {file_name} (file_id: {file_id})",
                "upsert_text_chunks", module_name))
            upsert_vectors(sparse_index, sparse_list, namespace, logger)
            logger.info(_log_message(
                f"[{namespace}] Sparse upsertion completed. Proceeding with Dense upsertion - {len(sparse_list)} valid sparse vectors out of {total_no_of_chunks} chunks.",
                "upsert_text_chunks", module_name))

            # logger.info(_log_message(
            #     f"First 3 sparse vectors: {sparse_list[:3]}",
            #     "upsert_text_chunks", module_name))

            logger.info(_log_message(
                f"[{namespace}] Starting Dense upsertion of {total_no_of_chunks} chunks from file {file_name} (file_id: {file_id})",
                "upsert_text_chunks", module_name))
            dense_index = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX)
            upsert_vectors(dense_index, dense_list, namespace, logger)
        else:
            msg = f"[{namespace}] No valid vectors generated to upsert."
            logger.warning(_log_message(msg, "upsert_text_chunks", module_name))
            raise ValueError(msg)

        return

    except Exception as e:
        logger.error(_log_message(
            f"[org_id: {org_id}, file_id: {file_id}] Error in upsert_text_chunks: {str(e)}",
            "upsert_text_chunks", module_name))
        raise

@mlflow.trace(name="Process and Upsert Chunks")
def process_and_upsert(org_id, user_id, file_id, file_name, cleaned_chunks,all_chunks_tf_sum, tag_ids, logger):
    """Clean text chunks and initiate the upsert process."""
    try:
        logger.info(_log_message(
            f"[org_id: {org_id}, file_id: {file_id}] Received {len(cleaned_chunks)} cleaned chunks for upsertion.",
            "process_and_upsert", module_name))
        
        avgdl_value = get_avgdl_from_db(org_id, file_id, len(cleaned_chunks),all_chunks_tf_sum, logger)

        

        upsert_text_chunks(cleaned_chunks, file_id, user_id, org_id, file_name, avgdl_value, tag_ids, logger)

        # bm25_reindexing(org_id, logger)

        return

    except Exception as e:
        logger.error(_log_message(
            f"[org_id: {org_id}, file_id: {file_id}] Failed during upsert process: {str(e)}",
            "process_and_upsert", module_name))
        raise


@mlflow.trace(name="Get Average Document Length - Org Level")
def get_avgdl_from_db(org_id: int, file_id:str, current_chunk_count: int,all_chunks_tf_sum:int, logger):
    """Get the avgdl (average document length) for given org_id either via API or fallback DB."""
    function_name = "get_avgdl_from_db"
    url = GET_BM25_ORG_AVGDL_URL  # replace with actual API endpoint

    payload = {
        "org_id": str(org_id),
        "file_id":file_id, # sending file_id as well to further help investigate the logs properly
        "total_chunks": current_chunk_count,
        "total_document_length": all_chunks_tf_sum,
        "action": "UPDATE"
    }

    # --- Step 1: Try API with retries ---
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                avgdl = data.get("avgdl")
                logger.info(_log_message(f"[API] Got avgdl={avgdl} from external API", function_name, module_name))
                return avgdl
            else:
                logger.warning(_log_message(f"[API] Failed attempt {attempt+1}: {response.status_code} {response.text}", function_name, module_name))
        except RequestException as e:
            logger.warning(_log_message(f"[API] Exception on attempt {attempt+1}: {e}", function_name, module_name))

    # --- Step 2: Fallback to DB ---
    start_time = time.time()
    try:
        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT COUNT(*), SUM(tf) 
                    FROM bm25_index 
                    WHERE is_archived IS NULL AND org_id = %s;
                """, (org_id,))
                
                result = cursor.fetchone()
                logger.info(_log_message(f"[DB] Raw result: {result}", function_name, module_name))

                if isinstance(result, dict):
                    n = result.get("COUNT(*)")
                    avgdl = float(result.get("SUM(tf)") / n) if n > 0 else 0
                else:
                    n = result[0]
                    avgdl = (result[1] / n) if n > 0 else 0

                logger.info(_log_message(f"[DB] Calculated avgdl: {avgdl}", function_name, module_name))
                return avgdl

            conn.commit()
    except Exception as e:
        logger.error(_log_message(f"[DB] Error while fetching avgdl: {e}", function_name, module_name))
        raise e
    finally:
        logger.info(_log_message(f"Time taken (DB fallback) = {time.time() - start_time:.2f} sec", function_name, module_name))
    

@mlflow.trace(name="Get Org Term Frequeny")
def get_org_tf(org_id: int, logger):
    """Get the term frequencies for all chunks for the provided org_id"""
    try:
        start_time = time.time()
        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            with conn.cursor() as cursor:
                # Fetch all term frequencies and related data for the organization
                cursor.execute(f"""
                    SELECT file_id, chunk_id, term_frequency, tf 
                    FROM {BM25_CORPUS_DB_TABLE} 
                    WHERE is_archived IS NULL AND org_id = %s;
                """, (org_id,))
                
                results = cursor.fetchall()
                
                if not results:
                    logger.warning(f"No term frequency data found for org_id={org_id}")
                    return None
                
                # Process results into a structured format
                org_tf_data = []
                total_docs = len(results)
                total_tf_sum = 0
                
                for row in results:
                    if isinstance(row, dict):
                        file_id = row['file_id']
                        chunk_id = row['chunk_id']
                        term_freq_json = row['term_frequency']
                        tf = row['tf']
                    else:
                        file_id, chunk_id, term_freq_json, tf = row
                    
                    # Parse the JSON term frequency
                    term_freq_dict = json.loads(term_freq_json) if isinstance(term_freq_json, str) else term_freq_json
                    
                    org_tf_data.append({
                        'file_id': file_id,
                        'chunk_id': chunk_id,
                        'term_frequency': term_freq_dict,
                        'tf': tf
                    })
                    
                    total_tf_sum += tf
                
                # Calculate average document length
                avgdl = total_tf_sum / total_docs if total_docs > 0 else 0
                
                elapsed_time = time.time() - start_time
                logger.info(f"Retrieved TF data for org_id={org_id}: {total_docs} documents, avgdl={avgdl:.2f}, time={elapsed_time:.3f}s")
                
                return {
                    'org_tf_data': org_tf_data,
                    'avgdl': avgdl,
                    'total_docs': total_docs
                }

    except Exception as e:
        logger.error(f"Error in get_org_tf for org_id={org_id}: {e}", exc_info=True)
        raise e


def bm25_reindexing(org_id: int, logger):
    try:
        bm25 = CustomBM25Encoder()

        tf_retrive_start_time = time.time()

        org_data = get_org_tf(org_id, logger)
        logger.info(f"Retrieved org TF data for org_id={org_id} in {time.time() - tf_retrive_start_time:.2f} seconds.")

        if not org_data:
            logger.warning(f"No data found for org_id={org_id}, skipping reindexing")
            return

        


        reindexing_start_time = time.time()

        avgdl = org_data['avgdl']
        org_tf_data = org_data['org_tf_data']
        
        reindexed_data = []
        
        # Process each document/chunk
        for chunk_data in org_tf_data:
            chunk_id = chunk_data['chunk_id']
            file_id = chunk_data['file_id']
            term_freq_dict = chunk_data['term_frequency']
            

            indices = [int(idx) for idx in term_freq_dict.keys()]  # Convert string keys back to int
            values = list(term_freq_dict.values())  # Term frequency counts
            
            # Recalculate sparse vectors using BM25
            try:
                recalculated_sparse_vectors = bm25._recalculate_indices_scores(avgdl, indices, values)
                
                # Store the recalculated data
                reindexed_data.append({
                    'file_id': file_id,
                    'chunk_id': chunk_id,
                    'sparse_vector': recalculated_sparse_vectors
                })
                
            except Exception as e:
                logger.error(f"Error recalculating sparse vectors for chunk_id={chunk_id}: {e}")
                logger.debug(f"Failed data - indices: {indices}, values: {values}, avgdl: {avgdl}")
                continue
                
        logger.info(f"BM25 reindexing completed for org_id={org_id} in {time.time() - reindexing_start_time:.2f} seconds. Total reindexed chunks: {len(reindexed_data)}")

        if reindexed_data:
            upsertion_start_time = time.time()
            upsert_bm25_reindexed_data(reindexed_data, org_id, logger)

            logger.info(f"BM25 reindexed data pinecone upsert completed for org_id={org_id} in {time.time() - upsertion_start_time:.2f} seconds.")
        # if reindexed_data:
        #     def background_upsert(data, org, log):
        #         try:
        #             upsert_bm25_reindexed_data(data, org, log)
        #             log.info(f"Completed background upsert for org_id={org}")
        #         except Exception as e:
        #             log.error(f"Error in background upsert for org_id={org}: {e}", exc_info=True)

        #     thread = threading.Thread(target=background_upsert, args=(reindexed_data, org_id, logger))
        #     thread.daemon = True
        #     thread.start()
        #     logger.info(f"Started background upsert thread for org_id={org_id}")
                                   
    except Exception as e:
        logger.error(f"Error in bm25_reindexing for org_id={org_id}: {e}", exc_info=True)
        raise e
    

def upsert_bm25_reindexed_data(reindexed_data, org_id, logger, batch_size=100):
    """
    Batch upsert BM25 reindexed vectors to Pinecone, keeping existing metadata intact.
    """
    try:
        if not reindexed_data:
            logger.warning(f"No reindexed data provided for org_id={org_id}, skipping upsert")
            return

        namespace = f"org_id_{org_id}#"
        index = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX_SPARSE)

        # Process in batches
        for i in range(0, len(reindexed_data), batch_size):
            batch = reindexed_data[i:i+batch_size]

            # Fetch existing metadata for all chunk_ids in this batch
            chunk_ids = [str(item['chunk_id']) for item in batch]
            fetched = index.fetch(ids=chunk_ids, namespace=namespace)

            vectors_to_upsert = []
            for item in batch:
                chunk_id = str(item['chunk_id'])
                new_vector = item['sparse_vector']

                # Use existing metadata if present, else empty dict
                metadata = fetched['vectors'].get(chunk_id, {}).get('metadata', {})

                vectors_to_upsert.append({
                    'id': chunk_id,
                    'values': [],  # Required field even for sparse-only vectors
                    "sparse_values": new_vector,
                    'metadata': metadata
                })

            # Upsert batch
            index.upsert(vectors=vectors_to_upsert, namespace=namespace)

            logger.info(f"Upserted batch {i // batch_size + 1} ({len(vectors_to_upsert)} vectors) in namespace={namespace}")

        logger.info(f"Completed upsert of {len(reindexed_data)} reindexed vectors for org_id={org_id}")

    except Exception as e:
        logger.error(f"Error in upserting BM25 reindexed data for org_id={org_id}: {e}", exc_info=True)
        raise e