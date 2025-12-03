import pickle
import pandas as pd
# from pinecone_text.sparse import BM25Encoder
from src.common.file_insights.hybrid_vector_handler.bm25_encoder import CustomBM25Encoder
from src.config.base_config import config
from src.common.db_connection_manager import DBConnectionManager
from src.common.logger import _log_message
import logging
from openai import OpenAI
from pinecone import Pinecone
import time
from src.common.file_insights.chunking import ClusterSemanticChunker
from datetime import datetime
import json
from src.common.file_insights.hybrid_vector_handler.bm25_handler import calculate_tf_update_db
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.common.llm.factory import get_llm_client

client = get_llm_client(async_mode=False)

OPENAI_API_KEY = config.OPENAI_API_KEY
CONTRACT_GENERATION_DB = config.CONTRACT_GENERATION_DB
CONTRACT_INTEL_DB = config.CONTRACT_INTEL_DB
CONTEXTUAL_RAG_DB = config.CONTEXTUAL_RAG_DB
# DOCUMENT_SUMMARY_AND_CHUNK_INDEX = "bm25-testing-dense-index"
# DOCUMENT_SUMMARY_AND_CHUNK_INDEX_SPARSE = "bm25-testing-sparse-index"

DOCUMENT_SUMMARY_AND_CHUNK_INDEX = config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX
DOCUMENT_SUMMARY_AND_CHUNK_INDEX_SPARSE = config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX_SPARSE
fitted_corpus_table = "bm25_fitted_corpus"
module_name = "bm25_testing.py"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BM25ReIndexing:
    def __init__(self, org_id: int = None, logger=None):
        """
        Initialize the BM25ReIndexing class with logger and org_id.
        
        Args:
            org_id (int, optional): Organization ID to work with
        """
        # self.logger = logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.org_id = org_id
    
    def set_org_id(self, org_id: int):
        """Set or update the organization ID."""
        self.org_id = org_id
    
    def estimate_legal_pages(self, word_count: int, words_per_page: int = 2500) -> int:
        """Estimate number of legal pages based on word count."""
        num_pages = (word_count // words_per_page) + (1 if word_count % words_per_page > 0 else 0)
        return num_pages


    def retrieve_file_text_org_level(self):
        """
        Returns an org_id's entire corpus including:
        ci_file_guid, file_name, texts, metadata fields (char_length for each chunk, 
        chunks_count, org_id, page_no, user_id)
        """
        function_name = "retrieve_file_text_org_level"

        if not self.org_id:
            raise ValueError("org_id must be set before calling this method")

        try:
            # Step 1: get all the file_ids associated with that org
            with DBConnectionManager(CONTRACT_INTEL_DB, self.logger) as conn_intel:
                if conn_intel is None:
                    msg = "Failed to establish connection to CONTRACT_INTEL_DB."
                    self.logger.error(_log_message(msg, function_name, module_name))
                    raise ConnectionError(msg)

                with conn_intel.cursor() as cursor:
                    query = """
                        SELECT o.ci_org_guid, 
                            o.org_id, 
                            f.user_id, 
                            f.ci_file_guid as file_id, 
                            f.name as file_name
                        FROM files f
                        INNER JOIN organizations o 
                            ON f.ci_org_guid = o.ci_org_guid
                        WHERE o.org_id = %s 
                        AND f.type = 1 
                        AND o.is_archived IS NULL
                        AND f.name NOT LIKE 'temp/%%'
                        AND f.is_archived IS NULL
                        AND EXISTS (
                            SELECT 1
                            FROM llm_process_status lps
                            WHERE lps.ci_file_guid = f.ci_file_guid
                                AND lps.process_type = 3
                                AND lps.completed_steps = 3
                        );
                    """
                    cursor.execute(query, (self.org_id,))
                    rows = cursor.fetchall()
                    if not rows:
                        self.logger.warning(_log_message(
                            f"No active files found for org_id {self.org_id}", 
                            function_name, module_name
                        ))
                        return pd.DataFrame()

            # Build a lookup dict for later
            metadata_lookup = {row["file_id"]: row for row in rows}
            ci_file_guids = list(metadata_lookup.keys())

            self.logger.info(_log_message(
                f"Total files found for org_id {self.org_id}: {len(ci_file_guids)}", 
                function_name, module_name
            ))

            if not ci_file_guids:
                self.logger.warning(_log_message(
                    f"No file IDs to fetch from file_extractions for org_id {self.org_id}", 
                    function_name, module_name
                ))
                return pd.DataFrame()

            # Step 2: Fetch all file extractions
            with DBConnectionManager(CONTRACT_GENERATION_DB, self.logger) as conn_gen:
                if conn_gen is None:
                    msg = "Failed to establish connection to CONTRACT_GENERATION_DB."
                    self.logger.error(_log_message(msg, function_name, module_name))
                    raise ConnectionError(msg)

                with conn_gen.cursor() as cursor:
                    placeholders = ",".join(["%s"] * len(ci_file_guids))
                    query = f"""
                        SELECT ci_file_guid, extracted_text, unstructured_elements
                        FROM file_extractions
                        WHERE ci_file_guid IN ({placeholders})
                    """
                    cursor.execute(query, tuple(ci_file_guids))
                    results = cursor.fetchall()

            if not results:
                self.logger.warning(_log_message(
                    f"No data found for file_extractions for org_id {self.org_id}", 
                    function_name, module_name
                ))
                return pd.DataFrame()

            records = []
            document_chunker = ClusterSemanticChunker(self.logger)

            # Collect TF update tasks
            tf_tasks = []

            self.logger.info(_log_message(
                f"Total file extractions fetched: {len(results)}",
                function_name, module_name
            ))

            for result in results:
                ci_file_guid = result["ci_file_guid"]
                extracted_text = result.get("extracted_text", "") or ""
                text = extracted_text

                self.logger.info(_log_message(
                    f"Extracted text length for {ci_file_guid}: {len(text)}", 
                    function_name, module_name
                ))

                extracted_list = document_chunker.create_chunks(text)

                self.logger.info(_log_message(
                    f"Number of chunks extracted for {ci_file_guid}: {len(extracted_list)}", 
                    function_name, module_name
                ))

                if extracted_list:
                    meta = metadata_lookup.get(ci_file_guid, {})
                    chunks_count = len(extracted_list)

                    for i, chunk in enumerate(extracted_list):
                        records.append({
                            "ci_file_guid": ci_file_guid,
                            "chunk_id": f"{self.org_id}#{ci_file_guid}#{i+1}",
                            "ci_org_guid": meta.get("ci_org_guid"),
                            "file_name": meta.get("file_name"),
                            "org_id": self.org_id,
                            "user_id": meta.get("user_id"),
                            "text": chunk,
                            "chunks_count": chunks_count,
                            "char_length": len(chunk),
                            "page_no": i+1
                        })

                    # Queue TF update task instead of calling directly
                    tf_tasks.append((self.org_id, meta.get("user_id"), ci_file_guid, extracted_list))

            # Step 3: Run TF updates in parallel
            with ThreadPoolExecutor(max_workers=50) as executor:  # adjust max_workers as needed
                futures = {
                    executor.submit(calculate_tf_update_db, org_id, user_id, file_id, chunks, self.logger): file_id
                    for org_id, user_id, file_id, chunks in tf_tasks
                }

                for future in as_completed(futures):
                    file_id = futures[future]
                    try:
                        future.result()
                        self.logger.info(_log_message(
                            f"TF update completed for {file_id}", function_name, module_name
                        ))
                    except Exception as e:
                        self.logger.error(_log_message(
                            f"TF update failed for {file_id}: {str(e)}", function_name, module_name
                        ))

            # Step 4: Return final DataFrame
            df = pd.DataFrame(records)
            
            return df

        except Exception as e:
            self.logger.error(_log_message(
                f"Unhandled exception: {str(e)}", function_name, module_name
            ))
            raise e
        

    def get_bm25_model_from_db(self):
        """
        Retrieve BM25 fitted object from database for the org_id.
        
        Returns:
            dict: The BM25 model dictionary (with avgdl, n_docs, doc_freq, etc.) or None if not found.
        """
        function_name = "get_bm25_model_from_db"
        
        if not self.org_id:
            raise ValueError("org_id must be set before calling this method")
            
        self.logger.info(_log_message(
            f"Retrieving BM25 fitted object from Database for {self.org_id}", 
            function_name, module_name
        ))
        
        with DBConnectionManager(CONTEXTUAL_RAG_DB, self.logger) as conn_gen:
            if conn_gen is None:
                msg = "Failed to establish connection to CONTEXTUAL_RAG_DB."
                self.logger.error(_log_message(msg, function_name, module_name))
                raise ConnectionError(msg)

            with conn_gen.cursor() as cursor:
                query = f"""
                        SELECT doc_freq 
                        FROM {fitted_corpus_table}
                        WHERE org_id = %s
                    """
                
                cursor.execute(query, (self.org_id,))
                result = cursor.fetchone()
                
                if result and result.get("doc_freq"):
                    bm25_json = result["doc_freq"]
                    bm25_dict = json.loads(bm25_json)


                    bm25 = CustomBM25Encoder()
                    """
                        do not use the load method directly because we are reading from DB not from json otherwise it throws error
                        TypeError: expected str, bytes or os.PathLike object, not dict
                        check logs for error
                        http://44.199.74.243:8000/#/experiments/4?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D&compareRunsMode=TRACES&selectedTraceId=d1a87d6584a5444eb3ddfd3693f89651
                        # BM25Encoder.load method expects a file path string, opens it with with open(path, "r"). 
                        # bm25.load(bm25_dict) 
                    """
                    # set_params() updates its internal attributes (avgdl, n_docs, doc_freq, etc.) from the dictionary.
                    # bm25.set_params(**bm25_dict)                    
                    self.logger.info(_log_message(
                        f"BM25 model retrieved successfully for {self.org_id}", 
                        function_name, module_name
                    ))
                    # #print(bm25)
                    return bm25
                
                self.logger.warning(_log_message(
                    f"No BM25 model found for org_id {self.org_id}", 
                    function_name, module_name
                ))
                return None
        
    def save_bm25_model_to_db(self, bm25_model: dict):
        """
        Save BM25 fitted object into database.
        
        Args:
            bm25_model (dict): The BM25 model dictionary (with avgdl, n_docs, doc_freq, etc.).
        """
        function_name = "save_bm25_model_to_db"
        
        if not self.org_id:
            raise ValueError("org_id must be set before calling this method")
            
        self.logger.info(_log_message(
            f"Saving New fitted BM25 object to Database for {self.org_id}", 
            function_name, module_name
        ))
        
        # Convert dict -> JSON string
        avg_doc_len = bm25_model.avgdl
        total_docs = bm25_model.n_docs
        updated_ts = datetime.now()
        
        print(bm25_model.__dict__)

        bm25_dict = {
            "avgdl": bm25_model.avgdl,
            "n_docs": bm25_model.n_docs,
            # "doc_freq": bm25_model.doc_freq ,
            "doc_freq": {
                "indices": list(bm25_model.doc_freq.keys()),
                "values": list(bm25_model.doc_freq.values())
            },
            "b": bm25_model.b,
            "k1": bm25_model.k1,
            "lower_case": True,
            "remove_punctuation": True,
            "remove_stopwords": True,
            "stem": False,
            "language": "english",
            "files": {}
        }

        bm25_json = json.dumps(bm25_dict)

        with DBConnectionManager(CONTEXTUAL_RAG_DB, self.logger) as conn_gen:
            if conn_gen is None:
                msg = "Failed to establish connection to CONTEXTUAL_RAG_DB."
                self.logger.error(_log_message(msg, function_name, module_name))
                raise ConnectionError(msg)

            with conn_gen.cursor() as cursor:
                # Check if the corpus row exists previously for the same org_id
                query = f"""
                        SELECT corpus_id, org_id 
                        FROM {fitted_corpus_table}
                        WHERE org_id = %s
                    """
                
                cursor.execute(query, (self.org_id,))
                exists = cursor.fetchone()
                
                # If the corpus doesn't exist for user, insert new row
                if not exists:
                    self.logger.info(_log_message(
                        f"No corpus exists previously for {self.org_id} | Inserting Corpus", 
                        function_name, module_name
                    ))
                    
                    insert_query = f"""
                                    INSERT INTO {fitted_corpus_table} (org_id, avg_doc_len, total_docs, doc_freq, updated_ts) 
                                    VALUES (%s, %s, %s, %s, %s)
                                """
                    values = (self.org_id, avg_doc_len, total_docs, bm25_json, updated_ts)
                    cursor.execute(insert_query, values)
                    
                    self.logger.info(_log_message(
                        f"New Corpus Inserted for {self.org_id}", 
                        function_name, module_name
                    ))
                else:
                    # Update if it exists previously 
                    self.logger.info(_log_message(
                        f"Corpus exists for {self.org_id} | Updating new corpus values at {updated_ts}", 
                        function_name, module_name
                    ))
                    
                    update_query = f"""
                            UPDATE {fitted_corpus_table}
                            SET doc_freq = %s,
                                avg_doc_len = %s,
                                total_docs = %s,
                                updated_ts = %s
                            WHERE org_id = %s
                    """
                    update_values = (bm25_json, avg_doc_len, total_docs, updated_ts, self.org_id)
                    cursor.execute(update_query, update_values)
                    
                    self.logger.info(_log_message(
                        f"Update completed for new corpus for {self.org_id}", 
                        function_name, module_name
                    ))
        
    def get_custom_bm25_encoder(self, df):
        """Fit BM25 encoder and encode documents."""
        function_name = "get_custom_bm25_encoder"
        
        # Step 6: BM25 fit and encode
        bm25 = CustomBM25Encoder()
        bm25.fit(df["text"].tolist())
        df["sparse_values"] = df["text"].apply(lambda x: bm25.encode_documents([x])[0])
        
        # # Save to files for debugging/analysis
        # df.to_csv("bm25_encoded_data.csv", index=False)
        # df.to_excel("bm25_encoded_data.xlsx", index=False)
        
        self.logger.info(_log_message(
            f"BM25 encoding completed and saved to CSV/Excel| Total length - {len(df)}", 
            function_name, module_name
        ))
        
        total_unique_files = df["ci_file_guid"].nunique()
        self.logger.info(_log_message(
            f"Total unique files processed: {total_unique_files}", 
            function_name, module_name
        ))
        
        return df, bm25
    
    def generate_single_embedding(self, text: str):
        """Generate embedding for a single text chunk using OpenAI directly."""
        try:

            response = client.embeddings.create(
                model=config.OPENAI_EMBEDING_MODEL_NAME,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Error generating embedding: {e}")
    

    def generate_embeddings_batch(self, df, batch_size: int = 100):
        """Generate embeddings in batches - most efficient approach."""
        texts = df["text"].tolist()
        all_embeddings = []
        
        print(f"Processing {len(texts)} texts in batches of {batch_size}...")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            
            try:
                response = client.embeddings.create(
                    model=config.OPENAI_EMBEDING_MODEL_NAME,
                    input=batch_texts
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Fallback to individual processing for this batch
                for text in batch_texts:
                    try:
                        embedding = self.generate_single_embedding(text)
                        all_embeddings.append(embedding)
                    except Exception as inner_e:
                        print(f"Failed to process individual text: {inner_e}")
                        # Add None or zero vector as fallback
                        all_embeddings.append(None)
        
        df = df.copy()
        df["dense_values"] = all_embeddings
        return df
    

    def generate_embeddings(self):
        """
        Generate dense (OpenAI) and sparse (BM25) embeddings for the org.
        
        Returns:
        - DataFrame with columns: ci_file_guid, ci_org_guid, org_id, text, sparse_values, dense_values
        - BM25 model
        """
        function_name = "generate_embeddings"
        
        if not self.org_id:
            raise ValueError("org_id must be set before calling this method")
            
        try:
            # Step 1: Get all texts for the org
            df = self.retrieve_file_text_org_level()
            if df.empty:
                self.logger.warning(_log_message(
                    f"No data found for org_id {self.org_id}", 
                    function_name, module_name
                ))
                return pd.DataFrame(), None
    
            # Step 2: Fit BM25 and create sparse vectors for all rows
            df, bm25 = self.get_custom_bm25_encoder(df)

            df = self.generate_embeddings_batch(df)


            # Save BM25 model to database
            # self.save_bm25_model_to_db(bm25)
            return df, bm25
    
        except Exception as e:
            self.logger.error(_log_message(
                f"Unhandled exception in generate_embeddings: {str(e)}", 
                function_name, module_name
            ))
            raise e
        
    def upsert_vectors(self, index, vectors, namespace, batch_size=100):
        """Upsert vectors into Pinecone in batches and ensure they are added correctly."""
        function_name = "upsert_vectors"
        
        try:
            initial_count = index.describe_index_stats().get('namespaces', {}).get(namespace, {}).get('vector_count', 0)
            self.logger.info(_log_message(
                f"[{namespace}] Initial vector count: {initial_count}",
                function_name, module_name
            ))
    
            total_vectors = len(vectors)
            self.logger.info(_log_message(
                f"[{namespace}] Total vectors to upsert: {total_vectors}",
                function_name, module_name
            ))
    
            for i in range(0, total_vectors, batch_size):
                batch = vectors[i:i + batch_size]
                self.logger.info(_log_message(
                    f"Upserting batch {i//batch_size + 1} with {len(batch)} vectors",
                    function_name, module_name
                ))
                
                index.upsert(vectors=batch, namespace=namespace)
                self.logger.info(_log_message(
                    f"[{namespace}] Upserted batch {i//batch_size + 1} with {len(batch)} vectors.",
                    function_name, module_name
                ))
                time.sleep(0.25)
    
            # Retry to verify upsert success
            expected_min = initial_count + 1
            for attempt in range(1, 11):
                time.sleep(5)
                current_count = index.describe_index_stats().get('namespaces', {}).get(namespace, {}).get('vector_count', 0)
                self.logger.info(_log_message(
                    f"[{namespace}] Retry {attempt}: Current = {current_count}, Expected >= {expected_min}",
                    function_name, module_name
                ))
                if current_count >= expected_min:
                    break
    
            self.logger.info(_log_message(
                f"[{namespace}] Vectors upserted successfully. Final count: {current_count}",
                function_name, module_name
            ))
            
        except Exception as e:
            self.logger.error(_log_message(
                f"[{namespace}] Error during upsert: {str(e)}",
                function_name, module_name
            ))
            raise
      
    def process_and_upsert_text_chunks(self):
        """
        Generate sparse + dense embeddings for the org and upsert to Pinecone.
        """
        function_name = "process_and_upsert_text_chunks"
        
        if not self.org_id:
            raise ValueError("org_id must be set before calling this method")
    
        # Step 1: Generate embeddings
        df, bm25 = self.generate_embeddings()
        if df.empty:
            self.logger.warning(_log_message(
                f"No text found for org_id {self.org_id}",
                function_name, module_name
            ))
            return None
    
        sparse_total_no_of_chunks = len(df)
        self.logger.info(_log_message(
            f"[org_id_{self.org_id}] Total sparse: {sparse_total_no_of_chunks}",
            function_name, module_name
        ))
    
        # Step 3: Build Pinecone-ready vector payloads
        sparse_list = [
            self.build_vector(row, "sparse")
            for _, row in df.iterrows()
        ]

        dense_list = [
            self.build_vector(row, "dense")
            for _, row in df.iterrows()
        ]
    
        sparse_list = [v for v in sparse_list if v]
        dense_list = [v for v in dense_list if v]
    
        namespace = f"org_id_{self.org_id}#"
    
        if sparse_list:
            sparse_index = Pinecone(api_key=config.PINECONE_API_KEY).Index(DOCUMENT_SUMMARY_AND_CHUNK_INDEX_SPARSE)
            self.logger.info(_log_message(
                f"[{namespace}] Sparse upsertion of {len(sparse_list)} chunks for org_id ({self.org_id})",
                function_name, module_name
            ))
            # Uncomment the line below to actually perform the upsert
            self.upsert_vectors(sparse_index, sparse_list, namespace)
    
        if not sparse_list:
            msg = f"[{namespace}] No valid vectors generated to upsert for org_id {self.org_id}"
            self.logger.warning(_log_message(msg, function_name, module_name))
            raise ValueError(msg)
        

        if dense_list:
            dense_index = Pinecone(api_key=config.PINECONE_API_KEY).Index(DOCUMENT_SUMMARY_AND_CHUNK_INDEX)
            self.logger.info(_log_message(
                f"[{namespace}] Dense upsertion of {len(dense_list)} chunks for org_id ({self.org_id})",
                function_name, module_name
            ))
            # Uncomment the line below to actually perform the upsert
            self.upsert_vectors(dense_index, dense_list, namespace)     
        
        if not dense_list:
            msg = f"[{namespace}] No valid dense vectors generated to upsert for org_id {self.org_id}"
            self.logger.warning(_log_message(msg, function_name, module_name))
            raise ValueError(msg)
    
        return bm25
    
    def build_vector(self, row, vector_type):
        """
        Build a single Pinecone vector entry from a dataframe row.
        """
        if not self.org_id:
            raise ValueError("org_id must be set before calling this method")
            
        file_id = row["ci_file_guid"]
        file_name = row.get("file_name", "")
        text = row["text"]
    
        base_metadata = {
            "file_id": file_id,
            "file_name": file_name,
            "user_id": row.get('user_id'),
            "org_id": self.org_id,
            "text": text,
            "chunks_count": row.get('chunks_count'),
            "char_length": row.get('char_length'),
            "page_no": row.get('page_no')
        }

    
        if vector_type == "sparse":
            sparse = row["sparse_values"]
            if not sparse or "indices" not in sparse or "values" not in sparse or len(sparse["indices"]) == 0:
                return None
            return {
                "id": f"{row.get('chunk_id')}",
                "values": [],  # Required field even for sparse-only vectors
                "sparse_values": sparse,
                "metadata": base_metadata
            }
        
        elif vector_type == "dense":
            dense = row["dense_values"]
            if not dense or len(dense) == 0:
                return None
            return {
                "id": f"{row.get('chunk_id')}",
                "values": dense,
                "metadata": base_metadata
            }
    
        return None


# # Example usage:
# bm25_encoder = BM25ReIndexing(org_id=844)
# bm25_encoder.process_and_upsert_text_chunks()

# # Or if you need to set org_id later:
# bm25_encoder = BM25ReIndexing(1297,logger)
# bm25_encoder.set_org_id(1297)
# bm25_encoder.process_and_upsert_text_chunks()