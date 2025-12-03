
import os,sys
import json
from src.common.db_connection_manager import DBConnectionManager
from src.common.file_insights.hybrid_vector_handler.bm25_encoder import CustomBM25Encoder
from typing import List, Dict, Union, Optional,Any
import time
from src.config.base_config import config
import mlflow
mlflow.openai.autolog()

CONTEXTUAL_RAG_DB = config.CONTEXTUAL_RAG_DB
BM25_CORPUS_DB_TABLE = config.BM25_CORPUS_DB_TABLE
BM25_METADATA_DB_TABLE = config.BM25_METADATA_DB_TABLE
BM25_SUMMARY_DB_TABLE = config.BM25_SUMMARY_DB_TABLE
CONTRACT_INTEL_DB = config.CONTRACT_INTEL_DB

@mlflow.trace(name="Delete TF")
def delete_tf(org_id:int, user_id:int,file_ids:Union[str,List[str]],logger=None):
    try:
        logger.info(f"[BM25] Starting TF deletion for org_id={org_id}, user_id={user_id}, file_ids={file_ids}")
        # BM25_CORPUS_DB_TABLE = config.BM25_CORPUS_DB_TABLE
        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            with conn.cursor() as cursor:
                # Check if the table exists, if not create it
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {BM25_CORPUS_DB_TABLE} (
                        file_id VARCHAR(255),
                        org_id INTEGER,
                        user_id INTEGER,
                        chunk_id VARCHAR(255),
                        term_frequency JSON,
                        tf INTEGER,
                        PRIMARY KEY (file_id, chunk_id)
                    );
                """)
                if isinstance(file_ids, str):
                    file_ids = [file_ids]
                
                # Delete records for the given file_ids
                cursor.execute(f"""
                    DELETE FROM {BM25_CORPUS_DB_TABLE} WHERE org_id = %s AND user_id = %s AND file_id IN %s;
                """, (org_id, user_id, tuple(file_ids),))
            

    except Exception as e:
        logger.error(f"[BM25] Error in delete_tf: {e}", exc_info=True)
        raise e
    

@mlflow.trace(name="Calculate IDF")
def calculate_idf(query: str, org_id: int, file_ids: Union[None, List[str]] = None, tag_ids=[], logger=None):
    try:
        logger.info(f"[BM25-IDF] Starting IDF calculation for org_id={org_id}, query='{query}', file_ids={file_ids}")

        # Step 1: Get file_temp_ids values from file_tags
        with DBConnectionManager(CONTRACT_INTEL_DB, logger) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT file_temp_id 
                    FROM file_tags 
                    WHERE tag_id IN %s
                    """,
                    (tuple(tag_ids),)
                )
                file_temp_ids = [row["file_temp_id"] for row in cursor.fetchall()]  # collect as list

        logger.debug(f"[BM25-IDF] Retrieved file_temp_ids from tags: {file_temp_ids}")

        if not file_temp_ids:
            logger.info("[BM25-IDF] No file_temp_ids found for given tag_ids, returning empty result.")
            return {}

    
        if file_ids:
            file_temp_ids = [fid for fid in file_ids if fid in file_temp_ids] # Use only file_ids that are present in file_temp_ids
            logger.debug(f"[BM25-IDF] Filtered file_temp_ids with provided file_ids: {file_temp_ids}")


        result = []
        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            with conn.cursor() as cursor:
                if file_temp_ids:
                    cursor.execute(
                        f"""
                        SELECT term_frequency 
                        FROM {BM25_CORPUS_DB_TABLE} 
                        WHERE is_archived IS NULL 
                        AND file_id IN %s;
                        """,
                        (tuple(file_temp_ids),)
                    )
                    result = cursor.fetchall()
                else:
                    logger.warning("[BM25-IDF] No file_temp_ids provided, skipping query.")
                    return {}

                logger.debug(f"[BM25-IDF] Fetched {len(result)} records from DB for IDF calculation")

                # Extract list of term indices from each document
                term_lists = []
                for row in result:
                    tf_json = row.get("term_frequency") if isinstance(row, dict) else row[0]
                    terms = list(json.loads(tf_json).keys())
                    term_lists.append(terms)

                logger.debug(f"[BM25-IDF] Collected {len(term_lists)} documents for IDF computation")

                # Calculate document frequency
                bm25 = CustomBM25Encoder()
                doc_freq = bm25.count_values_across_lists(term_lists)
                total_docs = len(term_lists)

                # Compute IDF values for the query
                idf_vals = bm25.custom_doc_query(query, doc_freq=doc_freq, N=total_docs)

                logger.info(f"[BM25-IDF] Completed IDF computation for query: '{query}'")
                return idf_vals

    except Exception as e:
        logger.error(f"[BM25-IDF] Error in calculate_idf: {e}", exc_info=True)
        return {}


@mlflow.trace(name="Calculate TF")
def calculate_tf_update_db(org_id: int, user_id: int, file_id: str, chunks: List[str], logger):
    """Efficiently update the Term-Frequency BM25 index for a given file_id and its chunks."""
    logger.info(f"[BM25] Starting optimized TF update for file_id={file_id}, org_id={org_id}, user_id={user_id}, total_chunks={len(chunks)}")
    # BM25_CORPUS_DB_TABLE = config.BM25_CORPUS_DB_TABLE
    try:
        bm25 = CustomBM25Encoder()
        chunk_data = []
        chunk_ids = []
        #-------------------------------
        # sums up all the chunks's term frequency 
        all_chunks_tf_sum = 0
        #------------------------------
        # Preprocess all chunks
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{org_id}#{file_id}#{idx+1}"
            chunk_text = chunk
            indices, values = bm25._tf(chunk)
            term_freq = {i: val for i, val in zip(indices, values)}
            term_freq_json = json.dumps(term_freq)
            total_tf = sum(values)
            all_chunks_tf_sum += total_tf
            chunk_data.append((file_id, org_id, user_id, chunk_id, chunk_text, term_freq_json, total_tf))
            chunk_ids.append(chunk_id)

        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            with conn.cursor() as cursor:
                logger.debug("[BM25] Ensuring bm25_index table exists")
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {BM25_CORPUS_DB_TABLE} (
                        file_id VARCHAR(255),
                        org_id INTEGER,
                        user_id INTEGER,
                        chunk_id VARCHAR(255),
                        term_frequency JSON,
                        tf INTEGER,
                        PRIMARY KEY (file_id, chunk_id)
                    );
                """)

                # Step 1: Fetch existing chunk_ids for this file
                logger.debug("[BM25] Fetching existing chunk_ids from DB")
                cursor.execute(f"""
                    SELECT chunk_id FROM {BM25_CORPUS_DB_TABLE} 
                    WHERE is_archived IS NULL AND file_id = %s AND chunk_id IN %s;
                """, (file_id, tuple(chunk_ids)))
                existing = cursor.fetchall()
                existing_ids = set(row['chunk_id'] if isinstance(row, dict) else row[0] for row in existing)

                # Step 2: Split data into inserts and updates
                to_insert = [row for row in chunk_data if row[3] not in existing_ids]
                to_update = [row for row in chunk_data if row[3] in existing_ids]

                logger.debug(f"[BM25] {len(to_insert)} new records to insert, {len(to_update)} records to update")

                # Step 3: Execute batch insert
                if to_insert:
                    cursor.executemany(f"""
                        INSERT INTO {BM25_CORPUS_DB_TABLE} (file_id, org_id, user_id, chunk_id, chunk_text, term_frequency, tf)
                        VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """, to_insert)

                # Step 4: Execute batch update
                if to_update:
                    update_data = [(row[4], row[5], row[6], row[0], row[3]) for row in to_update]  
                     

                    cursor.executemany(f"""
                        UPDATE {BM25_CORPUS_DB_TABLE} 
                        SET chunk_text = %s, term_frequency = %s, tf = %s 
                        WHERE file_id = %s AND chunk_id = %s;
                    """, update_data)

                conn.commit()
                logger.info(f"[BM25] TF update complete for file_id={file_id}, inserts={len(to_insert)}, updates={len(to_update)}")

        return all_chunks_tf_sum # returning total tf of all chunks for further to hit the get-org-avgdl api
    except Exception as e:
        logger.error(f"[BM25] Error in calculate_tf_update_db for file_id={file_id}: {e}", exc_info=True)
        raise e




# -------------------------------------------------------------------------------------------------------------

@mlflow.trace(name="Delete TF Metadata")
def delete_tf_metadata(org_id:int, user_id:int,file_ids:Union[str,List[str]],logger=None):
    try:
        logger.info(f"[BM25] Starting TF deletion for org_id={org_id}, user_id={user_id}, file_ids={file_ids}")
        # BM25_METADATA_DB_TABLE = config.BM25_METADATA_DB_TABLE
        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            with conn.cursor() as cursor:
                # Check if the table exists, if not create it
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {BM25_METADATA_DB_TABLE} (
                        file_id VARCHAR(255),
                        org_id INTEGER,
                        user_id INTEGER,
                        chunk_id VARCHAR(255),
                        term_frequency JSON,
                        tf INTEGER,
                        PRIMARY KEY (file_id, chunk_id)
                    );
                """)
                if isinstance(file_ids, str):
                    file_ids = [file_ids]
                
                # Delete records for the given file_ids
                cursor.execute(f"""
                    DELETE FROM {BM25_METADATA_DB_TABLE} WHERE org_id = %s AND user_id = %s AND file_id IN %s;
                """, (org_id, user_id, tuple(file_ids),))
            

    except Exception as e:
        logger.error(f"[BM25] Error in delete_tf: {e}", exc_info=True)
        raise e
    

@mlflow.trace(name="Calculate IDF Metadata")
def calculate_idf_metadata(query: str, org_id: int, file_ids: Union[None, List[str]] = None, logger=None):
    try:
        logger.info(f"[BM25-IDF] Starting IDF calculation for org_id={org_id}, query='{query}'")
        # BM25_METADATA_DB_TABLE = config.BM25_METADATA_DB_TABLE
        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            with conn.cursor() as cursor:
                if isinstance(file_ids, list) and len(file_ids) > 0:
                    logger.debug(f"[BM25-IDF] Fetching term_frequency for file_ids={file_ids}")
                    cursor.execute(f"""
                        SELECT term_frequency FROM {BM25_METADATA_DB_TABLE} WHERE file_id IN %s;
                    """, (tuple(file_ids),))
                else:
                    logger.debug(f"[BM25-IDF] Fetching term_frequency for org_id={org_id}")
                    cursor.execute(f"""
                        SELECT term_frequency FROM {BM25_METADATA_DB_TABLE} WHERE org_id = %s;
                    """, (org_id,))

                result = cursor.fetchall()
                if not result:
                    logger.warning(f"[BM25-IDF] No records found for given input. Returning empty result.")
                    return {}

                # Extract list of term indices from each document
                term_lists = []
                for row in result:
                    tf_json = row.get("term_frequency") if isinstance(row, dict) else row[0]
                    terms = list(json.loads(tf_json).keys())
                    term_lists.append(terms)
                
                logger.debug(f"[BM25-IDF] Collected {len(term_lists)} documents for IDF computation")

                # Calculate document frequency
                bm25 = CustomBM25Encoder()
                doc_freq = bm25.count_values_across_lists(term_lists)
                total_docs = len(term_lists)


                # Compute IDF values for the query
                idf_vals = bm25.custom_doc_query(query, doc_freq=doc_freq, N=total_docs)
                
                logger.info(f"[BM25-IDF] Completed IDF computation for query: '{query}'")
                return idf_vals

    except Exception as e:
        logger.error(f"[BM25-IDF] Error in calculate_idf_metadata: {e}", exc_info=True)
        return {}


@mlflow.trace(name="Calculate TF Metadata")
def calculate_tf_update_db_metadata(org_id: int, user_id: int, file_id: str, chunks: List[str], logger):
    """Efficiently update the Term-Frequency BM25 index for a given file_id and its chunks."""
    logger.info(f"[BM25] Starting optimized TF update for file_id={file_id}, org_id={org_id}, user_id={user_id}, total_chunks={len(chunks)}")
    # BM25_METADATA_DB_TABLE = config.BM25_METADATA_DB_TABLE
    try:
        bm25 = CustomBM25Encoder()
        chunk_data = []
        chunk_ids = []

        # Preprocess all chunks
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{file_id}_{idx+1}"
            indices, values = bm25._tf(chunk)
            term_freq = {i: val for i, val in zip(indices, values)}
            term_freq_json = json.dumps(term_freq)
            total_tf = sum(values)
            chunk_data.append((file_id, org_id, user_id, chunk_id, term_freq_json, total_tf))
            chunk_ids.append(chunk_id)

        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            with conn.cursor() as cursor:
                logger.debug("[BM25] Ensuring bm25_index table exists")
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {BM25_METADATA_DB_TABLE} (
                        file_id VARCHAR(255),
                        org_id INTEGER,
                        user_id INTEGER,
                        chunk_id VARCHAR(255),
                        term_frequency JSON,
                        tf INTEGER,
                        PRIMARY KEY (file_id, chunk_id)
                    );
                """)

                # Step 1: Fetch existing chunk_ids for this file
                logger.debug("[BM25] Fetching existing chunk_ids from DB")
                cursor.execute(f"""
                    SELECT chunk_id FROM {BM25_METADATA_DB_TABLE} 
                    WHERE file_id = %s AND chunk_id IN %s;
                """, (file_id, tuple(chunk_ids)))
                existing = cursor.fetchall()
                existing_ids = set(row['chunk_id'] if isinstance(row, dict) else row[0] for row in existing)

                # Step 2: Split data into inserts and updates
                to_insert = [row for row in chunk_data if row[3] not in existing_ids]
                to_update = [row for row in chunk_data if row[3] in existing_ids]

                logger.debug(f"[BM25] {len(to_insert)} new records to insert, {len(to_update)} records to update")

                # Step 3: Execute batch insert
                if to_insert:
                    cursor.executemany(f"""
                        INSERT INTO {BM25_METADATA_DB_TABLE} (file_id, org_id, user_id, chunk_id, term_frequency, tf)
                        VALUES (%s, %s, %s, %s, %s, %s);
                    """, to_insert)

                # Step 4: Execute batch update
                if to_update:
                    update_data = [(row[4], row[5], row[0], row[3]) for row in to_update]  # (term_freq_json, tf, file_id, chunk_id)
                    cursor.executemany(f"""
                        UPDATE {BM25_METADATA_DB_TABLE} SET term_frequency = %s, tf = %s 
                        WHERE file_id = %s AND chunk_id = %s;
                    """, update_data)

                conn.commit()
                logger.info(f"[BM25] TF update complete for file_id={file_id}, inserts={len(to_insert)}, updates={len(to_update)}")

    except Exception as e:
        logger.error(f"[BM25] Error in calculate_tf_update_db_metadata for file_id={file_id}: {e}", exc_info=True)
        raise e


@mlflow.trace(name="Get AvgDL from DB Metadata")
def get_avgdl_from_db_metadata(file_id: str, logger):
    """Get the tf_part of BM25 for given chunk and file_id"""
    try:
    
        start_time = time.time()
        # BM25_METADATA_DB_TABLE = config.BM25_METADATA_DB_TABLE
        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            with conn.cursor() as cursor:
                # Fetch all rows for the given file_id
                cursor.execute(f"""
                    SELECT COUNT(*), SUM(tf) FROM {BM25_METADATA_DB_TABLE} WHERE file_id = %s;
                """, (file_id,))
                
                result = cursor.fetchone()
                print(result)
                if isinstance(result, dict):
                    n = result.get("COUNT(*)")
                    avgdl = float(result.get("SUM(tf)") / n) if n > 0 else 0
                else:
                    n = result[0]
                    avgdl = result[1] / n if n > 0 else 0
                
                return avgdl
            
            conn.commit()
        print(f"Time taken to update document_metadata: {time.time() - start_time} seconds")

    except Exception as e:
        print(f"Error in update document: {e}")
        raise e
    



# -------------------------------------------------------------------------------------------------------------------

@mlflow.trace(name="Delete TF Summary")
def delete_tf_summary(org_id:int, user_id:int,file_ids:Union[str,List[str]],logger=None):
    try:
        logger.info(f"[BM25] Starting TF deletion for org_id={org_id}, user_id={user_id}, file_ids={file_ids}")
        # BM25_SUMMARY_DB_TABLE = config.BM25_SUMMARY_DB_TABLE
        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            with conn.cursor() as cursor:
                # Check if the table exists, if not create it
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {BM25_SUMMARY_DB_TABLE} (
                        file_id VARCHAR(255),
                        org_id INTEGER,
                        user_id INTEGER,
                        chunk_id VARCHAR(255),
                        term_frequency JSON,
                        tf INTEGER,
                        PRIMARY KEY (file_id, chunk_id)
                    );
                """)
                if isinstance(file_ids, str):
                    file_ids = [file_ids]
                
                # Delete records for the given file_ids
                cursor.execute(f"""
                    DELETE FROM {BM25_SUMMARY_DB_TABLE} WHERE org_id = %s AND user_id = %s AND file_id IN %s;
                """, (org_id, user_id, tuple(file_ids),))
            

    except Exception as e:
        logger.error(f"[BM25] Error in delete_tf_summary: {e}", exc_info=True)
        raise e
    

@mlflow.trace(name="Calculate IDF Summary")
def calculate_idf_summary(query: str, org_id: int, file_ids: Union[None, List[str]] = None, logger=None):
    try:
        logger.info(f"[BM25-IDF] Starting IDF calculation for org_id={org_id}, query='{query}'")
        # BM25_SUMMARY_DB_TABLE = config.BM25_SUMMARY_DB_TABLE
        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            with conn.cursor() as cursor:
                if isinstance(file_ids, list) and len(file_ids) > 0:
                    logger.debug(f"[BM25-IDF] Fetching term_frequency for file_ids={file_ids}")
                    cursor.execute(f"""
                        SELECT term_frequency FROM {BM25_SUMMARY_DB_TABLE} WHERE file_id IN %s;
                    """, (tuple(file_ids),))
                else:
                    logger.debug(f"[BM25-IDF] Fetching term_frequency for org_id={org_id}")
                    cursor.execute(f"""
                        SELECT term_frequency FROM {BM25_SUMMARY_DB_TABLE} WHERE org_id = %s;
                    """, (org_id,))

                result = cursor.fetchall()
                if not result:
                    logger.warning(f"[BM25-IDF] No records found for given input. Returning empty result.")
                    return {}

                # Extract list of term indices from each document
                term_lists = []
                for row in result:
                    tf_json = row.get("term_frequency") if isinstance(row, dict) else row[0]
                    terms = list(json.loads(tf_json).keys())
                    term_lists.append(terms)
                
                logger.debug(f"[BM25-IDF] Collected {len(term_lists)} documents for IDF computation")

                # Calculate document frequency
                bm25 = CustomBM25Encoder()
                doc_freq = bm25.count_values_across_lists(term_lists)
                total_docs = len(term_lists)


                # Compute IDF values for the query
                idf_vals = bm25.custom_doc_query(query, doc_freq=doc_freq, N=total_docs)
                
                logger.info(f"[BM25-IDF] Completed IDF computation for query: '{query}'")
                return idf_vals

    except Exception as e:
        logger.error(f"[BM25-IDF] Error in calculate_idf_summary: {e}", exc_info=True)
        return {}


@mlflow.trace(name="Calculate TF Summary")
def calculate_tf_update_db_summary(org_id: int, user_id: int, file_id: str, chunks: List[str], logger):
    """Efficiently update the Term-Frequency BM25 index for a given file_id and its chunks."""
    logger.info(f"[BM25] Starting optimized TF update for file_id={file_id}, org_id={org_id}, user_id={user_id}, total_chunks={len(chunks)}")
    # BM25_SUMMARY_DB_TABLE = config.BM25_SUMMARY_DB_TABLE
    try:
        bm25 = CustomBM25Encoder()
        chunk_data = []
        chunk_ids = []

        # Preprocess all chunks
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{file_id}_{idx+1}"
            indices, values = bm25._tf(chunk)
            term_freq = {i: val for i, val in zip(indices, values)}
            term_freq_json = json.dumps(term_freq)
            total_tf = sum(values)
            chunk_data.append((file_id, org_id, user_id, chunk_id, term_freq_json, total_tf))
            chunk_ids.append(chunk_id)

        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            with conn.cursor() as cursor:
                logger.debug("[BM25] Ensuring bm25_index table exists")
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {BM25_SUMMARY_DB_TABLE} (
                        file_id VARCHAR(255),
                        org_id INTEGER,
                        user_id INTEGER,
                        chunk_id VARCHAR(255),
                        term_frequency JSON,
                        tf INTEGER,
                        PRIMARY KEY (file_id, chunk_id)
                    );
                """)

                # Step 1: Fetch existing chunk_ids for this file
                logger.debug("[BM25] Fetching existing chunk_ids from DB")
                cursor.execute(f"""
                    SELECT chunk_id FROM {BM25_SUMMARY_DB_TABLE} 
                    WHERE file_id = %s AND chunk_id IN %s;
                """, (file_id, tuple(chunk_ids)))
                existing = cursor.fetchall()
                existing_ids = set(row['chunk_id'] if isinstance(row, dict) else row[0] for row in existing)

                # Step 2: Split data into inserts and updates
                to_insert = [row for row in chunk_data if row[3] not in existing_ids]
                to_update = [row for row in chunk_data if row[3] in existing_ids]

                logger.debug(f"[BM25] {len(to_insert)} new records to insert, {len(to_update)} records to update")

                # Step 3: Execute batch insert
                if to_insert:
                    cursor.executemany(f"""
                        INSERT INTO {BM25_SUMMARY_DB_TABLE} (file_id, org_id, user_id, chunk_id, term_frequency, tf)
                        VALUES (%s, %s, %s, %s, %s, %s);
                    """, to_insert)

                # Step 4: Execute batch update
                if to_update:
                    update_data = [(row[4], row[5], row[0], row[3]) for row in to_update]  # (term_freq_json, tf, file_id, chunk_id)
                    cursor.executemany(f"""
                        UPDATE {BM25_SUMMARY_DB_TABLE} SET term_frequency = %s, tf = %s 
                        WHERE file_id = %s AND chunk_id = %s;
                    """, update_data)

                conn.commit()
                logger.info(f"[BM25] TF update complete for file_id={file_id}, inserts={len(to_insert)}, updates={len(to_update)}")

    except Exception as e:
        logger.error(f"[BM25] Error in calculate_tf_update_db_summary for file_id={file_id}: {e}", exc_info=True)
        raise e



@mlflow.trace(name="Get AvgDL from DB Summary")
def get_avgdl_from_db_summary(file_id: str, logger):
    """Get the tf_part of BM25 for given chunk and file_id"""
    try:
    
        start_time = time.time()
        # BM25_SUMMARY_DB_TABLE = config.BM25_SUMMARY_DB_TABLE
        with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            with conn.cursor() as cursor:
                # Fetch all rows for the given file_id
                cursor.execute(f"""
                    SELECT COUNT(*), SUM(tf) FROM {BM25_SUMMARY_DB_TABLE} WHERE file_id = %s;
                """, (file_id,))
                
                result = cursor.fetchone()
                print(result)
                if isinstance(result, dict):
                    n = result.get("COUNT(*)")
                    avgdl = float(result.get("SUM(tf)") / n) if n > 0 else 0
                else:
                    n = result[0]
                    avgdl = result[1] / n if n > 0 else 0
                
                return avgdl
            
            conn.commit()
        print(f"Time taken to update document: {time.time() - start_time} seconds")

    except Exception as e:
        print(f"Error in update document: {e}")
        raise e
    



