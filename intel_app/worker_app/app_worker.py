from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, timedelta
import json, os
from pydantic import BaseModel
from worker_app.utils.db_connection_manager import DBConnectionManager
from src.config.base_config import config
from worker_app.utils.bm25_encoder import CustomBM25Encoder
from worker_app.logger import request_logger, _log_message, flush_all_logs
from pinecone import Pinecone
from enum import Enum
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
import json
import logging
import time
import uuid

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# In-memory storage 
# Global state
org_state = {}
# lock = asyncio.Lock()
lock_map: dict[str, asyncio.Lock] = {}

TIMEOUT = 30  # seconds
DATA_DIR = "/data"
BATCH_SIZE = 10
# Ensure folder exists
os.makedirs(DATA_DIR, exist_ok=True)

CONTEXTUAL_RAG_DB = config.CONTEXTUAL_RAG_DB
CONTRACT_INTEL_DB= config.CONTRACT_INTEL_DB
BM25_CORPUS_DB_TABLE = config.BM25_CORPUS_DB_TABLE

# this will be only calculated at the app startup only, so there is no processing time wasted here at the time of request processing
async def get_org_tf(org_id: int, logger):
    """Get the term frequencies for all chunks for the provided org_id"""
    try:
        start_time = time.time()
        async with DBConnectionManager(CONTEXTUAL_RAG_DB, logger) as conn:
            async with conn.cursor() as cursor:
                # Fetch all term frequencies and related data for the organization
                await cursor.execute(f"""
                    SELECT file_id, chunk_id, term_frequency, tf,chunk_text
                    FROM {BM25_CORPUS_DB_TABLE}
                    WHERE is_archived IS NULL AND org_id = %s;
                """, (org_id,))
                
                results = await cursor.fetchall()
                
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
                #print(f"for org {org_id} sum_document_length is {total_tf_sum} \n avgdl is {avgdl} \n total_document_count is {total_docs}")
                return {
                    'org_tf_data': org_tf_data,
                    'avgdl':avgdl,
                    'sum_document_length': total_tf_sum,
                    'total_document_count': total_docs
                }
 
    except Exception as e:
        #print(f"error in data {e}")
        logger.error(f"Error in get_org_tf for org_id={org_id}: {e}", exc_info=True)
        raise e

async def bm25_reindexing(org_id: str, logger):
    try:
        bm25 = CustomBM25Encoder()

        tf_retrive_start_time = time.time()
        org_data = await get_org_tf(int(org_id), logger)
    
        logger.info(f"Retrieved org TF data for org_id={org_id} in {time.time() - tf_retrive_start_time:.2f} seconds.")

        if not org_data:
            logger.warning(f"No data found for org_id={org_id}, skipping reindexing")
            return

        reindexing_start_time = time.time()

        # using the previously calculated values during the api call (not using the values from db because even if files are deleted the chunks will still be there so that will mismatch the counts)
        # also in case if there is another api call writing to the table might disturb  
        file_path = get_file_path(str(org_id))
        org_state_data = await load_org_data(file_path)
        avgdl = org_state_data['avgdl']  
 
        logger.info(f"Average dl value before reindexing {avgdl} (value stored in the global state) Using the global state value currently | value recieved from db {org_data['avgdl']}")
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
                # print(chunk_id)
                recalculated_sparse_vectors = await asyncio.to_thread(bm25._recalculate_indices_scores,avgdl, indices, values)
                # print(recalculated_sparse_vectors)
                # Store the recalculated data
                reindexed_data.append({
                    'file_id': file_id,
                    'chunk_id': chunk_id,
                    'sparse_vector': recalculated_sparse_vectors
                })

                # print("success for",chunk_id)
                # #print(reindexed_data)
                
            except Exception as e:
                logger.error(f"Error recalculating sparse vectors for chunk_id={chunk_id}: {e}")
                logger.debug(f"Failed data - indices: {indices}, values: {values}, avgdl: {avgdl}")
                continue
                
        logger.info(f"BM25 reindexing completed for org_id={org_id} in {time.time() - reindexing_start_time:.2f} seconds. Total reindexed chunks: {len(reindexed_data)}")

        if reindexed_data:
            upsertion_start_time = time.time()
            logger.info(f"About to upsert {len(reindexed_data)} chunks for org_id={org_id}")
            print(f"About to upsert {len(reindexed_data)} chunks for org_id={org_id}")
            await asyncio.to_thread(upsert_bm25_reindexed_data, reindexed_data, org_id, logger)

            # upsert_bm25_reindexed_data(reindexed_data, org_id, logger)

            logger.info(f"BM25 reindexed data pinecone upsert completed for org_id={org_id} in {time.time() - upsertion_start_time:.2f} seconds.")
        else:
            logger.info(f"BM25 reindexed data not found for org_id={org_id} in {time.time() - upsertion_start_time:.2f} seconds.")
            #print(f"BM25 Reindexed data not found")

    except Exception as e:
        logger.error(f"Error in bm25_reindexing for org_id={org_id}: {e}", exc_info=True)
        raise e
    finally:
        flush_all_logs(f"{org_id}", str(config.BM25_LOG_GROUP_NAME), "BM25_WORKER")

    
def upsert_bm25_reindexed_data(reindexed_data, org_id, logger, batch_size=100):
    """
    Batch upsert BM25 reindexed vectors to Pinecone, keeping existing metadata intact.
    """
    try:
        # print(f"Received {len(reindexed_data)} vectors to upsert for org_id={org_id}")
        logger.info(f"Received {len(reindexed_data)} vectors to upsert for org_id={org_id}")

        if not reindexed_data:
            logger.warning(f"No reindexed data provided for org_id={org_id}, skipping upsert")
            return

        namespace = f"org_id_{org_id}#"
        index = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX_SPARSE)
        # with open("reindexed.json",'w') as f:
        #     json.dump(reindexed_data,f,indent=2)

        # Process in batches
        for i in range(0, len(reindexed_data), batch_size):
            batch = reindexed_data[i:i+batch_size]
            
            # Fetch existing metadata for all chunk_ids in this batch
            chunk_ids = [str(item['chunk_id']) for item in batch]
            fetched = index.fetch(ids=chunk_ids, namespace=namespace)
            # print("fetched object",fetched)
            vectors_to_upsert = []
            for item in batch:
                chunk_id = str(item['chunk_id'])
                new_vector = item['sparse_vector']
                logger.info(f"chunk id {chunk_id} | sparse vector {new_vector}")
                metadata = {}
                if chunk_id in fetched.vectors:
                    # print("chunk_id",chunk_id)
                    metadata = fetched.vectors[chunk_id].metadata  #  only metadata

                vectors_to_upsert.append({
                    'id': chunk_id,
                    'values': [],  # Required field even for sparse-only vectors
                    "sparse_values": new_vector,
                    'metadata': metadata
                })
            # vectors_to_upsert.append({'vector total count':len(vectors_to_upsert)})
            # with open(f"vector_batch_{i}.json",'w') as f:
            #     json.dump(vectors_to_upsert,f,indent=2)

            # Upsert batch
            index.upsert(vectors=vectors_to_upsert, namespace=namespace)

            logger.info(f"Upserted batch {i // batch_size + 1} ({len(vectors_to_upsert)} vectors) in namespace={namespace}")
            print(f"Upserted batch {i // batch_size + 1} ({len(vectors_to_upsert)} vectors) in namespace={namespace}")
        logger.info(f"Completed upsert of {len(reindexed_data)} reindexed vectors for org_id={org_id}")
        print(f"Completed upsert of {len(reindexed_data)} reindexed vectors for org_id={org_id}")

    except Exception as e:
        logger.error(f"Error in upserting BM25 reindexed data for org_id={org_id}: {e}", exc_info=True)
        raise e
    finally:
        print(f"processed all batches for {org_id}")
        flush_all_logs(f"{org_id}", str(config.BM25_LOG_GROUP_NAME), "BM25_WORKER")

async def get_all_org():
    try:
        org_values = []
        async with DBConnectionManager(CONTRACT_INTEL_DB, None) as conn:
            async with conn.cursor() as cursor:
                # Fetch all the organizations
                await cursor.execute(f"""
                    select org_id from organizations where is_archived is null and org_active = 1;
                """)
                
                results = await cursor.fetchall()
                
                if not results:
                    # logger.warning(f"No organization found in DB")
                    return []
                
                org_values = [org['org_id'] for org in results]
                if not org_values:
                    return []   

        placeholders = ",".join(["%s"] * len(org_values))  # dynamically expand placeholders
        # print(org_values)
        
        # logger.info(f"Fetched org data for these organizations {org_values}")
        async with DBConnectionManager(CONTEXTUAL_RAG_DB, None) as conn:
            async with conn.cursor() as cursor:
                start = time.time()
                # Fetch all term frequencies and related data for the organization
                await cursor.execute(f"""
                    SELECT org_id, file_id, chunk_id, term_frequency, tf 
                    FROM {BM25_CORPUS_DB_TABLE}
                    WHERE org_id IN ({placeholders}) and is_archived is NULL;
                """, (org_values))
                
                results = await cursor.fetchall()
                
                if not results:
                    return []
                    # logger.warning(f"No term frequency data found for given orgs {placeholders}")
                
                org_tf_map = defaultdict(lambda: {"total_docs": 0, "total_tf_sum": 0})

                for row in results:
                    if isinstance(row, dict):
                        org_id = row['org_id']
                        file_id = row['file_id']
                        chunk_id = row['chunk_id']
                        term_freq_json = row['term_frequency']
                        tf = row['tf']
                    else:
                        org_id, file_id, chunk_id, term_freq_json, tf = row
                    
                    org_tf_map[org_id]["total_docs"] += 1
                    org_tf_map[org_id]["total_tf_sum"] += tf
                # logger.info(f"org_tf_map for all org {org_tf_map}")
                loop = asyncio.get_event_loop()
                os.makedirs(DATA_DIR, exist_ok=True)

                new_time = time.time()
                for org_id, stats in org_tf_map.items():
                    # logger.info(f"inside for org_id")
                    avgdl = stats["total_tf_sum"] / stats["total_docs"] if stats["total_docs"] > 0 else 1
                    filtered_data = {
                        "avgdl": avgdl,
                        "sum_document_length": stats["total_tf_sum"],
                        "total_document_count": stats["total_docs"]
                    }
                    file_path = os.path.join(DATA_DIR, f"{org_id}.json")
                    # logger.debug(f"file path {file_path}")
                    await loop.run_in_executor(None, save_json, file_path, filtered_data)
                # logger.info(f"Time taken to create new json for all org {time.time()- new_time}")
                # logger.info(f"Done fetching data for all orgs | time taken {time.time()-start}")
        return org_values
    except Exception as e:
        # logger.exception(f"Error getting all organizations from {CONTRACT_INTEL_DB}")
        #print(f"error getting organizations")
        return []
                
def get_file_path(org_id: str):
    return os.path.join(DATA_DIR, f"{org_id}.json")

async def load_org_data(file_path:str):
    """load the json data for each org id at the startup only"""
    # try:
        # logger.debug(f"Loading all org data at startup")

    # read updated values and return 
    with open(file_path, "r") as f:
        return json.load(f)
        
    # except Exception as e:
    #     logger.exception(f"Error while loading org_data {e}",exc_info=True)
    # finally:
    #     flush_all_logs(f"worker_app_lifespan", str(config.BM25_LOG_GROUP_NAME), "BM25_WORKER")

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

async def save_org_data(org_id: str, data: dict):
    """once worker queue for that org is empty, this method will save the updated values of the org in the json"""
    # try:
    file_path = get_file_path(org_id)
    loop = asyncio.get_event_loop()
    # await loop.run_in_executor(None, lambda: json.dump(data, open(file_path, "w"), indent=2))
    await loop.run_in_executor(None, save_json, file_path, data)
        # with open(file_path, "w") as f:
        #     json.dump(data, f, indent=2)  
    # except Exception as e:
    #     logger.error(f"error in save_org_data() for {org_id} - {str(e)}")
    # finally:
    #     flush_all_logs(f"worker_app_lifespan", str(config.BM25_LOG_GROUP_NAME), "BM25_WORKER")

def create_org_file(file_path, org_id):
    initial_data = {
        "avgdl": 1,
        "sum_document_length": 0,
        "total_document_count": 0
    }
    with open(file_path, "w") as f:
        json.dump(initial_data, f, indent=2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # --- Startup ---
    # logger = request_logger(
    #         f"worker_app_lifespan",
    #         str(config.BM25_LOG_GROUP_NAME),
    #         "BM25_WORKER"
    #     )
    # logger.info(">>> Lifespan startup triggered")   
    print(f"Lifespan initiating for worker app, wait till completion")
    global org_state
    try:
        # get all org from the db and create empty json if not exists
        list_all_orgs = await get_all_org()
        for org_id in list_all_orgs:
            file_path = get_file_path(org_id)
            # print(file_path)
            # if the file does not exists, create one
            if not os.path.exists(file_path): 
                create_org_file(file_path, org_id)

            file_data = await load_org_data(file_path)
            
            # print(file_data)
            if not file_data:
                continue # if no file data was found continue to initialize others
            org_state[str(org_id)] = {
                "total_document_count": file_data["total_document_count"],
                "sum_document_length": file_data["sum_document_length"],
                "avgdl":file_data['avgdl'],
                "last_activity": datetime.now(),
                "waiting_requests": [],
                "queue_task": None
            }
        print(F"lifespan completely initiated")
        print(f"Loaded org states at startup: {list(org_state.keys())}")
        # logger.info(f"Loaded org states at startup: {list(org_state.keys())}")
        yield  # <-- App runs here
    except Exception as e:
        print(f"error in lifespane {str(e)}")
        # logger.error(f"error in lifespane {str(e)}")
        raise e

    # --- Shutdown ---
    # print("Shutting down secondary app, saving states...")
    # logger.info("Shutting down secondary app, saving states...")
    for org_id, state in org_state.items():
        await save_org_data(org_id, {
            "total_document_count": state["total_document_count"],
            "sum_document_length": state["sum_document_length"],
            "avgdl":state['avgdl']
        })

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ActionEnum(str, Enum):
    add = "UPDATE"
    delete = "DELETE"

class ProcessBM25Request(BaseModel):
    """
    request will contain org_id to identify the request
    total_document_length - is the sum of all the term frequencies for the file (each chunk included)
    total_chunks: is total number of chunks in that file 
    action: states whether the new document is getted added or deleted
    """
    org_id: str
    file_id:str
    total_chunks:int
    total_document_length: int
    action: ActionEnum = ActionEnum.add   # default value

"""
Load tested the worker under three conditions 
1. when multiple parallel request for same org without time difference
Remarks: working properly, processes 10 requests first returns, processes next 10 and so on

2. when multiple request for same org with time difference > defined timeout 
Remarks: can handle properly

3. when multiple request for same org where request is received just before timeout could finish

Remarks: Working 
"""

@app.post("/bm25/get-org-avgdl")
async def process_request(payload: ProcessBM25Request):
    
    try:

        logger = request_logger(
            f"{payload.org_id}",
            str(config.BM25_LOG_GROUP_NAME),
            "BM25_WORKER"
        )
        org_id = payload.org_id
        file_id=payload.file_id
        total_chunks = payload.total_chunks
        total_document_length = payload.total_document_length
        action = payload.action
        res_time = time.time()
        logger.info(f"Request recieved for org_id {org_id} | file_id {file_id} | resquest : {payload.dict}")
        # Ensure org-specific lock exists
        if org_id not in lock_map:
            print(f"locking for {org_id}")
            lock_map[org_id] = asyncio.Lock()

        # logger.info(f"org_state {org_id} {org_state}")
        async with lock_map[org_id]: 
        # async with lock:
            # if request is coming for organization which did not have any data in db at the time of app startup
            if org_id not in org_state:
                # this is to ensure that organization which is new will also have it's initialized json
                new_file_path = get_file_path(org_id)
                if not os.path.exists(new_file_path): 
                    logger.info(f"Previously No data found for {org_id} | Loading data from DB")
                    create_org_file(new_file_path, org_id) 
                
                
                new_org =  await get_org_tf(int(org_id),logger)

                logger.debug(f"Data retreived for {org_id} from DB")
                if not new_org: 
                    return {
                        "org_id":org_id,
                        "total_document_count": 0,
                        "sum_document_length": 0,
                        "avgdl": 1, # to avoid divide by 0 error,
                        "msg":f"Error : No organization bm25 data found in database for {org_id}"
                    }

                # Need of below condition : deletion action is called after the chunks are already deleted from pinecone and database
                # so in case (for any reason) the previous state of a preexisting organization is not saved or created, we try to fetch the fresh data from db
                # in case of deletion the data is already deleted and we are reading the deleted data so the count that we read here in case of deletion will always be actual value - total recieved chunks to delete
                # the process after this will again subtract the deleted chunks from the final state, so the count in the org state will always be lesser than actually deleted. deletion is happening two times
                if action == "DELETE":
                    new_org['sum_document_length'] = new_org['sum_document_length']+total_document_length
                    new_org['total_document_count'] = new_org['total_document_count'] + total_chunks
                    new_org['avgdl'] = new_org['sum_document_length']/new_org['total_document_count'] 
                    logger.info(F"Existing state did not have data for {org_id}: so the actual values are {new_org}")
                # same as delete, updation has already happended so it will sum up two times 
                elif action == "UPDATE":
                    new_org['sum_document_length'] = new_org['sum_document_length']-total_document_length
                    new_org['total_document_count'] = new_org['total_document_count'] - total_chunks
                    new_org['avgdl'] = new_org['sum_document_length'] / new_org['total_document_count'] if new_org['total_document_count'] > 0 else 1
                    logger.info(F"Existing state did not have data for {org_id}: so the actual values are {new_org}")
                # on startup read the updated values from DB for that org_id and dump in json and return the values
                # keep only required keys
                keys_to_keep = ['avgdl', 'sum_document_length', 'total_document_count']
                filtered_data = {k: new_org[k] for k in keys_to_keep}

                # Save to file
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: json.dump(filtered_data, open(new_file_path, "w"), indent=2))

                # read updated values and return 
                with open(new_file_path, "r") as f:
                    new_org_data =  json.load(f)

                if not new_org_data:
                    return {
                        "org_id":org_id,
                        "total_document_count": 0,
                        "sum_document_length": 0,
                        "avgdl": 1, # to avoid divide by 0 error,
                        "msg":f"Error : No organization bm25 data found in database for {org_id}"
                    }
                
                org_state[org_id] = {
                    "total_document_count": new_org_data['total_document_count'],
                    "sum_document_length": new_org_data['sum_document_length'],
                    "last_activity": datetime.now(),
                    "waiting_requests": [],
                    "queue_task": None
                }
                logger.info(f"Data Saved for {org_id} | total chunks : {new_org_data['total_document_count']}, sum document length {new_org_data['sum_document_length']}, avgdl {new_org_data['sum_document_length']/new_org_data['total_document_count']} ")
                await save_org_data(org_id, {
                    "total_document_count": new_org_data['total_document_count'],
                    "sum_document_length": new_org_data['sum_document_length'],
                    "avgdl": new_org_data['sum_document_length']/new_org_data['total_document_count']
                })

            state = org_state[org_id] # a global state maintained for all org_id just after the app startup

            logger.info(f"Loaded values before processing for {org_id} | total document count {state['total_document_count']} | sum document length {state['sum_document_length']}")
            # if the payload action is update, means new file is getting added into corpus so add values to existing
            if action == "UPDATE":
                state["total_document_count"] += total_chunks
                state["sum_document_length"] += total_document_length
            # if action is delete, that means old file is getting deleted so subtract values
            elif action == "DELETE":
                state["total_document_count"] -= total_chunks
                state["sum_document_length"] -= total_document_length
            
            if state['total_document_count'] > 0:
                print("total document count is 0")
                state['avgdl'] = state["sum_document_length"] / state["total_document_count"]
            else:
                state['avgdl'] = 1
            # print(f"last activity for response {org_id} is {datetime.now()}")
            state["last_activity"] = datetime.now()
            logger.info(f"to add or delete -> total chunks {total_chunks}, total document length {total_document_length} \n values after {action} opeation -  Most recent values for {org_id} | avgdl {state['avgdl']} | total document count {state['total_document_count']} | sum document length {state['sum_document_length']}")
            loop = asyncio.get_event_loop()
            fut = loop.create_future()
            state["waiting_requests"].append(fut)

            if not state["queue_task"] or state["queue_task"].done():
                state["queue_task"] = asyncio.create_task(worker(org_id,logger))

        result = await fut
        logger.info(f"Response returned for org_id {org_id} | Response time: {time.time()-res_time} | Response - {result}")
        #
        return result
    except Exception as e:
        logger.exception(f"Error in [process_request] for {org_id} - {str(e)}",exc_info=True)
        # flush_all_logs(f"{search_id}", str(config.BM25_LOG_GROUP_NAME), "BM25_WORKER")
        raise
    finally:
        flush_all_logs(f"{payload.org_id}", str(config.BM25_LOG_GROUP_NAME), "BM25_WORKER")

async def worker(org_id: str,logger):
    """
    Worker will handle all the incoming requests from multiple organizations
    for each organization there will be a separate waiting_list , the worker will keep tagging each recieved request with a timestamp and keep checking 
    the current time and the last checked time is more than the timeout if yes it will process whatever requests are already there in waiting list
    if timeout isnt reached but waiting list is already more than batch size then process batch size and leftovers will be processed in next batch
    """
    try:
        state = org_state[org_id]
        while True:
            # The sleep call is there to prevent the worker from continuously busy-looping (CPU spinning) when there are no tasks in the queue.
            await asyncio.sleep(1)
            # Condition 1: Timeout reached
            timeout_reached = (datetime.now() - state["last_activity"]) > timedelta(seconds=TIMEOUT)

            # Condition 2: Enough requests accumulated
            batch_ready = len(state["waiting_requests"]) >= BATCH_SIZE

            if timeout_reached or batch_ready:
                # Pop the first batch (up to BATCH_SIZE), leave rest in queue
                to_process = state["waiting_requests"][:BATCH_SIZE]
                # print(f"to_process for woker: {to_process}")
                state["waiting_requests"] = state["waiting_requests"][BATCH_SIZE:]
                # print(f"remaining leftovers {state["waiting_requests"]}")

                result = {
                    "org_id": org_id,
                    "total_document_count": state["total_document_count"],
                    "sum_document_length": state["sum_document_length"],
                    "avgdl": state["avgdl"],
                    "msg":"success: updated values calculated"
                }

                """
                BELOW VALUES ARE TO DEBUG THE RESULTS, NOT NECESSARY TO SEND IN THE RESPONSE
                # "last_activity": state["last_activity"],
                    # "timeout_value":datetime.now() - state["last_activity"],
                    # "requests_to_process": summarize_futures(waiting_req_before_process),
                    # "requests_processed": summarize_futures(to_process),
                    # "remaining_to_process":summarize_futures(state["waiting_requests"]),
                    # "processed_batch_length":len(to_process)

                    def summarize_futures(futures):
                        return [
                            {
                                "done": f.done(),
                                "cancelled": f.cancelled(),
                                "result": f.result() if f.done() and not f.cancelled() else None,
                                "id": id(f)  # just a unique identifier
                            }
                            for f in futures
                        ]
                """
                for fut in to_process:
                    if not fut.done():
                        fut.set_result(result)

                await save_org_data(org_id, {
                    "total_document_count": state["total_document_count"],
                    "sum_document_length": state["sum_document_length"],
                    "avgdl": state["avgdl"],
                })

                # -------------------------------------------------------------------
                #Trigger background reindex for this org after each batch
                logger.info(f"Starting reindexing for {org_id}")
                asyncio.create_task(bm25_reindexing(org_id, logger))
                # -------------------------------------------------------------------

                # If timeout triggered, flush *all* and exit loop
                if timeout_reached:
                    # process any leftovers
                    leftovers = state["waiting_requests"]
                    for fut in leftovers:
                        if not fut.done():
                            fut.set_result(result)
                    await save_org_data(org_id, {
                        "total_document_count": state["total_document_count"],
                        "sum_document_length": state["sum_document_length"],
                        "avgdl": state["avgdl"],
                    })
                    state["waiting_requests"] = []
                    break
    except Exception as e:
        logger.exception(f"Error in [worker] for org_id {org_id} - {str(e)}")
    finally:
        flush_all_logs(f"{org_id}", str(config.BM25_LOG_GROUP_NAME), "BM25_WORKER")
