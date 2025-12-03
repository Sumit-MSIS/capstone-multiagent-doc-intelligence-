import boto3
import json
from boto3.dynamodb.conditions import Key
from src.common.llm_status_handler.status_handler import set_websocket_stream_db
from asyncio import to_thread
from datetime import datetime
from src.config.base_config import config

# Initialize DynamoDB table once
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(config.DYNAMODB_MESSAGE_TABLE_NAME)

async def save_chunk_to_dynamo(connection_id, chat_id, chunk_order, chunk_answer, request_id, stream_completed, related_questions, client_id, enable_agent=False):

    return 
    # Handle final answer logic
    if stream_completed and related_questions:       
        chunk_answer = json.dumps({"related_questions": related_questions})


    # Save the chunk
    await to_thread(table.put_item, Item={
        'client_id': client_id, 
        'chat_id': chat_id,
        'chunk_order': str(chunk_order),  # Ensure string for consistent key format
        'chunk_answer': chunk_answer,
        'request_id': request_id,
        'stream_completed_status': stream_completed,
        'enable_agent': enable_agent,  # Store whether this is an agent response
        'inserted_at': datetime.utcnow().isoformat() + 'Z'  #
    })
    
    # print(f"✅ Stored chunk {chunk_order} for chat_id: {chat_id}")

    # Trigger full message handling if stream is complete
    # if stream_completed:
    #     return await handle_stream_complete(connection_id, request_id, chunk_order, chat_id, client_id, related_questions, enable_agent)

# async def handle_stream_complete(connection_id, request_id, final_order, chat_id, client_id, related_questions, enable_agent):
#     try:
#         # Fetch all chunks for the given chat_id
#         response = await to_thread(table.query, KeyConditionExpression=Key('chat_id').eq(chat_id))
#         items = response.get('Items', [])
        
#         # Ensure all chunks are present
#         received_orders = sorted(int(item['chunk_order']) for item in items)
#         expected_orders = list(range(1, final_order + 1))

#         if received_orders == expected_orders:
#             # Sort and combine chunks
#             sorted_chunks = sorted(items, key=lambda x: int(x['chunk_order']))
#             full_message = ''.join(str(item['chunk_answer']) for item in sorted_chunks)

#             # print(f"✅ Assembled full message for chat_id {chat_id}.")

#             final_response = {
#                 "answers": [{"answer": full_message, "page_no": 1, "paragraph_no": 1, "score": 1}],
#                 "related_questions": related_questions
#             }

#             # Send message and clean up
#             await to_thread(set_websocket_stream_db, connection_id, request_id, final_response, chat_id, client_id, enable_agent)
#             # await delete_chunks(chat_id, items)

#             # print("🧹 Deleted all chunks from DynamoDB.")
#             return success_response("Message forwarded and cleaned up.")
#         else:
#             # print("⏳ Waiting for remaining chunks...")
#             return success_response("Waiting for remaining chunks.")
    
#     except Exception as e:
#         # print(f"❌ Error in stream handling: {e}")
#         return error_response("Internal error during stream completion.")

async def delete_chunks(chat_id, items):
    # Batch delete chunks
    def batch_delete():
        with table.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={
                    'chat_id': chat_id,
                    'chunk_order': item['chunk_order']
                })
    await to_thread(batch_delete)

# Utility functions
def success_response(message: str):
    return {
        "statusCode": 200,
        "body": json.dumps({"message": message})
    }

def error_response(message: str):
    return {
        "statusCode": 500,
        "body": json.dumps({"message": message})
    }
