import os
import time
import pandas  as pd
from mlflow_trace.mlflow_utils import get_experiment_ids,create_org_user_id_schema,update_total_cost,update_cost_tags,log_data_with_user_ids, log_data_with_user_ids_intel_chat
from mlflow import MlflowClient
import mlflow
# from dotenv import load_dotenv
# load_dotenv()
from src.config.base_config import config

TRACKING_URI = config.MLFLOW_TRACKING_URI
EMBEDDING_TRACE = config.MLFLOW_EMBEDDING_TRACE
CHAT_TRACE = config.MLFLOW_CHAT_TRACE
MLFLOW_EXPERIMENT_NAMES = config.MLFLOW_EXPERIMENT_NAMES

client = MlflowClient(TRACKING_URI)

costs_per_million_tokens = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "cache": 1.25},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600, "cache": 0.075},
    "gpt-4o-mini-search-preview": {"input": 0.150, "output": 0.600, "cache": 0.00},
    "gpt-4o-search-preview": {"input": 2.50, "output": 10.00, "cache": 0.00},
    "text-embedding-3-large": {"input": 0.13},  
    "text-embedding-3-small": {"input": 0.02},  
    "text-embedding-ada-002": {"input": 0.10}
}

# Convert per 1M token costs to per-token costs
costs_per_token = {
    model: {k: v / 1_000_000 for k, v in costs.items()}
    for model, costs in costs_per_million_tokens.items()
}

search_cost_per_thousand_calls = {
    "gpt-4o-mini-search-preview": {
        'low': 25.00, 
        'medium': 27.50 ,
        'high':30.00 
    },

    "gpt-4o-search-preview": {
        'low': 30.00,
        'medium': 35.00,
        'high': 50.00
    }
}

search_cost_per_call = {
    model: {k: v / 1000 for k, v in costs.items()}
    for model, costs in search_cost_per_thousand_calls.items()
}


gpt_mini_search = 'gpt-4o-mini-search-preview_requests'
gpt_4o_search = 'gpt-4o_search-preview_requests'


def main(exps, env:str,max_traces=10):
    try:

        start_time = time.time()
        for experiment_name in exps:
            experiment_name_env = f"{experiment_name} - {env}"
            EXPERIMENT_IDS = get_experiment_ids([experiment_name_env],client=client)
            if not EXPERIMENT_IDS :
                print(f"Experiment {experiment_name} not found")
                continue 

            target_request_df = mlflow.search_traces(experiment_ids=EXPERIMENT_IDS,filter_string="tag.cost = 'False'",max_results=max_traces)
            target_request_ids = target_request_df["trace"].to_list()
            
            if not target_request_ids or  len(target_request_ids) == 0:
                print(f"No traces found in {experiment_name} with cost = False")
                continue 
            
            data = {}

            for trace_obj in target_request_ids:
                org_id = f"org_id_{trace_obj.info.tags['org_id']}"
                user_id = f"user_id_{trace_obj.info.tags['user_id']}"
                
                request_input = trace_obj.info.request_metadata.get('mlflow.traceInputs',{})
                request_output = trace_obj.info.request_metadata.get('mlflow.traceOutputs',{})
                request_timestamp_ms = trace_obj.info.timestamp_ms
                request_id = trace_obj.info.request_id
                request_execution_time = trace_obj.info.execution_time_ms

                org_id_val = trace_obj.info.tags.get('org_id', '')
                user_id_val = trace_obj.info.tags.get('user_id','')
                    
                if org_id not in data:
                    data = create_org_user_id_schema(data,org_id,user_id,"XYZ",flag="org_id")
                
                if user_id not in data[org_id]:
                    data = create_org_user_id_schema(data,org_id,user_id,"XYZ",flag="user_id")

                spans = trace_obj.search_spans()
                data[org_id][user_id]['requests'] += 1

                for span in spans:
                    if EMBEDDING_TRACE in span.name and len(span.events) == 0:
                        inputs = span.inputs
                        if not inputs:
                            print("Inputs are empty")
                            continue

                        if inputs['model'] not in data[org_id][user_id]['embedding']:
                            data = create_org_user_id_schema(data,org_id,user_id,model_name=inputs['model'],flag="embedding")
                        input_tokens = len(inputs['input'][0])

                        data[org_id][user_id]['embedding'][inputs['model']]['input_tokens'] += input_tokens
                        data[org_id][user_id]['embedding'][inputs['model']]['requests'] += 1

                        exec_time = span.end_time_ns - span.start_time_ns
                        temp = {
                            'org_id': org_id_val,
                            'user_id':user_id_val,
                            'request_id': request_id,
                            'request_execution_time': request_execution_time,
                            'request_input': request_input,
                            'request_output': request_output,
                            'request_timestamp': request_timestamp_ms,
                            'execution_time': exec_time,
                            'service':experiment_name,
                            'env': env,
                            'input_tokens': input_tokens,
                            'model': inputs['model'],
                            'type': "embedding",
                            'total_tokens': input_tokens,
                            'total_cost': costs_per_token[inputs['model']]['input'] * input_tokens
                        }
                        
                        data[org_id][user_id]['query'].append(temp)

                    elif CHAT_TRACE in span.name and len(span.events) == 0:
                        temp = {}
                        outputs = span.outputs
                        inputs = span.inputs

                        if not inputs or not outputs:
                            print("Inputs or Outputs are empty")
                            continue

                        usage_metadata =  outputs['usage']
                        model_name = outputs["model"]
                        if "gpt" in model_name:
                            model_name = model_name[:-11]

                        input_tokens = usage_metadata['prompt_tokens']
                        output_tokens = usage_metadata['completion_tokens'] 
                        if model_name in ["gpt-4o-mini","gpt-4o","gpt-4o-mini-search-preview","gpt-4o-search-preview"]:
                            cache_read = usage_metadata['prompt_tokens_details']['cached_tokens']
                        else:
                            cache_read = 0
                        
                        processed_tokens = input_tokens - cache_read  ### Processed tokens = input tokens - cache read

                        if model_name not in data[org_id][user_id]['chat']:
                            data = create_org_user_id_schema(data,org_id,user_id,model_name,flag="chat")
                        
                        data[org_id][user_id]['chat'][model_name]['input_tokens'] += processed_tokens
                        data[org_id][user_id]['chat'][model_name]['output_tokens'] += output_tokens
                        data[org_id][user_id]['chat'][model_name]['cached_tokens'] += cache_read
                        data[org_id][user_id]['chat'][model_name]['requests'] += 1

                        if 'search-preview' in model_name:
                            search_cost = search_cost_per_call.get(model_name,{}).get('medium',0)
                        else:
                            search_cost = 0.0


                        messages_user = ""
                        messages_system = ""

                        input_messages = inputs.get("messages", None)
                        if input_messages:
                            for message in input_messages:
                                if message.get("role") == "system" and isinstance(message.get("content"), str):
                                    messages_system += message.get("content") + "\n"
                                elif message.get("role") == "user" and isinstance(message.get("content"), str):
                                    messages_user += message.get("content") + "\n\n"

                        
                        exec_time = span.end_time_ns - span.start_time_ns

                        temp = {

                            'org_id': org_id_val,
                            'user_id':user_id_val,
                            'service':experiment_name,
                            'env': env,
                            'request_input': request_input,
                            'request_output': request_output,
                            'request_timestamp': request_timestamp_ms,
                            'request_id': request_id,
                            'request_execution_time': request_execution_time,
                            'execution_time': exec_time,
                            'system_msg': messages_system,
                            'user_msg': messages_user,
                            'output_msg': outputs.get("choices",{})[0].get("message",{}).get("content"," "),
                            'input_tokens': processed_tokens,
                            'output_tokens': output_tokens,
                            'cached_tokens': cache_read,
                            'model': model_name,
                            'type': "chat",
                            'input_cost': costs_per_token[model_name]['input'] * processed_tokens,
                            'output_cost': costs_per_token[model_name]['output'] * output_tokens,
                            'cached_cost': costs_per_token[model_name]['cache'] * cache_read,
                            'total_tokens': processed_tokens + output_tokens + cache_read,
                            'total_cost': (costs_per_token[model_name]['input'] * processed_tokens) + (costs_per_token[model_name]['output'] * output_tokens) + (costs_per_token[model_name]['cache'] * cache_read) + search_cost
                        
                        }


                        data[org_id][user_id]['query'].append(temp)
                  
            data = update_total_cost(data)
            if not data:
                print("Error in updating total cost")
                return 
            log_data_with_user_ids(experiment_name_env, data, TRACKING_URI)
            
            update_cost_tags(target_request_ids,client,"cost")

        print(f"Time taken :",time.time()-start_time)
    
    except Exception as e:
        print("Error in Mlflow Runner main function:",str(e))
        raise e




def main_intel(env:str,max_traces=10):
    try:

        start_time = time.time()
        experiment_name = "Intel Chat"
        experiment_name_env = f"{experiment_name} - {env}"
        EXPERIMENT_IDS = get_experiment_ids([experiment_name_env],client=client)
        if not EXPERIMENT_IDS :
            print(f"Experiment {experiment_name} not found")
            return 
             

        target_request_df = mlflow.search_traces(experiment_ids=EXPERIMENT_IDS,filter_string="tag.cost = 'False' and tag.agent = 'None'",max_results=max_traces)
        target_request_ids = target_request_df["trace"].to_list()
        
        if not target_request_ids or  len(target_request_ids) == 0:
            print(f"No traces found in {experiment_name} with cost = False")
            return 
             
        
        data = {}

        for trace_obj in target_request_ids:
            org_id = f"org_id_{trace_obj.info.tags['org_id']}"
            user_id = f"user_id_{trace_obj.info.tags['user_id']}"
            
            request_input = trace_obj.info.request_metadata.get('mlflow.traceInputs',{})
            request_output = trace_obj.info.request_metadata.get('mlflow.traceOutputs',{})
            request_timestamp_ms = trace_obj.info.timestamp_ms
            request_id = trace_obj.info.request_id
            request_execution_time = trace_obj.info.execution_time_ms

            org_id_val = trace_obj.info.tags.get('org_id', '')
            user_id_val = trace_obj.info.tags.get('user_id','')
                
            if org_id not in data:
                data = create_org_user_id_schema(data,org_id,user_id,"XYZ",flag="org_id")
            
            if user_id not in data[org_id]:
                data = create_org_user_id_schema(data,org_id,user_id,"XYZ",flag="user_id")

            spans = trace_obj.search_spans()
            data[org_id][user_id]['requests'] += 1

            for span in spans:
                if EMBEDDING_TRACE in span.name and len(span.events) == 0:
                    inputs = span.inputs
                    if not inputs:
                        print("Inputs are empty")
                        continue

                    if inputs['model'] not in data[org_id][user_id]['embedding']:
                        data = create_org_user_id_schema(data,org_id,user_id,model_name=inputs['model'],flag="embedding")
                    input_tokens = len(inputs['input'][0])

                    data[org_id][user_id]['embedding'][inputs['model']]['input_tokens'] += input_tokens
                    data[org_id][user_id]['embedding'][inputs['model']]['requests'] += 1

                    exec_time = span.end_time_ns - span.start_time_ns
                    temp = {
                        'org_id': org_id_val,
                        'user_id':user_id_val,
                        'request_id': request_id,
                        'request_execution_time': request_execution_time,
                        'request_input': request_input,
                        'request_output': request_output,
                        'request_timestamp': request_timestamp_ms,
                        'execution_time': exec_time,
                        'service':experiment_name,
                        'env': env,
                        'input_tokens': input_tokens,
                        'model': inputs['model'],
                        'type': "embedding",
                        'total_tokens': input_tokens,
                        'total_cost': costs_per_token[inputs['model']]['input'] * input_tokens
                    }
                    
                    data[org_id][user_id]['query'].append(temp)

                elif CHAT_TRACE in span.name and span.name != "Completions_Usage":
                    temp = {}
                    outputs = span.outputs
                    inputs = span.inputs
                    
                    # print(f"\n\n-------------\n{span.inputs}--------------\n\n")
                    if not inputs:
                        print("Inputs or Outputs are empty")
                        continue

                    if inputs.get("stream",False):
                        # print("Streamed response detected, skipping trace")
                        model_name = inputs.get("model", "Unknown")
                        message_system = ""
                        message_user = ""
                        if inputs.get("messages") and isinstance(inputs["messages"], list):
                            for message in inputs["messages"]:
                                if message.get("role") == "system" and isinstance(message.get("content"), str):
                                    message_system += message.get("content") + "\n"
                                elif message.get("role") == "user" and isinstance(message.get("content"), str):
                                    message_user += message.get("content") + "\n\n"
                        
                        exec_time = span.end_time_ns - span.start_time_ns

                        if model_name not in data[org_id][user_id]['chat']:
                            data = create_org_user_id_schema(data,org_id,user_id,model_name,flag="chat")
                        
                        print(f"\n>>>>> {model_name} - {data[org_id][user_id]['chat'][model_name]['requests']}<<<<\n")
                        data[org_id][user_id]['chat'][model_name]['input_tokens'] += 0
                        data[org_id][user_id]['chat'][model_name]['output_tokens'] += 0
                        data[org_id][user_id]['chat'][model_name]['cached_tokens'] += 0
                        data[org_id][user_id]['chat'][model_name]['requests'] += 1

                        temp = {
                            'org_id': org_id_val,
                            'user_id': user_id_val,
                            'service': experiment_name,
                            'env': env,
                            'request_input': request_input,
                            'request_output': request_output,
                            'request_timestamp': request_timestamp_ms,
                            'request_id': request_id,
                            'request_execution_time': request_execution_time,
                            'execution_time': exec_time,
                            'system_msg': message_system,
                            'user_msg': message_user,
                            'output_msg': outputs,
                            'model': model_name,
                            'type': "chat",
                            'input_tokens': 0,  # No input tokens for streamed responses
                            'output_tokens': 0,  # No output tokens for streamed responses
                            'cached_tokens': 0,  # No cached tokens for streamed responses
                            'input_cost': 0,  # No cost for streamed responses
                            'output_cost': 0,  # No cost for streamed responses
                            'cached_cost': 0,  # No cached cost for streamed responses
                            'total_tokens': 0,  # No tokens for streamed responses
                            'total_cost': 0  # No cost for streamed responses
                        }

                        data[org_id][user_id]['query'].append(temp)


                    else:
                        usage_metadata =  outputs['usage']
                        model_name = outputs["model"]
                        if "gpt" in model_name:
                            model_name = model_name[:-11]

                        input_tokens = usage_metadata['prompt_tokens']
                        output_tokens = usage_metadata['completion_tokens'] 
                        if model_name in ["gpt-4o-mini","gpt-4o","gpt-4o-mini-search-preview","gpt-4o-search-preview"]:
                            cache_read = usage_metadata['prompt_tokens_details']['cached_tokens']
                        else:
                            cache_read = 0
                        
                        processed_tokens = input_tokens - cache_read  ### Processed tokens = input tokens - cache read

                        if model_name not in data[org_id][user_id]['chat']:
                            data = create_org_user_id_schema(data,org_id,user_id,model_name,flag="chat")
                        
                        data[org_id][user_id]['chat'][model_name]['input_tokens'] += processed_tokens
                        data[org_id][user_id]['chat'][model_name]['output_tokens'] += output_tokens
                        data[org_id][user_id]['chat'][model_name]['cached_tokens'] += cache_read
                        data[org_id][user_id]['chat'][model_name]['requests'] += 1

                        if 'search-preview' in model_name:
                            search_cost = search_cost_per_call.get(model_name,{}).get('medium',0)
                        else:
                            search_cost = 0.0


                        messages_user = ""
                        messages_system = ""

                        input_messages = inputs.get("messages", None)
                        if input_messages:
                            for message in input_messages:
                                if message.get("role") == "system" and isinstance(message.get("content"), str):
                                    messages_system += message.get("content") + "\n"
                                elif message.get("role") == "user" and isinstance(message.get("content"), str):
                                    messages_user += message.get("content") + "\n\n"

                        
                        exec_time = span.end_time_ns - span.start_time_ns

                        temp = {

                            'org_id': org_id_val,
                            'user_id':user_id_val,
                            'service':experiment_name,
                            'env': env,
                            'request_input': request_input,
                            'request_output': request_output,
                            'request_timestamp': request_timestamp_ms,
                            'request_id': request_id,
                            'request_execution_time': request_execution_time,
                            'execution_time': exec_time,
                            'system_msg': messages_system,
                            'user_msg': messages_user,
                            'output_msg': outputs.get("choices",{})[0].get("message",{}).get("content"," "),
                            'input_tokens': processed_tokens,
                            'output_tokens': output_tokens,
                            'cached_tokens': cache_read,
                            'model': model_name,
                            'type': "chat",
                            'input_cost': costs_per_token[model_name]['input'] * processed_tokens,
                            'output_cost': costs_per_token[model_name]['output'] * output_tokens,
                            'cached_cost': costs_per_token[model_name]['cache'] * cache_read,
                            'total_tokens': processed_tokens + output_tokens + cache_read,
                            'total_cost': (costs_per_token[model_name]['input'] * processed_tokens) + (costs_per_token[model_name]['output'] * output_tokens) + (costs_per_token[model_name]['cache'] * cache_read) + search_cost
                        
                        }


                        data[org_id][user_id]['query'].append(temp)
                
                elif span.name == "Completions_Usage":
                    inputs = span.inputs
                    outputs = span.outputs

                    if not inputs:
                        print("Inputs or Outputs are empty")
                        continue

                    model_name = inputs["chunk_obj"]["model"]
                    if "gpt" in model_name:
                        model_name = model_name[:-11]
                    
                    input_tokens = outputs["prompt_tokens"]
                    output_tokens = outputs["completion_tokens"]
                    cache_read = outputs["prompt_tokens_details"].get("cached_tokens", 0)
                    processed_tokens = input_tokens - cache_read  ### Processed tokens = input tokens - cache read

                    if model_name not in data[org_id][user_id]['chat']:
                        data = create_org_user_id_schema(data,org_id,user_id,model_name,flag="chat")
                    
                    data[org_id][user_id]['chat'][model_name]['input_tokens'] += processed_tokens
                    data[org_id][user_id]['chat'][model_name]['output_tokens'] += output_tokens
                    data[org_id][user_id]['chat'][model_name]['cached_tokens'] += cache_read
                    data[org_id][user_id]['chat'][model_name]['requests'] += 0

                    if 'search-preview' in model_name:
                        search_cost = search_cost_per_call.get(model_name,{}).get('medium',0)
                    else:
                        search_cost = 0.0

                    exec_time = span.end_time_ns - span.start_time_ns

                    temp = {

                        'org_id': org_id_val,
                        'user_id':user_id_val,
                        'service':experiment_name,
                        'env': env,
                        'request_input': request_input,
                        'request_output': request_output,
                        'request_timestamp': request_timestamp_ms,
                        'request_id': request_id,
                        'request_execution_time': request_execution_time,
                        'execution_time': exec_time,
                        'input_tokens': processed_tokens,
                        'output_tokens': output_tokens,
                        'cached_tokens': cache_read,
                        'model': model_name,
                        'type': "chat",
                        'input_cost': costs_per_token[model_name]['input'] * processed_tokens,
                        'output_cost': costs_per_token[model_name]['output'] * output_tokens,
                        'cached_cost': costs_per_token[model_name]['cache'] * cache_read,
                        'total_tokens': processed_tokens + output_tokens + cache_read,
                        'total_cost': (costs_per_token[model_name]['input'] * processed_tokens) + (costs_per_token[model_name]['output'] * output_tokens) + (costs_per_token[model_name]['cache'] * cache_read) + search_cost
                    
                    }


                    data[org_id][user_id]['query'].append(temp)
        
                
        data = update_total_cost(data)
        if not data:
            print("Error in updating total cost")
            return 
        log_data_with_user_ids_intel_chat(experiment_name_env, data, TRACKING_URI, agent_flag="None")
        
        update_cost_tags(target_request_ids,client,"cost")

        print(f"Time taken :",time.time()-start_time)
    
    except Exception as e:
        print("Error in Mlflow Runner main_intel function:",str(e))
        raise e




def main_agent(env:str,max_traces=10):
    try:

        start_time = time.time()
        experiment_name = "Intel Chat"
        experiment_name_env = f"{experiment_name} - {env}"
        EXPERIMENT_IDS = get_experiment_ids([experiment_name_env],client=client)
        print(EXPERIMENT_IDS)
        if not EXPERIMENT_IDS :
            print(f"Experiment {experiment_name} not found")
            return 
             

        target_request_df = mlflow.search_traces(experiment_ids=EXPERIMENT_IDS,filter_string="tag.cost = 'False' and tag.agent = 'True' and tag.user_id = '671'",max_results=4)
        target_request_ids = target_request_df["trace"].to_list()
        print(f"Found {len(target_request_ids)} traces")
        if not target_request_ids or  len(target_request_ids) == 0:
            print(f"No traces found in {experiment_name} with cost = False")
            return 
             
        
        data = {}

        for trace_obj in target_request_ids:
            print(f"Processing trace {trace_obj.info.trace_id}")
            org_id = f"org_id_{trace_obj.info.tags['org_id']}"
            user_id = f"user_id_{trace_obj.info.tags['user_id']}"
            
            request_input = trace_obj.info.request_metadata.get('mlflow.traceInputs',{})
            request_output = trace_obj.info.request_metadata.get('mlflow.traceOutputs',{})
            request_timestamp_ms = trace_obj.info.timestamp_ms
            request_id = trace_obj.info.request_id
            request_execution_time = trace_obj.info.execution_time_ms

            org_id_val = trace_obj.info.tags.get('org_id', '')
            user_id_val = trace_obj.info.tags.get('user_id','')
                
            if org_id not in data:
                data = create_org_user_id_schema(data,org_id,user_id,"XYZ",flag="org_id")
            
            if user_id not in data[org_id]:
                data = create_org_user_id_schema(data,org_id,user_id,"XYZ",flag="user_id")

            spans = trace_obj.search_spans()
            data[org_id][user_id]['requests'] += 1

            for span in spans:
                print(f"Processing span {span.name}")
                if EMBEDDING_TRACE in span.name and len(span.events) == 0:
                    inputs = span.inputs
                    if not inputs:
                        print("Inputs are empty")
                        continue

                    if inputs['model'] not in data[org_id][user_id]['embedding']:
                        data = create_org_user_id_schema(data,org_id,user_id,model_name=inputs['model'],flag="embedding")
                    input_tokens = len(inputs['input'][0])

                    data[org_id][user_id]['embedding'][inputs['model']]['input_tokens'] += input_tokens
                    data[org_id][user_id]['embedding'][inputs['model']]['requests'] += 1

                    exec_time = 0
                    temp = {
                        'org_id': org_id_val,
                        'user_id':user_id_val,
                        'request_id': request_id,
                        'request_execution_time': request_execution_time,
                        'request_input': request_input,
                        'request_output': request_output,
                        'request_timestamp': request_timestamp_ms,
                        'execution_time': exec_time,
                        'service':experiment_name,
                        'env': env,
                        'input_tokens': input_tokens,
                        'model': inputs['model'],
                        'type': "embedding",
                        'total_tokens': input_tokens,
                        'total_cost': costs_per_token[inputs['model']]['input'] * input_tokens
                    }
                    
                    data[org_id][user_id]['query'].append(temp)

                elif CHAT_TRACE in span.name and "Completions_Usage" != span.name:
                    temp = {}
                    outputs = span.outputs
                    inputs = span.inputs

                    if not inputs:
                        print("Inputs or Outputs are empty")
                        continue

                    if inputs.get("stream",False):
                        # print("Streamed response detected, skipping trace")
                        model_name = inputs.get("model", "Unknown")
                        message_system = ""
                        message_user = ""
                        if inputs.get("messages") and isinstance(inputs["messages"], list):
                            for message in inputs["messages"]:
                                if message.get("role") == "system" and isinstance(message.get("content"), str):
                                    message_system += message.get("content") + "\n"

                                elif message.get("role") == "user" and isinstance(message.get("content"), str):
                                    message_user += message.get("content") + "\n\n"
                        
                        exec_time = 0

                        if model_name not in data[org_id][user_id]['chat']:
                            data = create_org_user_id_schema(data,org_id,user_id,model_name,flag="chat")
                        
                        data[org_id][user_id]['chat'][model_name]['input_tokens'] += 0
                        data[org_id][user_id]['chat'][model_name]['output_tokens'] += 0
                        data[org_id][user_id]['chat'][model_name]['cached_tokens'] += 0
                        data[org_id][user_id]['chat'][model_name]['requests'] += 1

                        temp = {
                            'org_id': org_id_val,
                            'user_id': user_id_val,
                            'service': experiment_name,
                            'env': env,
                            'request_input': request_input,
                            'request_output': request_output,
                            'request_timestamp': request_timestamp_ms,
                            'request_id': request_id,
                            'request_execution_time': request_execution_time,
                            'execution_time': exec_time,
                            'system_msg': message_system,
                            'user_msg': message_user,
                            'output_msg': outputs,
                            'model': model_name,
                            'type': "chat",
                            'input_tokens': 0,  # No input tokens for streamed responses
                            'output_tokens': 0,  # No output tokens for streamed responses
                            'cached_tokens': 0,  # No cached tokens for streamed responses
                            'input_cost': 0,  # No cost for streamed responses
                            'output_cost': 0,  # No cost for streamed responses
                            'cached_cost': 0,  # No cached cost for streamed responses
                            'total_tokens': 0,  # No tokens for streamed responses
                            'total_cost': 0  # No cost for streamed responses
                        }

                        data[org_id][user_id]['query'].append(temp)


                    else:
                        usage_metadata =  outputs['usage']
                        model_name = outputs["model"]
                        if "gpt" in model_name:
                            model_name = model_name[:-11]

                        input_tokens = usage_metadata['prompt_tokens']
                        output_tokens = usage_metadata['completion_tokens'] 
                        if model_name in ["gpt-4o-mini","gpt-4o","gpt-4o-mini-search-preview","gpt-4o-search-preview"]:
                            cache_read = usage_metadata['prompt_tokens_details']['cached_tokens']
                        else:
                            cache_read = 0
                        
                        processed_tokens = input_tokens - cache_read  ### Processed tokens = input tokens - cache read

                        if model_name not in data[org_id][user_id]['chat']:
                            data = create_org_user_id_schema(data,org_id,user_id,model_name,flag="chat")
                        
                        data[org_id][user_id]['chat'][model_name]['input_tokens'] += processed_tokens
                        data[org_id][user_id]['chat'][model_name]['output_tokens'] += output_tokens
                        data[org_id][user_id]['chat'][model_name]['cached_tokens'] += cache_read
                        data[org_id][user_id]['chat'][model_name]['requests'] += 1

                        if 'search-preview' in model_name:
                            search_cost = search_cost_per_call.get(model_name,{}).get('medium',0)
                        else:
                            search_cost = 0.0


                        messages_user = ""
                        messages_system = ""

                        input_messages = inputs.get("messages", None)
                        if input_messages:
                            for message in input_messages:
                                if message.get("role") == "system" and isinstance(message.get("content"), str):
                                    messages_system += message.get("content") + "\n"
                                elif message.get("role") == "user" and isinstance(message.get("content"), str):
                                    messages_user += message.get("content") + "\n\n"

                        
                        exec_time = 0

                        temp = {

                            'org_id': org_id_val,
                            'user_id':user_id_val,
                            'service':experiment_name,
                            'env': env,
                            'request_input': request_input,
                            'request_output': request_output,
                            'request_timestamp': request_timestamp_ms,
                            'request_id': request_id,
                            'request_execution_time': request_execution_time,
                            'execution_time': exec_time,
                            'system_msg': messages_system,
                            'user_msg': messages_user,
                            'output_msg': outputs.get("choices",{})[0].get("message",{}).get("content"," "),
                            'input_tokens': processed_tokens,
                            'output_tokens': output_tokens,
                            'cached_tokens': cache_read,
                            'model': model_name,
                            'type': "chat",
                            'input_cost': costs_per_token[model_name]['input'] * processed_tokens,
                            'output_cost': costs_per_token[model_name]['output'] * output_tokens,
                            'cached_cost': costs_per_token[model_name]['cache'] * cache_read,
                            'total_tokens': processed_tokens + output_tokens + cache_read,
                            'total_cost': (costs_per_token[model_name]['input'] * processed_tokens) + (costs_per_token[model_name]['output'] * output_tokens) + (costs_per_token[model_name]['cache'] * cache_read) + search_cost
                        
                        }


                        data[org_id][user_id]['query'].append(temp)
                
                elif span.name == "Completions_Usage":
                    inputs = span.inputs
                    outputs = span.outputs

                    if not inputs:
                        print("Inputs or Outputs are empty")
                        continue
                    print(f"Outputs: {outputs}")
                    team_leader_model_name = "gpt-4o"
                    agent_model_name = "gpt-4o"
                    print(f"Team Leader Model: {team_leader_model_name}, Agent Model: {agent_model_name}")

                    # 1. Add all team leader tokens 

                    input_tokens = outputs["team_leader_input_tokens"]
                    output_tokens = outputs["team_leader_output_tokens"]
                    cache_read = outputs["team_leader_cached_tokens"]
                    processed_tokens = input_tokens - cache_read  ### Processed tokens = input tokens - cache read

                    if team_leader_model_name not in data[org_id][user_id]['chat']:
                        data = create_org_user_id_schema(data,org_id,user_id,team_leader_model_name,flag="chat")

                    data[org_id][user_id]['chat'][team_leader_model_name]['input_tokens'] += processed_tokens
                    data[org_id][user_id]['chat'][team_leader_model_name]['output_tokens'] += output_tokens
                    data[org_id][user_id]['chat'][team_leader_model_name]['cached_tokens'] += cache_read
                    data[org_id][user_id]['chat'][team_leader_model_name]['requests'] += 0

                    if 'search-preview' in team_leader_model_name:
                        search_cost = search_cost_per_call.get(team_leader_model_name,{}).get('medium',0)
                    else:
                        search_cost = 0.0

                    exec_time = 0

                    temp = {

                        'org_id': org_id_val,
                        'user_id':user_id_val,
                        'service':experiment_name,
                        'env': env,
                        'request_input': request_input,
                        'request_output': request_output,
                        'request_timestamp': request_timestamp_ms,
                        'request_id': request_id,
                        'request_execution_time': request_execution_time,
                        'execution_time': exec_time,
                        'input_tokens': processed_tokens,
                        'output_tokens': output_tokens,
                        'cached_tokens': cache_read,
                        'model': team_leader_model_name,
                        'type': "chat",
                        'input_cost': costs_per_token[team_leader_model_name]['input'] * processed_tokens,
                        'output_cost': costs_per_token[team_leader_model_name]['output'] * output_tokens,
                        'cached_cost': costs_per_token[team_leader_model_name]['cache'] * cache_read,
                        'total_tokens': processed_tokens + output_tokens + cache_read,
                        'total_cost': (costs_per_token[team_leader_model_name]['input'] * processed_tokens) + (costs_per_token[team_leader_model_name]['output'] * output_tokens) + (costs_per_token[team_leader_model_name]['cache'] * cache_read) + search_cost

                    }


                    data[org_id][user_id]['query'].append(temp)

                    # 2. Add all agent tokens 
                    input_tokens = outputs["agent_input_tokens"]
                    output_tokens = outputs["agent_output_tokens"]
                    cache_read = outputs["agent_cached_tokens"]
                    processed_tokens = input_tokens - cache_read  ### Processed tokens = input tokens - cache read

                    if agent_model_name not in data[org_id][user_id]['chat']:
                        data = create_org_user_id_schema(data,org_id,user_id,agent_model_name,flag="chat")

                    data[org_id][user_id]['chat'][agent_model_name]['input_tokens'] += processed_tokens
                    data[org_id][user_id]['chat'][agent_model_name]['output_tokens'] += output_tokens
                    data[org_id][user_id]['chat'][agent_model_name]['cached_tokens'] += cache_read
                    data[org_id][user_id]['chat'][agent_model_name]['requests'] += 0

                    if 'search-preview' in agent_model_name:
                        search_cost = search_cost_per_call.get(agent_model_name,{}).get('medium',0)
                    else:
                        search_cost = 0.0

                    exec_time = 0

                    temp = {

                        'org_id': org_id_val,
                        'user_id':user_id_val,
                        'service':experiment_name,
                        'env': env,
                        'request_input': request_input,
                        'request_output': request_output,
                        'request_timestamp': request_timestamp_ms,
                        'request_id': request_id,
                        'request_execution_time': request_execution_time,
                        'execution_time': exec_time,
                        'input_tokens': processed_tokens,
                        'output_tokens': output_tokens,
                        'cached_tokens': cache_read,
                        'model': agent_model_name,
                        'type': "chat",
                        'input_cost': costs_per_token[agent_model_name]['input'] * processed_tokens,
                        'output_cost': costs_per_token[agent_model_name]['output'] * output_tokens,
                        'cached_cost': costs_per_token[agent_model_name]['cache'] * cache_read,
                        'total_tokens': processed_tokens + output_tokens + cache_read,
                        'total_cost': (costs_per_token[agent_model_name]['input'] * processed_tokens) + (costs_per_token[agent_model_name]['output'] * output_tokens) + (costs_per_token[agent_model_name]['cache'] * cache_read) + search_cost

                    }


                    data[org_id][user_id]['query'].append(temp)
        
                
        data = update_total_cost(data)
        # import pandas as pd
        # d = pd.DataFrame(data[org_id][user_id]['query'])
        # d.to_csv("agent_trace_data.csv")
        # import pickle
        # with open("agent_trace_data.pkl","wb") as f:
        #     pickle.dump(data,f)
        
        # for org_id, users in data.items():
        #     for user_id, user_data in users.items():
        #         df = pd.DataFrame(user_data["query"])
        #         df = df.fillna(0)
        #         df.to_csv(f"agent_trace_data_{org_id}_{user_id}.csv", index=False)

        if not data:
            print("Error in updating total cost")
            return 
        log_data_with_user_ids_intel_chat(experiment_name_env, data, TRACKING_URI, agent_flag="True")
        
        update_cost_tags(target_request_ids,client,"cost")

        print(f"Time taken :",time.time()-start_time)
    
    except Exception as e:
        print("Error in Mlflow Runner main_agent function:",str(e))
        raise e




# if __name__ == "__main__":
    # main_intel(env="DEV_AVIVO",max_traces=1)
    # main_agent(env="DEV_AVIVO",max_traces=1)