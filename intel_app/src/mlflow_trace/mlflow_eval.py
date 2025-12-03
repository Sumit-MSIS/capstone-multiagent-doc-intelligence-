import os,sys
import time
import sqlite3
import pandas  as pd
import json
from mlflow_trace.mlflow_utils import get_experiment_ids,call_llm,log_eval,user_prompt_parser
from mlflow import MlflowClient
import mlflow
import concurrent.futures
import time
from dotenv import load_dotenv
from src.config.base_config import config
load_dotenv()

TRACKING_URI = config.MLFLOW_TRACKING_URI
EMBEDDING_TRACE = config.MLFLOW_EMBEDDING_TRACE
CHAT_TRACE = config.MLFLOW_CHAT_TRACE

client = MlflowClient(TRACKING_URI)

def main(experiment_name:str, request_id , org_id:int, user_id:int):
    try:
        time.sleep(2)
        start_time = time.time()
        EXPERIMENT_IDS = get_experiment_ids([experiment_name],client=client)
        if not EXPERIMENT_IDS:
            return 

        target_request_df = mlflow.search_traces(experiment_ids=EXPERIMENT_IDS)
        target_request_id = target_request_df[target_request_df["request_id"] == request_id]
        target_trace = target_request_id["trace"].to_list()
        trace_obj = target_trace[0]
        spans = trace_obj.search_spans()
        data = {

            "inputs": [],
            "contexts": [],
            "outputs": [],
            "FF_score": [],
            "RL_score": [],
            "AR_score": [],
            "FF_justification": [],
            "RL_justification": [],
            "AR_justification": [],
            
        }
        metrics = ["FF","RL","AR"]
        for span in spans:
            if CHAT_TRACE in span.name and len(span.events) == 0:
                inputs = span.inputs
                messages_system = " "
                messages_user = " "
                input_messages = inputs.get("messages",None)
                if input_messages:
                    for message in input_messages:
                        if message.get("role") == "system":
                            messages_system += message.get("content") + "\n"
                        elif message.get("role") == "user":
                            messages_user += message.get("content") + "\n\n"

                messages_system_parsed, messages_user_parsed = user_prompt_parser(messages_system, messages_user) or (" "," ")
                outputs = span.outputs
                message_assistant = outputs.get("choices",{})[0].get("message",{}).get("content"," ")
                data["inputs"].append(messages_system_parsed)
                data["contexts"].append(messages_user_parsed)
                data["outputs"].append(message_assistant)

                t1 = time.time()
                def process_metric(metric):
                    start_time = time.time()
                    
                    score, justification = call_llm(
                        inputs=messages_system, 
                        outputs=message_assistant, 
                        contexts=messages_user, 
                        metric=metric
                    )
                    
                    return metric, score, justification

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {executor.submit(process_metric, metric): metric for metric in metrics}
                    
                    for future in concurrent.futures.as_completed(futures):
                        metric, score, justification = future.result()
                        data[f"{metric}_score"].append(score)
                        data[f"{metric}_justification"].append(justification)

        df = pd.DataFrame(data)
        df_mean_list = df[["FF_score","RL_score","AR_score"]].mean().to_list()
        t2 = time.time()
        log_eval(experiment_name,request_id,df,TRACKING_URI,org_id,user_id,df_mean_list)

    except Exception as e:
        print("Error in  Mlflow Runner  Evaluation :",str(e))
        raise e



# if __name__ == "__main__":
#     main("Intel Chat - Get Answer", "b693fbbe87984441b69c560ea6588f07", 506,647)


