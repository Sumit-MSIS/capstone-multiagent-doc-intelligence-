import os
import json
import re
import pandas as pd
from typing import List, Dict
import mlflow
import random
from mlflow import MlflowClient
from  datetime import datetime
import pytz
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from src.config.base_config import config
# from dotenv import load_dotenv
# load_dotenv()

TRACKING_URI = config.MLFLOW_TRACKING_URI
ENV = config.MLFLOW_ENV
MLFLOW_EXPERIMENT_NAMES = config.MLFLOW_EXPERIMENT_NAMES


def update_total_cost(data: dict):
    try:
        costs_per_million_tokens = {
            "gpt-4o": {"input": 2.50, "output": 10.00, "cache": 1.25},
            "gpt-4o-mini": {"input": 0.150, "output": 0.600, "cache": 0.075},
            "gpt-4o-mini-search-preview": {"input": 0.150, "output": 0.600, "cache": 0.00},
            "gpt-4o-search-preview": {"input": 2.50, "output": 10.00, "cache": 0.00},
            "text-embedding-3-large": {"input": 0.13},  
            "text-embedding-3-small": {"input": 0.02},  
            "text-embedding-ada-002": {"input": 0.10}
        }

        costs_per_token = {
            model: {k: v / 1_000_000 for k, v in costs.items()}
            for model, costs in costs_per_million_tokens.items()
        }
        search_cost_per_thousand_calls = {
            "gpt-4o-mini-search-preview": {
                'low': 25.00, 
                'medium': 27.50 ,  ## bydefault
                'high':30.00 
            },

            "gpt-4o-search-preview": {
                'low': 30.00,
                'medium': 35.00, ## by default
                'high': 50.00
            }
        }

        search_cost_per_call = {
            model: {k: v / 1000 for k, v in costs.items()}
            for model, costs in search_cost_per_thousand_calls.items()
        }

        for org_id in data:
            for user_id in data[org_id]:
                for model_type in data[org_id][user_id]:
                    if model_type == "embedding":
                        for model_name, stats in data[org_id][user_id][model_type].items():
                            input_tokens = stats["input_tokens"]
                            cost_per_token = costs_per_token.get(model_name, {}).get("input", 0)
                            total_cost = input_tokens * cost_per_token
                            data[org_id][user_id]["total_embedding_cost"] += total_cost

                    elif model_type == "chat":
                        for model_name, stats in data[org_id][user_id][model_type].items():
                            input_tokens = stats["input_tokens"]
                            output_tokens = stats["output_tokens"]
                            cache_read = stats["cached_tokens"]
                            requests = stats["requests"]
                            cost_per_input_token = costs_per_token.get(model_name, {}).get("input", 0)
                            cost_per_output_token = costs_per_token.get(model_name, {}).get("output", 0)
                            cost_per_cache_token = costs_per_token.get(model_name, {}).get("cache", 0)
                            
                            if "search" in model_name:  ### will change as later
                                search_cost = search_cost_per_call.get(model_name, {}).get("medium", 0)
                                total_cost = (
                                    (input_tokens * cost_per_input_token)
                                    + (output_tokens * cost_per_output_token)
                                    + (cache_read * cost_per_cache_token)
                                    + (requests * search_cost)
                                )
                            else:
                                total_cost = (
                                    (input_tokens * cost_per_input_token)
                                    + (output_tokens * cost_per_output_token)
                                    + (cache_read * cost_per_cache_token)
                                )
                            data[org_id][user_id]["total_chat_cost"] += total_cost
        return data

    except Exception as e:
        print(f"Error in updating total cost: {e}")
        return None


def generate_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def read_json(filename):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None

def write_json(filename, data):
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        return
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None


def update_cost_tags(target_request_ids,client,flag:str):
    """
    Update the 'cost' tag in the 'trace_tags' table based on the presence of request_id in error_request_ids.

    Parameters:
    - target_request_ids: List of objects containing request_id attributes.
    - error_request_ids: List of request_ids that encountered errors.
    """
    try:
        if len(target_request_ids) == 0:
            return
        for trace_obj in target_request_ids:
            r_id = trace_obj.info.request_id  # Extract request_id
            if flag == "cost":
                client.set_trace_tag(r_id,"cost","True")
                

    except Exception as e:
        raise e
    
def get_experiment_ids(experiment_name: List[str], client):
    try:
        exp_ids = []
        for exp_name in experiment_name:
            exp = client.get_experiment_by_name(exp_name)
            if exp:
                exp_ids.append(exp.experiment_id)
            else:
                return None
        return exp_ids

    except Exception as e:
        print(f"Error in fetching experiment IDs: {e}")
        return None
 
   
def create_org_user_id_schema(
    data: Dict, org_id: str, user_id: str, model_name: str, flag: str
):
    try:
        if flag == "org_id":
            data[org_id] = {
                user_id: {"chat": {}, "embedding": {}, "total_chat_cost": 0, "total_embedding_cost": 0,"requests": 0, 'query': []}
            }

            return data

        elif flag == "user_id":
            data[org_id][user_id] = {"chat": {}, "embedding": {},"total_chat_cost": 0, "total_embedding_cost": 0,"requests": 0, 'query': [] }
            return data

        elif flag == "embedding":
            data[org_id][user_id]["embedding"][model_name] = {
                "input_tokens": 0,
                "dimensions": 3072,
                "requests": 0
            }
            return data

        elif flag == "chat":
            data[org_id][user_id]["chat"][model_name] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "requests": 0
            }
            return data

    except Exception as e:
        print(f"Error in creating org-user schema: {e}")
        return None


def log_data_with_user_ids(experiment_name:str, data: Dict, TRACKING_URI: str):
    """Logs MLflow metrics for each org-user pair while ensuring total cost remains the same for all users in an organization."""
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(experiment_name)

        t = pytz.timezone('Asia/Kolkata')
        timestamp = datetime.now(tz=t)

        run_name = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        date_ = datetime.now().strftime("%Y-%m-%d")
        year,month,date = date_.split("-")
        time = datetime.now().strftime("%H-%M-%S")

        for org_id, users in data.items():
            for user_id, user_data in users.items():

                total_embedding_cost = user_data.get("total_embedding_cost", 0)
                total_chat_cost = user_data.get("total_chat_cost", 0)
                total_requests = user_data.get("requests", 0)
                df = pd.DataFrame(user_data["query"])
                df = df.fillna(0)
                with mlflow.start_run(run_name=run_name,tags={"mlflow.runColor": generate_random_color()}):  # One run per org-user pair
                    _,_,org_id_ = org_id.split("_")
                    _,_,user_id_ = user_id.split("_")

                    mlflow.log_table(df, artifact_file="query.json")

                    mlflow.log_param("org_id", org_id_)
                    mlflow.log_param("user_id", user_id_)
                    mlflow.log_param("Year", year)
                    mlflow.log_param("Month", month)
                    mlflow.log_param("Date", date)
                    mlflow.log_param("Time", time)
                    mlflow.log_param("IST Timestamp", timestamp)
                    mlflow.set_tags({
                        "cost": "True",
                        "eval": "False",
                        "hour": "False",
                        "day": "False",
                        "week": "False",
                        "month": "False",
                    })

                    if "embedding" in user_data:
                        for model_name, details in user_data["embedding"].items():
                            mlflow.log_metric(
                                f"{model_name}_embedding_input_tokens",
                                details.get("input_tokens", 0),
                            )

                            mlflow.log_metric(
                                f"{model_name}_requests",
                                details.get("requests", 0),
                            )

                    if "chat" in user_data:
                        for model_name, details in user_data["chat"].items():
                            mlflow.log_metric(
                                f"{model_name}_chat_input_tokens",
                                details.get("input_tokens", 0),
                            )
                            mlflow.log_metric(
                                f"{model_name}_chat_output_tokens",
                                details.get("output_tokens", 0),
                            )
                            mlflow.log_metric(
                                f"{model_name}_chat_cached_tokens",
                                details.get("cached_tokens", 0),
                            )

                            mlflow.log_metric(
                                f"{model_name}_requests",
                                details.get("requests", 0),
                            )

                    mlflow.log_metric("Total_Embedding_Cost", total_embedding_cost)
                    mlflow.log_metric("Total_Chat_Cost", total_chat_cost)
                    mlflow.log_metric("Total Requests", total_requests)


    except Exception as e:
        print(f"Error in logging Cost data: {e}")
        raise e

def log_data_with_user_ids_intel_chat(experiment_name:str, data: Dict, TRACKING_URI: str, agent_flag):
    """Logs MLflow metrics for each org-user pair while ensuring total cost remains the same for all users in an organization."""
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(experiment_name)

        t = pytz.timezone('Asia/Kolkata')
        timestamp = datetime.now(tz=t)

        run_name = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        date_ = datetime.now().strftime("%Y-%m-%d")
        year,month,date = date_.split("-")
        time = datetime.now().strftime("%H-%M-%S")

        for org_id, users in data.items():
            for user_id, user_data in users.items():

                total_embedding_cost = user_data.get("total_embedding_cost", 0)
                total_chat_cost = user_data.get("total_chat_cost", 0)
                total_requests = user_data.get("requests", 0)
                df = pd.DataFrame(user_data["query"])
                df = df.fillna(0)
                with mlflow.start_run(run_name=run_name,tags={"mlflow.runColor": generate_random_color()}):  # One run per org-user pair
                    _,_,org_id_ = org_id.split("_")
                    _,_,user_id_ = user_id.split("_")

                    mlflow.log_param("org_id", org_id_)
                    mlflow.log_param("user_id", user_id_)
                    mlflow.log_param("Year", year)
                    mlflow.log_param("Month", month)
                    mlflow.log_param("Date", date)
                    mlflow.log_param("Time", time)
                    mlflow.log_param("IST Timestamp", timestamp)
                    mlflow.set_tags({
                        "cost": "True",
                        "eval": "False",
                        "hour": "False",
                        "day": "False",
                        "week": "False",
                        "month": "False",
                        "agent": agent_flag
                    })

                    if "embedding" in user_data:
                        for model_name, details in user_data["embedding"].items():
                            mlflow.log_metric(
                                f"{model_name}_embedding_input_tokens",
                                details.get("input_tokens", 0),
                            )

                            mlflow.log_metric(
                                f"{model_name}_requests",
                                details.get("requests", 0),
                            )

                    if "chat" in user_data:
                        for model_name, details in user_data["chat"].items():
                            mlflow.log_metric(
                                f"{model_name}_chat_input_tokens",
                                details.get("input_tokens", 0),
                            )
                            mlflow.log_metric(
                                f"{model_name}_chat_output_tokens",
                                details.get("output_tokens", 0),
                            )
                            mlflow.log_metric(
                                f"{model_name}_chat_cached_tokens",
                                details.get("cached_tokens", 0),
                            )

                            mlflow.log_metric(
                                f"{model_name}_requests",
                                details.get("requests", 0),
                            )

                    mlflow.log_metric("Total_Embedding_Cost", total_embedding_cost)
                    mlflow.log_metric("Total_Chat_Cost", total_chat_cost)
                    mlflow.log_metric("Total Requests", total_requests)
                    mlflow.log_table(df, artifact_file="query.json")


    except Exception as e:
        print(f"Error in logging Cost data: {e}")
        raise e

def call_llm(inputs:str, outputs:str, contexts:str, metric:str):
    try:
        mlflow.openai.autolog(disable=True)
        metric_data_json = read_json("mlflow_trace/mlflow_metric.json")
        metric_data = metric_data_json[metric]

        system_prompt = """
            Task:
            You must return the following fields in your response in two lines, one below the other:
            score: Your numerical score for the model's {name} based on the rubric
            justification: Your reasoning about the model's {name} score

            You are an impartial judge. You will be given an input that was sent to a machine
            learning model, and you will be given an output that the model produced. You
            may also be given additional information that was used by the model to generate the output.

            Your task is to determine a numerical score called {name} based on the input and output.
            A definition of {name} and a grading rubric are provided below.
            You must use the grading rubric to determine your score. You must also justify your score.

            Examples could be included below for reference. Make sure to use them as references and to
            understand them before completing the task. 
            *** Strictly if NOT find any  Input or Context or Output Don't use the example data to give response. Give as not provided or not mentioned that should be meaningful and calculate the correct score as per it. ***\n

            Input:
            {input} 

            \n
            
            Output:
            {output}
            \n
            
            {context}
            
            \n
            
            Metric definition:
            {definition}
            \n
            
            Grading rubric:
            {grading_prompt}
            \n
            
            {examples}

            \n
            You must return the following fields in your response in two lines, one below the other:
            score: Your numerical score for the model's {name} based on the rubric
            justification: Your reasoning about the model's {name} score

            ### Do not add additional new lines. Do not add any other fields.Don't add any special characters or emojis .

            """

        prompt_template = PromptTemplate.from_template( system_prompt )


        llm = ChatOpenAI(model="gpt-4o-mini")

        # llm = get_llm_client()

        

        output_parser = StrOutputParser()

        chain = prompt_template | llm | output_parser

        response = chain.invoke(
            {
                "name":metric_data["name"],
                "input":inputs,
                "output":outputs,
                "context":contexts,
                "definition":metric_data["definition"],
                "grading_prompt":metric_data["grading_prompt"],
                "examples":metric_data["examples"]
            }
        )

        print(response)
        score_match = re.search(r"score:\s*(\d+)", response)
        score = int(score_match.group(1)) if score_match else None

        justification_match = re.search(r"justification:\s*(.*)", response, re.DOTALL)
        justification = justification_match.group(1).strip() if justification_match else None
 
        mlflow.openai.autolog(disable=False)

        return score,justification

    except Exception as e:
        print(f"Error in calling LLM: {e}")
        raise e

    
def log_eval(experiment_name:str, request_id ,data: pd.DataFrame, TRACKING_URI: str,org_id:int,user_id:int,mean_list:List):
    """Logs MLflow metrics for each org-user pair while ensuring total cost remains the same for all users in an organization."""
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        print("Inside Log Data.")

        # metrics = ["FF","RL","AR"]

        AS = "Answer Similarity/Mean"
        FF = "Faithfulness/Mean"
        AC = "Answer Correctness/Mean"
        RL = "Relevance/Mean"
        AR = "Answer Relevance/Mean"
    
        run_name = datetime.now().strftime("%Y-%m-%d")
        year,month,date = run_name.split("-")

        with mlflow.start_run(run_name=run_name,tags={"mlflow.runColor": generate_random_color()}):  
            print(f"Logging Data for Org: {org_id}, User: {user_id}")
        
            mlflow.log_param("org_id", org_id)
            mlflow.log_param("user_id", user_id)
            mlflow.log_param("Year", int(year))
            mlflow.log_param("Month", int(month))
            mlflow.log_param("Date", int(date))
            mlflow.log_param("request_id",request_id)  
            mlflow.set_tags({
                "cost": "False",
                "eval": "True",
                "service": "True",   #### After move it ti False
                "total": "True" 
            })

            mlflow.log_input(mlflow.data.from_pandas(data), context="eval_results")
            mlflow.log_table(data, artifact_file="eval_results_table.json")
            mlflow.log_metric(FF, mean_list[0])
            mlflow.log_metric(RL, mean_list[1])
            mlflow.log_metric(AR, mean_list[2])
            # mlflow.log_metric(AS, mean_list[3])
            # mlflow.log_metric(AC, mean_list[4])    

    except Exception as e:
        print(f"Error in logging Evaluation data: {e}")
        raise e


def user_prompt_parser(msg_system:str, msg_user:str):
    try:

        inst_st, inst_ed = "<inst_st>", "<inst_ed>"
        ctx_st, ctx_ed = "<ctx_st>", "<ctx_ed>"
        ctx_data = ""  # Store extracted context data separately

        start = 0
        while True:
            inst_st_idx = msg_user.find(inst_st, start)
            inst_ed_idx = msg_user.find(inst_ed, inst_st_idx + len(inst_st))
            
            if inst_st_idx == -1 or inst_ed_idx == -1:
                break
            
            msg_system += msg_user[inst_st_idx + len(inst_st):inst_ed_idx] + "\n"
            start = inst_ed_idx + len(inst_ed)

        start = 0
        while True:
            ctx_st_idx = msg_user.find(ctx_st, start)
            ctx_ed_idx = msg_user.find(ctx_ed, ctx_st_idx + len(ctx_st))
            
            if ctx_st_idx == -1 or ctx_ed_idx == -1:
                break
            
            ctx_data += msg_user[ctx_st_idx + len(ctx_st):ctx_ed_idx] + "\n"
            start = ctx_ed_idx + len(ctx_ed)

        return msg_system, ctx_data

    except Exception as e:
        print(f"Error in user prompt parser: {e}")
        raise e



# if __name__ == "__main__":
#     s = fetch_request_org_user_ids()
#     print(s)
