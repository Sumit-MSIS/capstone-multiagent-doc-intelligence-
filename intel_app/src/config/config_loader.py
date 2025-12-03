import boto3
import json
import os
from botocore.exceptions import ClientError

env_prefix = os.getenv("ENVIRONMENT", "DEV")
def get_secret(secret_name: str, region_name: str = "us-east-1") -> str:
    """
    Fetch a secret string from AWS Secrets Manager.
    """
    session = boto3.session.Session()
    client = session.client(
        service_name="secretsmanager",
        region_name=region_name
    ) 

    try:
        response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise RuntimeError(f"Error retrieving secret {secret_name}: {str(e)}")

    secret_string = response.get("SecretString")
    if not secret_string:
        raise RuntimeError(f"No SecretString found in {secret_name}")

    return secret_string


def parse_db_credentials(secret_string: str) -> dict:
    """
    Parse DB_CREDENTIALS which may be invalid JSON like:
    {username:avivo,host:db-url,port:3306,password:xyz}

    Returns a dictionary with proper keys.
    """
    try:
        # Try JSON first (preferred)
        return json.loads(secret_string)
    except json.JSONDecodeError:
        # Handle pseudo-JSON without quotes
        cleaned = secret_string.strip("{}")
        pairs = [item.strip() for item in cleaned.split(",")]
        result = {}
        for pair in pairs:
            if ":" in pair:
                k, v = pair.split(":", 1)
                result[k.strip()] = v.strip()
        return result

def load_secret_to_env(secret_name: str, region_name: str = "us-east-1", env_prefix: str = None):
    if env_prefix is None:
        env_prefix = os.getenv("ENVIRONMENT", "DEV")
    env_prefix = env_prefix.upper()
    if not env_prefix.endswith("_"):
        env_prefix += "_"

    secret_string = get_secret(secret_name, region_name=region_name)
    print(f"Loaded secret for {secret_name} | secret_string: {secret_string} | type: {type(secret_string)}")
    
    try:
        parsed_secret = json.loads(secret_string)
        if isinstance(parsed_secret, dict):
            secret_dict = parsed_secret
        else:
            key_name = secret_name.replace(env_prefix, "", 1)
            secret_dict = {key_name: str(parsed_secret)}
    except json.JSONDecodeError:
        key_name = secret_name.replace(env_prefix, "", 1)
        secret_dict = {key_name: secret_string}

    for key, value in secret_dict.items():
        clean_key = key.replace(env_prefix, "", 1)
        if key in ["host", "port", "password", "username"]:
            os.environ[f"DB_{clean_key.upper()}"] = str(value)
        else:
            os.environ[clean_key] = str(value)

    return secret_dict




def load_rds_credentials(secret_name: str = f"{env_prefix.upper()}_DB_CREDENTIALS", region_name: str = "us-east-1"):
    """
    Load DB_CREDENTIALS secret and export:
    DB_USER, DB_HOST, DB_PORT, DB_PASSWORD
    """
    secret_string = get_secret(secret_name, region_name=region_name)
    creds = parse_db_credentials(secret_string)

    mapping = {
        "username": "DB_USER",
        "host": "DB_HOST",
        "port": "DB_PORT",
        "password": "DB_PASSWORD"
    }

    for key, env_key in mapping.items():
        if key in creds:
            os.environ[env_key] = str(creds[key])

    return creds
