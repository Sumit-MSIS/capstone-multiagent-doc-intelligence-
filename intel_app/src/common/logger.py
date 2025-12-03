import logging
import boto3
import os
from datetime import datetime
from src.config.base_config import config
from time import time
import gc
import re

import logging

# Define custom log levels
START_LEVEL = 21  # Between INFO (20) and WARNING (30)
END_LEVEL = 22



logging.addLevelName(START_LEVEL, "START")
logging.addLevelName(END_LEVEL, "END")


class CustomLogger(logging.Logger):
    def start(self, msg, *args, **kwargs):
        if self.isEnabledFor(START_LEVEL):
            self._log(START_LEVEL, msg, args, **kwargs)

    def end(self, msg, *args, **kwargs):
        if self.isEnabledFor(END_LEVEL):
            self._log(END_LEVEL, msg, args, **kwargs)

# Override the default logger class
logging.setLoggerClass(CustomLogger)



MAX_BATCH_SIZE_BYTES = 1048576  # 1 MB
MAX_EVENT_SIZE = 262144         # 256 KB

class CloudWatchHandler(logging.Handler):
    def __init__(self, log_group_name, log_stream_name, region_name, batch_size=250, batch_interval=10):
        super().__init__()
        self.boto_client = boto3.client('logs', region_name=region_name)
        self.log_group_name = log_group_name
        self.log_stream_name = log_stream_name
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.log_events = []
        self.last_flush_time = time()

        try:
            self.boto_client.create_log_group(logGroupName=self.log_group_name)
        except self.boto_client.exceptions.ResourceAlreadyExistsException:
            pass

        try:
            self.boto_client.create_log_stream(logGroupName=self.log_group_name, logStreamName=self.log_stream_name)
        except self.boto_client.exceptions.ResourceAlreadyExistsException:
            pass

        self.sequence_token = self._get_sequence_token()

    def _get_sequence_token(self):
        try:
            response = self.boto_client.describe_log_streams(
                logGroupName=self.log_group_name,
                logStreamNamePrefix=self.log_stream_name
            )
            streams = response.get('logStreams', [])
            if streams:
                return streams[0].get('uploadSequenceToken')
        except Exception as e:
            print(f"Error fetching sequence token: {e}")
        return None

    def emit(self, record):
        message = self.format(record)

        # Truncate message if too large
        if len(message.encode('utf-8')) > MAX_EVENT_SIZE:
            message = message[:MAX_EVENT_SIZE - 3] + '...'

        log_event = {
            'timestamp': int(record.created * 1000),
            'message': message
        }
        self.log_events.append(log_event)

        if len(self.log_events) >= self.batch_size or time() - self.last_flush_time >= self.batch_interval:
            self.flush_logs()

    def flush_logs(self):
        if not self.log_events:
            return

        try:
            self.log_events.sort(key=lambda x: x['timestamp'])

            batch = []
            size = 0

            for event in self.log_events:
                event_size = len(event['message'].encode('utf-8')) + 26  # rough overhead
                if size + event_size > MAX_BATCH_SIZE_BYTES:
                    self._send_batch(batch)
                    batch = [event]
                    size = event_size
                else:
                    batch.append(event)
                    size += event_size

            if batch:
                self._send_batch(batch)

            self.log_events = []
            self.last_flush_time = time()

        except Exception as e:
            print(f"Error sending logs to CloudWatch: {e}")

    def _send_batch(self, batch):
        if not batch:
            return

        try:
            put_args = {
                'logGroupName': self.log_group_name,
                'logStreamName': self.log_stream_name,
                'logEvents': batch
            }

            if self.sequence_token:
                put_args['sequenceToken'] = self.sequence_token

            response = self.boto_client.put_log_events(**put_args)
            self.sequence_token = response['nextSequenceToken']

        except self.boto_client.exceptions.InvalidSequenceTokenException as e:
            self.sequence_token = e.response['expectedSequenceToken']
            self._send_batch(batch)  # Retry with correct token
        except Exception as e:
            print(f"Error sending batch to CloudWatch: {e}")


    def close(self):
        self.flush_logs()
        # if hasattr(self.boto_client, '_endpoint') and hasattr(self.boto_client._endpoint, 'http_session'):
        #     try:
        #         self.boto_client._endpoint.http_session.close()
        #     except Exception as e:
        #         print(f"Error closing boto3 HTTP session: {e}")
        super().close()

def get_logger(logger_name, dir_name, service_type):
    service_config = {
        "CONTRACT": (config.CONTRACT_LOG_GROUP_NAME, config.CONTRACT_LOG_GROUP_REGION),
        "MEDICAL": (config.MEDICAL_LOG_GROUP_NAME, config.MEDICAL_LOG_GROUP_REGION),
        "INTEL_CHAT": (config.INTELCHAT_LOG_GROUP_NAME, config.INTELCHAT_LOG_GROUP_REGION),
        "CONCEPTUAL_SEARCH": (config.CONCEPTUAL_SEARCH_LOG_GROUP_NAME, config.CONCEPTUAL_SEARCH_LOG_GROUP_REGION),
        "COMPARE_CONTRACT": (config.COMPARECONTRACT_LOG_GROUP_NAME, config.COMPARECONTRACT_LOG_GROUP_REGION),
        "AI_CONTRACT_CREATION": (config.AI_CONTRACT_CREATION_LOG_GROUP_NAME, config.AI_CONTRACT_CREATION_LOG_GROUP_REGION),
        "DELETE_FILES": (config.DELETE_FILES_LOG_GROUP_NAME, config.DELETE_FILES_LOG_GROUP_REGION),
        "SUMMARY_TEMPLATE": (config.DOC_SUMMARY_TEMPLATE_LOG_GROUP_NAME, config.DOC_SUMMARY_TEMPLATE_LOG_GROUP_REGION),
        "AI_SUPPORT_REWRITE_CONTENT": (config.AI_SUPPORT_LOG_GROUP_NAME, config.AI_SUPPORT_LOG_GROUP_REGION),
        "RECURRING_PAYMENT": (config.RECURRING_PAYMENT_LOG_GROUP_NAME, config.RECURRING_PAYMENT_LOG_GROUP_REGION),
        "MLFLOW": (config.MLFLOW_LOG_GROUP_NAME, config.MLFLOW_LOG_GROUP_REGION),
        "CELERY": ("dev-celery-tasks", config.CONTRACT_LOG_GROUP_REGION),
        "UPDATE_ACCESS_TAG": (config.ACCESS_TAG_LOG_GROUP_NAME, config.ACCESS_TAG_LOG_GROUP_REGION)

    }

    log_group_name, log_group_region = service_config.get(service_type, (None, None))

    logger = logging.getLogger(logger_name)
    logger.__class__ = CustomLogger
    logger.setLevel(logging.DEBUG)

    # CloudWatch logging if config is available
    if log_group_name and log_group_region:
        # cloud_handler = CloudWatchHandler(log_group_name, sanitize_log_stream_name(f"{datetime.now().strftime('%Y-%m-%d')}/{logger_name}"), log_group_region)

        cloud_handler = CloudWatchHandler(log_group_name, f"{datetime.now().strftime('%Y-%m-%d')}/{logger_name}", log_group_region)
        cloud_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(cloud_handler)
    else:
        print(f"Warning: No CloudWatch config for service '{service_type}'. Logs will be stored locally only.")
    return logger

def request_logger(file_id, dir_name, service_type):
    return get_logger(file_id, dir_name, service_type)


def flush_all_logs(file_id, dir_name, service_type):
    logger = logging.getLogger(file_id)
    for handler in list(logger.handlers):  # Use list() to avoid modifying during iteration
        if hasattr(handler, 'flush'):
            try:
                handler.flush()
            except Exception as e:
                print(f"Error flushing logs for {file_id}: {e}")

        handler.close()
        logger.removeHandler(handler)

def sanitize_log_stream_name(user_query: str, max_length: int = 256) -> str:
    """
    Sanitizes a string to be used as a valid AWS CloudWatch log stream name.
    
    - Removes or replaces invalid characters (: * ? " < > | / \).
    - Collapses multiple spaces into one.
    - Trims leading/trailing whitespace.
    - Optionally truncates to the allowed max length (default 256).
    """

    # Replace invalid characters with '-'
    sanitized = re.sub(r'[:*?"<>|/\\]', '-', user_query)

    # Replace multiple spaces or consecutive '-' with a single '-'
    sanitized = re.sub(r'[\s\-]+', '-', sanitized)

    # Trim to max allowed length (CloudWatch allows up to 512 chars, safe to use 256)
    sanitized = sanitized.strip('- ')[:max_length]

    return sanitized

def _log_message(message: str, function_name: str, module_name: str) -> str:
    return f"[function={function_name} | module={module_name}] - {message}"




        