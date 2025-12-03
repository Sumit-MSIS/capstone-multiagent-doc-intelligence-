import time
from datetime import datetime
from src.common.file_insights.doc_summariser.doc_parser import DocParser
from src.common.file_insights.doc_summariser.doc_summariser import DocSummariser
from src.config.base_config import config
from src.common.llm_status_handler.status_handler import set_llm_file_status
from src.common.logger import request_logger, flush_all_logs
from src.common.file_insights.doc_summariser.utils import update_toc_to_db, save_summary_to_db, update_html_to_db
import markdown
from src.common.file_insights.doc_summariser.extract_toc import ExtractTOC
import mlflow
import re
import json
from src.common.file_insights.doc_summariser.llm_call import open_ai_llm_call
from opentelemetry import context as ot_context
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from mlflow import MlflowClient
from src.common.llm.factory import get_llm_client
# mlflow.config.enable_async_logging()
mlflow.openai.autolog()
TRACKING_URI = config.MLFLOW_TRACKING_URI
ENV = config.MLFLOW_ENV
OPENAI_MODEL_NAME = config.OPENAI_MODEL_NAME
mlflow.set_tracking_uri(TRACKING_URI)
OPENAI_API_KEY = config.OPENAI_API_KEY
client = MlflowClient(tracking_uri=TRACKING_URI)
openai_client = get_llm_client(async_mode=False)

cost_tags = {

    "cost": "False"
}

# Configuration
MODULE_NAME = "doc_process_gen.py"


def submit_with_context(executor, fn, *args, **kwargs):
    # Capture the current OpenTelemetry context (which includes the parent span)
    parent_ctx = ot_context.get_current()
    def wrapped():
        token = ot_context.attach(parent_ctx)
        try:
            return fn(*args, **kwargs)
        finally:
            ot_context.detach(token)
    return executor.submit(wrapped)


def _log_message(message: str, function_name: str, module_name: str) -> str:
    return f"[function={function_name} | module={module_name}] - {message}"

# @mlflow.trace(name="Process Document")
def process_document(file_id, file_name, file_type, user_id, org_id, markdown_text, raw_complete_file_toc,retry_count):
    mlflow.set_experiment(f"Doc Process Gen - {ENV}")
    with mlflow.start_span("Doc Process Gen") as span:
        request_id = span.request_id
        span.set_inputs({
            "file_id": file_id,
            "file_name": file_name,
            "file_type": file_type,
            "markdown_text": markdown_text,
            "complete_file_toc": raw_complete_file_toc,
        })
        span.set_attributes({
            "user_id": user_id,
            "org_id": org_id,
            "retry_count": retry_count
        })

        client.set_trace_tag(request_id, "org_id", org_id)
        client.set_trace_tag(request_id, "user_id", user_id)
        client.set_trace_tag(request_id, "machine", ENV)

        mlflow.update_current_trace(tags=cost_tags)

        logger = request_logger(f"{file_id}-{org_id}-{user_id}", str(config.DOC_SUMMARY_TEMPLATE_LOG_DIR_NAME), "SUMMARY_TEMPLATE")

        logger.info(_log_message("##################################################### DOC PROCESS GEN REQUEST STARTED #####################################################", "process_document", MODULE_NAME))
        logger.info(_log_message(f"Request received: {file_id=}, {file_name=}, {file_type=}", "process_document", MODULE_NAME))

        if not all([file_id, file_name, file_type, markdown_text]):
            error_msg = f"Missing required parameters - {file_id=}, {file_name=}, {file_type=}, {markdown_text=}"
            logger.error(_log_message(error_msg, "process_document", MODULE_NAME))
            flush_all_logs(file_id, str(config.DOC_SUMMARY_TEMPLATE_LOG_DIR_NAME), "SUMMARY_TEMPLATE")
            raise ValueError("Missing required parameters")

        try:
            start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            file_parser = DocParser(logger)
            toc_extracter = ExtractTOC(logger)
            summariser = DocSummariser(logger)

            set_llm_file_status(file_id, user_id, org_id, 4, True, retry_count, 1, "", start_datetime, "", False, False, False, 40, logger)

            @mlflow.trace(name="Process Document - Extract TOC and Update")
            def extract_toc_and_update():
                try:
                    system_prompt = """
                            You are an expert document formatter specializing in structured table of contents (TOC) organization. Your task is to clean up and reformat a JSON-structured table of contents while preserving meaningful hierarchical organization and maximizing user-friendliness, clarity, and accuracy.

                            INPUT FORMAT: The input is a string containing:
                                table_of_contents: An array of sections, where each section has:
                                section_name: The main heading text
                                sub_sections: An array of objects, each with a "title" field containing subheading text
                            TASK REQUIREMENTS:
                                Analyze and understand the heading structure to reveal the document's organization.
                                Remove sections that are not proper content headings:
                                Skip single-word entries like "TO", "AND", "USA", etc.
                                Skip addresses, company names in isolation.
                                Skip sections that appear to be document metadata rather than content.
                                Group introductory recitals, background information, or parties' descriptions (e.g., “Party A is...”) under a single "Recitals" or "Preamble" section, placed before numbered sections, instead of listing them as main sections.
                                Clean up each subheading by:
                                Removing truncated or incomplete text (do not include subheadings cut off mid-sentence).
                                Eliminating unnecessary text, formatting artifacts, or placeholder text.
                                Ensuring proper subheading formatting (capitalize main words, remove excess punctuation).
                                Removing numbering, bullet points, and leading/trailing whitespace, unless numbering is essential for clarity.
                                Giving every sub-section a meaningful, descriptive title; remove sub-sections with only a number or no meaningful content.
                                Preserve the hierarchical structure for meaningful content sections.
                                Consistently format section and subsection titles using clear, concise, and uniform language.
                                Use logical section order and flow: general contract basics (definitions, parties, term), then operational matters (pricing, support), then legal boilerplate (confidentiality, dispute resolution).
                                Exclude duplicates or overlapping section headings.
                                Do not add extra information, context, or content not present in the original TOC.
                                If a subheading is present in the original text, extract and clean that subheading; if not, omit or leave it empty.
                                Do not list empty or placeholder sub-sections (e.g., "3.1.1" with no title).
                                Visually distinguish main sections and sub-sections in the hierarchy (by nesting in the JSON structure).
                                Only include meaningful document sections that represent actual content.
                                Do not include addresses, non-content metadata, or signature blocks.
                            
                            OUTPUT FORMATTING RULES:
                                Strictly do not add orjson at the start or end, and do not add newlines outside the JSON.
                                Only include meaningful document sections that represent actual content.
                                Exclude addresses and non-content metadata.
                                OUTPUT FORMAT: {
                                "file_id": "file-id",
                                "table_of_contents": [
                                    {
                                    "section_name": "Properly Formatted Section Name",
                                    "sub_sections": [
                                        {
                                        "title": "Properly Formatted Subsection Title"
                                        },
                                        {
                                        "title": "Properly Formatted Subsection Title"
                                        }
                                    ]
                                    }
                                ]
                                }

                            OUTPUT INSTRUCTIONS: Return the entire JSON object with a clear, concise, and user-friendly TOC structure, excluding non-content sections such as single words, addresses, and signature pages. Format section names and subsection titles properly, making the TOC as user-friendly and accurate as possible for any document. 
                            """
                    user_prompt = f"""Please reformat the following JSON table of contents. Clean up the subheadings to be concise and clear while preserving their essential meaning. Understand the document structure and remove any truncated or unnecessary text. Return the result in the same JSON format.
                                    {raw_complete_file_toc}"""
                    response = openai_client.chat.completions.create(
                        model=OPENAI_MODEL_NAME,
                        temperature=0.1,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                    )
                    # structured_toc = str(structured_toc).replace("```json", "").replace("```", "").replace("\\", "").replace("\n", "")
                    logger.info(_log_message(f"LLM TOC response: {response.choices[0].message.content}", "process_document", MODULE_NAME))
                    structured_toc = json.loads(response.choices[0].message.content)
                    structured_toc =  {
                    "file_id": file_id,
                    "table_of_contents": structured_toc.get("table_of_contents", [])
                    }
                    extracted_toc = structured_toc
                    logger.info(_log_message(f"TOC extracted successfully: {extracted_toc}", "process_document", MODULE_NAME))
                    html_content = markdown.markdown(markdown_text)
                    update_html_to_db(file_id, html_content, logger)
                    if extracted_toc:
                        update_toc_to_db(file_id, extracted_toc, logger)
                        set_llm_file_status(file_id, user_id, org_id, 4, True, retry_count, 1, "", start_datetime, "", False, False, False, 60, logger)
                        logger.info(_log_message("Updated TOC successfully", "process_document", MODULE_NAME))
                    return {"toc": extracted_toc}
                except Exception as e:
                    logger.error(_log_message(f"TOC extraction failed: {e}", "process_document", MODULE_NAME))
                    return {"toc": None}

            @mlflow.trace(name="Process Document - Summarize and Upsert")
            def summarize_and_upsert():
                try:
                    summary = summariser.summarize_text(markdown_text)
                    if summary:
                        summariser.upsert_to_pinecone(file_id, file_name, summary, org_id, user_id)
                        summary_html_text = "<p>" + summary.replace("\n", "</p><p>") + "</p>"
                        save_summary_to_db(file_name, file_type, file_id, user_id, org_id, summary, summary_html_text, logger)
                        result = extract_toc_and_update()
                        return {"summary": summary}
                    else:
                        return {"summary": None}
                except Exception as e:
                    logger.error(_log_message(f"Summarization failed: {e}", "process_document", MODULE_NAME))
                    return {"summary": None}

            @mlflow.trace(name="Process Document - Parse Text")
            def parse_text():
                try:
                    return {"parsed_text": file_parser.process_file(markdown_text, file_name, file_type, user_id, org_id, file_id)}
                except Exception as e:
                    logger.error(_log_message(f"Markdown parsing failed: {e}", "process_document", MODULE_NAME))
                    return {"parsed_text": None}

            with ThreadPoolExecutor() as executor:
                futures = {
                    'parsed_text': submit_with_context(executor, parse_text),
                    'summary': submit_with_context(executor, summarize_and_upsert)
                }

                results = {key: future.result() for key, future in futures.items()}

            # --- Handle Summary Status ---
            summary = results.get('summary', {}).get('summary')
            if summary:
                set_llm_file_status(file_id, user_id, org_id, 4, True, retry_count, 3, "", start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), False, False, False, 100, logger)
                response_json = {"success": True, "error": "", "data": summary}
            else:
                error_msg = "Summarization failed. Upsert to Pinecone not performed. Received Summary as None from OpenAI!"
                set_llm_file_status(file_id, user_id, org_id, 4, True, retry_count, 2, error_msg, start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), False, False, False, 100, logger)
                response_json = {"success": False, "error": error_msg, "data": ""}

            logger.info(_log_message(f"Generated Summary - {summary}", "process_document", MODULE_NAME))
            logger.info(_log_message(f"Response sent successfully - {response_json}", "process_document", MODULE_NAME))
            logger.info(_log_message("Doc Process Generation task completed successfully.", "process_document", MODULE_NAME))

        except Exception as e:
            error_msg = f"An error occurred during doc process gen: {str(e)}"
            logger.error(_log_message(error_msg, "process_document", MODULE_NAME))
            set_llm_file_status(file_id, user_id, org_id, 4, True, retry_count, 2, error_msg, start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), False, False, False, 100, logger)
            response_json = {"success": False, "error": error_msg, "data": ""}

        finally:
            logger.info(_log_message("##################################################### END DOC PROCESS GEN #####################################################", "process_document", MODULE_NAME))
            flush_all_logs(f"{file_id}-{org_id}-{user_id}", str(config.DOC_SUMMARY_TEMPLATE_LOG_DIR_NAME), "SUMMARY_TEMPLATE")
        
        span.set_outputs(response_json)

    return response_json


