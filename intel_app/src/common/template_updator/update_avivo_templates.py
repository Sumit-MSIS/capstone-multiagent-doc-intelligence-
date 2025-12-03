from typing import List
from dotenv import load_dotenv
import os
import tempfile
import requests
import mammoth
import json
import pymysql
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from PIL import Image
# from io import BytesIO
# from src.config.base_config import config
from pinecone import Pinecone
from datetime import datetime
from src.config.base_config import config
import fitz  # from PyMuPDF
# Load local .env first (baseline values, can be overridden by secrets)
load_dotenv()

def convert_to_html(file_path, file_type):
    """Convert DOCX or PDF to HTML-like content."""
    try:
        if file_type == "docx":
            with open(file_path, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file)
                return result.value  # Clean HTML

        elif file_type == "pdf":
            return convert_pdf_to_html_like(file_path)

        else:
            print(f"Unsupported file type: {file_type}")
            return None

    except Exception as e:
        print(f"Error converting {file_type.upper()} to HTML: {e}")
        return None

def convert_pdf_to_html_like(pdf_path):
    """Extract text from PDF and wrap in minimal HTML."""
    try:
        doc = fitz.open(pdf_path)
        html = "<div>\n"
        for page in doc:
            text = page.get_text()
            if text:
                # html += f"<p>{text.strip().replace('\n', '<br>')}</p>\n"
                html += "<p>" + text.strip().replace("\n", "<br>") + "</p>\n"

        html += "</div>"
        return html
    except Exception as e:
        print(f"Error extracting PDF content: {e}")
        return None
    
def update_avivo_template(org_id, user_id, url, file_id, file_name, file_type):
    """Download DOCX file, convert it to HTML, return JSON, and clean up."""

    temp_path = url

    # Convert to HTML
    html_content = convert_to_html(temp_path, file_type)

    summary_text = retrieve_entire_chunks(file_id, org_id)
    print(f"File Name - {file_name} | Summary text: {summary_text}\\n\n\n")

    # if html_content:
    #     thumbnail_image = capture_thumbnail(html_content)
    # else:
    #     print(f"Skipping thumbnail generation for file {file_name} due to missing HTML content.")
    #     thumbnail_image = None

        

    print("Thumbnail saved as 'thumbnail.png'")

    current_date = datetime.now().strftime("%Y-%m-%d")

    file_name_with_ext = f"{file_name}.{file_type}"

    update_templates_data(file_id, file_name_with_ext, current_date, summary_text, None, html_content)
    # if os.path.exists(temp_path):
    #     os.remove(temp_path)
    # Return JSON response
    return {
        "success": True,
        "error": "",
        "data": {
            "org_id": org_id,
            "user_id": user_id,
            "file_id": file_id,
            "html_content": html_content
        }

    }

def retrieve_entire_chunks(file_id, org_id):
    pc = Pinecone(api_key =config.PINECONE_API_KEY)
    index = pc.Index(config.DOCUMENT_SUMMARY_INDEX)
    logical_partition=f"org_id_{org_id}#"
    initial_id = [f"{org_id}#{file_id}"]
    data = index.fetch(ids=initial_id, namespace=logical_partition)
    # Extract metadata if available
    if "vectors" in data and data["vectors"]:
        vector_data = data["vectors"].get(initial_id[0], {})
        metadata = vector_data.get("metadata", {})
        text = metadata.get("text", "")
        return text  # Return metadata dictionary

    return ""


def update_templates_data(file_id: str, file_name: str, upload_date: str, summary_text: str, thumbnail_image: str, html_content: str):
    print("LLM Debug: Inside BM25 index upload to MySQL")

    connection = None  # Initialize connection variable

    try:
        # Connect to MySQL
        connection = pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("CONTRACT_INTEL_DB"),
            port=int(os.getenv("DB_PORT"))
        )
        print("Database connection successful!")

        with connection.cursor() as cursor:
            # Check if file exists
            check_query = "SELECT COUNT(*) FROM templates WHERE fileId = %s"
            cursor.execute(check_query, (file_id,))
            file_exists = cursor.fetchone()[0] > 0

            if file_exists:
                # Update existing record
                update_query = """
                    UPDATE templates
                    SET fileName = %s, upload_date = %s, summary_text = %s, 
                        thumbnail_image = %s, html_content = %s
                    WHERE fileId = %s
                """
                update_query = """
                    UPDATE templates
                    SET summary_text = %s
                    WHERE fileId = %s
                """
                cursor.execute(update_query, (file_name, upload_date, summary_text, thumbnail_image, html_content, file_id))
                cursor.execute(update_query, (summary_text, file_id))
                print("File exists: Updated record.")
            else:
                # Insert new record
                insert_query = """
                    INSERT INTO templates 
                    (fileId, fileName, upload_date, summary_text, thumbnail_image, html_content)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (file_id, file_name, upload_date, summary_text, thumbnail_image, html_content))
                print("New file: Inserted record.")

            # Commit changes
            connection.commit()

    except pymysql.MySQLError as err:
        print(f"MySQL Error: {err}")
    finally:
        if connection:
            connection.close()
            print("Database connection closed.")