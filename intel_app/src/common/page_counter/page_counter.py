import os
import uuid
import httpx
import aiofiles
import asyncio
import fitz  # PyMuPDF
import glob
from fastapi import HTTPException
from unstructured.partition.docx import partition_docx


# 1. Download the file asynchronously and save it
async def fetch_file_async(url: str, file_type: str, retries: int = 3, delay: float = 2.0) -> str:
    file_path = None
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.get(url)
                response.raise_for_status()
                file_content = response.content

            file_path = await _create_temp_file_async(file_content, file_type)
            return file_path

        except httpx.RequestError as e:
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                if file_path:
                    await _cleanup_temp_file_async(file_path)
                raise HTTPException(status_code=502, detail=f"Failed to download file: {str(e)}")

        except Exception as e:
            if file_path:
                await _cleanup_temp_file_async(file_path)
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

async def _create_temp_file_async(file_content: bytes, file_type: str) -> str:
    os.makedirs("/tmp", exist_ok=True)
    filename = f"{uuid.uuid4()}.{file_type}"
    file_path = os.path.join("/tmp", filename)

    try:
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        return file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create temp file: {str(e)}")

async def _cleanup_temp_file_async(temp_filepath: str) -> None:
    if temp_filepath and os.path.exists(temp_filepath):
        try:
            os.remove(temp_filepath)
        except Exception as e:
            print(f"[WARN] Failed to delete temp file {temp_filepath}: {e}")


# 2. Page counter functions
async def get_pdf_page_count(file_path):
    try:
        def count_pages():
            with fitz.open(file_path) as doc:
                return len(doc)
        pages = await asyncio.to_thread(count_pages)
        return {"success": True, "data": {"page_count": pages, "is_parsable": True}}
    except Exception as e:
        return {"success": False, "error": f"Error reading PDF: {e}", "data": {"is_parsable": False}}

async def count_words_in_txt(file_path):
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            text = await f.read()

        word_count = len(text.split())
        estimated_pages = max(1, word_count // 500)

        return {"success": True, "data": {"page_count": estimated_pages, "is_parsable": True}}
    except Exception as e:
        return {"success": False, "error": f"Error reading TXT file: {e}", "data": {"is_parsable": False}}

def estimate_docx_pages_with_unstructured(file_path):
    try:
        elements = partition_docx(filename=file_path)
        page_breaks = sum(1 for element in elements if element.category == "PageBreak")
        return {"success": True, "data": {"page_count": page_breaks + 1, "is_parsable": True}}
    except Exception as e:
        return {"success": False, "error": f"Error using unstructured: {e}", "data": {"is_parsable": False}}

async def get_page_count_from_file(file_path):
    try:
        suffix = os.path.splitext(file_path)[-1].lower()
        output_dir = os.path.join(os.path.dirname(file_path), "converted_pdfs")

        if suffix in [".pdf", "pdf"]:
            return await get_pdf_page_count(file_path)

        elif suffix in [".docx", "doc"]:
            return await asyncio.to_thread(estimate_docx_pages_with_unstructured, file_path)

        elif suffix in [".txt", "txt"]:
            return await count_words_in_txt(file_path)

        else:
            return {"success": False, "error": "Unsupported file type", "data": {"is_parsable": False}}

    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}", "data": {"is_parsable": False}}


# 3. High-level endpoint that glues both
async def get_page_count(url: str, file_type: str):
    file_path = await fetch_file_async(url, file_type)
    try:
        result = await get_page_count_from_file(file_path)
        return result
    finally:
        await _cleanup_temp_file_async(file_path)
