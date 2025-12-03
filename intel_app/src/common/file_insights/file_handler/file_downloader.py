from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Optional

import aiofiles
import httpx
from fastapi import HTTPException, status

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

_DEFAULT_TMP_DIR = os.getenv("TMP_DIR", "/tmp")
_DEFAULT_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

async def fetch_file_async(
    url: str,
    file_type: str,
    retries: int = 3,
    delay: float = 2.0,
    timeout: float = _DEFAULT_TIMEOUT,
    tmp_dir: str = _DEFAULT_TMP_DIR,
    logger: logging.Logger = None,
) -> str:
    """
    Asynchronously downloads a file from a given URL and saves it to a temp file.

    Args:
        url: URL of the file to download.
        file_type: File extension (e.g., "jpg", "pdf").
        retries: Number of retry attempts on network failure.
        delay: Delay (in seconds) between retries.
        timeout: HTTP request timeout in seconds.
        tmp_dir: Directory path for temporary files.
        logger: Logger instance to use for structured logging.

    Returns:
        The full path to the saved temporary file.

    Raises:
        HTTPException(502): If the file cannot be downloaded after all retries.
        HTTPException(500): For unexpected internal errors.
    """

    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.debug("Downloading from %s (attempt %d/%d)", url, attempt, retries)
                response = await client.get(url)
                response.raise_for_status()
                file_content = response.content

            file_path = await _create_temp_file_async(file_content, file_type, tmp_dir, logger)
            logger.info("File downloaded successfully to %s", file_path)
            return file_path

        except httpx.RequestError as exc:
            logger.warning("Network error downloading %s: %s", url, exc)
            if attempt < retries:
                await asyncio.sleep(delay)
                continue
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to download file after {retries} attempts: {exc}",
            ) from exc

        except Exception as exc:
            logger.exception("Unexpected error while downloading %s", url)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error: {exc}",
            ) from exc


async def _create_temp_file_async(
    file_content: bytes,
    file_type: str,
    tmp_dir: str = _DEFAULT_TMP_DIR,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Asynchronously creates a temporary file with given content and returns its path.

    Args:
        file_content: Binary file content to write.
        file_type: File extension (without dot).
        tmp_dir: Directory path to store the temporary file.
        logger: Logger instance for structured logging.

    Returns:
        The full path of the created file.

    Raises:
        HTTPException(500): If the file cannot be created.
    """
    logger = logger or logging.getLogger(__name__)

    os.makedirs(tmp_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}.{file_type.lstrip('.')}"
    file_path = os.path.join(tmp_dir, filename)

    try:
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_content)
        return file_path
    except Exception as exc:
        logger.exception("Failed to create temp file %s", file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create temp file: {exc}",
        ) from exc


async def _cleanup_temp_file_async(
    temp_filepath: Optional[str],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Asynchronously deletes a temporary file if it exists.

    Args:
        temp_filepath: The path of the temporary file to remove.
        logger: Logger instance for structured logging.
    """
    logger = logger or logging.getLogger(__name__)

    if not temp_filepath:
        return

    if os.path.exists(temp_filepath):
        try:
            os.remove(temp_filepath)
            logger.debug("Deleted temp file: %s", temp_filepath)
        except Exception as exc:
            logger.warning("Failed to delete temp file %s: %s", temp_filepath, exc)
