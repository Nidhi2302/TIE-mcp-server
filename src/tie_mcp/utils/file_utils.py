"""
File utilities for TIE MCP Server
"""

import asyncio
import json
import logging
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

logger = logging.getLogger(__name__)


async def ensure_directory(path: Path):
    """Ensure a directory exists, create if it doesn't"""
    try:
        await aiofiles.os.makedirs(path, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


async def safe_file_operation(operation: Callable[[], Any], max_retries: int = 3) -> Any:
    """
    Safely perform a file operation with retries

    Args:
        operation: Function to execute
        max_retries: Maximum number of retries

    Returns:
        Operation result
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            # Run the operation in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, operation)
            return result
        except Exception as e:
            last_exception = e
            logger.warning(f"File operation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff

    logger.error(f"File operation failed after {max_retries} attempts")
    raise last_exception


async def read_json_file(file_path: Path) -> dict:
    """Read a JSON file asynchronously"""
    try:
        async with aiofiles.open(file_path, encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        raise


async def write_json_file(file_path: Path, data: dict, indent: int = 2):
    """Write data to a JSON file asynchronously"""
    try:
        # Ensure parent directory exists
        await ensure_directory(file_path.parent)

        # Write to temporary file first, then rename for atomic operation
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")

        async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
            content = json.dumps(data, indent=indent, ensure_ascii=False)
            await f.write(content)

        # Atomic rename
        await aiofiles.os.rename(temp_path, file_path)
        logger.debug(f"Successfully wrote JSON file: {file_path}")

    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {e}")
        # Clean up temp file if it exists
        try:
            await aiofiles.os.remove(temp_path)
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")
        raise


# Pickle functions removed for security reasons
# Use JSON serialization with read_json_file/write_json_file instead
# For complex objects, implement custom serialization methods


async def copy_file(src: Path, dst: Path):
    """Copy a file asynchronously"""
    try:
        # Ensure destination directory exists
        await ensure_directory(dst.parent)

        def _copy_file():
            shutil.copy2(src, dst)

        await safe_file_operation(_copy_file)
        logger.debug(f"Successfully copied file: {src} -> {dst}")

    except Exception as e:
        logger.error(f"Error copying file {src} to {dst}: {e}")
        raise


async def move_file(src: Path, dst: Path):
    """Move a file asynchronously"""
    try:
        # Ensure destination directory exists
        await ensure_directory(dst.parent)

        await aiofiles.os.rename(src, dst)
        logger.debug(f"Successfully moved file: {src} -> {dst}")

    except Exception as e:
        logger.error(f"Error moving file {src} to {dst}: {e}")
        raise


async def delete_file(file_path: Path):
    """Delete a file asynchronously"""
    try:
        if await aiofiles.os.path.exists(file_path):
            await aiofiles.os.remove(file_path)
            logger.debug(f"Successfully deleted file: {file_path}")
        else:
            logger.warning(f"File does not exist: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        raise


async def delete_directory(dir_path: Path, recursive: bool = True):
    """Delete a directory asynchronously"""
    try:
        if await aiofiles.os.path.exists(dir_path):

            def _delete_dir():
                if recursive:
                    shutil.rmtree(dir_path)
                else:
                    dir_path.rmdir()

            await safe_file_operation(_delete_dir)
            logger.debug(f"Successfully deleted directory: {dir_path}")
        else:
            logger.warning(f"Directory does not exist: {dir_path}")
    except Exception as e:
        logger.error(f"Error deleting directory {dir_path}: {e}")
        raise


async def get_file_size(file_path: Path) -> int:
    """Get file size asynchronously"""
    try:
        stat = await aiofiles.os.stat(file_path)
        return stat.st_size
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        raise


async def file_exists(file_path: Path) -> bool:
    """Check if file exists asynchronously"""
    try:
        return await aiofiles.os.path.exists(file_path)
    except Exception as e:
        logger.error(f"Error checking if file exists {file_path}: {e}")
        return False


async def list_files(directory: Path, pattern: str = "*", recursive: bool = False) -> list[Path]:
    """List files in a directory asynchronously"""
    try:

        def _list_files():
            if recursive:
                return list(directory.rglob(pattern))
            else:
                return list(directory.glob(pattern))

        return await safe_file_operation(_list_files)
    except Exception as e:
        logger.error(f"Error listing files in {directory}: {e}")
        raise


async def get_directory_size(directory: Path) -> int:
    """Get total size of a directory asynchronously"""
    try:

        def _get_dir_size():
            total_size = 0
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size

        return await safe_file_operation(_get_dir_size)
    except Exception as e:
        logger.error(f"Error getting directory size for {directory}: {e}")
        raise


class TemporaryDirectory:
    """Async context manager for temporary directories"""

    def __init__(self, prefix: str = "tie_mcp_", suffix: str = ""):
        self.prefix = prefix
        self.suffix = suffix
        self.path: Path | None = None

    async def __aenter__(self) -> Path:
        def _create_temp_dir():
            return tempfile.mkdtemp(prefix=self.prefix, suffix=self.suffix)

        temp_dir = await safe_file_operation(_create_temp_dir)
        self.path = Path(temp_dir)
        logger.debug(f"Created temporary directory: {self.path}")
        return self.path

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.path and await aiofiles.os.path.exists(self.path):
            await delete_directory(self.path, recursive=True)
            logger.debug(f"Cleaned up temporary directory: {self.path}")


class TemporaryFile:
    """Async context manager for temporary files"""

    def __init__(self, prefix: str = "tie_mcp_", suffix: str = "", delete: bool = True):
        self.prefix = prefix
        self.suffix = suffix
        self.delete = delete
        self.path: Path | None = None

    async def __aenter__(self) -> Path:
        def _create_temp_file():
            fd, temp_path = tempfile.mkstemp(prefix=self.prefix, suffix=self.suffix)
            # Close the file descriptor since we'll use async file operations
            import os

            os.close(fd)
            return temp_path

        temp_file = await safe_file_operation(_create_temp_file)
        self.path = Path(temp_file)
        logger.debug(f"Created temporary file: {self.path}")
        return self.path

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.delete and self.path and await file_exists(self.path):
            await delete_file(self.path)
            logger.debug(f"Cleaned up temporary file: {self.path}")


async def backup_file(file_path: Path, backup_suffix: str = ".bak") -> Path:
    """Create a backup of a file"""
    backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
    await copy_file(file_path, backup_path)
    logger.info(f"Created backup: {file_path} -> {backup_path}")
    return backup_path


async def rotate_files(file_path: Path, max_rotations: int = 5):
    """Rotate files (useful for logs)"""
    try:
        if not await file_exists(file_path):
            return

        # Rotate existing files
        for i in range(max_rotations - 1, 0, -1):
            old_file = file_path.with_suffix(f"{file_path.suffix}.{i}")
            new_file = file_path.with_suffix(f"{file_path.suffix}.{i + 1}")

            if await file_exists(old_file):
                if i == max_rotations - 1:
                    await delete_file(old_file)
                else:
                    await move_file(old_file, new_file)

        # Move current file to .1
        rotated_file = file_path.with_suffix(f"{file_path.suffix}.1")
        await move_file(file_path, rotated_file)

        logger.debug(f"Rotated file: {file_path}")

    except Exception as e:
        logger.error(f"Error rotating file {file_path}: {e}")
        raise
