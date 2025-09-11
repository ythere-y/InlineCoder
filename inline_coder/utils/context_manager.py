from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


@contextmanager
def ensure_directory_exists(directory: Union[Path, str]):
    """
    Context manager to ensure a directory exists.
    If the directory does not exist, it will be created.
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory {directory} created.")
        yield
    finally:
        pass
