import os
import time
from typing import List
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'.csv', '.xlsx'}
MAX_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


class FileUploadError(Exception):
    pass


class FileUploadHandler:
    def __init__(self, upload_dir: str = UPLOAD_DIR):
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def save(self, file: FileStorage) -> str:
        """Validate and persist an uploaded file. Returns the saved filepath."""
        self._validate(file)
        filename = self._unique_filename(file.filename)
        filepath = os.path.join(self.upload_dir, filename)
        file.save(filepath)

        # Verify size after save (stream may not expose size before)
        if os.path.getsize(filepath) > MAX_SIZE_BYTES:
            os.remove(filepath)
            raise FileUploadError("File exceeds the 50 MB size limit.")

        return filepath

    def list_datasets(self) -> List[str]:
        """Return all stored dataset filepaths."""
        return [
            os.path.join(self.upload_dir, f)
            for f in sorted(os.listdir(self.upload_dir))
            if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate(self, file: FileStorage) -> None:
        if not file or not file.filename:
            raise FileUploadError("No file provided.")

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise FileUploadError(
                f"Unsupported file type '{ext}'. Only CSV and Excel (.xlsx) are accepted."
            )

    def _unique_filename(self, original: str) -> str:
        timestamp = int(time.time())
        safe = secure_filename(original)
        return f"{timestamp}_{safe}"
