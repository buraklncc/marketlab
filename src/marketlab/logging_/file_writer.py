"""
file_writer.py — JSONL file writer with rotation.

Writes log entries to /logs/*.jsonl files.
Max 10MB per file, then rotates with date suffix.
Keeps last 7 days of logs.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any

from marketlab.config import LOGS_DIR

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
RETENTION_DAYS = 7


class FileWriter:
    """Writes JSON lines to log files with automatic rotation."""

    def __init__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def write(self, level: str, event: str, message: str,
              data: Any = None) -> None:
        """
        Write a log entry to the appropriate JSONL file.

        Parameters
        ----------
        level : str    "simple" | "detailed" | "debug"
        event : str    event type identifier
        message : str  human-readable message
        data : Any     technical payload
        """
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
            "message": message,
            "data": data,
        }

        filepath = LOGS_DIR / f"{level}.jsonl"
        self._maybe_rotate(filepath)

        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except IOError:
            pass

    def read(self, level: str, limit: int = 100,
             offset: int = 0) -> list[dict]:
        """
        Read the last `limit` entries from a log file.
        """
        filepath = LOGS_DIR / f"{level}.jsonl"
        if not filepath.exists():
            return []

        entries = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # Take last `limit` lines (after offset)
            relevant = lines[-(limit + offset):]
            if offset > 0:
                relevant = relevant[:limit]
            else:
                relevant = relevant[-limit:]
            for line in relevant:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except IOError:
            pass

        return entries

    def _maybe_rotate(self, filepath: Path) -> None:
        """Rotate file if it exceeds MAX_FILE_SIZE."""
        if not filepath.exists():
            return
        try:
            size = filepath.stat().st_size
            if size >= MAX_FILE_SIZE:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                rotated = filepath.with_suffix(f".{ts}.jsonl")
                filepath.rename(rotated)
                self._cleanup_old()
        except OSError:
            pass

    def _cleanup_old(self) -> None:
        """Remove log files older than RETENTION_DAYS."""
        cutoff = time.time() - (RETENTION_DAYS * 86400)
        try:
            for f in LOGS_DIR.iterdir():
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
        except OSError:
            pass

