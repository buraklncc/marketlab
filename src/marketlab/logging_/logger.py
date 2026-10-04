from typing import Any, Optional, Callable
from marketlab.logging_.file_writer import FileWriter

class Logger:
    _instance: Optional["Logger"] = None
    _broadcast_callback: Optional[Callable] = None

    def __new__(cls) -> "Logger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._writer = FileWriter()
        return cls._instance

    def set_broadcast_callback(self, callback: Callable):
        """Set a callback (usually for WebSocket broadcast)."""
        self._broadcast_callback = callback

    def _maybe_broadcast(self, level: str, event: str, message: str, data: Any = None):
        if self._broadcast_callback:
            try:
                import asyncio
                from datetime import datetime, timezone
                payload = {
                    "type": "log",
                    "payload": {
                        "level": level,
                        "event": event,
                        "message": message,
                        "data": data,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
                if asyncio.iscoroutinefunction(self._broadcast_callback):
                    asyncio.create_task(self._broadcast_callback(payload))
                else:
                    self._broadcast_callback(payload)
            except Exception:
                pass

    def simple(self, event: str, message: str, data: Any = None) -> None:
        self._writer.write("simple", event, message, data)
        self._maybe_broadcast("simple", event, message, data)

    def detailed(self, event: str, message: str, data: Any = None) -> None:
        self._writer.write("detailed", event, message, data)
        self._maybe_broadcast("detailed", event, message, data)

    def debug(self, event: str, message: str, data: Any = None) -> None:
        self._writer.write("debug", event, message, data)
        self._maybe_broadcast("debug", event, message, data)

    def get_logs(self, level: str = "simple", limit: int = 100) -> list[dict]:
        return self._writer.read(level, limit)

def get_logger() -> Logger:
    return Logger()

# Global singleton instance
logger = Logger()

