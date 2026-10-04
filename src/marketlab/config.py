"""Local paths for offline experiments; set MARKETLAB_HOME before importing."""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("MARKETLAB_HOME", str(Path.home() / ".marketlab")))
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

