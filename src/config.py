from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    log_dir: Path = Path(os.getenv("LOG_DIR", "logs"))
    model_path: Path = Path(os.getenv("MODEL_PATH", "models/cv_fit_model.pt"))
    borderline_low: float = float(os.getenv("BORDERLINE_LOW", "0.45"))
    borderline_high: float = float(os.getenv("BORDERLINE_HIGH", "0.55"))
    # Ollama LLM backend
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    @property
    def log_dir_abs(self) -> Path:
        return (self.project_root / self.log_dir).resolve()

    @property
    def model_path_abs(self) -> Path:
        return (self.project_root / self.model_path).resolve()


settings = Settings()
