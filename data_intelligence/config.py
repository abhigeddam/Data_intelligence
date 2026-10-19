"""
Global configuration settings for OmniInsight SMB Data Intelligence.
"""

from pathlib import Path
from pydantic import BaseModel, Field
import os

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    app_name: str = "OmniInsight AI SMB Data Intelligence"
    app_version: str = "1.0.0"
    env: str = os.getenv("APP_ENV", "development")
    
    # Storage directories
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
    samples_dir: Path = BASE_DIR / "data" / "samples"
    outputs_dir: Path = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs")
    reports_dir: Path = BASE_DIR / os.getenv("REPORTS_DIR", "reports")
    
    # Server network
    api_host: str = os.getenv("HOST", "0.0.0.0")
    api_port: int = int(os.getenv("PORT", "8000"))
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    # LLM keys (optional; rule/statistical agents provide full autonomy offline)
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY", None)
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY", None)
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY", None)

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        for path in [self.data_dir, self.samples_dir, self.outputs_dir, self.reports_dir]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
