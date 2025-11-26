"""
Application settings and configuration
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    DATABASE_URL: str = "postgresql://garmf:garmf123@localhost:5432/garmf_mpox"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Paths
    ARTEFACTS_PATH: str = "artifacts"
    DATASETS_PATH: str = "data/datasets"
    GENERATED_CODE_PATH: str = "generated"
    RUNS_PATH: str = "runs"
    
    # ML Settings
    RANDOM_SEED: int = 42
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/garmf.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
