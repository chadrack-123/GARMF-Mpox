"""
Configs API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from datetime import datetime

from core.database import get_db
from models.config import Config, ConfigType

router = APIRouter()


# Pydantic schemas
class ConfigCreate(BaseModel):
    study_id: int
    name: str
    yaml_text: str
    config_type: ConfigType = ConfigType.BASELINE


class ConfigResponse(BaseModel):
    id: int
    study_id: int
    name: str
    yaml_text: str
    created_at: datetime
    config_type: ConfigType
    
    class Config:
        from_attributes = True


@router.get("/studies/{study_id}/configs", response_model=List[ConfigResponse])
def get_study_configs(study_id: int, db: Session = Depends(get_db)):
    """Get all configs for a study"""
    configs = db.query(Config).filter(Config.study_id == study_id).all()
    return configs


@router.post("/studies/{study_id}/configs", response_model=ConfigResponse)
def create_config(study_id: int, config: ConfigCreate, db: Session = Depends(get_db)):
    """Create a new config for a study"""
    db_config = Config(**config.model_dump())
    db.add(db_config)
    db.commit()
    db.refresh(db_config)
    return db_config


@router.get("/{config_id}", response_model=ConfigResponse)
def get_config(config_id: int, db: Session = Depends(get_db)):
    """Get a specific config"""
    config = db.query(Config).filter(Config.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")
    return config
