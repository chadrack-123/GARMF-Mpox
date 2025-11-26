"""
Datasets API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel

from core.database import get_db
from models.dataset import Dataset, DatasetType

router = APIRouter()


# Pydantic schemas
class DatasetCreate(BaseModel):
    study_id: int
    type: DatasetType
    storage_uri: str
    checksum: str | None = None
    data_contract: dict | None = None
    license: str | None = None


class DatasetResponse(BaseModel):
    id: int
    study_id: int
    type: DatasetType
    storage_uri: str
    checksum: str | None
    data_contract: dict | None
    license: str | None
    
    class Config:
        from_attributes = True


@router.get("/studies/{study_id}/datasets", response_model=List[DatasetResponse])
def get_study_datasets(study_id: int, db: Session = Depends(get_db)):
    """Get all datasets for a study"""
    datasets = db.query(Dataset).filter(Dataset.study_id == study_id).all()
    return datasets


@router.post("", response_model=DatasetResponse)
def create_dataset(dataset: DatasetCreate, db: Session = Depends(get_db)):
    """Create a new dataset"""
    db_dataset = Dataset(**dataset.model_dump())
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Get a specific dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset
