"""
Studies API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel

from core.database import get_db
from models.study import Study, ModalityType

router = APIRouter()


# Pydantic schemas
class StudyCreate(BaseModel):
    title: str
    citation: str | None = None
    doi: str | None = None
    disease: str = "Mpox"
    modality: ModalityType
    notes: str | None = None


class StudyResponse(BaseModel):
    id: int
    title: str
    citation: str | None
    doi: str | None
    disease: str
    modality: ModalityType
    notes: str | None
    
    class Config:
        from_attributes = True


@router.get("", response_model=List[StudyResponse])
def get_studies(db: Session = Depends(get_db)):
    """Get all studies"""
    studies = db.query(Study).all()
    return studies


@router.post("", response_model=StudyResponse)
def create_study(study: StudyCreate, db: Session = Depends(get_db)):
    """Create a new study"""
    db_study = Study(**study.model_dump())
    db.add(db_study)
    db.commit()
    db.refresh(db_study)
    return db_study


@router.get("/{study_id}", response_model=StudyResponse)
def get_study(study_id: int, db: Session = Depends(get_db)):
    """Get a specific study"""
    study = db.query(Study).filter(Study.id == study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    return study


@router.delete("/{study_id}")
def delete_study(study_id: int, db: Session = Depends(get_db)):
    """Delete a study"""
    study = db.query(Study).filter(Study.id == study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    db.delete(study)
    db.commit()
    return {"message": "Study deleted successfully"}
