"""
Artefacts API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from pathlib import Path

from core.database import get_db
from models.artefact import Artefact, ArtefactType

router = APIRouter()


# Pydantic schemas
class ArtefactResponse(BaseModel):
    id: int
    run_id: int
    type: ArtefactType
    path: str
    metadata: dict | None
    
    class Config:
        from_attributes = True


@router.get("/runs/{run_id}/artefacts", response_model=List[ArtefactResponse])
def get_run_artefacts(run_id: int, db: Session = Depends(get_db)):
    """Get all artefacts for a run"""
    artefacts = db.query(Artefact).filter(Artefact.run_id == run_id).all()
    return artefacts


@router.get("/{artefact_id}/download")
def download_artefact(artefact_id: int, db: Session = Depends(get_db)):
    """Download an artefact file"""
    artefact = db.query(Artefact).filter(Artefact.id == artefact_id).first()
    if not artefact:
        raise HTTPException(status_code=404, detail="Artefact not found")
    
    file_path = Path(artefact.path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artefact file not found")
    
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream"
    )


@router.get("/runs/{run_id}/xai", response_model=List[ArtefactResponse])
def get_run_xai(run_id: int, db: Session = Depends(get_db)):
    """Get XAI artefacts for a run"""
    artefacts = db.query(Artefact).filter(
        Artefact.run_id == run_id,
        Artefact.type == ArtefactType.XAI
    ).all()
    return artefacts


@router.get("/runs/{run_id}/docs", response_model=List[ArtefactResponse])
def get_run_docs(run_id: int, db: Session = Depends(get_db)):
    """Get documentation artefacts for a run"""
    artefacts = db.query(Artefact).filter(
        Artefact.run_id == run_id,
        Artefact.type == ArtefactType.DOC
    ).all()
    return artefacts


@router.get("/bundles/{run_id}")
def get_bundle(run_id: int, db: Session = Depends(get_db)):
    """Download release bundle for a run"""
    artefact = db.query(Artefact).filter(
        Artefact.run_id == run_id,
        Artefact.type == ArtefactType.BUNDLE
    ).first()
    
    if not artefact:
        raise HTTPException(status_code=404, detail="Bundle not found")
    
    file_path = Path(artefact.path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Bundle file not found")
    
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/zip"
    )
