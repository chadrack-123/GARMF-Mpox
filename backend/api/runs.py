"""
Runs API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from datetime import datetime

from core.database import get_db
from models.run import Run, RunKind, RunStatus

router = APIRouter()


# Pydantic schemas
class RunCreate(BaseModel):
    study_id: int
    config_id: int
    kind: RunKind = RunKind.BASELINE


class RunResponse(BaseModel):
    id: int
    study_id: int
    config_id: int
    kind: RunKind
    status: RunStatus
    started_at: datetime | None
    finished_at: datetime | None
    metrics: dict | None
    reproducibility_metrics: dict | None
    environment_info: dict | None
    random_seeds: dict | None
    split_hash: str | None
    container_digest: str | None
    error_message: str | None
    
    class Config:
        from_attributes = True


@router.get("", response_model=List[RunResponse])
def get_runs(db: Session = Depends(get_db)):
    """Get all runs"""
    runs = db.query(Run).order_by(Run.started_at.desc()).all()
    return runs


@router.post("", response_model=RunResponse)
def create_run(
    run: RunCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create and execute a new run"""
    from ml.runner.executor import execute_run
    
    # Create run record
    db_run = Run(**run.model_dump(), status=RunStatus.PENDING)
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    
    # Execute run in background
    background_tasks.add_task(execute_run, db_run.id)
    
    return db_run


@router.get("/{run_id}", response_model=RunResponse)
def get_run(run_id: int, db: Session = Depends(get_db)):
    """Get a specific run"""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
