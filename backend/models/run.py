"""
Run database model
"""

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Enum, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from core.database import Base


class RunKind(str, enum.Enum):
    """Run types"""
    BASELINE = "baseline"
    FRAMEWORK_ENHANCED = "framework_enhanced"


class RunStatus(str, enum.Enum):
    """Run status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Run(Base):
    """Run model for tracking ML experiment executions"""
    
    __tablename__ = "runs"
    
    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(Integer, ForeignKey("studies.id"), nullable=False)
    config_id = Column(Integer, ForeignKey("configs.id"), nullable=False)
    kind = Column(Enum(RunKind), nullable=False)
    status = Column(Enum(RunStatus), nullable=False, default=RunStatus.PENDING)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    
    # Metrics
    metrics = Column(JSON, nullable=True)  # Predictive metrics
    reproducibility_metrics = Column(JSON, nullable=True)
    
    # Reproducibility tracking
    environment_info = Column(JSON, nullable=True)  # OS, Python version, packages
    random_seeds = Column(JSON, nullable=True)  # All seeds used
    split_hash = Column(String(64), nullable=True)  # Hash of train/test split
    container_digest = Column(String(128), nullable=True)  # Docker digest or env hash
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    
    # Relationships
    study = relationship("Study", back_populates="runs")
    config = relationship("Config", back_populates="runs")
    artefacts = relationship("Artefact", back_populates="run", cascade="all, delete-orphan")
