"""
Artefact database model
"""

from sqlalchemy import Column, Integer, String, ForeignKey, Enum, JSON
from sqlalchemy.orm import relationship
import enum

from core.database import Base


class ArtefactType(str, enum.Enum):
    """Artefact types"""
    LOG = "log"
    MODEL = "model"
    METRIC_PLOT = "metric_plot"
    XAI = "xai"
    DOC = "doc"
    BUNDLE = "bundle"


class Artefact(Base):
    """Artefact model for storing run outputs"""
    
    __tablename__ = "artefacts"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    type = Column(Enum(ArtefactType), nullable=False)
    path = Column(String(1000), nullable=False)
    meta = Column(JSON, nullable=True)  # Renamed from 'metadata' (SQLAlchemy reserved)
    
    # Relationships
    run = relationship("Run", back_populates="artefacts")
