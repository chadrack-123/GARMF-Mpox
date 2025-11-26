"""
Study database model
"""

from sqlalchemy import Column, Integer, String, Text, Enum
from sqlalchemy.orm import relationship
import enum

from core.database import Base


class ModalityType(str, enum.Enum):
    """Study modality types"""
    IMAGE = "image"
    TABULAR = "tabular"
    MIXED = "mixed"


class Study(Base):
    """Study model representing a published model/paper to be reproduced"""
    
    __tablename__ = "studies"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    citation = Column(Text, nullable=True)
    doi = Column(String(200), nullable=True)
    disease = Column(String(200), nullable=False, default="Mpox")
    modality = Column(Enum(ModalityType), nullable=False)
    notes = Column(Text, nullable=True)
    
    # Relationships
    datasets = relationship("Dataset", back_populates="study", cascade="all, delete-orphan")
    configs = relationship("Config", back_populates="study", cascade="all, delete-orphan")
    runs = relationship("Run", back_populates="study", cascade="all, delete-orphan")
