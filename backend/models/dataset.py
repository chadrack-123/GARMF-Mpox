"""
Dataset database model
"""

from sqlalchemy import Column, Integer, String, Text, ForeignKey, Enum, JSON
from sqlalchemy.orm import relationship
import enum

from core.database import Base


class DatasetType(str, enum.Enum):
    """Dataset types"""
    IMAGE = "image"
    TABULAR = "tabular"


class Dataset(Base):
    """Dataset model for storing dataset metadata"""
    
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(Integer, ForeignKey("studies.id"), nullable=False)
    type = Column(Enum(DatasetType), nullable=False)
    storage_uri = Column(String(1000), nullable=False)
    checksum = Column(String(64), nullable=True)  # SHA-256
    data_contract = Column(JSON, nullable=True)  # Schema validation rules
    license = Column(String(200), nullable=True)
    
    # Relationships
    study = relationship("Study", back_populates="datasets")
