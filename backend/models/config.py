"""
Config database model
"""

from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from core.database import Base


class ConfigType(str, enum.Enum):
    """Configuration types"""
    BASELINE = "baseline"
    FRAMEWORK_ENHANCED = "framework_enhanced"


class Config(Base):
    """Config model for storing experiment configurations"""
    
    __tablename__ = "configs"
    
    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(Integer, ForeignKey("studies.id"), nullable=False)
    name = Column(String(200), nullable=False)
    yaml_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    config_type = Column(Enum(ConfigType), nullable=False, default=ConfigType.BASELINE)
    
    # Relationships
    study = relationship("Study", back_populates="configs")
    runs = relationship("Run", back_populates="config", cascade="all, delete-orphan")
