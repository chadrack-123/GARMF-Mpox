"""
Data contract validation for datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import json


class DataContractValidator:
    """Validates datasets against their data contracts"""
    
    def __init__(self, contract: Dict[str, Any]):
        """
        Initialize validator with a data contract
        
        Args:
            contract: Dictionary containing validation rules
        """
        self.contract = contract
        self.validation_report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
    
    def validate_tabular(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate tabular dataset
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation report
        """
        # Check required columns
        if "required_columns" in self.contract:
            missing = set(self.contract["required_columns"]) - set(df.columns)
            if missing:
                self.validation_report["valid"] = False
                self.validation_report["errors"].append(
                    f"Missing required columns: {missing}"
                )
        
        # Check data types
        if "dtypes" in self.contract:
            for col, expected_dtype in self.contract["dtypes"].items():
                if col in df.columns:
                    actual_dtype = str(df[col].dtype)
                    if not self._dtype_matches(actual_dtype, expected_dtype):
                        self.validation_report["warnings"].append(
                            f"Column {col}: expected {expected_dtype}, got {actual_dtype}"
                        )
        
        # Check value ranges
        if "ranges" in self.contract:
            for col, range_spec in self.contract["ranges"].items():
                if col in df.columns:
                    min_val, max_val = range_spec.get("min"), range_spec.get("max")
                    col_min, col_max = df[col].min(), df[col].max()
                    
                    if min_val is not None and col_min < min_val:
                        self.validation_report["errors"].append(
                            f"Column {col}: min value {col_min} < {min_val}"
                        )
                        self.validation_report["valid"] = False
                    
                    if max_val is not None and col_max > max_val:
                        self.validation_report["errors"].append(
                            f"Column {col}: max value {col_max} > {max_val}"
                        )
                        self.validation_report["valid"] = False
        
        # Check missing values
        missing_counts = df.isnull().sum()
        if "allow_missing" in self.contract:
            for col, allowed in self.contract["allow_missing"].items():
                if col in df.columns and not allowed and missing_counts[col] > 0:
                    self.validation_report["errors"].append(
                        f"Column {col}: has {missing_counts[col]} missing values but none allowed"
                    )
                    self.validation_report["valid"] = False
        
        # Detect class imbalance
        if "target_column" in self.contract:
            target_col = self.contract["target_column"]
            if target_col in df.columns:
                class_counts = df[target_col].value_counts()
                imbalance_ratio = class_counts.max() / class_counts.min()
                
                self.validation_report["statistics"]["class_distribution"] = class_counts.to_dict()
                self.validation_report["statistics"]["imbalance_ratio"] = float(imbalance_ratio)
                
                if imbalance_ratio > 10:
                    self.validation_report["warnings"].append(
                        f"High class imbalance detected: ratio = {imbalance_ratio:.2f}"
                    )
        
        return self.validation_report
    
    def validate_image_dataset(self, image_dir: Path) -> Dict[str, Any]:
        """
        Validate image dataset
        
        Args:
            image_dir: Path to image directory
            
        Returns:
            Validation report
        """
        if not image_dir.exists():
            self.validation_report["valid"] = False
            self.validation_report["errors"].append(f"Directory not found: {image_dir}")
            return self.validation_report
        
        # Check image extensions
        expected_exts = self.contract.get("allowed_extensions", [".jpg", ".jpeg", ".png"])
        image_files = []
        for ext in expected_exts:
            image_files.extend(list(image_dir.rglob(f"*{ext}")))
        
        if len(image_files) == 0:
            self.validation_report["valid"] = False
            self.validation_report["errors"].append("No valid image files found")
        
        self.validation_report["statistics"]["total_images"] = len(image_files)
        
        # Check class distribution for classification tasks
        if self.contract.get("task") == "classification":
            class_dirs = [d for d in image_dir.iterdir() if d.is_dir()]
            class_counts = {d.name: len(list(d.rglob("*.jpg")) + list(d.rglob("*.png"))) 
                           for d in class_dirs}
            
            self.validation_report["statistics"]["class_distribution"] = class_counts
            
            if class_counts:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                if min_count > 0:
                    imbalance_ratio = max_count / min_count
                    self.validation_report["statistics"]["imbalance_ratio"] = imbalance_ratio
                    
                    if imbalance_ratio > 10:
                        self.validation_report["warnings"].append(
                            f"High class imbalance detected: ratio = {imbalance_ratio:.2f}"
                        )
        
        return self.validation_report
    
    @staticmethod
    def _dtype_matches(actual: str, expected: str) -> bool:
        """Check if data types match"""
        # Simplified type matching
        type_groups = {
            "int": ["int64", "int32", "int16", "int8"],
            "float": ["float64", "float32", "float16"],
            "object": ["object", "string"],
            "bool": ["bool"]
        }
        
        for group, types in type_groups.items():
            if expected in types and actual in types:
                return True
        
        return actual == expected


def generate_contract_from_data(df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
    """
    Generate a data contract from a DataFrame
    
    Args:
        df: DataFrame to analyze
        target_column: Name of target column for classification
        
    Returns:
        Generated contract dictionary
    """
    contract = {
        "required_columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "ranges": {},
        "allow_missing": {}
    }
    
    # Add numeric ranges
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        contract["ranges"][col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max())
        }
    
    # Check missing values
    for col in df.columns:
        contract["allow_missing"][col] = df[col].isnull().any()
    
    # Add target column info
    if target_column and target_column in df.columns:
        contract["target_column"] = target_column
    
    return contract
