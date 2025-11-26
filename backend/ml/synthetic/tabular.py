"""
Synthetic data generation for tabular datasets
"""

from imblearn.over_sampling import SMOTE, ADASYN, SMOTETomek
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class TabularSyntheticGenerator:
    """Generate synthetic samples for tabular data using imbalanced-learn"""
    
    def __init__(self, method: str = "smote", random_state: int = 42):
        """
        Initialize synthetic generator
        
        Args:
            method: Oversampling method ('smote', 'smote_tomek', 'adasyn')
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.random_state = random_state
        self.sampler = None
        self.augmentation_log = {}
    
    def generate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_ratio: float = 1.0,
        minority_multiplier: float = None
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Generate synthetic samples
        
        Args:
            X: Feature matrix
            y: Target labels
            target_ratio: Target ratio of minority to majority class
            minority_multiplier: Multiply minority class by this factor
            
        Returns:
            Tuple of (X_resampled, y_resampled, augmentation_log)
        """
        # Log before counts
        before_counts = y.value_counts().to_dict()
        logger.info(f"Class distribution before augmentation: {before_counts}")
        
        # Configure sampler
        sampling_strategy = self._compute_sampling_strategy(
            y, target_ratio, minority_multiplier
        )
        
        if self.method == "smote":
            self.sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                k_neighbors=5
            )
        elif self.method == "smote_tomek":
            self.sampler = SMOTETomek(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state
            )
        elif self.method == "adasyn":
            self.sampler = ADASYN(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                n_neighbors=5
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Perform resampling
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        
        # Log after counts
        after_counts = pd.Series(y_resampled).value_counts().to_dict()
        logger.info(f"Class distribution after augmentation: {after_counts}")
        
        # Create augmentation log
        self.augmentation_log = {
            "method": self.method,
            "before_counts": before_counts,
            "after_counts": after_counts,
            "samples_added": len(y_resampled) - len(y),
            "target_ratio": target_ratio,
            "minority_multiplier": minority_multiplier,
            "parameters": {
                "random_state": self.random_state,
                "sampling_strategy": sampling_strategy
            }
        }
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled), self.augmentation_log
    
    def _compute_sampling_strategy(
        self,
        y: pd.Series,
        target_ratio: float,
        minority_multiplier: float
    ) -> Dict[int, int]:
        """Compute sampling strategy for oversampling"""
        class_counts = y.value_counts()
        majority_class = class_counts.idxmax()
        majority_count = class_counts.max()
        
        if minority_multiplier is not None:
            # Multiply minority classes
            sampling_strategy = {}
            for cls, count in class_counts.items():
                if cls != majority_class:
                    sampling_strategy[cls] = int(count * minority_multiplier)
        else:
            # Use target ratio
            target_count = int(majority_count * target_ratio)
            sampling_strategy = {}
            for cls, count in class_counts.items():
                if cls != majority_class and count < target_count:
                    sampling_strategy[cls] = target_count
        
        return sampling_strategy if sampling_strategy else 'auto'


def balance_to_max(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Balance dataset to maximum class count using SMOTE
    
    Args:
        X: Feature matrix
        y: Target labels
        random_state: Random seed
        
    Returns:
        Tuple of (X_resampled, y_resampled, log)
    """
    generator = TabularSyntheticGenerator(method="smote", random_state=random_state)
    return generator.generate(X, y, target_ratio=1.0)
