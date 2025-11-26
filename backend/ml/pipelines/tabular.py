"""
Tabular ML pipelines using scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from typing import Dict, Any, Tuple
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class TabularPipeline:
    """Pipeline for tabular classification tasks"""
    
    def __init__(
        self,
        model_name: str = "random_forest",
        random_state: int = 42,
        test_size: float = 0.2,
        use_oversampling: bool = False,
        oversampling_method: str = "smote"
    ):
        """
        Initialize tabular pipeline
        
        Args:
            model_name: Model type ('random_forest', 'xgboost', 'lightgbm')
            random_state: Random seed
            test_size: Test set proportion
            use_oversampling: Whether to apply oversampling
            oversampling_method: Oversampling method if enabled
        """
        self.model_name = model_name
        self.random_state = random_state
        self.test_size = test_size
        self.use_oversampling = use_oversampling
        self.oversampling_method = oversampling_method
        
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoders = {}
        self.feature_names = []
    
    def _get_model(self) -> Any:
        """Get model instance"""
        if self.model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_name == "xgboost":
            return XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif self.model_name == "lightgbm":
            return LGBMClassifier(
                n_estimators=100,
                random_state=self.random_state,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def preprocess(
        self,
        df: pd.DataFrame,
        target_column: str,
        categorical_columns: list = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data
        
        Args:
            df: Input dataframe
            target_column: Target column name
            categorical_columns: List of categorical columns
            
        Returns:
            Tuple of (X, y)
        """
        df = df.copy()
        
        # Separate features and target
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        self.feature_names = list(X.columns)
        
        # Handle categorical columns
        if categorical_columns is None:
            categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_columns:
            if col in X.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Imputation
        X = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns
        )
        
        # Scaling
        X = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        # Encode target
        if y.dtype == 'object':
            if 'target' not in self.label_encoders:
                self.label_encoders['target'] = LabelEncoder()
                y = pd.Series(self.label_encoders['target'].fit_transform(y))
            else:
                y = pd.Series(self.label_encoders['target'].transform(y))
        
        logger.info(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Apply oversampling if enabled
        if self.use_oversampling:
            from ml.synthetic.tabular import TabularSyntheticGenerator
            
            generator = TabularSyntheticGenerator(
                method=self.oversampling_method,
                random_state=self.random_state
            )
            X_train, y_train, aug_log = generator.generate(X_train, y_train)
            logger.info(f"Applied oversampling: {aug_log}")
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        self.model = self._get_model()
        logger.info(f"Training {self.model_name} model...")
        
        self.model.fit(X_train, y_train)
        
        logger.info("Training completed")
        return self.model
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # AUC
        try:
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        except:
            auc = None
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc) if auc else None
        }
        
        logger.info(f"Test metrics: {metrics}")
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance.tolist()))
        return {}
    
    def save_model(self, path: Path):
        """Save trained model and preprocessing objects"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load trained model and preprocessing objects"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.imputer = data['imputer']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        logger.info(f"Model loaded from {path}")
