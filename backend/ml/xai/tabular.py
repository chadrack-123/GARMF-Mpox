"""
XAI (Explainable AI) module for tabular models using SHAP
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP explainer for tabular models"""
    
    def __init__(self, model: Any, X_train: pd.DataFrame):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model
            X_train: Training data for background distribution
        """
        self.model = model
        self.X_train = X_train
        
        # Select appropriate explainer
        if hasattr(model, 'predict_proba'):
            # Tree-based models
            try:
                self.explainer = shap.TreeExplainer(model)
            except:
                # Fallback to KernelExplainer
                self.explainer = shap.KernelExplainer(
                    model.predict_proba,
                    shap.sample(X_train, 100)
                )
        else:
            # Use KernelExplainer as fallback
            self.explainer = shap.KernelExplainer(
                model.predict,
                shap.sample(X_train, 100)
            )
    
    def explain(self, X_test: pd.DataFrame) -> shap.Explanation:
        """
        Generate SHAP values
        
        Args:
            X_test: Test data to explain
            
        Returns:
            SHAP Explanation object
        """
        logger.info("Computing SHAP values...")
        shap_values = self.explainer(X_test)
        return shap_values
    
    def plot_summary(
        self,
        shap_values: shap.Explanation,
        output_path: Path
    ):
        """
        Generate summary plot
        
        Args:
            shap_values: SHAP values
            output_path: Output file path
        """
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP summary plot to {output_path}")
    
    def plot_bar(
        self,
        shap_values: shap.Explanation,
        output_path: Path
    ):
        """
        Generate bar plot of mean absolute SHAP values
        
        Args:
            shap_values: SHAP values
            output_path: Output file path
        """
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP bar plot to {output_path}")
    
    def plot_waterfall(
        self,
        shap_values: shap.Explanation,
        index: int,
        output_path: Path
    ):
        """
        Generate waterfall plot for a single prediction
        
        Args:
            shap_values: SHAP values
            index: Sample index
            output_path: Output file path
        """
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[index], show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP waterfall plot to {output_path}")
    
    def get_feature_importance(
        self,
        shap_values: shap.Explanation
    ) -> pd.DataFrame:
        """
        Get feature importance from SHAP values
        
        Args:
            shap_values: SHAP values
            
        Returns:
            DataFrame with feature importance
        """
        # Mean absolute SHAP values
        importance = np.abs(shap_values.values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': shap_values.feature_names,
            'importance': importance
        })
        df = df.sort_values('importance', ascending=False)
        
        return df


def generate_shap_explanations(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: Path,
    max_samples: int = 100
) -> Dict[str, Any]:
    """
    Generate complete SHAP explanations
    
    Args:
        model: Trained model
        X_train: Training data
        X_test: Test data
        output_dir: Output directory
        max_samples: Maximum number of test samples to explain
        
    Returns:
        Dictionary with paths and feature importance
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize explainer
    explainer = SHAPExplainer(model, X_train)
    
    # Explain test set (sample if too large)
    if len(X_test) > max_samples:
        X_test_sample = X_test.sample(max_samples, random_state=42)
    else:
        X_test_sample = X_test
    
    shap_values = explainer.explain(X_test_sample)
    
    # Generate plots
    summary_path = output_dir / "shap_summary.png"
    bar_path = output_dir / "shap_importance.png"
    waterfall_path = output_dir / "shap_waterfall_sample.png"
    
    explainer.plot_summary(shap_values, summary_path)
    explainer.plot_bar(shap_values, bar_path)
    explainer.plot_waterfall(shap_values, 0, waterfall_path)
    
    # Get feature importance
    importance_df = explainer.get_feature_importance(shap_values)
    importance_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    
    logger.info(f"Generated SHAP explanations in {output_dir}")
    
    return {
        "summary_plot": str(summary_path),
        "bar_plot": str(bar_path),
        "waterfall_plot": str(waterfall_path),
        "feature_importance_csv": str(importance_path),
        "top_features": importance_df.head(10).to_dict('records')
    }
