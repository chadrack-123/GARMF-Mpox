"""
Automatic documentation generation (Data Cards, Model Cards, Reproducibility Checklists)
"""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def generate_data_card(
    dataset_info: Dict[str, Any],
    output_path: Path
) -> Path:
    """
    Generate Data Card documentation
    
    Args:
        dataset_info: Dictionary with dataset metadata
        output_path: Output file path
        
    Returns:
        Path to generated file
    """
    content = f"""# Data Card

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Information

- **Name:** {dataset_info.get('name', 'N/A')}
- **Type:** {dataset_info.get('type', 'N/A')}
- **Source:** {dataset_info.get('source', 'N/A')}
- **License:** {dataset_info.get('license', 'N/A')}

## Dataset Statistics

- **Total Samples:** {dataset_info.get('total_samples', 'N/A')}
- **Features:** {dataset_info.get('num_features', 'N/A')}
- **Classes:** {dataset_info.get('num_classes', 'N/A')}

## Class Distribution

{_format_class_distribution(dataset_info.get('class_distribution', {}))}

## Data Quality

- **Missing Values:** {dataset_info.get('missing_values', 'N/A')}
- **Duplicates:** {dataset_info.get('duplicates', 'N/A')}
- **Imbalance Ratio:** {dataset_info.get('imbalance_ratio', 'N/A')}

## Data Contract

{_format_data_contract(dataset_info.get('data_contract', {}))}

## Preprocessing Applied

{_format_preprocessing(dataset_info.get('preprocessing', []))}

## Ethical Considerations

- **Sensitive Attributes:** {dataset_info.get('sensitive_attributes', 'None identified')}
- **Bias Assessment:** {dataset_info.get('bias_assessment', 'Not assessed')}

## Limitations

{dataset_info.get('limitations', 'No known limitations documented.')}
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    logger.info(f"Generated Data Card: {output_path}")
    
    return output_path


def generate_model_card(
    model_info: Dict[str, Any],
    output_path: Path
) -> Path:
    """
    Generate Model Card documentation
    
    Args:
        model_info: Dictionary with model metadata
        output_path: Output file path
        
    Returns:
        Path to generated file
    """
    content = f"""# Model Card

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information

- **Model Name:** {model_info.get('name', 'N/A')}
- **Model Type:** {model_info.get('type', 'N/A')}
- **Architecture:** {model_info.get('architecture', 'N/A')}
- **Version:** {model_info.get('version', '1.0')}

## Intended Use

- **Primary Use:** {model_info.get('primary_use', 'Research')}
- **Target Domain:** {model_info.get('domain', 'Infectious Disease Modeling')}
- **Target Users:** {model_info.get('users', 'Researchers, Public Health Officials')}

## Training Data

- **Dataset:** {model_info.get('dataset_name', 'N/A')}
- **Training Samples:** {model_info.get('train_samples', 'N/A')}
- **Validation Samples:** {model_info.get('val_samples', 'N/A')}
- **Test Samples:** {model_info.get('test_samples', 'N/A')}

## Model Performance

### Predictive Metrics

{_format_metrics(model_info.get('metrics', {}))}

### Reproducibility Metrics

{_format_reproducibility_metrics(model_info.get('reproducibility_metrics', {}))}

## Training Details

- **Training Duration:** {model_info.get('training_duration', 'N/A')}
- **Hardware:** {model_info.get('hardware', 'N/A')}
- **Hyperparameters:**

{_format_hyperparameters(model_info.get('hyperparameters', {}))}

## Explainability

- **XAI Method:** {model_info.get('xai_method', 'N/A')}
- **Feature Importance Available:** {model_info.get('feature_importance_available', 'Yes')}

## Limitations

{model_info.get('limitations', 'No known limitations documented.')}

## Ethical Considerations

- **Fairness Assessment:** {model_info.get('fairness_assessment', 'Not assessed')}
- **Privacy Considerations:** {model_info.get('privacy', 'No PII in training data')}

## References

{model_info.get('references', 'N/A')}
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    logger.info(f"Generated Model Card: {output_path}")
    
    return output_path


def generate_reproducibility_checklist(
    run_info: Dict[str, Any],
    output_path: Path
) -> Path:
    """
    Generate Reproducibility Checklist (EPIFORGE/IDMRC style)
    
    Args:
        run_info: Dictionary with run metadata
        output_path: Output file path
        
    Returns:
        Path to generated file
    """
    content = f"""# Reproducibility Checklist

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Run ID:** {run_info.get('run_id', 'N/A')}  
**Study:** {run_info.get('study_name', 'N/A')}

## Environment Reproducibility

- [{'x' if run_info.get('python_version') else ' '}] **Python Version Recorded:** {run_info.get('python_version', 'Not recorded')}
- [{'x' if run_info.get('packages') else ' '}] **Package Versions Pinned:** {len(run_info.get('packages', []))} packages
- [{'x' if run_info.get('os_info') else ' '}] **OS Information:** {run_info.get('os_info', 'Not recorded')}
- [{'x' if run_info.get('hardware') else ' '}] **Hardware Specification:** {run_info.get('hardware', 'Not recorded')}
- [{'x' if run_info.get('container_digest') else ' '}] **Container Digest/Environment Hash:** {run_info.get('container_digest', 'Not available')}

## Data Reproducibility

- [{'x' if run_info.get('data_checksum') else ' '}] **Data Checksum:** {run_info.get('data_checksum', 'Not computed')}
- [{'x' if run_info.get('data_contract') else ' '}] **Data Contract Validated:** {run_info.get('data_contract_valid', 'No')}
- [{'x' if run_info.get('split_hash') else ' '}] **Train/Test Split Hash:** {run_info.get('split_hash', 'Not computed')}
- [{'x' if run_info.get('preprocessing_logged') else ' '}] **Preprocessing Steps Logged:** {run_info.get('preprocessing_logged', 'No')}

## Code Reproducibility

- [{'x' if run_info.get('config_snapshot') else ' '}] **Configuration Snapshot:** Available
- [{'x' if run_info.get('random_seeds') else ' '}] **Random Seeds Recorded:** {run_info.get('random_seeds', {})}
- [{'x' if run_info.get('code_version') else ' '}] **Code Version/Commit:** {run_info.get('code_version', 'Not tracked')}
- [{'x' if run_info.get('deterministic_mode') else ' '}] **Deterministic Mode Enabled:** {run_info.get('deterministic_mode', 'Unknown')}

## Documentation

- [{'x' if run_info.get('data_card') else ' '}] **Data Card Generated:** {run_info.get('data_card', 'No')}
- [{'x' if run_info.get('model_card') else ' '}] **Model Card Generated:** {run_info.get('model_card', 'No')}
- [{'x' if run_info.get('xai_outputs') else ' '}] **XAI Outputs Available:** {run_info.get('xai_outputs', 'No')}
- [{'x' if run_info.get('logs') else ' '}] **Execution Logs Saved:** {run_info.get('logs', 'No')}

## Release Bundle

- [{'x' if run_info.get('bundle_created') else ' '}] **Release Bundle Created:** {run_info.get('bundle_path', 'Not created')}
- [{'x' if run_info.get('reproduction_instructions') else ' '}] **Reproduction Instructions:** Available

## Reproducibility Score

**Overall Score:** {run_info.get('reproducibility_score', 0)}/100

### Metric Concordance

- **Accuracy Reproducible:** {run_info.get('metric_concordance', {}).get('accuracy', 'N/A')}
- **F1 Score Reproducible:** {run_info.get('metric_concordance', {}).get('f1', 'N/A')}

## Notes

{run_info.get('notes', 'No additional notes.')}
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    logger.info(f"Generated Reproducibility Checklist: {output_path}")
    
    return output_path


# Helper functions

def _format_class_distribution(dist: Dict) -> str:
    """Format class distribution"""
    if not dist:
        return "Not available"
    
    lines = []
    for cls, count in dist.items():
        lines.append(f"- **{cls}:** {count}")
    return "\n".join(lines)


def _format_data_contract(contract: Dict) -> str:
    """Format data contract"""
    if not contract:
        return "No contract specified"
    
    return f"""
- **Required Columns:** {len(contract.get('required_columns', []))}
- **Type Constraints:** {len(contract.get('dtypes', {}))} columns
- **Value Ranges:** {len(contract.get('ranges', {}))} columns
"""


def _format_preprocessing(steps: list) -> str:
    """Format preprocessing steps"""
    if not steps:
        return "No preprocessing applied"
    
    return "\n".join([f"- {step}" for step in steps])


def _format_metrics(metrics: Dict) -> str:
    """Format metrics"""
    if not metrics:
        return "No metrics available"
    
    lines = []
    for metric, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}")
        else:
            lines.append(f"- **{metric.replace('_', ' ').title()}:** {value}")
    return "\n".join(lines)


def _format_reproducibility_metrics(metrics: Dict) -> str:
    """Format reproducibility metrics"""
    if not metrics:
        return "Not computed"
    
    return f"""
- **Environment Determinism:** {metrics.get('environment_determinism', 'N/A')}
- **Documentation Completeness:** {metrics.get('documentation_completeness', 'N/A')}
- **Reproducibility Success Rate:** {metrics.get('success_rate', 'N/A')}
"""


def _format_hyperparameters(params: Dict) -> str:
    """Format hyperparameters"""
    if not params:
        return "Not specified"
    
    lines = []
    for param, value in params.items():
        lines.append(f"  - `{param}`: {value}")
    return "\n".join(lines)
