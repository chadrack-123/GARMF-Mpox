"""
Release bundle builder
"""

import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def create_release_bundle(run_id: int, run_dir: Path) -> Path:
    """
    Create a complete release bundle for a run
    
    Args:
        run_id: Run identifier
        run_dir: Run directory containing artefacts
        
    Returns:
        Path to created bundle ZIP file
    """
    # Create bundle directory
    bundle_dir = Path("artifacts/release") / str(run_id)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    
    temp_bundle_dir = bundle_dir / f"run_{run_id}_bundle"
    temp_bundle_dir.mkdir(exist_ok=True)
    
    logger.info(f"Creating release bundle for run {run_id}")
    
    # Copy model weights
    model_files = list(run_dir.glob("*.pth")) + list(run_dir.glob("*.joblib"))
    if model_files:
        models_dir = temp_bundle_dir / "models"
        models_dir.mkdir(exist_ok=True)
        for model_file in model_files:
            shutil.copy(model_file, models_dir / model_file.name)
    
    # Copy config files
    config_files = list(run_dir.glob("*.yaml")) + list(run_dir.glob("*.yml"))
    if config_files:
        config_dir = temp_bundle_dir / "config"
        config_dir.mkdir(exist_ok=True)
        for config_file in config_files:
            shutil.copy(config_file, config_dir / config_file.name)
    
    # Copy documentation
    docs_dir = run_dir / "docs"
    if docs_dir.exists():
        target_docs_dir = temp_bundle_dir / "docs"
        shutil.copytree(docs_dir, target_docs_dir)
    
    # Copy XAI outputs
    xai_dir = run_dir / "xai"
    if xai_dir.exists():
        target_xai_dir = temp_bundle_dir / "xai"
        shutil.copytree(xai_dir, target_xai_dir)
    
    # Copy metrics
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        shutil.copy(metrics_file, temp_bundle_dir / "metrics.json")
    
    # Copy environment info
    env_file = run_dir / "environment.json"
    if env_file.exists():
        shutil.copy(env_file, temp_bundle_dir / "environment.json")
    
    # Create manifest
    manifest = _create_manifest(run_id, temp_bundle_dir)
    manifest_path = temp_bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    
    # Create reproduction instructions
    instructions = _create_reproduction_instructions(run_id, manifest)
    instructions_path = temp_bundle_dir / "REPRODUCE.md"
    instructions_path.write_text(instructions)
    
    # Create README
    readme = _create_readme(run_id, manifest)
    readme_path = temp_bundle_dir / "README.md"
    readme_path.write_text(readme)
    
    # Create ZIP archive
    zip_path = bundle_dir / f"run_{run_id}_bundle"
    shutil.make_archive(str(zip_path), 'zip', temp_bundle_dir)
    
    # Clean up temp directory
    shutil.rmtree(temp_bundle_dir)
    
    final_zip = zip_path.with_suffix('.zip')
    logger.info(f"Created release bundle: {final_zip}")
    
    return final_zip


def _create_manifest(run_id: int, bundle_dir: Path) -> Dict[str, Any]:
    """Create bundle manifest"""
    manifest = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat(),
        "bundle_version": "1.0",
        "contents": {
            "models": [f.name for f in (bundle_dir / "models").glob("*")] if (bundle_dir / "models").exists() else [],
            "config": [f.name for f in (bundle_dir / "config").glob("*")] if (bundle_dir / "config").exists() else [],
            "docs": [f.name for f in (bundle_dir / "docs").glob("*")] if (bundle_dir / "docs").exists() else [],
            "xai": [f.name for f in (bundle_dir / "xai").rglob("*") if f.is_file()] if (bundle_dir / "xai").exists() else []
        }
    }
    
    return manifest


def _create_reproduction_instructions(run_id: int, manifest: Dict) -> str:
    """Create reproduction instructions"""
    return f"""# Reproduction Instructions for Run {run_id}

## Prerequisites

1. Python 3.11+
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Setup

1. Extract this bundle to a directory
2. Set up environment variables (if needed)
3. Review `environment.json` for exact package versions

## Reproduction Steps

### Option 1: Using Framework

```bash
# Set environment variables
export RUN_ID={run_id}

# Run reproduction
python -m ml.runner.reproduce --run-id {run_id} --config config/*.yaml
```

### Option 2: Manual Reproduction

1. Load the configuration from `config/` directory
2. Load the model from `models/` directory
3. Follow data preprocessing steps in documentation
4. Evaluate using the same test set split (see `manifest.json` for split hash)

## Verification

Compare your results with the metrics in `metrics.json`:

- Check predictive metrics (accuracy, F1, AUC)
- Verify reproducibility metrics match

## Troubleshooting

- Ensure exact package versions from `environment.json`
- Use the same random seeds specified in config
- Verify data checksums match original dataset

## Documentation

- `docs/model_card.md` - Model details
- `docs/data_card.md` - Dataset details
- `docs/reproducibility_checklist.md` - Reproducibility verification

## XAI Outputs

Explainability outputs are available in `xai/` directory:
- SHAP plots (for tabular models)
- Grad-CAM visualizations (for image models)

## Contact

For questions about reproduction, refer to the study documentation or contact the framework maintainers.

---
Generated: {datetime.utcnow().isoformat()}
"""


def _create_readme(run_id: int, manifest: Dict) -> str:
    """Create bundle README"""
    return f"""# GARMF-Mpox Run {run_id} - Release Bundle

This bundle contains all artefacts, documentation, and instructions needed to reproduce Run {run_id}.

## Bundle Contents

### Models
{_format_list(manifest['contents'].get('models', []))}

### Configuration
{_format_list(manifest['contents'].get('config', []))}

### Documentation
{_format_list(manifest['contents'].get('docs', []))}

### XAI Outputs
{len(manifest['contents'].get('xai', []))} explainability artefacts

## Quick Start

See `REPRODUCE.md` for detailed reproduction instructions.

## Manifest

Full bundle manifest available in `manifest.json`.

## Framework

This bundle was created using **GARMF-Mpox** (GenAI-Assisted Reproducible Modelling Framework for Mpox).

For more information: [Framework Documentation](#)

---
**Bundle Version:** {manifest['bundle_version']}  
**Created:** {manifest['created_at']}
"""


def _format_list(items: list) -> str:
    """Format list for markdown"""
    if not items:
        return "- None"
    return "\n".join([f"- {item}" for item in items])
