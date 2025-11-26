"""
Run executor - orchestrates ML pipeline execution
"""

import sys
import platform
import hashlib
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np
import torch

from core.database import SessionLocal
from models.run import Run, RunStatus
from models.config import Config
from models.artefact import Artefact, ArtefactType
from models.dataset import Dataset
from ml.contracts.validator import DataContractValidator
from ml.pipelines.tabular import TabularPipeline
from ml.pipelines.imaging import ImagingPipeline
from ml.xai.tabular import generate_shap_explanations
from ml.xai.imaging import generate_gradcam_for_model
from ml.docs.generator import generate_data_card, generate_model_card, generate_reproducibility_checklist
from ml.bundles.builder import create_release_bundle

logger = logging.getLogger(__name__)


def execute_run(run_id: int):
    """
    Execute a complete ML run
    
    Args:
        run_id: Run identifier
    """
    db = SessionLocal()
    
    try:
        # Load run
        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            logger.error(f"Run {run_id} not found")
            return
        
        # Update status
        run.status = RunStatus.RUNNING
        run.started_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Starting run {run_id}")
        
        # Load config
        config = db.query(Config).filter(Config.id == run.config_id).first()
        config_dict = yaml.safe_load(config.yaml_text)
        
        # Load dataset
        dataset = db.query(Dataset).filter(
            Dataset.study_id == run.study_id
        ).first()
        
        # Set deterministic seeds
        random_seeds = _set_deterministic_seeds(config_dict.get('random_seed', 42))
        run.random_seeds = random_seeds
        
        # Capture environment info
        run.environment_info = _capture_environment()
        
        # Create run directory
        run_dir = Path(f"runs/{run_id}")
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate data contract
        if dataset.data_contract:
            _validate_data_contract(dataset, run_dir)
        
        # Execute based on dataset type
        if dataset.type.value == "tabular":
            metrics = _execute_tabular_run(
                config_dict, dataset, run_dir, run_id, db
            )
        elif dataset.type.value == "image":
            metrics = _execute_imaging_run(
                config_dict, dataset, run_dir, run_id, db
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset.type}")
        
        # Store metrics
        run.metrics = metrics
        
        # Compute reproducibility metrics
        run.reproducibility_metrics = _compute_reproducibility_metrics(run, run_dir)
        
        # Generate documentation
        _generate_documentation(run, dataset, metrics, run_dir, db)
        
        # Create release bundle
        bundle_path = create_release_bundle(run_id, run_dir)
        
        # Store bundle as artefact
        artefact = Artefact(
            run_id=run_id,
            type=ArtefactType.BUNDLE,
            path=str(bundle_path),
            metadata={"size_mb": bundle_path.stat().st_size / (1024 * 1024)}
        )
        db.add(artefact)
        
        # Mark run as completed
        run.status = RunStatus.COMPLETED
        run.finished_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Run {run_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Run {run_id} failed: {e}", exc_info=True)
        
        run.status = RunStatus.FAILED
        run.error_message = str(e)
        run.finished_at = datetime.utcnow()
        db.commit()
    
    finally:
        db.close()


def _set_deterministic_seeds(seed: int) -> Dict[str, int]:
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return {
        "python": seed,
        "numpy": seed,
        "torch": seed
    }


def _capture_environment() -> Dict[str, Any]:
    """Capture environment information"""
    import pkg_resources
    
    packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": sys.version,
        "packages": packages,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
    }


def _validate_data_contract(dataset: Dataset, run_dir: Path):
    """Validate dataset against contract"""
    validator = DataContractValidator(dataset.data_contract)
    
    if dataset.type.value == "tabular":
        df = pd.read_csv(dataset.storage_uri)
        report = validator.validate_tabular(df)
    else:
        report = validator.validate_image_dataset(Path(dataset.storage_uri))
    
    # Save validation report
    report_path = run_dir / "data_validation.json"
    import json
    report_path.write_text(json.dumps(report, indent=2))
    
    logger.info(f"Data validation: {'PASSED' if report['valid'] else 'FAILED'}")


def _execute_tabular_run(
    config: Dict,
    dataset: Dataset,
    run_dir: Path,
    run_id: int,
    db
) -> Dict[str, float]:
    """Execute tabular ML run"""
    # Load data
    df = pd.read_csv(dataset.storage_uri)
    
    # Initialize pipeline
    pipeline = TabularPipeline(
        model_name=config.get('model', 'random_forest'),
        random_state=config.get('random_seed', 42),
        test_size=config.get('test_size', 0.2),
        use_oversampling=config.get('use_oversampling', False)
    )
    
    # Preprocess
    X, y = pipeline.preprocess(df, config['target_column'])
    
    # Split
    X_train, X_test, y_train, y_test = pipeline.split_data(X, y)
    
    # Compute split hash
    split_hash = hashlib.sha256(
        str(X_test.index.tolist()).encode()
    ).hexdigest()
    
    run = db.query(Run).filter(Run.id == run_id).first()
    run.split_hash = split_hash
    db.commit()
    
    # Train
    pipeline.train(X_train, y_train)
    
    # Evaluate
    metrics = pipeline.evaluate(X_test, y_test)
    
    # Save model
    model_path = run_dir / "model.joblib"
    pipeline.save_model(model_path)
    
    artefact = Artefact(
        run_id=run_id,
        type=ArtefactType.MODEL,
        path=str(model_path),
        metadata={"model_type": config.get('model')}
    )
    db.add(artefact)
    db.commit()
    
    # Generate XAI
    if config.get('generate_xai', True):
        xai_dir = run_dir / "xai" / "tabular"
        xai_results = generate_shap_explanations(
            pipeline.model, X_train, X_test, xai_dir
        )
        
        for xai_file in xai_dir.glob("*"):
            artefact = Artefact(
                run_id=run_id,
                type=ArtefactType.XAI,
                path=str(xai_file),
                metadata=xai_results
            )
            db.add(artefact)
        db.commit()
    
    return metrics


def _execute_imaging_run(
    config: Dict,
    dataset: Dataset,
    run_dir: Path,
    run_id: int,
    db
) -> Dict[str, float]:
    """Execute imaging ML run"""
    # Initialize pipeline
    pipeline = ImagingPipeline(
        model_name=config.get('model', 'resnet18'),
        image_size=config.get('image_size', 224),
        batch_size=config.get('batch_size', 32),
        num_epochs=config.get('num_epochs', 10),
        random_state=config.get('random_seed', 42)
    )
    
    # Prepare data
    dataset_path = Path(dataset.storage_uri)
    train_loader, val_loader = pipeline.prepare_data(dataset_path)
    
    # Train
    history = pipeline.train(train_loader, val_loader)
    
    # Evaluate
    metrics = pipeline.evaluate(val_loader)
    
    # Save model
    model_path = run_dir / "model.pth"
    pipeline.save_model(model_path)
    
    artefact = Artefact(
        run_id=run_id,
        type=ArtefactType.MODEL,
        path=str(model_path),
        metadata={"model_type": config.get('model')}
    )
    db.add(artefact)
    db.commit()
    
    # Generate XAI (Grad-CAM for first few test images)
    if config.get('generate_xai', True):
        xai_dir = run_dir / "xai" / "images"
        xai_dir.mkdir(parents=True, exist_ok=True)
        
        # Get sample images
        sample_images = list(dataset_path.rglob("*.jpg"))[:5]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for img_path in sample_images:
            xai_result = generate_gradcam_for_model(
                pipeline.model, img_path, xai_dir, device,
                image_size=config.get('image_size', 224),
                model_name=config.get('model', 'resnet18')
            )
            
            # Store XAI artefacts
            for key, path in xai_result.items():
                if key != "image":
                    artefact = Artefact(
                        run_id=run_id,
                        type=ArtefactType.XAI,
                        path=path,
                        metadata={"image": str(img_path), "type": key}
                    )
                    db.add(artefact)
        db.commit()
    
    return metrics


def _compute_reproducibility_metrics(run: Run, run_dir: Path) -> Dict[str, Any]:
    """Compute reproducibility metrics"""
    metrics = {
        "environment_determinism": 1.0 if run.environment_info else 0.0,
        "documentation_completeness": 0.0,
        "success_rate": 1.0 if run.status == RunStatus.COMPLETED else 0.0
    }
    
    # Check documentation completeness
    doc_files = list(run_dir.glob("docs/*"))
    metrics["documentation_completeness"] = min(len(doc_files) / 3.0, 1.0)
    
    return metrics


def _generate_documentation(
    run: Run,
    dataset: Dataset,
    metrics: Dict,
    run_dir: Path,
    db
):
    """Generate all documentation"""
    docs_dir = run_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Data card
    data_card_path = generate_data_card(
        {
            "name": f"Dataset {dataset.id}",
            "type": dataset.type.value,
            "source": dataset.storage_uri,
            "license": dataset.license
        },
        docs_dir / "data_card.md"
    )
    
    # Model card
    model_card_path = generate_model_card(
        {
            "name": f"Run {run.id}",
            "type": run.kind.value,
            "metrics": metrics,
            "reproducibility_metrics": run.reproducibility_metrics
        },
        docs_dir / "model_card.md"
    )
    
    # Reproducibility checklist
    checklist_path = generate_reproducibility_checklist(
        {
            "run_id": run.id,
            "python_version": run.environment_info.get('python_version'),
            "packages": run.environment_info.get('packages'),
            "random_seeds": run.random_seeds,
            "split_hash": run.split_hash
        },
        docs_dir / "reproducibility_checklist.md"
    )
    
    # Store as artefacts
    for path in [data_card_path, model_card_path, checklist_path]:
        artefact = Artefact(
            run_id=run.id,
            type=ArtefactType.DOC,
            path=str(path),
            metadata={"doc_type": path.stem}
        )
        db.add(artefact)
    
    db.commit()
