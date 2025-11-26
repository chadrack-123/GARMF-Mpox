# GARMF-Mpox

**GenAI-Assisted Reproducible Modelling Framework for Mpox**

A complete full-stack ML research framework for reproducible infectious disease AI modelling, supporting both tabular and imaging datasets.

## 🌟 Features

### Core Capabilities
- **Study Management**: Track published models and papers for reproduction
- **Dataset Management**: Handle tabular and imaging datasets with data contracts
- **Configuration System**: YAML-based experiment configuration
- **Run Execution**: Automated ML pipeline execution with full tracking
- **Artefact Storage**: Store models, logs, XAI outputs, and documentation
- **Release Bundles**: Package everything needed for reproduction

### Reproducibility Features
- Deterministic seed control (Python, NumPy, PyTorch)
- Environment logging (OS, Python, packages, hardware)
- Data checksums and contract validation
- Train/test split hashing
- Container/environment digest tracking
- Comprehensive documentation generation

### ML Capabilities
- **Tabular Models**: RandomForest, XGBoost, LightGBM
- **Imaging Models**: ResNet18, DenseNet201, ViT
- **Synthetic Data**: SMOTE, ADASYN, image augmentation
- **XAI**: SHAP for tabular, Grad-CAM for images
- **Metrics**: Accuracy, F1, AUC, Precision, Recall

### Documentation
- Auto-generated Data Cards
- Auto-generated Model Cards
- Reproducibility Checklists (EPIFORGE/IDMRC style)

## 🏗️ Architecture

### Backend (Python 3.11 + FastAPI)
```
backend/
├── api/              # FastAPI routers
├── core/             # Settings, database, logging
├── models/           # SQLAlchemy ORM models
├── ml/
│   ├── contracts/    # Data validation
│   ├── synthetic/    # Data augmentation
│   ├── pipelines/    # ML pipelines (tabular & imaging)
│   ├── xai/          # Explainability (SHAP, Grad-CAM)
│   ├── docs/         # Documentation generation
│   ├── codegen/      # LLM code generation (stub)
│   ├── runner/       # Run executor
│   └── bundles/      # Release bundle builder
└── config_templates/ # YAML templates
```

### Frontend (React + TypeScript + Vite)
```
frontend/
├── src/
│   ├── api/          # API client
│   ├── components/   # Reusable components
│   ├── pages/        # Application pages
│   └── types/        # TypeScript types
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Docker & Docker Compose (optional)

### Option 1: Docker Compose (Recommended)

```bash
# Clone repository
cd Framework

# Start all services
docker-compose up -d

# Backend will be at http://localhost:8000
# Frontend will be at http://localhost:3000
# API docs at http://localhost:8000/docs
```

### Option 2: Manual Setup

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Unix

# Install dependencies
pip install -r requirements.txt

# Copy environment file
copy .env.example .env

# Edit .env with your database credentials

# Run database migrations (if using Alembic)
# alembic upgrade head

# Start backend
uvicorn main:app --reload
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

#### Database Setup

```bash
# Using PostgreSQL
createdb garmf_mpox
createuser garmf -P  # Password: garmf123

# Or use Docker
docker run -d \
  --name garmf_postgres \
  -e POSTGRES_USER=garmf \
  -e POSTGRES_PASSWORD=garmf123 \
  -e POSTGRES_DB=garmf_mpox \
  -p 5432:5432 \
  postgres:15
```

## 📖 Usage

### 1. Create a Study

Navigate to the Studies Dashboard and click "New Study". Fill in:
- Title
- Citation
- DOI
- Disease (e.g., "Mpox")
- Modality (tabular, image, or mixed)

### 2. Register Dataset

Go to Dataset Manager:
- Upload or register dataset URI
- Specify type (tabular/image)
- Define data contract (validation rules)
- Compute checksum

### 3. Create Configuration

Use Config Builder:
- Select study
- Choose model architecture
- Configure hyperparameters
- Set random seeds
- Enable XAI
- Save as YAML

### 4. Execute Run

In Runs section:
- Select study and config
- Choose run kind (baseline or framework_enhanced)
- Start execution

The framework will:
1. Validate data contract
2. Set deterministic seeds
3. Load and preprocess data
4. Train model
5. Evaluate performance
6. Generate XAI outputs
7. Create documentation
8. Build release bundle

### 5. View Results

- **Run Monitor**: Check status and metrics
- **XAI Explorer**: View SHAP plots or Grad-CAM heatmaps
- **Documentation**: Read data/model cards
- **Bundles**: Download complete reproduction package

## 📁 Configuration Templates

### Tabular Pipeline (`params_tabular.yaml`)
```yaml
dataset:
  type: "tabular"
  target_column: "diagnosis"

model: "random_forest"

preprocessing:
  imputation: "mean"
  encoding: "label"
  scaling: "standard"

oversampling:
  enabled: true
  method: "smote"

random_seed: 42

xai:
  enabled: true
  method: "shap"
```

### Imaging Pipeline (`params_imaging.yaml`)
```yaml
dataset:
  type: "image"

model: "resnet18"

image_size: 224
batch_size: 32
num_epochs: 10

xai:
  enabled: true
  method: "gradcam"

random_seed: 42
```

## 🔬 API Endpoints

### Studies
- `GET /api/studies` - List all studies
- `POST /api/studies` - Create study
- `GET /api/studies/{id}` - Get study details

### Datasets
- `GET /api/datasets/studies/{id}/datasets` - Get study datasets
- `POST /api/datasets` - Register dataset

### Configs
- `GET /api/configs/studies/{id}/configs` - Get study configs
- `POST /api/configs/studies/{id}/configs` - Create config

### Runs
- `GET /api/runs` - List all runs
- `POST /api/runs` - Execute run
- `GET /api/runs/{id}` - Get run details

### Artefacts
- `GET /api/artefacts/runs/{id}/artefacts` - Get run artefacts
- `GET /api/artefacts/runs/{id}/xai` - Get XAI outputs
- `GET /api/artefacts/runs/{id}/docs` - Get documentation
- `GET /api/artefacts/bundles/{id}` - Download release bundle

## 🧪 Development

### Backend Testing
```bash
cd backend
pytest
```

### Frontend Testing
```bash
cd frontend
npm test
```

### Code Quality
```bash
# Backend
black backend/
pylint backend/

# Frontend
npm run lint
```

## 📊 Reproducibility Metrics

Each run computes:
- **Metric Concordance**: Are results reproducible?
- **Documentation Completeness**: All docs generated?
- **Environment Determinism**: Environment fully specified?
- **Success Rate**: Overall reproducibility score

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

Built for reproducible infectious disease AI research, inspired by EPIFORGE and IDMRC best practices.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**GARMF-Mpox** - Making ML research reproducible, one experiment at a time.
