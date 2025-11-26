# GARMF-Mpox Quick Start Guide

## Initial Setup

### 1. Start PostgreSQL Database

**Option A: Using Docker**
```powershell
docker run -d `
  --name garmf_postgres `
  -e POSTGRES_USER=garmf `
  -e POSTGRES_PASSWORD=garmf123 `
  -e POSTGRES_DB=garmf_mpox `
  -p 5432:5432 `
  postgres:15
```

**Option B: Using Docker Compose**
```powershell
docker-compose up -d postgres
```

### 2. Setup Backend

```powershell
cd backend

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
copy .env.example .env

# Start backend server
uvicorn main:app --reload
```

Backend will run at: http://localhost:8000  
API Documentation: http://localhost:8000/docs

### 3. Setup Frontend

```powershell
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will run at: http://localhost:3000

## Using Docker Compose (All-in-One)

```powershell
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

## Next Steps

1. Open http://localhost:3000
2. Create your first study
3. Register a dataset
4. Create a configuration
5. Execute a run
6. Explore XAI outputs

## Troubleshooting

### Backend won't start
- Check PostgreSQL is running: `docker ps`
- Verify DATABASE_URL in `.env`
- Check logs: `docker-compose logs backend`

### Frontend won't start
- Delete `node_modules` and run `npm install` again
- Check port 3000 is available

### Database connection error
- Ensure PostgreSQL container is healthy
- Wait 10 seconds after starting PostgreSQL
- Check credentials match in `.env`

## Directory Structure

```
Framework/
├── backend/           # FastAPI backend
│   ├── api/          # API endpoints
│   ├── core/         # Core configuration
│   ├── models/       # Database models
│   ├── ml/           # ML modules
│   └── main.py       # Entry point
├── frontend/         # React frontend
│   └── src/
│       ├── pages/    # Application pages
│       ├── api/      # API client
│       └── main.tsx  # Entry point
├── data/             # Datasets
├── artifacts/        # Generated artifacts
├── runs/             # Run outputs
└── docker-compose.yml
```

## Configuration Files

### Backend Environment (`.env`)
```
DATABASE_URL=postgresql://garmf:garmf123@localhost:5432/garmf_mpox
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### Config Templates
- `backend/config_templates/params_tabular.yaml` - Tabular pipeline
- `backend/config_templates/params_imaging.yaml` - Imaging pipeline
- `backend/config_templates/params_unified.yaml` - Mixed modality

## Support

For issues or questions:
- Check README.md for detailed documentation
- Review API docs at http://localhost:8000/docs
- Check logs in `logs/` directory
