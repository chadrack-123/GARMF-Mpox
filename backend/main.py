"""
GARMF-Mpox - GenAI-Assisted Reproducible Modelling Framework for Mpox
Main FastAPI application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.database import engine, Base
from core.settings import settings
from core.logging import setup_logging
from api import studies, datasets, configs, runs, artefacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    setup_logging()
    Base.metadata.create_all(bind=engine)
    yield
    # Shutdown
    pass


app = FastAPI(
    title="GARMF-Mpox API",
    description="GenAI-Assisted Reproducible Modelling Framework for Mpox",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(studies.router, prefix="/api/studies", tags=["Studies"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(configs.router, prefix="/api/configs", tags=["Configs"])
app.include_router(runs.router, prefix="/api/runs", tags=["Runs"])
app.include_router(artefacts.router, prefix="/api/artefacts", tags=["Artefacts"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GARMF-Mpox API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}
