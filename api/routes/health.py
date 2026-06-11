"""
api/routes/health.py

Health check endpoint for Railway uptime monitoring.
"""

from fastapi import APIRouter
from api.schemas import HealthResponse
from config import APP_NAME

router = APIRouter()


@router.get("/health")
async def health():
    """Health check for Railway uptime monitoring."""
    return HealthResponse(app=APP_NAME, version="0.4.0")