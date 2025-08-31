import logging  # Add this import
from fastapi import HTTPException, Depends, status, Request
from fastapi.security import APIKeyHeader
from .config import settings
from .exceptions import UnauthorizedIPException
from .models import ModelID

# Create a logger instance for this module
logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(
    name="X-API-Key", 
    auto_error=False,
    description="API key for authentication"
)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
            headers={"WWW-Authenticate": "APIKey"}
        )
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return True

def validate_model_id(model_id: str):
    """Validate requested model ID"""
    if model_id not in settings.MODEL_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model ID '{model_id}' not found"
        )
    return model_id