import logging
from .utils import setup_logging
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

# Local imports
from .config import settings
from .dependencies import verify_api_key, validate_model_id
from .models import PromptRequest, HealthResponse, ModelListResponse
from .services import generation_service
from .utils import setup_logging
from .exceptions import (
    TimeoutException,
    GPUOutOfMemoryException,
    GenerationException,
    ModelLoadException,
    UnauthorizedIPException
)
from .model_registry import model_registry

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Multi-Model LLM API",
    description="API for multiple Vietnamese language models",
    version="1.0.0",
    docs_url="/docs" if settings.LOG_LEVEL == "DEBUG" else None,
    redoc_url=None
)

# Security middleware
if settings.SSL_CERT_PATH and settings.SSL_KEY_PATH:
    app.add_middleware(HTTPSRedirectMiddleware)

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    try:
        if settings.PRELOAD_MODELS:
            model_registry.load_all_models()
        logger.info("API startup complete")
    except Exception as e:
        logger.critical("Critical startup failure: %s", str(e))

# --- Exception Handlers ---
# ... (same as before) ...

# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    status = generation_service.health_status()
    return {
        "status": "ok",
        "version": "1.0.0",
        **status
    }

@app.get("/models", response_model=ModelListResponse)
async def list_models():
    """List available and loaded models"""
    models = model_registry.list_models()
    return {
        "available_models": models["available"],
        "loaded_models": models["loaded"]
    }

@app.post("/generate")
@limiter.limit(settings.RATE_LIMIT)
async def generate_text(
    request: Request,
    prompt_req: PromptRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Generate text using specified model
    - model_id: Model to use (default: 'ancient')
    """
    # Validate model ID
    model_id = validate_model_id(prompt_req.model_id)
    
    # Generate response
    result = generation_service.generate(model_id, prompt_req.prompt)
    return {
        "model_id": model_id,
        "response": result
    }

# ... (global exception handler) ...