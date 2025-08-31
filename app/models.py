from pydantic import BaseModel, Field
from enum import Enum

class ModelID(str, Enum):
    ancient_finetuned = "ancient_finetuned"
    ancient_deepseek_api = "ancient_deepseek_api"
    ancient_openai_api = "ancient_openai_api"
    qb_finetuned = "qb_finetuned"
    qb_deepseek_api = "qb_deepseek_api"
    qb_openai_api = "qb_openai_api"

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    model_id: ModelID = Field(default=ModelID.qb_finetuned, description="Model to use for generation")

class HealthResponse(BaseModel):
    status: str
    models_loaded: list
    gpu_available: bool
    gpu_memory: str
    version: str = "1.0.0"

class ModelListResponse(BaseModel):
    available_models: list
    loaded_models: list