# services.py
import os
import torch
import logging
from .config import settings
from .utils import time_limit, get_gpu_status
from .exceptions import GPUOutOfMemoryException, GenerationException
from .model_registry import model_registry

logger = logging.getLogger(__name__)

class GenerationService:
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.translator = None  # Translator instance holder
    
    # def _get_translator(self):
    #     """Lazy-load the translator when needed"""
    #     if self.translator is None:
    #         from .deepseek_rag import QuangBinhDialectTranslator
    #         # Get API key from environment
    #         api_key = os.getenv(settings.MODEL_REGISTRY["deepseek_rag"]["api_key_env"])
    #         if not api_key:
    #             raise ValueError("DeepSeek API key not set in environment")
            
    #         # Initialize translator
    #         self.translator = QuangBinhDialectTranslator(deepseek_api_key=api_key)
    #         logger.info("DeepSeek + RAG model initialized")
        
        return self.translator

    def generate(self, model_id: str, prompt: str) -> str:
        """Generate text using specified model"""
        try:
            with time_limit(settings.GENERATION_TIMEOUT):
                model = model_registry.get_model(model_id)
                config = settings.MODEL_REGISTRY[model_id]
                
                # Handle API models
                if config["type"] == "api":
                    return model.translate_sentence(prompt)[1]  # Return only translated text
                
                # Handle local models
                tokenizer = model_registry.get_tokenizer(model_id)
                prompt_template = model_registry.get_prompt_template(model_id)
                
                formatted_prompt = prompt_template.format(prompt, "")
                inputs = tokenizer(
                    [formatted_prompt], 
                    return_tensors="pt"
                ).to(model.device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=settings.MODEL_REGISTRY[model_id]["max_seq_length"],
                    do_sample=True,
                    use_cache=True
                )
                
                answer = tokenizer.batch_decode(
                    outputs, 
                    skip_special_tokens=True
                )[0]
                return answer.split("### Response:")[-1].strip()
                
        except torch.cuda.OutOfMemoryError:
            logger.exception("CUDA out of memory")
            raise GPUOutOfMemoryException("GPU memory exhausted")
        except Exception as e:
            logger.exception("Generation failed")
            raise GenerationException(f"Text generation failed: {str(e)}")
    
    def health_status(self):
        return {
            "models_loaded": model_registry.list_models()["loaded"],
            "gpu_available": self.gpu_available,
            "gpu_memory": get_gpu_status()
        }

# Initialize service
generation_service = GenerationService()