import os
import torch
import unsloth
import logging
from unsloth import FastLanguageModel
from .config import settings
from .exceptions import ModelLoadException

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self):
        self.local_models = {}
        self.tokenizers = {}
        self.api_models = {}  # New store for API models
        self.prompt_templates = {}
        self.loaded = False
        
    def load_all_models(self):
        """Preload all models during startup"""
        if self.loaded:
            return
            
        for model_id, config in settings.MODEL_REGISTRY.items():
            try:
                logger.info(f"Loading model: {model_id}")
                
                if config["type"] == "local":
                    # Local model loading
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=config["name"],
                        max_seq_length=config["max_seq_length"],
                        load_in_4bit=config.get("load_in_4bit", True),
                        attn_implementation="flash_attention_2",
                        device_map="auto"
                    )
                    self.local_models[model_id] = FastLanguageModel.for_inference(model)
                    self.tokenizers[model_id] = tokenizer
                    self.prompt_templates[model_id] = config["prompt_template"]
                    
                elif config["type"] == "api":
                    # API model initialization
                    from .deepseek_rag import get_api_model
                    self.api_models[model_id] = get_api_model(model_id)
                    
                logger.info(f"Model {model_id} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {str(e)}")
                raise ModelLoadException(f"Model {model_id} loading failed")
        
        self.loaded = True
        logger.info("All models loaded")
        
    def load_model(self, model_id: str):
        """Dynamically load a model if not preloaded"""
        if model_id in self.models:
            return
            
        if model_id not in settings.MODEL_REGISTRY:
            raise ModelLoadException(f"Model ID {model_id} not found in registry")
            
        config = settings.MODEL_REGISTRY[model_id]
        try:
            logger.info(f"Dynamically loading model: {model_id}")
            if config["type"] == "local":
                # Local model loading
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=config["name"],
                    max_seq_length=config["max_seq_length"],
                    load_in_4bit=config.get("load_in_4bit", True),
                    attn_implementation="flash_attention_2"
                )
                self.local_models[model_id] = FastLanguageModel.for_inference(model)
                self.tokenizers[model_id] = tokenizer
                self.prompt_templates[model_id] = config["prompt_template"]
                
            elif config["type"] == "api":
                # API model initialization
                from .deepseek_rag import get_api_model
                self.api_models[model_id] = get_api_model(model_id)
            logger.info(f"Model {model_id} loaded dynamically")
        except Exception as e:
            logger.error(f"Dynamic load failed for {model_id}: {str(e)}")
            raise ModelLoadException(f"Model {model_id} dynamic load failed")
    
    def get_model(self, model_id: str):
        """Retrieve loaded model with validation"""
        # First check local models
        if model_id in self.local_models:
            return self.local_models[model_id]
        
        # Then check API models
        if model_id in self.api_models:
            return self.api_models[model_id]
        
        # If not found in either, try to load it
        self.load_model(model_id)
        
        # Return the newly loaded model
        if model_id in self.local_models:
            return self.local_models[model_id]
        if model_id in self.api_models:
            return self.api_models[model_id]
        
        raise ModelLoadException(f"Model {model_id} could not be loaded")
    
    def get_tokenizer(self, model_id: str):
        """Retrieve tokenizer with validation"""
        if model_id not in self.tokenizers:
            self.load_model(model_id)
        return self.tokenizers.get(model_id)
    
    def get_prompt_template(self, model_id: str):
        """Get prompt template for specific model"""
        return self.prompt_templates.get(
            model_id, 
            "Hãy đóng vai là một người viết chuyên nghiệp...\n### Input:\n{}\n### Response:\n{}"
        )
    
    def list_models(self):
        """List all available models"""
        return {
            "loaded": list(self.local_models.keys()) + list(self.api_models.keys()),
            "available": list(settings.MODEL_REGISTRY.keys())
        }

# Global registry instance
model_registry = ModelRegistry()