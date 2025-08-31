import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Security
    API_KEY = os.getenv("API_KEY")
    ALLOWED_IPS = os.getenv("ALLOWED_IPS", "").split(",")
    
    # Model Registry
    MODEL_REGISTRY = {
        "ancient_finetuned": {
            "type": "local",
            "name": "DiligentPenguinn/vietnamese-paraphraser-acient-style",
            "max_seq_length": 1024,
            "prompt_template": """Hãy viết lại câu sau theo phong cách tiếng Việt cổ trang:
            ### Input:
            {}
            
            ### Response:
            {}""",
            "load_in_4bit": True
        },
        "qb_finetuned": {
            "type": "local",
            "name": "DiligentPenguinn/vietnamese-paraphraser-qb-dialect",
            "max_seq_length": 1024,
            "prompt_template": """Hãy viết lại câu sau theo phong cách phương ngữ Quảng Bình, miền Trung Việt Nam:
            ### Input:
            {}
            
            ### Response:
            {}""",
            "load_in_4bit": True
        },
        "qb_deepseek_api": {
            "type": "api",
            "api_key_env": "DEEPSEEK_API_KEY",
            "prompt_template": None # already stored elsewhere
        },
        # "qb_openai_api": {
        #     "type": "api",
        #     "api_key_env": "OPENAI_API_KEY",
        #     "prompt_template": None # already stored elsewhere
        # },
        "ancient_deepseek_api": {
            "type": "api",
            "api_key_env": "DEEPSEEK_API_KEY",
            "prompt_template": None # already stored elsewhere
        },
        # "ancient_openai_api": {
        #     "type": "api",
        #     "api_key_env": "OPENAI_API_KEY",
        #     "prompt_template": None # already stored elsewhere
        # },
        
    }
    
    # Performance
    RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")
    GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", 30))
    SSL_CERT_PATH = os.getenv("SSL_CERT_PATH")
    SSL_KEY_PATH = os.getenv("SSL_KEY_PATH")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "true").lower() == "true"

settings = Settings()