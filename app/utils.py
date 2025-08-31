import signal
import logging
from contextlib import contextmanager
from .exceptions import TimeoutException
from .config import settings  # Import settings here

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("llm_api.log")
        ]
    )

@contextmanager
def time_limit(seconds: int):
    def signal_handler(signum, frame):
        raise TimeoutException("Operation timed out")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def get_gpu_status():
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return f"{allocated:.2f}MB / {total:.2f}GB"
    return "N/A"