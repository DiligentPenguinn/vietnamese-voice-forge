class TimeoutException(Exception):
    pass

class ModelLoadException(Exception):
    pass

class GenerationException(Exception):
    pass

class GPUOutOfMemoryException(Exception):
    pass

class UnauthorizedIPException(Exception):
    pass