"""Configuration classes for image processing operations."""

class Config:
    def __init__(self, hardware_accelerated: bool = False, fattened_size_offset: int = 0, white_threshold: int = 200) -> None:
        self.hardware_accelerated = hardware_accelerated
        self.fattened_size_offset = fattened_size_offset
        self.white_threshold = 200